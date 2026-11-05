"""Per-episode verification panels for what the policy actually receives.

Opt-in via ``--save_diagnostic_plots``. Two PNGs per episode start:

* one strip per depth camera: raw sim depth -> (when ``augment_depth`` is set)
  resize-only @224 before live aug -> the augmented depth on the wire ->
  the dataset's recorded depth for the same episode, their signed difference,
  and an overlaid histogram of both;
* one state panel: every ``agent_pos`` dim against the ``[q01, q99]`` training
  band the server normalizes it into, with out-of-band dims flagged.

The point is to make a unit / range / layout mismatch visible instead of it
showing up only as bad actions. So the panels are rendered from the arrays the
transform *actually emits* (:meth:`ManiflowObsTransform.transform` and
``_depth_inputs`` are called once for the whole batch and sliced per env),
never from a client-side re-derivation of what the model does with them. What
happens inside the encoder is measured on the server, on the real tensor:
``maniflow_serving_utils.instrument_ptmap_saturation`` logs the pointmap clamp
saturation from the encoder's own first forward pass.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from toyblocks.serving.maniflow_serving_utils import STATE_NORM_STAT_KEYS
from toyblocks.serving.obs_transforms import _depth_as_nhw
from toyblocks.serving.obs_transforms.depth_ptmap_bounds import z_bounds_for_depth_key
from toyblocks.serving.obs_transforms.maniflow import ManiflowObsTransform
from toyblocks.utils.dataset_provenance import camera_short_name
from toyblocks.utils.episode_dataset import decode_expert_depth_meters

logger = logging.getLogger(__name__)

_DEPTH_KEY_PREFIX = "observation.depth."

# The ``agent_pos`` layout, as ``(stats_key, label, width, is_normalized)``.
# Mirrors ``ManiFlowNormalizationMixin._normalize_state`` exactly, including
# that the gripper and pd_mode dims are passed through RAW despite
# ``observation.state.gripper_position`` existing in stats.json -- drawing a
# band around a dim the server never normalizes would invent a failure mode.
_STATE_BLOCKS_19 = (
    ("observation.state.joint_position", "arm", 7, True),
    ("observation.state.joint_velocity", "jvel", 7, True),
    ("observation.state.ee_position", "ee", 3, True),
    ("observation.state.gripper_position", "gripper", 1, False),
    (None, "pd_mode", 1, False),
)
_STATE_BLOCKS_9 = (
    ("observation.state.joint_position", "arm", 7, True),
    ("observation.state.gripper_position", "gripper", 1, False),
    (None, "pd_mode", 1, False),
)
_STATE_BLOCKS_8 = (
    ("observation.state.joint_position", "arm", 7, True),
    ("observation.state.gripper_position", "gripper", 1, False),
)
# Keyed by exact width, mirroring ``STATE_DIM_LAYOUTS``: the width is what
# decides the blocks, so a width with no entry has no labelling to guess at.
_STATE_BLOCKS_BY_DIM = {
    8: _STATE_BLOCKS_8,
    9: _STATE_BLOCKS_9,
    19: _STATE_BLOCKS_19,
}


def _depth_panel_imshow(
    fig: Any,
    ax: Any,
    depth_hw: np.ndarray,
    *,
    title: str,
) -> None:
    """Turbo depth column with per-panel vmin/vmax (valid pixels only)."""
    valid = depth_hw > 0
    vmin = float(depth_hw[valid].min()) if valid.any() else 0.0
    vmax = float(depth_hw[valid].max()) if valid.any() else 1.0
    im = ax.imshow(
        np.where(valid, depth_hw, np.nan),
        cmap="turbo",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)


class PolicyInputDebugRecorder:
    """Renders the policy-input verification panels for an episode start.

    Args:
        output_dir: Eval output dir; panels land in ``policy_input_debug/``.
        transform: The live :class:`ManiflowObsTransform` the eval feeds the
            policy server through. Used, not re-created, so the panel shows the
            same arrays inference sends.
        server_meta: The policy server's advertised metadata (``state_norm``).
    """

    def __init__(
        self,
        output_dir: str | Path,
        transform: Any,
        server_meta: dict[str, Any],
    ) -> None:
        # Both this and the state_norm check below are invariants rather than
        # user-facing errors: the caller capability-gates on the same two
        # conditions and skips the panels instead of constructing a recorder.
        if not isinstance(transform, ManiflowObsTransform):
            raise TypeError(
                "the policy-input panels render ManiFlow policy inputs "
                "(model-input depth, agent_pos bands) and need the "
                f"maniflow obs transform; got {type(transform).__name__}. Run "
                "with --obs_transform maniflow (or maniflow_blocks_table)."
            )
        self._transform = transform
        self._out_dir = Path(output_dir) / "policy_input_debug"
        self._out_dir.mkdir(parents=True, exist_ok=True)

        self._state_norm = server_meta.get("state_norm") or {}
        if not self._state_norm:
            raise ValueError(
                "the policy-input panels need the server's state "
                "normalization bands, but its metadata advertises no "
                f"'state_norm' (keys: {sorted(server_meta)}). The state panel "
                "compares each raw dim against the [q01, q99] band the server "
                "applies; without it there is nothing to compare against. "
                "Serve with a build of serve_maniflow_policy.py that publishes "
                "state_norm."
            )
        missing = [k for k in self._state_norm if k not in STATE_NORM_STAT_KEYS]
        if missing:
            logger.warning("server advertised unrecognized state_norm keys: %s", missing)

    # ------------------------------------------------------------------

    def capture(
        self,
        obs: dict[str, Any],
        env_indices: list[int],
        episode_indices: dict[int, int],
        setups: list[dict | None],
    ) -> dict[int, dict[str, str]]:
        """Render panels for ``env_indices``; returns ``{env_idx: {column: png}}``.

        ``obs`` is the live (possibly torch / batched) observation dict;
        ``episode_indices`` maps env -> the per-env episode counter used in the
        filename, and ``setups`` is the queue setup per env (source of the
        dataset depth clip and the recorded frame-0 joint state).
        """
        from toyblocks.utils.common_utils import to_numpy

        obs_np = to_numpy(obs)
        wire = self._transform.transform(obs_np)

        out: dict[int, dict[str, str]] = {}
        for env_idx in env_indices:
            setup = setups[env_idx] if env_idx < len(setups) else None
            ep_idx = episode_indices.get(env_idx, 0)
            base = f"env{env_idx}_episode_{ep_idx:04d}"
            cols: dict[str, str] = {}

            for cam_row, depth_key in enumerate(self._transform.depth_keys):
                cam = getattr(self._transform, ManiflowObsTransform.DEPTH_KEY_CAMERAS[depth_key])
                png = self._out_dir / f"{base}_{cam}.png"
                self._render_depth_panel(
                    png,
                    obs_np=obs_np,
                    wire=wire,
                    env_idx=env_idx,
                    cam=cam,
                    depth_key=depth_key,
                    cam_row=cam_row,
                    setup=setup,
                )
                cols[f"policy_input_{_panel_column(cam)}"] = str(png)

            state_png = self._out_dir / f"{base}_state.png"
            self._render_state_panel(state_png, wire=wire, env_idx=env_idx, setup=setup)
            cols["policy_input_state"] = str(state_png)
            out[env_idx] = cols
        return out

    # ------------------------------------------------------------------
    # depth panel
    # ------------------------------------------------------------------

    def _render_depth_panel(
        self,
        png: Path,
        *,
        obs_np: dict[str, Any],
        wire: dict[str, Any],
        env_idx: int,
        cam: str,
        depth_key: str,
        cam_row: int,
        setup: dict | None,
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        raw_batch = _depth_as_nhw(self._transform.get_depth(obs_np, cam))
        raw = raw_batch[env_idx] if raw_batch.ndim == 3 else raw_batch
        model_depth = np.asarray(wire[depth_key])[env_idx, 0, 0]  # (H, W) meters
        fx, fy, cx, cy = np.asarray(wire["camera_intrinsics"])[env_idx, 0, cam_row]
        dataset_depth = _dataset_depth_meters(setup, cam, model_depth.shape)

        augment_depth = getattr(self._transform, "augment_depth", None)
        z_min, z_max = z_bounds_for_depth_key(
            depth_key,
            self._transform._per_camera_ptmap_bounds_for_masking(),
        )
        resized_plane: np.ndarray | None = None
        ejbl_note = ""
        if augment_depth:
            resized_batch = self._transform.process_depth(
                raw_batch,
                self._transform.image_size,
                max_depth_m=self._transform.depth_max_m,
            )
            resized_plane = resized_batch[env_idx] if resized_batch.ndim == 3 else resized_batch
            both_aug = (model_depth > 0) & (resized_plane > 0)
            mae_ejbl = (
                float(np.abs(model_depth - resized_plane)[both_aug].mean())
                if both_aug.any()
                else float("nan")
            )
            ejbl_note = (
                f"  ·  depth_aug={augment_depth!r}  "
                f"MAE(aug−resize_only)={mae_ejbl:.4f} m"
            )
        ejbl_note += f"  ·  ptmap Z mask [{z_min:g}, {z_max:g}] m"

        n_cols = 2 + (1 if resized_plane is not None else 0) + (2 if dataset_depth is not None else 0) + 1
        fig, axes = plt.subplots(1, n_cols, figsize=(3.4 * n_cols, 3.9))
        fig.suptitle(
            f"policy input · env {env_idx} · {cam} · {depth_key}   "
            f"K(224)=[fx {fx:.1f}, fy {fy:.1f}, cx {cx:.1f}, cy {cy:.1f}]"
            f"{ejbl_note}",
            fontsize=10,
        )
        ax_hist = axes[-1]
        col = 0

        finite = np.isfinite(raw)
        pct_inf = 100.0 * (1.0 - finite.mean())
        vmin = float(raw[finite].min()) if finite.any() else 0.0
        vmax = float(raw[finite].max()) if finite.any() else 1.0
        im = axes[col].imshow(np.where(finite, raw, np.nan), cmap="turbo", vmin=vmin, vmax=vmax)
        axes[col].set_title(
            f"raw sim depth {raw.shape[0]}x{raw.shape[1]}\n"
            f"{vmin:.3f}–{vmax:.3f} m · {pct_inf:.1f}% non-finite",
            fontsize=8,
        )
        fig.colorbar(im, ax=axes[col], fraction=0.046)
        col += 1

        if resized_plane is not None:
            _depth_panel_imshow(
                fig,
                axes[col],
                resized_plane,
                title=(
                    f"resize only @224 (no {augment_depth})\n"
                    f"clean grid before live depth aug"
                ),
            )
            col += 1

        zeroed = 100.0 * float((model_depth == 0).mean())
        valid = model_depth > 0
        aug_title = (
            f"model input @224 + {augment_depth}\n"
            if augment_depth
            else f"model input depth {model_depth.shape[0]}x{model_depth.shape[1]}\n"
        )
        invalid_note = f"outside ptmap Z [{z_min:g}, {z_max:g}] m or coarse invalid"
        _depth_panel_imshow(
            fig,
            axes[col],
            model_depth,
            title=(
                f"{aug_title}"
                f"{zeroed:.1f}% zeroed ({invalid_note})"
            ),
        )
        col += 1

        ax_hist.hist(
            model_depth[valid].ravel(),
            bins=64,
            alpha=0.6,
            label="model input" if augment_depth else "eval (model input)",
            color="tab:blue",
        )
        if resized_plane is not None:
            ro_valid = resized_plane > 0
            if ro_valid.any():
                ax_hist.hist(
                    resized_plane[ro_valid].ravel(),
                    bins=64,
                    alpha=0.5,
                    label="resize only @224",
                    color="tab:green",
                )

        if dataset_depth is not None:
            ds_valid = dataset_depth > 0
            _depth_panel_imshow(
                fig,
                axes[col],
                dataset_depth,
                title="dataset depth (frame 0, meters)",
            )
            col += 1

            both = valid & ds_valid
            diff = np.where(both, model_depth - dataset_depth, np.nan)
            lim = float(np.nanmax(np.abs(diff))) if both.any() else 1.0
            im = axes[col].imshow(diff, cmap="bwr", vmin=-lim, vmax=lim)
            mae = float(np.abs(diff[both]).mean()) if both.any() else float("nan")
            axes[col].set_title(
                f"model − dataset (m)\nMAE {mae:.4f} m over {100 * both.mean():.0f}% of pixels",
                fontsize=8,
            )
            fig.colorbar(im, ax=axes[col], fraction=0.046)
            col += 1

            ax_hist.hist(
                dataset_depth[ds_valid].ravel(),
                bins=64,
                alpha=0.6,
                label="dataset",
                color="tab:orange",
            )

        ax_hist.set_title("depth distribution (valid samples)", fontsize=8)
        ax_hist.set_xlabel("meters", fontsize=8)
        ax_hist.legend(fontsize=7)

        for ax in axes[:-1]:
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout()
        fig.savefig(png, dpi=110)
        plt.close(fig)

    # ------------------------------------------------------------------
    # state panel
    # ------------------------------------------------------------------

    def _render_state_panel(
        self,
        png: Path,
        *,
        wire: dict[str, Any],
        env_idx: int,
        setup: dict | None,
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        state = np.asarray(wire["agent_pos"])[env_idx, 0].astype(np.float32)
        dataset_state = _dataset_state(setup)
        rows = _state_rows(state, self._state_norm, dataset_state)

        fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 1.6))
        n_out = 0
        for i, row in enumerate(rows):
            y = len(rows) - 1 - i
            if row["q01"] is None:
                ax.text(0.5, y, "raw (server does not normalize this dim)",
                        fontsize=7, color="gray", va="center", ha="center",
                        transform=ax.get_yaxis_transform())
            else:
                lo, hi = row["q01"], row["q99"]
                out_of_band = not (lo <= row["value"] <= hi)
                n_out += int(out_of_band)
                ax.plot([lo, hi], [y, y], color="tab:blue", lw=6, alpha=0.25, solid_capstyle="butt")
                ax.plot([lo, lo], [y - 0.25, y + 0.25], color="tab:blue", lw=1)
                ax.plot([hi, hi], [y - 0.25, y + 0.25], color="tab:blue", lw=1)
                ax.scatter(
                    [row["value"]], [y],
                    color="tab:red" if out_of_band else "tab:green",
                    zorder=3, s=28,
                )
                ax.annotate(
                    f"{row['value']:.4f}",
                    (row["value"], y), textcoords="offset points", xytext=(6, 6),
                    fontsize=7, color="tab:red" if out_of_band else "black",
                )
            if row["dataset"] is not None:
                ax.scatter([row["dataset"]], [y], marker="|", color="tab:purple", s=110, zorder=2)

        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r["label"] for r in reversed(rows)], fontsize=8)
        ax.set_xlabel("raw value (blue band = [q01, q99] training range; purple = dataset frame 0)")
        ax.set_title(
            f"agent_pos ({state.shape[-1]} dims) · env {env_idx} · "
            f"{n_out} dim(s) outside the training band"
            + ("" if dataset_state else "  ·  no dataset state in setup"),
            fontsize=10,
        )
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        fig.savefig(png, dpi=110)
        plt.close(fig)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _panel_column(cam: str) -> str:
    """``front_camera`` -> ``front``: the wandb column suffix for a camera."""
    return cam.replace("_camera", "") or cam


def _state_rows(
    state: np.ndarray,
    state_norm: dict[str, Any],
    dataset_state: dict[str, float],
) -> list[dict[str, Any]]:
    """One row per ``agent_pos`` dim: label, raw value, band.

    The band is ``None`` for dims the server passes through raw (gripper,
    pd_mode), so the panel does not draw a constraint the normalizer never
    applies. See ``ManiFlowNormalizationMixin._normalize_state``.
    """
    dim = int(state.shape[-1])
    blocks = _STATE_BLOCKS_BY_DIM.get(dim)
    if blocks is None:
        raise ValueError(
            f"agent_pos has {dim} dims but the known ManiFlow state layouts are "
            f"{sorted(_STATE_BLOCKS_BY_DIM)}; cannot label its dims for the "
            "debug panel."
        )

    rows: list[dict[str, Any]] = []
    offset = 0
    for stats_key, label, width, normalized in blocks:
        band = state_norm.get(stats_key) if stats_key else None
        for j in range(width):
            value = float(state[offset + j])
            lo = hi = None
            if normalized:
                if band is None:
                    raise ValueError(
                        f"the server normalizes the {label!r} block with "
                        f"{stats_key!r} but advertised no state_norm entry for "
                        f"it (has: {sorted(state_norm)}); the panel cannot show "
                        "the band this dim is mapped into."
                    )
                lo = float(band["q01"][j])
                hi = float(band["q99"][j])
            row_label = f"{label}[{j}]" if width > 1 else label
            rows.append({
                "label": row_label,
                "value": value,
                "q01": lo,
                "q99": hi,
                "dataset": dataset_state.get(row_label),
            })
        offset += width
    return rows


def _dataset_state(setup: dict | None) -> dict[str, float]:
    """Recorded frame-0 joint / gripper state from a queue setup, by row label."""
    if not setup:
        return {}
    out: dict[str, float] = {}
    joints = setup.get("robot_joint_positions")
    if joints:
        for j, v in enumerate(joints[:7]):
            out[f"arm[{j}]"] = float(v)
    gripper = setup.get("robot_gripper_position")
    if gripper is not None:
        out["gripper"] = float(gripper)
    return out


def _dataset_depth_meters(
    setup: dict | None,
    cam: str,
    target_hw: tuple[int, int],
) -> np.ndarray | None:
    """The dataset's frame-0 depth for ``cam``, in meters at ``target_hw``.

    ``None`` when the episode's setup carries no depth clip for this camera (a
    dataset recorded without depth) or when the clip is the non-metric
    3-channel encoding ``lerobot_recorder`` writes, which has no absolute scale
    to compare against. Either way the comparison columns are dropped and the
    panel still renders: this is a diagnostic, and killing a rollout over one
    unusable clip loses the run it was meant to explain.
    """
    if not setup:
        return None
    clips = setup.get("expert_depth_videos") or {}
    entry = None
    for key, value in sorted(clips.items()):
        if camera_short_name(key, _DEPTH_KEY_PREFIX) == cam:
            entry = value
            break
    if entry is None:
        if clips:
            logger.warning(
                "episode setup has depth clips %s but none for camera %r; "
                "the policy-input panel will omit the dataset comparison.",
                sorted(clips),
                cam,
            )
        return None

    # Checked here rather than caught from decode_expert_depth_meters, which
    # raises the same ValueError for a failed seek and an undecodable stream --
    # real failures that must still surface. The check also costs no decode.
    if entry.get("channels") != 1:
        logger.warning(
            "depth clip %r for camera %r has channels=%s, not the 1-channel "
            "uint16-millimetre encoding; omitting the dataset comparison.",
            entry.get("path"),
            cam,
            entry.get("channels"),
        )
        return None

    depth = decode_expert_depth_meters(entry)
    if depth.shape != target_hw:
        import torch
        import torch.nn.functional as F

        t = torch.from_numpy(depth)[None, None]
        depth = F.interpolate(t, size=target_hw, mode="nearest")[0, 0].numpy()
    return depth
