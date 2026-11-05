"""Per-handover dump of the exact tensors an LL / HL policy receives.

LL panels are rendered inside the policy-server process, after
``select_and_validate_obs`` and immediately before the model call, so a panel is
literally the dict the model receives: post state-normalization, post
key-dropping, post any server-side override of ``prev_front_rgb``. HL panels are
rendered at the Molmo2 invocation with the exact question string (including the
``Previous action:`` prefix) and the front frame Molmo2 was given. A client-side
panel (see ``toyblocks.eval.policy_input_debug``) shows what the eval *sends*,
which is a different question the moment the server rewrites or drops a key.

One LL PNG per handover -- i.e. per surviving ``ActionChunkBroker`` inference,
which is exactly the set of steps whose observation reaches the LL. One HL PNG
per Molmo2 forward (K-gated, so not every chunk). The row keys a consumer needs
(episode, sim time) are in the filename, so no cross-process sidecar is
required: the eval client only asks for the trace, and the wandb consolidator
globs the directory afterwards.
"""

from __future__ import annotations

import math
import pathlib
from typing import Any

import numpy as np
import torch

# Rendered left to right, so the aged frame sits next to the live one it is
# supposed to differ from.
_RGB_KEYS = ("front_rgb", "prev_front_rgb", "gripper_rgb")


def render_ll_input_panel(
    torch_obs: dict[str, Any],
    *,
    out_dir: str,
    env_idx: int,
    sim_time_ms: float,
    episode_id: int,
    raw_state: np.ndarray | None,
    dropped: list[str],
    summary: str,
) -> pathlib.Path:
    """Write one panel PNG for a single ``predict_action`` handover.

    Args:
        torch_obs: The validated obs dict about to be handed to the model.
        out_dir: Directory the PNG lands in (created if missing). This is the
            *server's* filesystem view, so it only reaches the eval client when
            the two share a filesystem.
        env_idx: Which batch element to slice. The unbatched client sends
            ``B == 1`` requests, so it is 0 there.
        sim_time_ms: Client-stamped sim time of the request (``hl_sim_time_ms``).
        episode_id: Client-stamped episode counter (``hl_episode_id``).
        raw_state: The pre-normalization ``agent_pos``, drawn next to the
            normalized vector so a normalizer mismatch is visible.
        dropped: Keys ``select_and_validate_obs`` removed, reported verbatim.
        summary: ``summarize_model_inputs(torch_obs)`` for the same dict.

    Returns:
        The path written.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels: list[tuple[str, Any]] = []

    rgb: dict[str, np.ndarray] = {}
    for key in _RGB_KEYS:
        if key in torch_obs:
            rgb[key] = _image_hwc_uint8(torch_obs[key], env_idx, key)

    for key, img in rgb.items():
        panels.append((f"{key}  {img.shape[1]}x{img.shape[0]}", _draw_rgb(img)))

    if "front_rgb" in rgb and "prev_front_rgb" in rgb:
        diff = np.abs(rgb["front_rgb"].astype(np.int16) - rgb["prev_front_rgb"].astype(np.int16))
        mean_abs = float(diff.mean())
        panels.append(
            (
                f"|front − prev_front|  mean={mean_abs:.2f}/255"
                + ("  IDENTICAL" if mean_abs == 0.0 else ""),
                _draw_gray(diff.max(axis=-1).astype(np.uint8), vmin=0, vmax=255),
            )
        )

    hm = torch_obs.get("target_heatmap")
    if hm is not None:
        hm_2d = _heatmap_hw(hm, env_idx)
        peak_y, peak_x = np.unravel_index(int(np.argmax(hm_2d)), hm_2d.shape)
        panels.append(
            (
                f"target_heatmap  peak=({peak_x}, {peak_y}) "
                f"max={hm_2d.max():.3f} sum={hm_2d.sum():.1f}",
                _draw_heatmap(hm_2d, peak_x, peak_y),
            )
        )
        if "front_rgb" in rgb:
            panels.append(
                (
                    "target_heatmap over front_rgb",
                    _draw_heatmap_overlay(rgb["front_rgb"], hm_2d, peak_x, peak_y),
                )
            )

    for key in sorted(k for k in torch_obs if k.endswith("_depth")):
        depth = _depth_hw(torch_obs[key], env_idx, key)
        panels.append(
            (
                f"{key}  min={depth.min():.3f} max={depth.max():.3f} "
                f"zeroed={100.0 * float((depth == 0).mean()):.1f}%",
                _draw_depth(depth),
            )
        )

    if "agent_pos" in torch_obs:
        norm_state = _state_vector(torch_obs["agent_pos"], env_idx, "agent_pos")
        raw_vec = _state_vector(raw_state, env_idx, "raw_state") if raw_state is not None else None
        panels.append(
            (
                f"agent_pos  dim={norm_state.size}"
                + ("" if raw_vec is not None else "  (no raw state sent)"),
                _draw_state(norm_state, raw_vec),
            )
        )

    n_img = len(panels)
    ncols = min(4, max(1, n_img))
    nrows = max(1, math.ceil(n_img / ncols))

    fig = plt.figure(figsize=(3.6 * ncols, 3.7 * nrows + 3.4))
    grid = fig.add_gridspec(nrows + 1, ncols, height_ratios=[*([3.4] * nrows), 3.0])
    for idx, (title, draw) in enumerate(panels):
        ax = fig.add_subplot(grid[idx // ncols, idx % ncols])
        draw(ax)
        ax.set_title(title, fontsize=8)

    task_name = torch_obs.get("task_name")
    if isinstance(task_name, (list, tuple)):
        task = repr(task_name[env_idx]) if env_idx < len(task_name) else f"<no entry {env_idx}>"
    else:
        task = repr(task_name)

    ax_text = fig.add_subplot(grid[nrows, :])
    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        "\n".join(
            [
                f"task_name[{env_idx}] = {task}",
                f"dropped by select_and_validate_obs: {dropped or 'none'}",
                "",
                summary,
            ]
        ),
        family="monospace",
        fontsize=7,
        va="top",
        ha="left",
        transform=ax_text.transAxes,
    )

    fig.suptitle(
        f"LL predict_action inputs · episode {episode_id} · sim_time {sim_time_ms:.0f} ms · "
        f"env {env_idx}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"ll_input_ep{int(episode_id):04d}_t{float(sim_time_ms):07.0f}ms.png"
    fig.savefig(png, dpi=110)
    plt.close(fig)
    return png


def render_hl_input_panel(
    *,
    front_rgb: Any,
    gripper_rgb: Any,
    question: str,
    prev_action: str,
    hl_output: str,
    out_dir: str,
    env_idx: int,
    sim_time_ms: float,
    episode_id: int,
) -> pathlib.Path:
    """Write one panel PNG for a single Molmo2 HL invocation.

    The images and the question string are what ``Molmo2HLPolicy.get_action``
    hands to the preprocessor — not a client-side reconstruction — so a missing
    gripper frame or ``Previous action:`` prefix is visible as a missing panel,
    not inferred.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    front = _obs_front_hwc_uint8(front_rgb, env_idx)
    gripper = _obs_front_hwc_uint8(gripper_rgb, env_idx)
    prev_label = prev_action if prev_action else "(none — first HL call of episode)"

    fig = plt.figure(figsize=(8.4, 6.6))
    grid = fig.add_gridspec(2, 2, height_ratios=[3.4, 2.8])
    ax_front = fig.add_subplot(grid[0, 0])
    ax_front.imshow(front)
    ax_front.set_xticks([])
    ax_front.set_yticks([])
    ax_front.set_title(f"front_rgb  {front.shape[1]}x{front.shape[0]}", fontsize=9)
    ax_grip = fig.add_subplot(grid[0, 1])
    ax_grip.imshow(gripper)
    ax_grip.set_xticks([])
    ax_grip.set_yticks([])
    ax_grip.set_title(f"gripper_rgb  {gripper.shape[1]}x{gripper.shape[0]}", fontsize=9)

    ax_text = fig.add_subplot(grid[1, :])
    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        "\n".join(
            [
                f"PREV  {prev_label}",
                "",
                f"PROMPT  {question}",
                "",
                f"OUT  {hl_output}",
            ]
        ),
        family="monospace",
        fontsize=8,
        va="top",
        ha="left",
        transform=ax_text.transAxes,
        wrap=True,
    )
    fig.suptitle(
        f"HL Molmo2 inputs · episode {episode_id} · sim_time {sim_time_ms:.0f} ms · "
        f"env {env_idx}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"hl_input_ep{int(episode_id):04d}_t{float(sim_time_ms):07.0f}ms.png"
    fig.savefig(png, dpi=110)
    plt.close(fig)
    return png


def _obs_front_hwc_uint8(x: Any, env_idx: int) -> np.ndarray:
    """One (H, W, 3) uint8 frame from a raw obs camera Molmo2 sees.

    Handles the layouts ``Molmo2HLPolicy.get_action`` accepts: HWC uint8, CHW
    float, and batched / sequenced variants. An unexpected layout raises — the
    panel is the diagnostic, so guessing a transpose would invent a mismatch.
    """
    if x is None:
        raise ValueError("HL input panel has no front_rgb")
    arr = _numpy(x)
    if arr.ndim == 5:
        if env_idx >= arr.shape[0]:
            raise IndexError(f"front_rgb: env_idx={env_idx} out of range for shape {arr.shape}")
        arr = arr[env_idx, -1]
    elif arr.ndim == 4:
        if env_idx >= arr.shape[0]:
            raise IndexError(f"front_rgb: env_idx={env_idx} out of range for shape {arr.shape}")
        arr = arr[env_idx]
    if arr.ndim != 3:
        raise ValueError(f"front_rgb: expected a 3-d frame after slicing, got shape {arr.shape}")
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        hwc = np.transpose(arr, (1, 2, 0))
        if hwc.dtype != np.uint8:
            peak = float(hwc.max()) if hwc.size else 0.0
            hwc = np.rint(hwc.astype(np.float64) * (255.0 if peak <= 1.01 else 1.0))
            return np.clip(hwc, 0, 255).astype(np.uint8)
        return hwc
    if arr.shape[-1] != 3:
        raise ValueError(f"front_rgb: expected HWC or CHW RGB, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        peak = float(arr.max()) if arr.size else 0.0
        scaled = np.rint(arr.astype(np.float64) * (255.0 if peak <= 1.01 else 1.0))
        return np.clip(scaled, 0, 255).astype(np.uint8)
    return arr


# ---------------------------------------------------------------------------
# Drawing (each returns a closure so the panel list stays declarative)
# ---------------------------------------------------------------------------


def _draw_rgb(img: np.ndarray):
    def draw(ax) -> None:
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])

    return draw


def _draw_gray(img: np.ndarray, *, vmin: float, vmax: float):
    def draw(ax) -> None:
        ax.imshow(img, cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])

    return draw


def _draw_heatmap(hm: np.ndarray, peak_x: int, peak_y: int):
    def draw(ax) -> None:
        ax.imshow(hm, cmap="jet")
        ax.plot(peak_x, peak_y, marker="+", ms=12, mew=1.6, color="white")
        ax.set_xticks([])
        ax.set_yticks([])

    return draw


def _draw_heatmap_overlay(img: np.ndarray, hm: np.ndarray, peak_x: int, peak_y: int):
    def draw(ax) -> None:
        ax.imshow(img)
        peak = float(hm.max())
        ax.imshow(
            hm,
            cmap="jet",
            alpha=np.clip(hm / peak, 0.0, 1.0) * 0.7 if peak > 0 else 0.0,
            extent=(0, img.shape[1], img.shape[0], 0),
        )
        ax.plot(peak_x, peak_y, marker="+", ms=12, mew=1.6, color="white")
        ax.set_xticks([])
        ax.set_yticks([])

    return draw


def _draw_depth(depth: np.ndarray):
    def draw(ax) -> None:
        im = ax.imshow(depth, cmap="viridis")
        ax.figure.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks([])
        ax.set_yticks([])

    return draw


def _draw_state(norm: np.ndarray, raw: np.ndarray | None):
    """Normalized vector on the left axis, raw on the right (different units)."""

    def draw(ax) -> None:
        x = np.arange(norm.size)
        ax.vlines(x - 0.12, 0.0, norm, color="tab:orange", lw=1.4)
        ax.plot(x - 0.12, norm, ls="none", marker="s", ms=4, color="tab:orange",
                label="normalized (sent to model)")
        ax.axhline(0.0, color="0.6", lw=0.6)
        ax.set_ylabel("normalized", color="tab:orange", fontsize=8)
        ax.tick_params(axis="y", labelcolor="tab:orange", labelsize=7)
        ax.set_xlabel("agent_pos dim", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(alpha=0.2)
        if raw is not None:
            xr = np.arange(raw.size)
            ax_raw = ax.twinx()
            ax_raw.vlines(xr + 0.12, 0.0, raw, color="tab:blue", lw=1.4)
            ax_raw.plot(xr + 0.12, raw, ls="none", marker="o", ms=4, color="tab:blue",
                        label="raw (as received)")
            ax_raw.set_ylabel("raw", color="tab:blue", fontsize=8)
            ax_raw.tick_params(axis="y", labelcolor="tab:blue", labelsize=7)
            handles = ax.get_legend_handles_labels()
            raw_handles = ax_raw.get_legend_handles_labels()
            ax.legend(handles[0] + raw_handles[0], handles[1] + raw_handles[1], fontsize=6)
        else:
            ax.legend(fontsize=6)

    return draw


# ---------------------------------------------------------------------------
# Slicing (strict: an unexpected layout is a bug in the caller, not a fallback)
# ---------------------------------------------------------------------------


def _numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def _leading_slice(arr: np.ndarray, env_idx: int, want_ndim: int, key: str) -> np.ndarray:
    """Drop leading batch/sequence axes down to *want_ndim*: env first, then last frame."""
    if arr.ndim < want_ndim:
        raise ValueError(f"{key}: expected at least {want_ndim} dims, got shape {arr.shape}")
    if arr.ndim > want_ndim:
        if env_idx >= arr.shape[0]:
            raise IndexError(f"{key}: env_idx={env_idx} out of range for shape {arr.shape}")
        arr = arr[env_idx]
    while arr.ndim > want_ndim:
        arr = arr[-1]
    return arr


def _image_hwc_uint8(x: Any, env_idx: int, key: str) -> np.ndarray:
    """One (H, W, 3) uint8 frame from a ``(B, S, 3, H, W)`` float model input.

    Rounded rather than truncated: the model input is a float32 ``uint8 / 255``,
    and multiplying that back by 255 lands a hair below the integer for most
    bytes, so truncation would render the whole panel one level dark.
    """
    chw = _leading_slice(_numpy(x), env_idx, 3, key)
    if chw.shape[0] != 3:
        raise ValueError(f"{key}: expected a 3-channel CHW frame, got shape {chw.shape}")
    scaled = np.rint(np.transpose(chw, (1, 2, 0)).astype(np.float64) * 255.0)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _heatmap_hw(x: Any, env_idx: int) -> np.ndarray:
    """Channel 0 of the target heatmap as (H, W) -- the 3 channels are identical."""
    chw = _leading_slice(_numpy(x), env_idx, 3, "target_heatmap")
    return chw[0]


def _depth_hw(x: Any, env_idx: int, key: str) -> np.ndarray:
    arr = _numpy(x)
    if arr.ndim > 2 and arr.shape[-3] == 1:
        arr = _leading_slice(arr, env_idx, 3, key)[0]
    else:
        arr = _leading_slice(arr, env_idx, 2, key)
    return arr


def _state_vector(x: Any, env_idx: int, key: str) -> np.ndarray:
    return _leading_slice(_numpy(x), env_idx, 1, key)
