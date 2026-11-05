"""Shared utilities and classes for ManiFlow-based policy serving.

``hvla`` is imported lazily (inside :func:`load_maniflow_policy_common`) rather
than at module scope: everything else here -- the obs contract derivation, the
normalization mixin, ``select_and_validate_obs``, ``postprocess_actions`` -- is
plain numpy/torch and is covered by ``tests/serving/`` on CPU boxes where the
maniflow extra is not installed.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import pathlib
import re
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

if TYPE_CHECKING:
    from hvla.model.modules.maniflow.flow_policy import ManiFlowMultiModalPolicy

logger = logging.getLogger(__name__)

# The model's square input grid (not the camera resolution, which stays separate as
# orig_cam_w/orig_cam_h). Asserted against the checkpoint's shape_meta at load.
MANIFLOW_IMAGE_SIZE = 224
# Depth cutoff used by training's decode path. Keep synchronized with
# hvla/dataloader/franka_dataset.py:MAX_DEPTH_M.
TRAIN_MAX_DEPTH_M = 2.5


def depth_aug_supported(profile: str) -> bool:
    """True when live eval depth aug for ``profile`` (currently ``ejbl``) is importable."""
    name = str(profile).strip().lower()
    if name != "ejbl":
        return False
    try:
        from hvla.training.trainer_utils.depth_aug.elastic_jbl_depth import (  # noqa: F401
            add_ejbl_depth_noise,
        )
        from hvla.training.trainer_utils.depth_bake_profiles import resolve_profile
    except ImportError:
        return False
    try:
        resolved = resolve_profile("ejbl")
    except Exception:
        return False
    return resolved is not None


def require_depth_aug_supported(profile: str) -> None:
    """Fail fast when ``--depth-aug`` was requested but hvla lacks EJBL profiles."""
    name = str(profile).strip().lower()
    if depth_aug_supported(name):
        return
    raise RuntimeError(
        f"--depth-aug {name!r} needs hvla.training.trainer_utils.depth_bake_profiles, which the "
        "installed maniflow-tb checkout does not ship. Omit --depth-aug, or point the maniflow-tb "
        "submodule at a revision that provides the EJBL depth-augmentation profiles."
    )


# Obs keys the policy server synthesizes from the standard eval client payload
# (``front_rgb`` + ``target_heatmap``). Omitted from advertised ``obs_keys`` so
# every compared checkpoint uses the same client-side transform.
SERVER_DERIVED_OBS_KEYS = frozenset({"target_front_crop_rgb"})


def client_advertised_obs_keys(obs_shapes: dict[str, Any]) -> list[str]:
    """``obs_keys`` metadata: declared model inputs minus server-derived ones."""
    return sorted(k for k in obs_shapes if k not in SERVER_DERIVED_OBS_KEYS)


def parse_target_front_crop(value: Any) -> int:
    """Bottom-share percent for target-centered front crop, or -1 to disable.

    Matches ``hvla.dataloader.franka_dataset.parse_target_front_crop``.
    """
    if value is None:
        return -1
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("", "none", "null"):
            return -1
        value = int(float(text))
    else:
        value = int(value)
    if value == -1:
        return -1
    if value < 0 or value >= 100:
        raise ValueError(
            f"use_target_front_crop must be -1 (off) or 0-99 (bottom-share percent), "
            f"got {value}"
        )
    return value


def crop_eval_active(args: Any, server_meta: dict[str, Any]) -> bool:
    """Whether the eval client should crop front RGB and inject native ``target_xy``."""
    if bool(getattr(args, "enable_crop", False)):
        return True
    return int(server_meta.get("use_target_front_crop", -1)) >= 0


def crop_centered_on_target(
    img: np.ndarray,
    cx: float,
    cy: float,
    size: int,
    bottom_pct: int,
) -> np.ndarray:
    """Square ``size x size`` crop from HWC uint8 ``img`` (training ``_crop_centered``)."""
    h_img, w_img = img.shape[:2]
    top_px = int(round(size * (100 - bottom_pct) / 100.0))
    x0 = int(round(float(cx))) - size // 2
    y0 = int(round(float(cy))) - top_px

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x0 + size - w_img)
    pad_bottom = max(0, y0 + size - h_img)

    crop = img[max(0, y0) : min(h_img, y0 + size), max(0, x0) : min(w_img, x0 + size)]
    if pad_left or pad_top or pad_right or pad_bottom:
        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))
        if img.ndim == 3:
            pad_width = pad_width + ((0, 0),)
        crop = np.pad(crop, pad_width, mode="constant", constant_values=0)
    if crop.shape[0] != size or crop.shape[1] != size:
        out_shape = (size, size, img.shape[2]) if img.ndim == 3 else (size, size)
        out = np.zeros(out_shape, dtype=img.dtype)
        h, w = min(size, crop.shape[0]), min(size, crop.shape[1])
        out[:h, :w] = crop[:h, :w]
        return out
    return crop


def _heatmap_peak_xy_batch(hm: torch.Tensor) -> list[tuple[float, float] | None]:
    """Per-batch (x, y) argmax on channel 0 of a target heatmap."""
    if hm.ndim == 5:
        ch = hm[:, 0, 0]
    elif hm.ndim == 4 and hm.shape[1] == 3:
        ch = hm[:, 0]
    elif hm.ndim == 4:
        ch = hm[:, 0, 0]
    else:
        raise ValueError(f"unexpected target_heatmap shape {tuple(hm.shape)}")
    peaks: list[tuple[float, float] | None] = []
    for i in range(ch.shape[0]):
        flat = ch[i]
        peak_val = float(flat.max().item())
        if peak_val <= 0.0:
            peaks.append(None)
            continue
        idx = int(flat.argmax().item())
        y, x = divmod(idx, flat.shape[1])
        peaks.append((float(x), float(y)))
    return peaks


def inject_target_front_crop_rgb(
    torch_obs: dict[str, Any],
    bottom_pct: int,
) -> None:
    """Build ``target_front_crop_rgb`` on the server from ``front_rgb``.

    Crop-trained checkpoints read this view during training; the ToyBlocks eval
    client does not emit it so compared models share one obs transform. The crop
    is taken from the same ``front_rgb`` tensor and centered on the
    ``target_heatmap`` peak (224 grid), matching the oracle spatial goal.
    """
    if bottom_pct < 0:
        return
    if "target_front_crop_rgb" in torch_obs:
        return
    front = torch_obs.get("front_rgb")
    if front is None or not torch.is_tensor(front):
        raise ValueError(
            "checkpoint sets use_target_front_crop but the batch has no front_rgb tensor"
        )
    if front.ndim != 5 or front.shape[2] != 3:
        raise ValueError(
            f"front_rgb must be (B, 1, 3, H, W) to derive target_front_crop_rgb, "
            f"got {tuple(front.shape)}"
        )
    batch_size, _, _, height, width = front.shape
    if height != width:
        raise ValueError(
            f"target_front_crop expects square front_rgb, got H={height} W={width}"
        )
    crop_size = height
    hm = torch_obs.get("target_heatmap")
    if hm is None or not torch.is_tensor(hm):
        raise ValueError(
            "checkpoint sets use_target_front_crop but the batch has no "
            "target_heatmap tensor; the server derives the crop from the "
            "heatmap peak so oracle / target-conditioned HL must be enabled."
        )
    peaks = _heatmap_peak_xy_batch(hm)
    if len(peaks) != batch_size:
        raise ValueError(
            f"target_heatmap batch {len(peaks)} != front_rgb batch {batch_size}"
        )

    device, dtype = front.device, front.dtype
    crops: list[torch.Tensor] = []
    for i in range(batch_size):
        hwc = (
            front[i, 0]
            .permute(1, 2, 0)
            .detach()
            .float()
            .cpu()
            .numpy()
            * 255.0
        )
        img = np.clip(hwc, 0.0, 255.0).astype(np.uint8)
        peak = peaks[i]
        if peak is None:
            crop = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        else:
            crop = crop_centered_on_target(
                img, peak[0], peak[1], crop_size, bottom_pct
            )
        crops.append(_pil_to_tensor(crop, size=crop_size).to(device=device, dtype=dtype))
    torch_obs["target_front_crop_rgb"] = torch.stack(crops, dim=0).unsqueeze(1)


def _target_xy_batch_numpy(target_xy: Any, batch_size: int) -> np.ndarray:
    """``(B, 2)`` native-pixel targets; ``None`` means no target (NaN rows -> zero crop)."""
    if target_xy is None:
        return np.full((batch_size, 2), np.nan, dtype=np.float32)
    arr = np.asarray(target_xy, dtype=np.float32)
    if arr.ndim == 1 and arr.shape[0] == 2:
        arr = arr.reshape(1, 2)
    if arr.shape[0] == 1 and batch_size > 1:
        arr = np.repeat(arr, batch_size, axis=0)
    if arr.shape[0] != batch_size:
        raise ValueError(f"target_xy batch {arr.shape[0]} != native_front_rgb batch {batch_size}")
    return arr


def inject_target_front_crop_from_native(
    torch_obs: dict[str, Any],
    bottom_pct: int,
    native_nhwc: np.ndarray | torch.Tensor,
    target_xy: Any,
) -> None:
    """Build ``target_front_crop_rgb`` from the native front frame + pixel ``(x, y)`` targets.

    The ``--hl model`` server path: the client sends ``native_front_rgb``
    (``pass_native_front_for_server_crop``) and the crop is centered on the HL's
    native-resolution target, matching training's ``_crop_centered`` rather than
    the 224-grid approximation of :func:`inject_target_front_crop_rgb`. Rows
    with a non-finite target get an all-zero crop.
    """
    if bottom_pct < 0:
        return
    if "target_front_crop_rgb" in torch_obs:
        return
    if isinstance(native_nhwc, torch.Tensor):
        native = native_nhwc.detach().cpu().numpy()
    else:
        native = np.asarray(native_nhwc)
    if native.ndim == 3:
        native = native[None, ...]
    if native.ndim != 4 or native.shape[-1] != 3:
        raise ValueError(f"native_front_rgb must be (B, H, W, 3) uint8, got {tuple(native.shape)}")
    batch_size = native.shape[0]
    xy_batch = _target_xy_batch_numpy(target_xy, batch_size)
    crop_size = MANIFLOW_IMAGE_SIZE
    front = torch_obs.get("front_rgb")
    if front is not None and torch.is_tensor(front) and front.ndim == 5:
        device, dtype = front.device, front.dtype
    else:
        device, dtype = torch.device("cpu"), torch.float32

    crops: list[torch.Tensor] = []
    for i in range(batch_size):
        img = native[i]
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        cx, cy = float(xy_batch[i, 0]), float(xy_batch[i, 1])
        if not np.isfinite(cx) or not np.isfinite(cy):
            crop = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        else:
            crop = crop_centered_on_target(img, cx, cy, crop_size, bottom_pct)
        crops.append(_pil_to_tensor(crop, size=crop_size).to(device=device, dtype=dtype))
    torch_obs["target_front_crop_rgb"] = torch.stack(crops, dim=0).unsqueeze(1)


def _pil_to_tensor(img: np.ndarray, size: int = MANIFLOW_IMAGE_SIZE) -> torch.Tensor:
    """HWC uint8 numpy -> [3, size, size] float32 tensor in [0,1]."""
    from torchvision.transforms.functional import to_tensor
    pil = Image.fromarray(img)
    pil = pil.resize((size, size), Image.BILINEAR)
    return to_tensor(pil)  # [3, H, W] float32 [0,1]

def _summarize_value(x: Any) -> str:
    """Exact, compact one-line summary of a single model input.

    Reports shape/dtype/device plus min/max/mean for tensors and arrays so a
    debug log shows *exactly* what numbers a model is being fed (not just that a
    key is present). Lists (e.g. ``task_name``) are printed verbatim.
    """
    if x is None:
        return "None"
    if isinstance(x, (list, tuple)):
        return f"list[{len(x)}]={list(x)!r}"
    if torch.is_tensor(x):
        xf = x.detach().float()
        dtype = str(x.dtype).replace("torch.", "")
        return (
            f"Tensor{tuple(x.shape)} {dtype} dev={x.device} "
            f"min={xf.min().item():.4f} max={xf.max().item():.4f} mean={xf.mean().item():.4f}"
        )
    if isinstance(x, np.ndarray):
        return (
            f"ndarray{tuple(x.shape)} {x.dtype} "
            f"min={float(x.min()):.4f} max={float(x.max()):.4f} mean={float(x.mean()):.4f}"
        )
    return f"{type(x).__name__}={x!r}"


def _summarize_heatmap_peaks(hm: Any) -> str:
    """Per-sample peak location of a target heatmap, for spatial-goal debugging.

    Accepts (3,H,W), (B,3,H,W) or (B,S,3,H,W). Uses channel 0 (the 3 channels are
    identical Gaussians) and reports the argmax pixel ``(x,y)``, peak value and
    mass ``sum`` for each leading (batch/seq) index, so you can confirm the
    heatmap peak lands on the object the HL policy intended.
    """
    if hm is None:
        return "None"
    t = hm.detach().float().cpu() if torch.is_tensor(hm) else torch.as_tensor(np.asarray(hm)).float()
    if t.ndim == 5:  # (B, S, 3, H, W)
        flat = t.reshape(t.shape[0] * t.shape[1], *t.shape[2:])
    elif t.ndim == 4:  # (B, 3, H, W)
        flat = t
    elif t.ndim == 3:  # (3, H, W)
        flat = t.unsqueeze(0)
    else:
        return f"unexpected ndim={t.ndim} shape={tuple(t.shape)}"
    peaks = []
    for i in range(flat.shape[0]):
        ch = flat[i, 0]
        h, w = ch.shape
        idx = int(ch.argmax().item())
        y, x = divmod(idx, w)
        peaks.append(f"(x={x},y={y},peak={ch.max().item():.3f},sum={ch.sum().item():.1f})")
    return "[" + ", ".join(peaks) + "]"


def summarize_model_inputs(torch_obs: dict[str, Any]) -> str:
    """Multi-line breakdown of every key fed to a ManiFlow LL model.

    ``target_heatmap`` additionally gets a per-sample peak-pixel summary so the
    spatial goal is legible, not just its tensor stats.
    """
    lines = []
    for k in sorted(torch_obs):
        lines.append(f"    {k}: {_summarize_value(torch_obs[k])}")
        if k == "target_heatmap":
            lines.append(f"        peaks: {_summarize_heatmap_peaks(torch_obs[k])}")
    return "\n".join(lines)


# sim2real ``GOAL_ACTION_EXAMPLES`` / ``format_language_goal`` contract. Longest
# first so ``place on table`` is not parsed as ``place`` plus leftover words.
_KNOWN_HL_ACTIONS: tuple[str, ...] = (
    "place on table",
    "flip backward",
    "flip forward",
    "flip right",
    "flip left",
    "rotate ccw",
    "rotate cw",
    "pick",
    "place",
    "done",
)
# The only known action that may omit a point (sim2real robot loop: bare phrase
# + no heatmap). ``done`` is handled separately by the HL adapter.
PLACE_ON_TABLE_ACTION = "place on table"

# Molmo2 form: ``pick <320, 180>``. sim2real ``format_language_goal`` form:
# ``pick 320,180`` (no brackets, coords trailing). Optional ``.0`` fractions
# are accepted on each component (autoregressive HL output).
_UV_NUM = r"\d+(?:\.\d+)?"
_UV_BRACKET_RE = re.compile(rf"<\s*({_UV_NUM})\s*,\s*({_UV_NUM})\s*>")
_UV_TRAILING_RE = re.compile(rf"(?:^|\s)({_UV_NUM})\s*,\s*({_UV_NUM})\s*$")


def _parse_coords(s: str) -> np.ndarray | None:
    """First ``<u,v>`` match, else a trailing ``u,v``; otherwise None."""
    m = _UV_BRACKET_RE.search(s)
    if m:
        return np.array(
            [round(float(m.group(1))), round(float(m.group(2)))],
            dtype=np.float32,
        )
    m = _UV_TRAILING_RE.search(s)
    if m:
        return np.array(
            [round(float(m.group(1))), round(float(m.group(2)))],
            dtype=np.float32,
        )
    return None


def _strip_coord_tokens(s: str) -> str:
    """Remove ``<u,v>`` and trailing ``u,v`` tokens; leave action words intact."""
    stripped = _UV_BRACKET_RE.sub("", s)
    stripped = _UV_TRAILING_RE.sub("", stripped)
    return re.sub(r"\s+", " ", stripped).strip()


def _parse_hl_action(action_str: str) -> tuple[str, np.ndarray | None]:
    """Split an HL string into a known (or leftover) action phrase and optional uv.

    Known multi-word actions are matched as a prefix so ``place on table`` stays
    one phrase. Coords may be ``<u,v>`` or trailing ``u,v``. A known-action
    prefix whose remainder is not empty after stripping coords is not accepted
    as that action (leftover words stay on the generic strip path).
    """
    raw = (action_str or "").strip()
    lowered = raw.lower()
    for action in _KNOWN_HL_ACTIONS:
        if lowered == action:
            return action, None
        if lowered.startswith(f"{action} ") or lowered.startswith(f"{action}<"):
            remainder = raw[len(action):]
            if _strip_coord_tokens(remainder):
                continue
            return action, _parse_coords(remainder)
    return _strip_coord_tokens(raw), _parse_coords(raw)


def _parse_target_xy(action_str: str) -> np.ndarray | None:
    """Parse the first ``<u,v>`` or trailing ``u,v`` from an HL action string."""
    return _parse_hl_action(action_str)[1]


def _strip_coords(action_str: str) -> str:
    """Strip coordinate tokens from an HL string, leaving the action phrase.

    ``place on table`` is a single known action and is never treated as coords.
    """
    return _parse_hl_action(action_str)[0]


def format_task_name(lang: str, target_xy: np.ndarray | None, language_mode: str) -> str:
    """Build the LL policy's ``task_name`` in the format the checkpoint was trained on.

    Mirrors sim2real ``goal_utils.format_language_goal`` / HVLA
    ``franka_dataset.py``: ``action_only`` is the bare action phrase,
    ``action_goal`` appends native front-camera pixels as
    ``f"{action} {x:.0f},{y:.0f}"`` (no brackets). Missing or non-finite pixels
    leave the bare phrase -- the same no-point contract as sim2real
    (``if u is None or v is None: return action_str``). The mode comes from the
    served checkpoint (``model_input_spec['language_mode']``), never from a guess.
    """
    if language_mode == "action_only":
        return lang
    if language_mode != "action_goal":
        raise ValueError(
            f"unsupported language_mode {language_mode!r}; expected 'action_goal' "
            "or 'action_only'"
        )
    if target_xy is None or not np.all(np.isfinite(np.asarray(target_xy, dtype=np.float32))):
        return lang
    return f"{lang} {float(target_xy[0]):.0f},{float(target_xy[1]):.0f}"


# Default Gaussian sigma for target heatmaps, in 14-patch units on the model's
# input grid (sigma_px = sigma * S/14). Matches training checkpoints (e.g. sigma0.8).
DEFAULT_TARGET_SIGMA = 0.8

def _build_heatmap(target_xy: np.ndarray, orig_cam_w: float,
                   orig_cam_h: float, target_sigma: float) -> np.ndarray:
    """Build (3, S, S) Gaussian heatmap from pixel coords in original cam resolution."""
    S = MANIFLOW_IMAGE_SIZE
    heatmap = np.zeros((S, S), dtype=np.float32)
    if not np.isnan(target_xy[0]):
        x = target_xy[0] * S / orig_cam_w
        y = target_xy[1] * S / orig_cam_h
        sigma_px = target_sigma * (S / 14.0)
        yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
        heatmap = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma_px ** 2))
    return np.stack([heatmap, heatmap, heatmap], axis=0)


# Stats keys ``ManiFlowNormalizationMixin._normalize_state`` reads, in the order
# their blocks appear in ``agent_pos``. Advertised to the client so a debug panel
# can show each raw state dim against the band it is normalized into instead of
# re-deriving the bands from a stats.json the client may not even have.
STATE_NORM_STAT_KEYS: tuple[str, ...] = (
    "observation.state.joint_position",
    "observation.state.joint_velocity",
    "observation.state.ee_position",
    "observation.state.gripper_position",
)


def state_norm_metadata(stats: dict[str, Any] | None) -> dict[str, dict[str, list[float]]]:
    """``{stats_key: {"q01": [...], "q99": [...]}}`` for the state normalizer.

    Keys the checkpoint's stats.json does not carry are omitted (an 8-dim
    checkpoint has no joint_velocity / ee_position entry), so the client sees
    exactly the bands the server would apply.
    """
    if not stats:
        return {}
    out: dict[str, dict[str, list[float]]] = {}
    for key in STATE_NORM_STAT_KEYS:
        entry = stats.get(key)
        if entry is None:
            continue
        out[key] = {
            q: np.atleast_1d(np.asarray(entry[q], dtype=np.float32)).tolist()
            for q in ("q01", "q99")
        }
    return out


def instrument_ptmap_saturation(policy: ManiFlowMultiModalPolicy) -> None:
    """Log pointmap clamp saturation from the encoder's own first forward pass.

    The scene sitting outside the encoder's clamp is invisible in the depth the
    eval sends -- both depth panels look healthy while the encoder sees a slab
    of +/-1. Measuring it here, on the tensor ``_normalize_ptmaps`` actually
    returns, is the only reading that cannot drift from what the model consumes;
    the alternative (publishing the bounds and re-deriving the pointmap in the
    client) reimplements two encoder methods to answer the same question.

    One-shot: the wrapper uninstalls itself before it runs, so this costs one
    reduction on the first inference and nothing after.
    """
    encoder = policy.obs_encoder
    if not encoder.use_ptmap:
        return
    original = encoder._normalize_ptmaps
    cam_keys = list(encoder.depth_keys if encoder.use_gpu_backproj else encoder.ptmap_keys)

    def wrapped(ptmaps_raw: torch.Tensor, B: int) -> torch.Tensor:
        # Restore first: a raise anywhere below must not leave the wrapper (and
        # its per-forward reduction) installed for the rest of the run.
        encoder._normalize_ptmaps = original
        out = original(ptmaps_raw, B)
        with torch.no_grad():
            raw = ptmaps_raw.detach().view(B, len(cam_keys), 3, *ptmaps_raw.shape[-2:])
            norm = out.detach().view_as(raw)
            for i, key in enumerate(cam_keys):
                # z > 0 is the encoder's own validity test (see _sparse_encode);
                # zeroed pixels back-project to the origin and would report as
                # saturated on every axis.
                cam_raw = raw[:, i].transpose(0, 1)  # [3, B, H, W]
                valid = cam_raw[2] > 0  # [B, H, W]
                if not bool(valid.any()):
                    logger.warning(
                        "ptmap saturation %s: no valid (z > 0) pixels in the first "
                        "batch -- the encoder sees an empty pointmap for this camera.",
                        key,
                    )
                    continue
                xyz = cam_raw[:, valid]  # [3, n_valid]
                pinned = (norm[:, i].transpose(0, 1).abs() >= 1.0)[:, valid]
                logger.info(
                    "ptmap saturation %s: raw xyz min=%s max=%s | clamped at +/-1 on "
                    "%s%% of valid pixels (X, Y, Z)",
                    key,
                    [round(v, 3) for v in xyz.amin(dim=1).tolist()],
                    [round(v, 3) for v in xyz.amax(dim=1).tolist()],
                    [round(100.0 * v, 1) for v in pinned.float().mean(dim=1).tolist()],
                )
        return out

    encoder._normalize_ptmaps = wrapped


def postprocess_actions(
    actions: np.ndarray,
    action_mode: str,
    raw_state: np.ndarray | None,
    action_dim8: str = "pd_mode",
) -> np.ndarray:
    """Shared post-inference action pipeline for the ManiFlow LL model.

    Assumes ``actions`` are already denormalized. Steps:
      * ``relative`` mode: convert per-step deltas to absolute joint targets via
        cumsum over the current state (arm dims 0:7 only);
      * binarize gripper (idx 7) — BinaryJointAction convention negative=close,
        positive=open — and round idx 8: pd_mode to {0, 1}, or the play mode of a
        ``pause_on_attempt`` checkpoint (``action_dim8="play_mode"``) to
        {-1 stop, 0 pause, 1 play};
      * transpose ``(B, T, D) -> (T, B, D)`` for the batched
        :class:`ActionChunkBroker` (unbatched ``(T, D)`` is left alone).
    """
    if action_mode == "relative" and raw_state is not None:
        actions[..., :7] = raw_state[..., :7] + np.cumsum(actions[..., :7], axis=-2)
    gripper_close = np.round(np.clip(actions[..., 7], 0, 1)) > 0.5
    actions[..., 7] = np.where(gripper_close, -1.0, 1.0)
    low = -1 if action_dim8 == "play_mode" else 0
    actions[..., 8] = np.round(np.clip(actions[..., 8], low, 1))
    if actions.ndim == 3:
        actions = np.swapaxes(actions, 0, 1)
    return actions


def action_stats_key(action_mode: str) -> str:
    """Stats key holding the arm-action band for ``action_mode``."""
    identifier = "relative_joint_target" if action_mode == "relative" else "joint_target"
    return f"action.{identifier}"


class ManiFlowNormalizationMixin:
    """Shared normalization and denormalization logic for ManiFlow policies."""
    
    def _init_stats(self, stats: dict[str, Any], action_mode: str) -> None:
        self._stats = stats
        self._action_mode = action_mode
        
        # State
        state_max = np.array(self._stats['observation.state.joint_position']['q99']).astype(np.float32)
        self.state_min = np.array(self._stats['observation.state.joint_position']['q01']).astype(np.float32)
        self.state_denominator = np.maximum(state_max - self.state_min, 1e-6).astype(np.float32)

        gripper_max = np.float32(self._stats["observation.state.gripper_position"]['q99'])
        self.gripper_min = np.float32(self._stats["observation.state.gripper_position"]['q01'])
        self.gripper_denominator = np.maximum(gripper_max - self.gripper_min, 1e-6).astype(np.float32)

        # Action
        action_key = action_stats_key(self._action_mode)
        if action_key not in self._stats:
            available = sorted(k for k in self._stats if k.startswith("action."))
            raise ValueError(
                f"Stats key {action_key!r} not found (action_mode={self._action_mode!r}). "
                f"Available action keys: {available}"
            )
        actions_max = np.array(self._stats[action_key]['q99']).astype(np.float32)
        self.actions_min = np.array(self._stats[action_key]['q01']).astype(np.float32)
        self.actions_denominator = np.maximum(actions_max - self.actions_min, 1e-6).astype(np.float32)

        action_gripper_max = np.float32(self._stats['action.gripper_position']['q99'])
        self.actions_gripper_min = np.float32(self._stats['action.gripper_position']['q01'])
        self.actions_gripper_denominator = np.maximum(action_gripper_max - self.actions_gripper_min, 1e-6).astype(np.float32)

        action_pd_mode_max = np.float32(self._stats['action.pd_mode']['q99'])
        self.actions_pd_mode_min = np.float32(self._stats['action.pd_mode']['q01'])
        self.actions_pd_mode_denominator = np.maximum(action_pd_mode_max - self.actions_pd_mode_min, 1e-6).astype(np.float32)

        # Joint-velocity / EE stats only exist for datasets that recorded the
        # extended 19-dim state. 8-dim checkpoints (7 arm + 1 gripper) train on
        # stats.json files without these keys, so they stay unset and
        # _normalize_state takes the 8-dim path.
        self._has_extended_state_stats = (
            'observation.state.joint_velocity' in self._stats
            and 'observation.state.ee_position' in self._stats
        )
        if self._has_extended_state_stats:
            jvel_max = np.array(self._stats['observation.state.joint_velocity']['q99']).astype(np.float32)
            self.jvel_min = np.array(self._stats['observation.state.joint_velocity']['q01']).astype(np.float32)
            self.jvel_denominator = np.maximum(jvel_max - self.jvel_min, 1e-6).astype(np.float32)

            ee_max = np.array(self._stats['observation.state.ee_position']['q99']).astype(np.float32)
            self.ee_min = np.array(self._stats['observation.state.ee_position']['q01']).astype(np.float32)
            self.ee_denominator = np.maximum(ee_max - self.ee_min, 1e-6).astype(np.float32)

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        arm_state = 2 * (state[..., :7] - self.state_min) / self.state_denominator - 1.0

        # Short state layouts:
        #   8-dim: arm(7) + gripper(1)
        #   9-dim: arm(7) + gripper(1) + pd_mode(1)
        # Both leave the tail (gripper, and pd_mode when present) raw, matching
        # training. Same code path since only the arm is normalized.
        if state.shape[-1] <= 9:
            return np.concatenate([arm_state, state[..., 7:]], axis=-1)

        if not self._has_extended_state_stats:
            raise ValueError(
                f"state has {state.shape[-1]} dims (extended 19-dim form) but "
                "stats.json is missing 'observation.state.joint_velocity' / "
                "'observation.state.ee_position'; the checkpoint's stats do not "
                "match the observation transform's state layout."
            )

        jvel_state = 2 * (state[..., 7:14] - self.jvel_min) / self.jvel_denominator - 1.0
        ee_state = 2 * (state[..., 14:17] - self.ee_min[:3]) / self.ee_denominator[:3] - 1.0

        # gripper (index 17) and pd_mode (index 18) are not normalized (per training)
        gripper_state = state[..., 17:18]
        pd_mode_state = state[..., 18:19]

        return np.concatenate([arm_state, jvel_state, ee_state, gripper_state, pd_mode_state], axis=-1)

    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        # action: (T, 9) -> [7 arm, 1 gripper, 1 pd_mode]
        arm_act = (action[..., :7] + 1.0) / 2 * self.actions_denominator + self.actions_min
        arm_act = np.clip(arm_act, self.actions_min, self.actions_min + self.actions_denominator)

        # Training concatenates action.gripper_position and action.pd_mode RAW
        # (only the arm is normalized), so denorm must invert the arm only and
        # pass gripper (idx 7) / pd_mode (idx 8) through as identity. Mirrors
        # _normalize_state, which likewise leaves state gripper/pd_mode raw.
        gripper_act = action[..., 7:8]
        pd_act = action[..., 8:9]
        return np.concatenate([arm_act, gripper_act, pd_act], axis=-1)

def select_and_validate_obs(
    torch_obs: dict[str, Any],
    obs_shapes: dict[str, tuple[int, ...]],
    conditioning_keys: frozenset[str],
) -> tuple[dict[str, Any], list[str]]:
    """Filter ``torch_obs`` to the model's declared inputs and validate dims.

    The model's ``shape_meta`` (already respecting ``include_ptmap``) is the true
    input contract, so this is the single boundary that both selects and checks
    obs before they reach ``predict_action``.

    - Drops keys not in ``obs_shapes | conditioning_keys`` (returns them, sorted,
      for logging). Extra keys (e.g. ptmaps the client always sends but a
      no-ptmap model never declares) are quietly dropped.
    - Raises on a missing required obs key (every key in ``obs_shapes`` must be
      present).
    - Raises on a trailing-shape mismatch: compares ``v.shape[-len(expected):]``
      against ``obs_shapes[k]`` so leading ``(B, S)`` dims are ignored.
    - Conditioning keys (e.g. ``task_name``, ``target_heatmap``) pass through
      untouched with no shape check.

    Returns ``(filtered_obs, dropped_keys)``.
    """
    allowed = set(obs_shapes) | set(conditioning_keys)
    dropped = sorted(k for k in torch_obs if k not in allowed)
    filtered = {k: v for k, v in torch_obs.items() if k in allowed}

    for k, expected in obs_shapes.items():
        if k not in filtered:
            raise ValueError(
                f"Missing required obs key {k!r} (model expects trailing shape "
                f"{tuple(expected)}); present keys: {sorted(filtered)}"
            )
        v = filtered[k]
        if not torch.is_tensor(v):
            raise ValueError(
                f"Obs key {k!r} must be a tensor to validate against expected "
                f"shape {tuple(expected)}, got {type(v).__name__}"
            )
        actual_trailing = tuple(v.shape[-len(expected):])
        if actual_trailing != tuple(expected):
            raise ValueError(
                f"Obs key {k!r} trailing-shape mismatch: expected {tuple(expected)}, "
                f"got {actual_trailing} (full shape {tuple(v.shape)})"
            )

    return filtered, dropped


# Framework keys that are not model architecture: registry/hydra bookkeeping
# (``name``, ``model_name``, ``qwenvl``, ``_target_``), the training-only ``ema``
# setting, and ``encoder_output_dim``, which the policy constructor only ever
# swallowed into ``**kwargs`` and ``DiTX.forward`` discards. The image
# augmentation settings (``color_jitter``, ``random_crop_ratio``,
# ``random_rotation_degrees``) are read only by HVLA's training dataloader.
# Listing a key here never stops it reaching the model -- an HVLA whose
# constructor names it still receives it -- it only says "absent is not a bug".
IGNORED_FRAMEWORK_KEYS = frozenset(
    {
        "_target_",
        "name",
        "model_name",
        "qwenvl",
        "ema",
        "encoder_output_dim",
        "previous_front_frame_lookback_weights",
        "color_jitter",
        "random_crop_ratio",
        "random_rotation_degrees",
    }
)

# Framework keys this module consumes itself instead of handing to the model.
# Exempt from the unconsumed check for the same reason as the ignored keys above
# -- they are not architecture -- but kept apart because "we read this" and
# "nothing reads this" are different claims. ``language_mode`` is not a
# training-only setting: it is the eval-time text contract, deciding whether the
# model was trained on ``"pick"`` or on ``"pick 409,210"``, so the server
# advertises it and whoever builds the string formats itself to match.
CONSUMED_FRAMEWORK_KEYS = frozenset({"train_data_stats", "language_mode"})

# The only two ``language_mode`` values HVLA's dataloader implements.
# ``action_only`` is the bare action phrase; ``action_goal`` appends the goal's
# native front-camera pixel coordinates as ``f"{action} {x:.0f},{y:.0f}"``
# (hvla/dataloader/franka_dataset.py). Any other value means the text format the
# checkpoint was trained on is unknown here.
LANGUAGE_MODES = frozenset({"action_goal", "action_only"})

# ``(stats key, min field, denominator field, required)`` for every band
# ``framework.train_data_stats`` records. The arm-action band is keyed by action
# mode, so ``stats_from_train_data_stats`` prepends it rather than listing it
# here. joint_velocity / ee_position are absent for 8-dim checkpoints, matching
# the ``_has_extended_state_stats`` branch in the normalization mixin.
_TRAIN_DATA_STATS_BANDS: tuple[tuple[str, str, str, bool], ...] = (
    ("observation.state.joint_position", "state_min", "state_denominator", True),
    ("observation.state.gripper_position", "gripper_min", "gripper_denominator", True),
    ("observation.state.joint_velocity", "jvel_min", "jvel_denominator", False),
    ("observation.state.ee_position", "ee_min", "ee_denominator", False),
    ("action.gripper_position", "actions_gripper_min", "actions_gripper_denominator", True),
    ("action.pd_mode", "actions_pd_mode_min", "actions_pd_mode_denominator", True),
)


def stats_from_train_data_stats(
    train_data_stats: dict[str, Any], action_mode: str
) -> dict[str, dict[str, Any]]:
    """``framework.train_data_stats`` -> the ``{key: {q01, q99}}`` shape stats.json has.

    Training records the normalizer it fitted -- ``q01`` and the clamped
    ``q99 - q01`` span -- into the run's own config, so a checkpoint that
    carries it needs no stats.json to serve. That matters for a multi-dataset
    run, where no single dataset's ``meta/stats.json`` holds the mixed bands.

    ``q99`` is reconstructed as ``min + denominator``, which ``_init_stats``
    reduces straight back to that denominator, so the served bands are the
    recorded ones up to float32 rounding.

    Only one, unlabelled, arm-action band is recorded, so ``action_mode`` (from
    the config's ``datasets.vla_data.action_mode``) is what decides whether it
    is keyed absolute or relative -- unlike stats.json, there is no second band
    to cross-check the mode against.
    """
    bands = (
        (action_stats_key(action_mode), "actions_min", "actions_denominator", True),
        *_TRAIN_DATA_STATS_BANDS,
    )
    stats: dict[str, dict[str, Any]] = {}
    for key, min_field, denom_field, required in bands:
        present = [f for f in (min_field, denom_field) if f in train_data_stats]
        if not present:
            if required:
                raise ValueError(
                    f"framework.train_data_stats has neither {min_field!r} nor "
                    f"{denom_field!r}, so the {key!r} band the checkpoint was "
                    "normalized with is unknown."
                )
            continue
        if len(present) == 1:
            raise ValueError(
                f"framework.train_data_stats has {present[0]!r} without its pair, "
                f"leaving the {key!r} band incomplete."
            )
        low = np.asarray(train_data_stats[min_field], dtype=np.float32)
        span = np.asarray(train_data_stats[denom_field], dtype=np.float32)
        if low.shape != span.shape:
            raise ValueError(
                f"framework.train_data_stats {min_field!r} has shape {low.shape} but "
                f"{denom_field!r} has shape {span.shape}; the {key!r} band is malformed."
            )
        stats[key] = {"q01": low.tolist(), "q99": (low + span).tolist()}
    return stats


def derive_model_input_spec(cfg: Any, fw: Any) -> dict[str, Any]:
    """The served model's input contract, read off the checkpoint's own config.

    ``fw`` -- ``cfg.framework`` merged with hvla's ``_MANIFLOW_MM_DEFAULTS`` --
    is a parameter rather than something derived here so that this function, and
    the tests that pin the contract, stay importable without the maniflow extra:
    the defaults import lives at the call site in
    :func:`load_maniflow_policy_common`.

    ``fw`` is mutated in place when ``datasets.vla_data.include_ptmap`` is False
    (the ``*_ptmap`` obs entries are deleted and ``use_ptmap`` cleared), because
    the caller builds the policy from that same object and the built model must
    not disagree with the contract about whether pointmaps exist.

    Returns ``shape_meta`` (the ptmap-stripped copy the policy is built from),
    ``obs_shapes``, ``conditioning_keys`` and ``state_dim``, plus
    ``language_mode`` for a language-conditioned checkpoint.
    """
    include_ptmap = OmegaConf.select(cfg, "datasets.vla_data.include_ptmap", default=None)
    if include_ptmap is False:
        OmegaConf.update(fw, "use_ptmap", False)
        sm = fw.shape_meta
        if hasattr(sm, "obs"):
            ptmap_keys = [k for k in sm.obs if k.endswith("_ptmap")]
            for k in ptmap_keys:
                del sm.obs[k]

    shape_meta = OmegaConf.to_container(fw.shape_meta, resolve=True)

    # Derive the model's input contract from the (already ptmap-stripped)
    # shape_meta so the server can select AND validate client obs against it.
    # obs_shapes are the per-key trailing feature shapes (e.g. agent_pos: (19,),
    # front_rgb: (3, 224, 224)); leading (B, S) dims are validated separately.
    obs_shapes = {k: tuple(v["shape"]) for k, v in shape_meta["obs"].items() if v is not None}
    # Depth and camera_intrinsics are in the contract solely to feed the
    # encoder's GPU back-projection into pointmaps (ManiflowObsTransform emits
    # them together as that path's ingredients), so a checkpoint with pointmaps
    # disabled reads neither. Leaving them declared makes the eval render depth
    # the model discards and asks --save_diagnostic_plots to reproduce a
    # pointmap the encoder never builds.
    if not OmegaConf.select(fw, "use_ptmap", default=True):
        for k in [k for k in obs_shapes if k.endswith("_depth")] + ["camera_intrinsics"]:
            obs_shapes.pop(k, None)
    # Depth is declared as 3 channels by some training configs (a leftover from
    # the precomputed-pointmap era; see submit_maniflow_train.py), but the
    # GPU back-projection path in obs_encoder reshapes every depth tensor to a
    # single channel, and nothing sizes a module from the declared count. Fix up
    # the validation contract only -- `shape_meta` itself is handed to the policy
    # untouched, so the checkpoint still builds exactly as it was trained.
    for k in [k for k in obs_shapes if k.endswith("_depth")]:
        obs_shapes[k] = (1, *obs_shapes[k][-2:])
    # MANIFLOW_IMAGE_SIZE is baked into image preprocessing and heatmap construction
    # on both the client and server, so a checkpoint trained at another resolution
    # must fail here rather than silently receive mis-sized inputs.
    rgb_shape = obs_shapes.get("front_rgb")
    if rgb_shape is not None and tuple(rgb_shape[-2:]) != (
        MANIFLOW_IMAGE_SIZE,
        MANIFLOW_IMAGE_SIZE,
    ):
        raise ValueError(
            f"checkpoint front_rgb shape {tuple(rgb_shape)} disagrees with "
            f"MANIFLOW_IMAGE_SIZE={MANIFLOW_IMAGE_SIZE}; image preprocessing and "
            "target heatmaps are built at that size."
        )
    # `use_previous_front_frame` appends `prev_front_rgb` to the model's rgb_keys
    # without ever adding a shape_meta entry for it, so the input contract has to
    # be completed here: without this the key is dropped as "not in shape_meta"
    # and the model KeyErrors on it inside obs_encoder. It carries the same
    # layout as front_rgb. Read via `select` because the flag only exists in
    # HVLA's defaults on the previous-front-frame branch.
    if OmegaConf.select(fw, "use_previous_front_frame", default=None):
        if "front_rgb" not in obs_shapes:
            raise ValueError(
                "checkpoint sets use_previous_front_frame but declares no front_rgb "
                "in shape_meta, so prev_front_rgb has no layout to mirror."
            )
        obs_shapes["prev_front_rgb"] = obs_shapes["front_rgb"]

    use_target_front_crop = parse_target_front_crop(
        OmegaConf.select(fw, "use_target_front_crop", default=-1)
    )
    if use_target_front_crop >= 0:
        if "front_rgb" not in obs_shapes:
            raise ValueError(
                "checkpoint sets use_target_front_crop but declares no front_rgb "
                "in shape_meta, so target_front_crop_rgb has no layout to mirror."
            )
        obs_shapes["target_front_crop_rgb"] = obs_shapes["front_rgb"]

    # Conditioning keys are valid model inputs NOT present in shape_meta.obs, so
    # filtering must never drop them. Tie them to the model flags so the set
    # self-corrects for any future conditioning change.
    conditioning_keys: list[str] = []
    if fw.language_conditioned:
        conditioning_keys.append("task_name")
    if fw.target_injection_mode:
        conditioning_keys.append("target_heatmap")
    model_input_spec: dict[str, Any] = {
        # The ptmap-stripped copy, so the policy the caller builds and the
        # contract the server validates against come from one object.
        "shape_meta": shape_meta,
        "obs_shapes": obs_shapes,
        "conditioning_keys": frozenset(conditioning_keys),
    }
    # Whoever builds `task_name` has to build it in the format the checkpoint was
    # trained on -- an `action_goal` model fed a bare verb has no spatial goal at
    # all, which looks like a working run that behaves badly. Advertise the mode
    # rather than letting the client guess.
    if fw.language_conditioned:
        language_mode = OmegaConf.select(fw, "language_mode", default=None)
        if language_mode not in LANGUAGE_MODES:
            raise ValueError(
                "checkpoint config is language_conditioned with framework.language_mode="
                f"{language_mode!r}, which is not one of {sorted(LANGUAGE_MODES)}. The "
                "text format this checkpoint was trained on is therefore unknown, and "
                "guessing it would condition the policy on out-of-distribution "
                "language."
            )
        model_input_spec["language_mode"] = language_mode

    if use_target_front_crop >= 0:
        model_input_spec["use_target_front_crop"] = use_target_front_crop

    # The state width is what the server advertises and the client's obs
    # transform assembles agent_pos to match, so a checkpoint that declares no
    # state has to fail here rather than let the client pick a default width the
    # model would then be fed.
    if "agent_pos" not in obs_shapes:
        raise ValueError(
            f"checkpoint shape_meta declares no agent_pos (obs keys: {sorted(obs_shapes)}), "
            "so the state layout the client must send is unknown."
        )
    model_input_spec["state_dim"] = int(obs_shapes["agent_pos"][-1])
    # Action (and state) dim 8 is the PD gain mode, or -- for a checkpoint trained
    # with hvla's pause_on_attempt -- the play mode (sim2real): 1 play, 0 pause,
    # -1 skill finished (predicted, never fed back). The client acts on the two
    # completely differently, so it is advertised rather than guessed.
    pause_on_attempt = OmegaConf.select(cfg, "datasets.vla_data.pause_on_attempt", default=False)
    model_input_spec["action_dim8"] = "play_mode" if pause_on_attempt else "pd_mode"
    pc_bounds = OmegaConf.select(fw, "per_camera_ptmap_bounds", default=None)
    if pc_bounds is not None:
        model_input_spec["per_camera_ptmap_bounds"] = OmegaConf.to_container(
            pc_bounds, resolve=True
        )
    return model_input_spec


def load_maniflow_policy_common(
    checkpoint: str,
    stats_path: str,
    config: str | None = None,
    train_config: str | None = None,
    action_mode_override: str | None = None,
    use_ema: bool = False,
    build_policy: bool = True,
    num_inference_steps: int | None = None,
) -> tuple[ManiFlowMultiModalPolicy | None, dict | None, str, dict[str, Any]]:
    """``num_inference_steps``: flow integration steps; ``None`` keeps the checkpoint's."""
    from safetensors.torch import load_file
    ckpt_path = pathlib.Path(checkpoint)
    
    # 1. Load config
    config_path = config
    if config_path is None:
        search_paths = [ckpt_path.parent / "config.yaml", ckpt_path.parent.parent / "config.yaml"]
        for p in search_paths:
            if p.exists():
                config_path = str(p)
                break

    if config_path is None or not os.path.exists(config_path):
        raise ValueError("Config file not found. Please provide --config.")

    logging.info("Loading config from %s", config_path)
    cfg = OmegaConf.load(config_path)
    ckpt_framework = OmegaConf.to_container(
        OmegaConf.select(cfg, "framework", default=OmegaConf.create({})),
        resolve=True,
    )

    # action_mode decides whether postprocess_actions adds the current joint state
    # back to the model output, so guessing it mis-denormalizes every action of a
    # checkpoint trained the other way -- silently, since both conventions produce
    # plausible joint targets. Require it to be recorded or passed explicitly.
    if action_mode_override is not None:
        effective_action_mode = action_mode_override
    else:
        cfg_action_mode_raw = OmegaConf.select(cfg, "datasets.vla_data.action_mode", default=None)
        if cfg_action_mode_raw is None:
            raise ValueError(
                f"{config_path} does not set datasets.vla_data.action_mode, so this "
                "checkpoint's action convention is unknown. Pass --action-mode "
                "absolute|relative explicitly."
            )
        effective_action_mode = (
            "absolute" if cfg_action_mode_raw.startswith("abs") else "relative"
        )

    # 1.1 Load train config
    train_config_path = train_config
    if train_config_path is None:
        search_paths = list(ckpt_path.parent.glob("train_*.yaml")) + list(ckpt_path.parent.parent.glob("train_*.yaml"))
        for p in search_paths:
            if p.exists():
                train_config_path = str(p)
                break

    if train_config_path and os.path.exists(train_config_path):
        logging.info("Loading train config from %s", train_config_path)
        train_cfg = OmegaConf.load(train_config_path)
        cfg = OmegaConf.merge(train_cfg, cfg)

    # 2. Instantiate policy
    from hvla.model.framework.ManiFlowMultiModal import _MANIFLOW_MM_DEFAULTS
    from hvla.model.modules.maniflow.flow_policy import ManiFlowMultiModalPolicy

    fw = OmegaConf.merge(_MANIFLOW_MM_DEFAULTS, cfg.framework)

    # Derives the input contract AND strips ptmaps from fw, so the policy below
    # is built from the same shape_meta the server validates client obs against.
    model_input_spec = derive_model_input_spec(cfg, fw)
    shape_meta = model_input_spec["shape_meta"]
    if not build_policy:
        return None, None, effective_action_mode, model_input_spec

    visual_cond_len_override = OmegaConf.select(cfg.framework, "visual_cond_len", default=None)

    if "n_action_steps" in ckpt_framework:
        n_action_steps = ckpt_framework["n_action_steps"]
    elif "horizon" in ckpt_framework:
        n_action_steps = ckpt_framework["horizon"]
    elif "n_action_steps" in fw:
        n_action_steps = fw.n_action_steps
    else:
        n_action_steps = fw.horizon

    # The architecture is whatever the checkpoint's `framework` block says, so the
    # constructor's own signature -- not a list maintained here -- decides which
    # fields get through. A hand-copied list silently drops any option this file
    # has never heard of (a pointmap branch sharing the RGB encoder, per-camera
    # pointmap bounds), building a differently-shaped model whose only symptom is
    # a `load_state_dict` failure well after the server thought it was fine.
    fw_container = OmegaConf.to_container(fw, resolve=True)
    ctor_params = inspect.signature(ManiFlowMultiModalPolicy.__init__).parameters
    named_params = {
        name
        for name, p in ctor_params.items()
        if name != "self" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }
    kwargs: dict[str, Any] = {k: v for k, v in fw_container.items() if k in named_params}
    # The three values that are deliberately not the raw config value: shape_meta
    # is the ptmap-stripped copy, n_action_steps comes from the checkpoint's own
    # framework block (not the defaults merge), and visual_cond_len stays read
    # from the config only -- passing None when it is absent, as before.
    kwargs["shape_meta"] = shape_meta
    kwargs["n_action_steps"] = n_action_steps
    kwargs["visual_cond_len"] = visual_cond_len_override

    # The constructor ends in **kwargs and stores unknowns without complaint (they
    # are splatted into DiTX.forward, which drops them), so a field going nowhere
    # has to be caught here or not at all.
    unconsumed = sorted(
        set(fw_container) - set(kwargs) - IGNORED_FRAMEWORK_KEYS - CONSUMED_FRAMEWORK_KEYS
    )
    if unconsumed:
        raise ValueError(
            f"{config_path} configures framework keys {unconsumed}, which the "
            "installed HVLA's ManiFlowMultiModalPolicy.__init__ does not accept. "
            "They would be silently ignored, giving a model that differs from the "
            "trained one. Either install an hvla revision that supports them, or "
            "add them to IGNORED_FRAMEWORK_KEYS if they are training-only settings."
        )

    logger.info(
        "Building ManiFlowMultiModalPolicy from %d framework keys; values set by "
        "the checkpoint itself: %s",
        len(kwargs),
        {k: v for k, v in sorted(kwargs.items()) if k in ckpt_framework and k != "shape_meta"},
    )

    policy = ManiFlowMultiModalPolicy(**kwargs)
    # The policy ctor ends in **kwargs and stores unknowns without touching
    # encoder construction, so an HVLA install that predates the feature builds a
    # model whose rgb_keys omits prev_front_rgb -- while the server still demands
    # the key from the client, validates its shape, and reports the run as
    # previous-frame-conditioned with the frame silently ignored.
    if (
        "prev_front_rgb" in model_input_spec["obs_shapes"]
        and "prev_front_rgb" not in policy.obs_encoder.rgb_keys
    ):
        raise RuntimeError(
            "checkpoint sets use_previous_front_frame but the built model's "
            f"rgb_keys is {list(policy.obs_encoder.rgb_keys)} (no 'prev_front_rgb'), "
            "so the installed HVLA predates the feature and would ignore the frame. "
            "Update the hvla install to a revision that supports "
            "use_previous_front_frame."
        )
    if (
        "target_front_crop_rgb" in model_input_spec["obs_shapes"]
        and "target_front_crop_rgb" not in policy.obs_encoder.rgb_keys
    ):
        raise RuntimeError(
            "checkpoint sets use_target_front_crop but the built model's "
            f"rgb_keys is {list(policy.obs_encoder.rgb_keys)} (no "
            "'target_front_crop_rgb'), so the installed HVLA predates the feature. "
            "Update the hvla install to a revision that supports "
            "use_target_front_crop."
        )

    # 4. Load weights
    state_dict = None
    if use_ema:
        # Every step here used to be a soft miss that fell through to the online
        # weights below, so a run asked for EMA, silently got non-EMA, and reported
        # the difference as a model result. `ema` is the EMAModel wrapper
        # (averaged_model / optimization_step / decay), not a state dict -- HVLA
        # reads ema["averaged_model"], and handing the wrapper to load_state_dict
        # is what made --use_ema a no-op.
        state_pt = ckpt_path.parent / (ckpt_path.name.replace("_model.safetensors", "") + "_training_state.pt")
        if not state_pt.exists():
            raise FileNotFoundError(
                f"use_ema was requested but {state_pt} does not exist, so there are no "
                "EMA weights to serve. Drop --use_ema to evaluate the online weights."
            )
        training_state = torch.load(state_pt, map_location="cpu", weights_only=False)
        if "ema" not in training_state:
            raise KeyError(
                f"{state_pt} has no 'ema' entry (keys: {sorted(training_state)}); this "
                "checkpoint was not trained with EMA."
            )
        ema = training_state["ema"]
        if "averaged_model" not in ema:
            raise KeyError(
                f"{state_pt}'s 'ema' entry has no 'averaged_model' (keys: {sorted(ema)}); "
                "the expected EMAModel layout changed, so refusing to guess which "
                "tensors are the EMA weights."
            )
        state_dict = ema["averaged_model"]

    if state_dict is None:
        state_dict = load_file(checkpoint)
        stripped_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("policy."):
                stripped_state_dict[key[len("policy."):]] = value
        state_dict = stripped_state_dict if stripped_state_dict else state_dict

    policy.load_state_dict(state_dict)

    if num_inference_steps is not None and num_inference_steps > 0:
        policy.num_inference_steps = int(num_inference_steps)

    # Both adapters build their policy through here, so one call covers them.
    instrument_ptmap_saturation(policy)

    # Normalization bands, most authoritative source first: an explicit
    # --stats_path, then the bands the run recorded in its own config, then the
    # stats.json training copied next to that config. Deriving the last one here
    # rather than in the caller keeps the checkpoint's own record ahead of a
    # path guessed from the run layout.
    train_data_stats = ckpt_framework.get("train_data_stats")
    sibling_stats = pathlib.Path(config_path).parent / "stats.json"
    if stats_path:
        logging.info("Loading normalization stats from %s", stats_path)
        with open(stats_path) as f:
            stats = json.load(f)
    elif train_data_stats is not None:
        logging.info("Using framework.train_data_stats from %s", config_path)
        stats = stats_from_train_data_stats(train_data_stats, effective_action_mode)
    elif sibling_stats.exists():
        logging.info("Loading normalization stats from %s", sibling_stats)
        with open(sibling_stats) as f:
            stats = json.load(f)
    else:
        raise ValueError(
            f"No normalization stats: --stats_path was not given, {config_path} has no "
            f"framework.train_data_stats, and {sibling_stats} does not exist."
        )

    return policy, stats, effective_action_mode, model_input_spec
