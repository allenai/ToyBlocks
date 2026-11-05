"""Language goal formatting and target heatmaps for point-conditioned inference."""

from __future__ import annotations

import cv2
import numpy as np

from blocksuite.serving.sim2real.rollout.inference_config import InferenceConfig, POINT_GOAL_FORMATS

POLICY_GOAL_WIDTH = 640
POLICY_GOAL_HEIGHT = 360
POLICY_IMG_SIZE = 224

# Square front crop centered on the goal point, sent as an extra camera for
# checkpoints trained with ``framework.use_target_front_crop``.
TARGET_CROP_KEY = "target_front_crop_rgb"

GOAL_ACTION_EXAMPLES: tuple[tuple[str, str, bool], ...] = (
    ("PnP grasp", "pick", True),
    ("PnP place (block-on-block)", "place", True),
    ("PnP place (on table)", "place on table", True),
    ("Reorient", "flip right", True),
    ("Reorient", "flip left", True),
    ("Reorient", "flip forward", True),
    ("Reorient", "flip backward", True),
    ("Reorient", "rotate cw", True),
    ("Reorient", "rotate ccw", True),
    ("Done", "done", False),
)
ALLOWED_GOAL_ACTIONS = frozenset(
    action.lower() for _phase, action, _requires_point in GOAL_ACTION_EXAMPLES
)
NO_POINT_ACTIONS = frozenset(
    action for _phase, action, requires_point in GOAL_ACTION_EXAMPLES if not requires_point
)


# Display order for the GUI's action-goal buttons. Any allowed action missing
# from this tuple is appended at the end, so the buttons always cover the full
# :data:`ALLOWED_GOAL_ACTIONS` set.
GOAL_ACTION_BUTTON_ORDER: tuple[str, ...] = (
    "pick",
    "place",
    "place on table",
    "rotate ccw",
    "rotate cw",
    "flip backward",
    "flip forward",
    "flip left",
    "flip right",
    "done",
)


def goal_action_buttons() -> list[dict[str, object]]:
    """Allowed action goals in button order, with their point requirement."""
    ordered = [a for a in GOAL_ACTION_BUTTON_ORDER if a in ALLOWED_GOAL_ACTIONS]
    ordered += sorted(ALLOWED_GOAL_ACTIONS - set(ordered))
    return [
        {"action": action, "requires_point": action_requires_point(action)}
        for action in ordered
    ]


def is_valid_goal_action(goal: str) -> bool:
    return goal.strip().lower() in ALLOWED_GOAL_ACTIONS


def allowed_goal_actions_display() -> str:
    return ", ".join(sorted(ALLOWED_GOAL_ACTIONS))


def rescale_point_to_policy_space(
    u: float, v: float, src_width: float, src_height: float
) -> tuple[float, float]:
    """Map camera pixel coordinates to the 640x360 space used during training."""
    return (
        u * POLICY_GOAL_WIDTH / src_width,
        v * POLICY_GOAL_HEIGHT / src_height,
    )


def rescale_point_from_policy_space(
    u: float, v: float, src_width: float, src_height: float
) -> tuple[float, float]:
    """Map 640x360 policy-space coordinates back to camera pixels.

    Inverse of :func:`rescale_point_to_policy_space`. Molmo2 HL emits its
    ``<u,v>`` target in the policy space it was trained on, whereas a point
    held by this client is always in source-camera pixels -- that is what the
    GUI overlay draws with and what :func:`target_heatmap_from_point` and
    :func:`policy_space_uv` expect to scale back down themselves.
    """
    return (
        u * src_width / POLICY_GOAL_WIDTH,
        v * src_height / POLICY_GOAL_HEIGHT,
    )


def src_frame_size_for_point(
    point: tuple[str, float, float] | None,
    frames: dict[str, np.ndarray],
) -> tuple[float, float] | None:
    if point is None:
        return None
    rgb = frames.get(point[0])
    if rgb is None:
        return None
    height, width = rgb.shape[:2]
    return (float(width), float(height))


def policy_space_uv(
    point: tuple[str, float, float] | None,
    src_frame_size: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if point is None:
        return None
    u, v = point[1], point[2]
    if src_frame_size is None:
        return u, v
    src_w, src_h = src_frame_size
    return rescale_point_to_policy_space(u, v, src_w, src_h)


def format_language_goal(
    action_str: str,
    point_goal_format: str,
    *,
    u: float | None = None,
    v: float | None = None,
) -> str:
    """Build the model language prompt."""
    action_str = action_str.strip()
    if point_goal_format == "action_only":
        return action_str
    if point_goal_format != "action_goal":
        raise ValueError(
            f"point_goal_format must be one of {POINT_GOAL_FORMATS}, got {point_goal_format!r}"
        )
    if u is None or v is None:
        return action_str
    return f"{action_str} {u:.0f},{v:.0f}"


def action_requires_point(action_str: str) -> bool:
    return action_str.strip().lower() not in NO_POINT_ACTIONS


def instruction_from_goal(
    cfg: InferenceConfig,
    goal: str,
    point: tuple[str, float, float] | None,
    *,
    src_frame_size: tuple[float, float] | None = None,
) -> str:
    """Format a submitted GUI/CLI goal into the policy language instruction."""
    instruction = goal.strip()
    if cfg.point_goal_format == "action_goal" and point is not None:
        policy_uv = policy_space_uv(point, src_frame_size)
        if policy_uv is None:
            return format_language_goal(instruction, "action_only")
        u, v = policy_uv
        return format_language_goal(instruction, "action_goal", u=u, v=v)
    return format_language_goal(instruction, "action_only")


def build_target_heatmap(
    target_xy: np.ndarray | tuple[float, float],
    *,
    orig_cam_w: float = POLICY_GOAL_WIDTH,
    orig_cam_h: float = POLICY_GOAL_HEIGHT,
    target_sigma: float = 0.8,
    img_size: int = POLICY_IMG_SIZE,
) -> np.ndarray:
    """Build a (3, img_size, img_size) Gaussian heatmap from pixel coordinates.

    ``target_xy`` is in the same 640x360 policy camera space used during training.
    """
    heatmap = np.zeros((img_size, img_size), dtype=np.float32)
    x, y = float(target_xy[0]), float(target_xy[1])
    if not (np.isnan(x) or np.isnan(y)):
        x = x * img_size / orig_cam_w
        y = y * img_size / orig_cam_h
        sigma_px = target_sigma * (img_size / 14.0)
        yy, xx = np.mgrid[0:img_size, 0:img_size].astype(np.float32)
        heatmap = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma_px ** 2))
    return np.stack([heatmap, heatmap, heatmap], axis=0)


def target_heatmap_from_point(
    point: tuple[str, float, float] | None,
    src_frame_size: tuple[float, float] | None,
    *,
    target_sigma: float = 0.8,
    img_size: int = POLICY_IMG_SIZE,
) -> np.ndarray | None:
    """Build a policy-space heatmap from a GUI click, or return None if unavailable."""
    if point is None:
        return None
    policy_uv = policy_space_uv(point, src_frame_size)
    if policy_uv is None:
        return None
    return build_target_heatmap(
        policy_uv,
        target_sigma=target_sigma,
        img_size=img_size,
    )


def crop_centered(
    img: np.ndarray,
    cx: float,
    cy: float,
    *,
    size: int = POLICY_IMG_SIZE,
    bottom_pct: int = 50,
) -> np.ndarray:
    """Crop a square ``size x size`` window from an HWC image.

    Port of hvla's ``FrankaDataset._crop_centered``, kept identical down to the
    zero padding so a checkpoint sees the framing it was trained on: the target
    ``(cx, cy)`` is centered horizontally, and ``bottom_pct`` is the percent of
    the crop below it (``0`` = 100% above, ``50`` = zero-centered).
    """
    H, W = img.shape[:2]
    top_px = int(round(size * (100 - bottom_pct) / 100.0))
    x0 = int(round(float(cx))) - size // 2
    y0 = int(round(float(cy))) - top_px

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x0 + size - W)
    pad_bottom = max(0, y0 + size - H)

    crop = img[max(0, y0):min(H, y0 + size), max(0, x0):min(W, x0 + size)]
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


def target_front_crop_from_point(
    front_rgb: np.ndarray | None,
    point: tuple[str, float, float] | None,
    *,
    bottom_pct: int,
    size: int = POLICY_IMG_SIZE,
) -> np.ndarray:
    """Build the :data:`TARGET_CROP_KEY` camera from a goal point.

    Training crops out of the dataset's native 640x360 frames, so the live
    frame (1280x720 on the ZED) is rescaled into that policy space first --
    cropping straight out of it would cover half the scene the checkpoint saw.
    Without a frame or a point the crop is all zeros, as it is in training for
    a sample whose target is missing.
    """
    if front_rgb is None:
        return np.zeros((size, size, 3), dtype=np.uint8)
    frame = np.asarray(front_rgb)
    src_h, src_w = frame.shape[:2]
    if (src_w, src_h) != (POLICY_GOAL_WIDTH, POLICY_GOAL_HEIGHT):
        frame = cv2.resize(
            frame,
            (POLICY_GOAL_WIDTH, POLICY_GOAL_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
    if point is None:
        return np.zeros((size, size, frame.shape[2]), dtype=frame.dtype)
    u, v = rescale_point_to_policy_space(point[1], point[2], src_w, src_h)
    return crop_centered(frame, u, v, size=size, bottom_pct=bottom_pct)


def policy_obs_target_heatmap(heatmap_3chw: np.ndarray) -> np.ndarray:
    """Wrap a (3, H, W) heatmap for ManiFlow ``target_heatmap`` obs."""
    return np.asarray(heatmap_3chw, dtype=np.float32)[None, None, ...]


def build_display_heatmap(
    x: float,
    y: float,
    frame_w: int,
    frame_h: int,
    *,
    target_sigma: float = 0.8,
) -> np.ndarray:
    """Build a visually round Gaussian in frame pixels for GUI overlay only."""
    sigma_px = target_sigma * (frame_h / 14.0)
    yy, xx = np.mgrid[0:frame_h, 0:frame_w].astype(np.float32)
    heatmap = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma_px ** 2))
    return np.stack([heatmap, heatmap, heatmap], axis=0)


def overlay_heatmap_on_rgb(
    rgb: np.ndarray,
    heatmap_3chw: np.ndarray,
    *,
    alpha_scale: float = 0.5,
) -> np.ndarray:
    """Overlay a heatmap on an RGB frame for GUI preview."""
    frame = np.asarray(rgb).copy()
    if frame.ndim != 3:
        return frame
    height, width = frame.shape[:2]
    hm = np.asarray(heatmap_3chw[0], dtype=np.float32)
    if hm.shape[0] == height and hm.shape[1] == width:
        hm_resized = hm
    else:
        # Fallback for legacy policy-shaped heatmaps: restore 640x360 aspect first.
        hm_policy_aspect = cv2.resize(
            hm,
            (POLICY_GOAL_WIDTH, POLICY_GOAL_HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )
        hm_resized = cv2.resize(
            hm_policy_aspect,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
    hm_color = cv2.applyColorMap(
        (hm_resized * 255).astype(np.uint8), cv2.COLORMAP_HOT
    )
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    alpha = (hm_resized * alpha_scale)[..., None]
    return (frame * (1.0 - alpha) + hm_color * alpha).astype(np.uint8)
