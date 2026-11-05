"""Hardware camera helpers and mock frames for vlagent RCS client."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rcs._core.common import BaseCameraConfig

from toyblocks.serving.sim2real.rollout.hardware_config import (
    INCLUDE_DEPTH,
    INCLUDE_RIGHT,
    REALSENSE_CAMERA_DICT,
    ZED_CAMERA_DICT,
)

logger = logging.getLogger(__name__)

INTERACTIVE_GUI_CAMERAS = ("front_rgb", "gripper_rgb")
# View-only depth panels shown in the interactive GUI when INCLUDE_DEPTH (ZED)
# or use_ffstereo is enabled. Keys match the ffstereo/ZED depth dict naming
# used by ``obs_rcs2agents`` and ``save_depth_outputs``.
GUI_DEPTH_CAMERAS = ("front_depth", "gripper_depth")
POINT_PICK_CAMERA = "front_rgb"


def display_available() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def mock_camera_frame(cam_name: str) -> np.ndarray:
    """Placeholder RGB frame for dry-run GUI testing without hardware."""
    height, width = 720, 1280
    palette = {
        "front_rgb": (48, 72, 120),
        "gripper_rgb": (72, 48, 120),
        "front_rgb_right": (48, 120, 72),
        "gripper_rgb_right": (120, 48, 72),
    }
    base = palette.get(cam_name, (64, 64, 64))
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:] = base
    grid_color = tuple(min(c + 35, 255) for c in base)
    for x in range(0, width, 80):
        cv2.line(frame, (x, 0), (x, height), grid_color, 1)
    for y in range(0, height, 80):
        cv2.line(frame, (0, y), (width, y), grid_color, 1)
    cv2.putText(
        frame,
        f"{cam_name} (no camera)",
        (24, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "click to pick u,v" if cam_name == POINT_PICK_CAMERA else "view only",
        (24, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )
    return frame


def build_hardware_camera_cfgs(*, connected_only: bool = False) -> dict[str, Any]:
    from rcs_fr3.creators import HardwareCameraCreatorConfig

    camera_cfgs: dict[str, HardwareCameraCreatorConfig] = {}
    if REALSENSE_CAMERA_DICT is not None:
        camera_cfgs["realsense"] = HardwareCameraCreatorConfig(
            camera_type_id="realsense",
            camera_cfgs={
                name: BaseCameraConfig(
                    identifier=identifier,
                    resolution_width=1280,
                    resolution_height=720,
                    frame_rate=30,
                )
                for name, identifier in REALSENSE_CAMERA_DICT.items()
            },
        )
    if ZED_CAMERA_DICT is not None:
        zed_camera_cfgs = {
            name: BaseCameraConfig(
                identifier=identifier,
                resolution_width=1280,
                resolution_height=720,
                frame_rate=30,
            )
            for name, identifier in ZED_CAMERA_DICT.items()
        }
        if connected_only:
            try:
                from rcs_zed.camera import ZEDCameraSet

                available = ZEDCameraSet.enumerate_connected_devices()
            except ImportError:
                available = {}
            connected = {
                name: cam_cfg
                for name, cam_cfg in zed_camera_cfgs.items()
                if cam_cfg.identifier in available
            }
            for name, cam_cfg in zed_camera_cfgs.items():
                if name not in connected:
                    logger.warning(
                        "dry-run: skipping ZED camera %s (serial %s not connected)",
                        name,
                        cam_cfg.identifier,
                    )
            zed_camera_cfgs = connected
        if zed_camera_cfgs:
            camera_cfgs["zed"] = HardwareCameraCreatorConfig(
                camera_type_id="zed",
                camera_cfgs=zed_camera_cfgs,
                kwargs={
                    "enable_depth": INCLUDE_DEPTH,
                    "enable_imu": False,
                    "include_right": INCLUDE_RIGHT,
                },
            )
    return camera_cfgs


def camera_color_to_rgb(data: np.ndarray) -> np.ndarray:
    """Return an RGB frame for GUI display.

    RCS camera drivers (ZED, RealSense, etc.) already store ``color.data`` in RGB
    order. Do not BGR-swap here or colors will be wrong in the browser.
    """
    return np.asarray(data)


def read_camera_frames(
    camera_set: Any,
    camera_names: tuple[str, ...] = INTERACTIVE_GUI_CAMERAS,
) -> dict[str, np.ndarray]:
    """Read the latest RGB (and depth, when the stream carries it) frames.
    """
    frameset = camera_set.get_latest_frames()
    if frameset is None:
        return {}

    latest: dict[str, np.ndarray] = {}
    for cam_name in camera_names:
        frame = frameset.frames.get(cam_name)
        if frame is None:
            continue
        latest[cam_name] = camera_color_to_rgb(frame.camera.color.data)
        if not cam_name.endswith("_right"):
            depth = getattr(frame.camera, "depth", None)
            depth_data = getattr(depth, "data", None) if depth is not None else None
            if depth_data is not None:
                latest[f"{cam_name.removesuffix('_rgb')}_depth"] = np.asarray(depth_data)
    return latest


def collect_saveable_frames(camera_set: Any) -> dict[str, np.ndarray]:
    """Extract enabled camera streams from a live hardware camera set.

    Reads directly from ``camera_set.get_latest_frames()`` (rather than an
    RCS obs dict), so both the real inference loop and the dry-run path go
    through the same source. Filters to :data:`INTERACTIVE_GUI_CAMERAS` for
    left cams, includes their ``_right`` twins when :data:`INCLUDE_RIGHT`,
    and attaches depth (left cams only) when :data:`INCLUDE_DEPTH`.
    """
    frameset = camera_set.get_latest_frames()
    if frameset is None:
        return {}

    out: dict[str, np.ndarray] = {}
    for cam_name, frame in frameset.frames.items():
        if cam_name.endswith("_right"):
            if not INCLUDE_RIGHT:
                continue
        elif cam_name not in INTERACTIVE_GUI_CAMERAS:
            continue

        if frame.camera.color.data is not None:
            out[cam_name] = np.asarray(frame.camera.color.data)

        if INCLUDE_DEPTH and not cam_name.endswith("_right"):
            depth = frame.camera.depth
            if depth is not None and depth.data is not None:
                out[f"{cam_name.removesuffix('_rgb')}_depth"] = np.asarray(depth.data)
    return out


def save_camera_frames(frames: dict[str, np.ndarray], output_dir: str | Path) -> Path:
    """Write RGB as PNG and depth as TIFF under ``output_dir``."""
    from PIL import Image

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for name, data in sorted(frames.items()):
        arr = np.asarray(data)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            Image.fromarray(arr).save(out / f"{name}.png")
            saved.append(name)
        elif arr.ndim == 2:
            Image.fromarray(arr).save(out / f"{name}.tiff")
            saved.append(name)
        else:
            logger.warning(
                "save-first-frame: skipping %s with unexpected shape %s",
                name,
                arr.shape,
            )
    logger.info(
        "save-first-frame: wrote %d file(s) [%s] to %s",
        len(saved),
        ", ".join(saved),
        out.resolve(),
    )
    return out


def maybe_save_first_frame(
    cfg: Any,
    *,
    camera_set: Any | None = None,
    first_frame_saved: bool,
) -> bool:
    if not cfg.save_first_frame or first_frame_saved:
        return first_frame_saved

    if camera_set is None:
        logger.warning("save-first-frame: no camera source available")
        return first_frame_saved

    frames = collect_saveable_frames(camera_set)
    if not frames:
        logger.warning("save-first-frame: no frames matched current hardware flags")
        return first_frame_saved

    save_camera_frames(frames, cfg.save_first_frame_dir)
    return True


def try_open_hardware_camera_set() -> Any | None:
    """Open configured hardware cameras without connecting to the robot."""
    from rcs_fr3.creators import _create_hardware_camera_set

    camera_cfgs = build_hardware_camera_cfgs(connected_only=True)
    if not camera_cfgs:
        return None

    camera_set = None
    try:
        camera_set = _create_hardware_camera_set(camera_cfgs)
        if camera_set is None:
            return None
        camera_set.start(warm_up=True)
        camera_set.wait_for_frames(timeout=10.0)
        logger.info(
            "dry-run opened hardware cameras: %s",
            ", ".join(camera_set.camera_names),
        )
        return camera_set
    except Exception:
        logger.warning(
            "dry-run: hardware cameras unavailable, using mock frames",
            exc_info=True,
        )
        if camera_set is not None:
            try:
                camera_set.close()
            except Exception:
                logger.exception("failed to close camera set after dry-run open error")
        return None
