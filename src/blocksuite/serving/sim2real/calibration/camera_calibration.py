"""Eye-in-hand / eye-to-hand ChArUco calibration for DROID-style ZED cameras.

Board is stationary on the ground. The wrist camera (``gripper_rgb``) moves with
the end-effector via small relative Cartesian steps; the side camera
(``front_rgb``) is fixed. A Tk GUI shows both streams with board detection
overlays, a Start button to run the procedure, and after calibration draws the
solved board origin in both views. Cameras run at HD1080.

Example:

    python -m blocksuite.serving.sim2real.calibration.camera_calibration



========== Calibration results ==========
Wrist camera → TCP (cam→TCP, T_ee_cam):
[[-0.023223  0.94676   0.321103 -0.072296]
 [-0.999709 -0.024089 -0.001275  0.031937]
 [ 0.006528 -0.321039  0.947043 -0.126536]
 [ 0.        0.        0.        1.      ]]

Side camera → robot base (cam→base, T_base_cam):
[[ 0.007767  0.670723 -0.741667  0.907875]
 [ 0.999959 -0.008588  0.002706 -0.003992]
 [-0.004554 -0.741658 -0.670763  0.489332]
 [ 0.        0.        0.        1.      ]]

Robot base → board (T_base_board):
[[-0.999397 -0.027105  0.021709  0.619178]
 [-0.027638  0.999314 -0.024663 -0.042821]
 [-0.021025 -0.025248 -0.99946   0.158573]
 [ 0.        0.        0.        1.      ]]
=========================================
"""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import Any

import cv2
import numpy as np
import rcs
from PIL import Image, ImageTk
from rcs._core.common import BaseCameraConfig, RobotPlatform
from rcs.envs.base import ControlMode, RelativeTo
from rcs_fr3.creators import HardwareCameraCreatorConfig
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)

ROBOT_INSTANCE = RobotPlatform.HARDWARE
ROBOT_IP = "192.168.1.12"
ROBOTIQ_SERIAL = "DAANTG8W"
HOME_POSE = rcs.HOME_POSITIONS["FR3_DROID"]
ROBOT_KEY = "right"

ZED_CAMERA_DICT = {
    "gripper_rgb": "14943057",
    "front_rgb": "35115330",
}
WRIST_CAM = "gripper_rgb"
SIDE_CAM = "front_rgb"

# ZED HD1080
CAM_WIDTH = 1920
CAM_HEIGHT = 1080
CAM_FPS = 30

# ChArUco board on the ground
# OpenCV (squaresX, squaresY): debug probe showed 9x14 → 0 corners, 14x9 → ~100.
SQUARES_X = 14
SQUARES_Y = 9
SQUARE_LENGTH = 0.020  # 20 mm
MARKER_LENGTH = 0.015  # 15 mm
ARUCO_DICT = cv2.aruco.DICT_5X5_1000

SETTLE_FRAMES = 10
CAPTURE_FRAMES = 15
MIN_VALID_FRAMES = 3
# ChArUco chessboard corners required for solvePnP (not ArUco marker count).
# Markers can show in the overlay even when this check fails.
MIN_CORNERS = 4
AXIS_LEN_SCALE = 3.0
GUI_PREVIEW_WIDTH = 640

OUTPUT_PATH = Path(__file__).with_name("camera_calibration_result.npz")
DEBUG_LOG_PATH = Path(__file__).with_name("camera_calibration_debug.log")
DEBUG_FRAME_DIR = Path(__file__).with_name("camera_calibration_debug_frames")

# Alternate dictionaries to probe when the configured one fails (logged only).
PROBE_DICTS = [
    ("DICT_5X5_1000", cv2.aruco.DICT_5X5_1000),
    ("DICT_5X5_250", cv2.aruco.DICT_5X5_250),
    ("DICT_5X5_100", cv2.aruco.DICT_5X5_100),
    ("DICT_5X5_50", cv2.aruco.DICT_5X5_50),
    ("DICT_4X4_1000", cv2.aruco.DICT_4X4_1000),
    ("DICT_6X6_250", cv2.aruco.DICT_6X6_250),
    ("DICT_APRILTAG_36H11", cv2.aruco.DICT_APRILTAG_36H11),
]


def _setup_debug_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # Avoid duplicate handlers on re-import / re-run in same process.
    for h in list(root.handlers):
        if getattr(h, "_calib_debug_file", False):
            root.removeHandler(h)
    fh = logging.FileHandler(DEBUG_LOG_PATH, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    fh._calib_debug_file = True  # type: ignore[attr-defined]
    root.addHandler(fh)
    # Keep console at INFO.
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers):
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        root.addHandler(sh)
    logger.info(
        "Debug log → %s | board=%dx%d square=%.3fm marker=%.3fm dict=DICT_5X5_1000 MIN_CORNERS=%d",
        DEBUG_LOG_PATH,
        SQUARES_X,
        SQUARES_Y,
        SQUARE_LENGTH,
        MARKER_LENGTH,
        MIN_CORNERS,
    )


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def get_aruco_board(dictionary_id: int = ARUCO_DICT, squares_x: int = SQUARES_X, squares_y: int = SQUARES_Y):
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        SQUARE_LENGTH,
        MARKER_LENGTH,
        dictionary,
    )
    detector = cv2.aruco.CharucoDetector(board)
    params = detector.getDetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector.setDetectorParameters(params)
    return board, detector


def camera_matrix_from_intrinsics(intrinsics: np.ndarray) -> np.ndarray:
    return np.asarray(intrinsics[:3, :3], dtype=np.float64)


def pose_from_rt(rot: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot
    T[:3, 3] = np.asarray(t, dtype=np.float64).ravel()
    return T


def invert_pose(T: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    R_t = T[:3, :3].T
    out[:3, :3] = R_t
    out[:3, 3] = -R_t @ T[:3, 3]
    return out


def average_rotations(matrices: list[np.ndarray]) -> np.ndarray:
    return R.from_matrix(matrices).mean().as_matrix()


def _ids_preview(ids: np.ndarray | None, limit: int = 12) -> str:
    if ids is None or len(ids) == 0:
        return "[]"
    flat = np.asarray(ids).ravel().tolist()
    head = flat[:limit]
    suffix = "..." if len(flat) > limit else ""
    return f"{head}{suffix} (n={len(flat)})"


def probe_dictionaries(gray: np.ndarray) -> list[tuple[str, int]]:
    """Try common ArUco dictionaries; return (name, marker_count) sorted desc."""
    hits: list[tuple[str, int]] = []
    for name, dict_id in PROBE_DICTS:
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        det = cv2.aruco.ArucoDetector(aruco_dict)
        _corners, ids, _rej = det.detectMarkers(gray)
        n = 0 if ids is None else int(len(ids))
        if n > 0:
            hits.append((name, n))
    hits.sort(key=lambda x: x[1], reverse=True)
    return hits


def detect_board(
    image_rgb: np.ndarray,
    intrinsics: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    detector: cv2.aruco.CharucoDetector,
    *,
    cam_name: str = "?",
    log: bool = False,
    tag: str = "",
) -> tuple[bool, np.ndarray | None, np.ndarray | None, np.ndarray, dict[str, Any]]:
    viz = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    n_markers = 0 if marker_ids is None else int(len(marker_ids))
    n_corners = 0 if charuco_corners is None else int(len(charuco_corners))
    n_match = 0
    pose_ok = False
    rvec = tvec = None
    fail_reason = "no_charuco_corners"

    if marker_ids is not None and len(marker_ids) > 0:
        cv2.aruco.drawDetectedMarkers(viz, marker_corners, marker_ids)

    if charuco_corners is not None and charuco_ids is not None and n_corners >= MIN_CORNERS:
        cv2.aruco.drawDetectedCornersCharuco(viz, charuco_corners, charuco_ids)
        obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
        n_match = 0 if obj_points is None else int(len(obj_points))
        if obj_points is not None and img_points is not None and n_match >= MIN_CORNERS:
            k = camera_matrix_from_intrinsics(intrinsics)
            dist = np.zeros(5, dtype=np.float64)
            ok, rvec, tvec = cv2.solvePnP(
                obj_points, img_points, k, dist, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if ok:
                pose_ok = True
                fail_reason = ""
                axis_len = float(board.getSquareLength()) * AXIS_LEN_SCALE
                cv2.drawFrameAxes(viz, k, dist, rvec, tvec, axis_len)
            else:
                fail_reason = "solvePnP_failed"
        else:
            fail_reason = f"matchImagePoints<{MIN_CORNERS} (n_match={n_match})"
    elif n_corners > 0:
        fail_reason = f"corners<{MIN_CORNERS} (n_corners={n_corners})"
    elif n_markers > 0:
        fail_reason = "markers_but_no_charuco_corners"
    else:
        fail_reason = "no_markers"

    status = f"markers={n_markers} corners={n_corners}/{MIN_CORNERS} {'OK' if pose_ok else 'FAIL'}"
    color = (0, 200, 0) if pose_ok else (0, 0, 220)
    cv2.putText(viz, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    if fail_reason:
        cv2.putText(viz, fail_reason, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 220), 2, cv2.LINE_AA)

    info: dict[str, Any] = {
        "cam": cam_name,
        "pose_ok": pose_ok,
        "n_markers": n_markers,
        "n_corners": n_corners,
        "n_match": n_match,
        "fail_reason": fail_reason,
        "marker_ids": _ids_preview(marker_ids),
        "charuco_ids": _ids_preview(charuco_ids),
        "shape": tuple(image_rgb.shape),
        "gray_mean": float(np.mean(gray)),
        "gray_std": float(np.std(gray)),
        "K_diag": camera_matrix_from_intrinsics(intrinsics).diagonal().tolist(),
        "tvec": None if tvec is None else np.asarray(tvec).ravel().tolist(),
    }

    if log:
        logger.debug(
            "[%s%s] cam=%s pose_ok=%s markers=%d corners=%d match=%d reason=%s "
            "marker_ids=%s charuco_ids=%s shape=%s gray=%.1f±%.1f Kdiag=%s tvec=%s",
            tag,
            "" if not tag else " ",
            cam_name,
            pose_ok,
            n_markers,
            n_corners,
            n_match,
            fail_reason or "ok",
            info["marker_ids"],
            info["charuco_ids"],
            info["shape"],
            info["gray_mean"],
            info["gray_std"],
            info["K_diag"],
            info["tvec"],
        )

    return pose_ok, rvec, tvec, viz, info


def save_debug_capture(
    pose_idx: int,
    cam_name: str,
    rgb: np.ndarray,
    viz_bgr: np.ndarray,
    info: dict[str, Any],
) -> None:
    DEBUG_FRAME_DIR.mkdir(parents=True, exist_ok=True)
    stem = DEBUG_FRAME_DIR / f"pose{pose_idx:02d}_{cam_name}"
    # Store RGB as BGR for cv2.imwrite
    cv2.imwrite(str(stem.with_name(stem.name + "_rgb.jpg")), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(stem.with_name(stem.name + "_viz.jpg")), viz_bgr)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hits = probe_dictionaries(gray)
    # Also probe swapped board size with configured dict
    swapped_hits = []
    for name, dict_id in PROBE_DICTS[:4]:
        for sx, sy in ((SQUARES_X, SQUARES_Y), (SQUARES_Y, SQUARES_X)):
            b, d = get_aruco_board(dict_id, sx, sy)
            cc, ci, _mc, mi = d.detectBoard(gray)
            nm = 0 if mi is None else int(len(mi))
            nc = 0 if cc is None else int(len(cc))
            if nm or nc:
                swapped_hits.append(f"{name} board={sx}x{sy} markers={nm} corners={nc}")

    logger.warning(
        "Saved debug frames %s_*.jpg | info=%s | dict_probe=%s | board_probe=%s",
        stem.name,
        info,
        hits,
        swapped_hits,
    )


def draw_frame_on_image(
    viz_bgr: np.ndarray,
    intrinsics: np.ndarray,
    T_cam_frame: np.ndarray,
    *,
    axis_len: float = 0.12,
    label: str = "robot base",
) -> np.ndarray:
    """Draw a coordinate frame that is expressed in the camera frame (T_cam_frame)."""
    if not np.isfinite(T_cam_frame).all():
        return viz_bgr
    # Skip if origin is behind the camera.
    if T_cam_frame[2, 3] <= 1e-4:
        return viz_bgr

    k = camera_matrix_from_intrinsics(intrinsics)
    dist = np.zeros(5, dtype=np.float64)
    rvec, _ = cv2.Rodrigues(T_cam_frame[:3, :3])
    tvec = T_cam_frame[:3, 3].reshape(3, 1)
    cv2.drawFrameAxes(viz_bgr, k, dist, rvec, tvec, axis_len)

    origin_img, _ = cv2.projectPoints(
        np.zeros((1, 3), dtype=np.float64),
        rvec,
        tvec,
        k,
        dist,
    )
    u, v = origin_img.reshape(2).astype(int)
    if 0 <= u < viz_bgr.shape[1] and 0 <= v < viz_bgr.shape[0]:
        cv2.circle(viz_bgr, (u, v), 8, (0, 255, 255), -1)
        cv2.putText(
            viz_bgr,
            label,
            (u + 10, v - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return viz_bgr


def generate_calibration_moves() -> list[np.ndarray]:
    """Small relative TCP moves (tquat), same strategy as the original script."""

    def create_act(pos: list[float], euler_deg: list[float]) -> np.ndarray:
        quat = R.from_euler("xyz", euler_deg, degrees=True).as_quat()
        return np.concatenate([pos, quat])

    moves: list[np.ndarray] = []

    # Translations (X, Y plane)
    moves.append(create_act([0.15, 0, 0], [0, 0, 0]))
    moves.append(create_act([-0.30, 0, 0], [0, 0, 0]))
    moves.append(create_act([0.15, 0, 0], [0, 0, 0]))

    moves.append(create_act([0, 0.08, 0], [0, 0, 0]))
    moves.append(create_act([0, -0.16, 0], [0, 0, 0]))
    moves.append(create_act([0, 0.08, 0], [0, 0, 0]))

    # Z-axis translations
    moves.append(create_act([0, 0, 0.10], [0, 0, 0]))
    moves.append(create_act([0, 0, -0.15], [0, 0, 0]))
    moves.append(create_act([0, 0, 0.05], [0, 0, 0]))

    # Rotations (pitch, roll, yaw)
    moves.append(create_act([0, 0, 0], [15, 0, 0]))
    moves.append(create_act([0, 0, 0], [-30, 0, 0]))
    moves.append(create_act([0, 0, 0], [15, 0, 0]))

    moves.append(create_act([0, 0, 0], [0, 15, 0]))
    moves.append(create_act([0, 0, 0], [0, -30, 0]))
    moves.append(create_act([0, 0, 0], [0, 15, 0]))

    moves.append(create_act([0, 0, 0], [0, 0, 25]))
    moves.append(create_act([0, 0, 0], [0, 0, -50]))
    moves.append(create_act([0, 0, 0], [0, 0, 25]))

    # Combined
    moves.append(create_act([0.05, 0.05, 0.05], [10, 10, 10]))
    moves.append(create_act([-0.10, -0.10, -0.10], [-20, -20, -20]))
    moves.append(create_act([0.05, 0.05, 0.05], [10, 10, 10]))

    return moves


def hold_action() -> dict[str, Any]:
    return {
        ROBOT_KEY: {
            "tquat": np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float64),
            "gripper": np.array([1.0]),
        }
    }


def move_action(tquat: np.ndarray) -> dict[str, Any]:
    return {
        ROBOT_KEY: {
            "tquat": np.asarray(tquat, dtype=np.float64),
            "gripper": np.array([1.0]),
        }
    }


def get_env():
    from rcs_fr3.configs import DROIDEnv

    env_creator = DROIDEnv()
    env_creator.robot_ip = ROBOT_IP
    env_creator.gripper_serial_number = ROBOTIQ_SERIAL
    hw_cfg = env_creator.config()
    hw_cfg.robot_cfgs[ROBOT_KEY].q_home = HOME_POSE
    hw_cfg.robot_cfgs[ROBOT_KEY].ignore_realtime = True
    hw_cfg.robot_cfgs[ROBOT_KEY].speed_factor = 0.3
    # Sync motion so each calibration pose is reached before the next step.
    hw_cfg.robot_cfgs[ROBOT_KEY].async_control = False
    if ROBOT_KEY in hw_cfg.gripper_cfgs:
        hw_cfg.gripper_cfgs[ROBOT_KEY].async_control = False
    hw_cfg.control_mode = ControlMode.CARTESIAN_TQuat
    hw_cfg.relative_to = RelativeTo.LAST_STEP
    hw_cfg.max_relative_movement = (0.5, np.deg2rad(90))
    hw_cfg.wrapper_cfg.include_depth = False
    hw_cfg.wrapper_cfg.binary_gripper = True
    hw_cfg.wrapper_cfg.home_on_reset = True
    hw_cfg.camera_cfgs = {
        "zed": HardwareCameraCreatorConfig(
            camera_type_id="zed",
            camera_cfgs={
                name: BaseCameraConfig(
                    identifier=identifier,
                    resolution_width=CAM_WIDTH,
                    resolution_height=CAM_HEIGHT,
                    frame_rate=CAM_FPS,
                )
                for name, identifier in ZED_CAMERA_DICT.items()
            },
            kwargs={"enable_depth": False, "enable_imu": False},
        )
    }
    env = env_creator.create_env(hw_cfg)
    robot = env.get_wrapper_attr("robot")[ROBOT_KEY]
    rcfg = robot.get_config()
    rcfg.kp = np.array([600, 600, 600, 600, 250, 150, 50])
    rcfg.kd = np.array([50, 50, 50, 50, 30, 25, 15])
    robot.set_config(rcfg)
    return env


@dataclass
class CalibrationResult:
    T_ee_cam_wrist: np.ndarray
    T_base_board: np.ndarray
    T_base_cam_side: np.ndarray
    methods_wrist: dict[str, np.ndarray]


class CalibrationApp:
    def __init__(self) -> None:
        self.board, self.detector = get_aruco_board()
        self.env = get_env()
        self.env.reset()
        self.robot = self.env.get_wrapper_attr("robot")[ROBOT_KEY]
        self.camera_set = self.env.get_wrapper_attr("camera_set")

        self._lock = threading.Lock()
        self._latest_viz: dict[str, np.ndarray] = {}
        self._status = "Ready — position wrist over the board, then Start Calibration"
        self._calibrating = False
        self._guiding = False
        self._shutdown = threading.Event()
        self._result: CalibrationResult | None = None
        self._start_requested = threading.Event()
        self._load_saved_result()

        self.root = tk.Tk()
        self.root.title("ZED ChArUco Calibration")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.status_var = tk.StringVar(value=self._status)
        tk.Label(self.root, textvariable=self.status_var, font=("Helvetica", 12)).pack(pady=6)

        btn_row = tk.Frame(self.root)
        btn_row.pack(pady=4)
        self.start_btn = tk.Button(
            btn_row,
            text="Start Calibration",
            command=self._request_start,
            font=("Helvetica", 14),
            width=20,
            height=2,
        )
        self.start_btn.pack(side=tk.LEFT, padx=8)
        tk.Button(btn_row, text="Quit", command=self._on_close, width=10, height=2).pack(side=tk.LEFT, padx=8)

        cams = tk.Frame(self.root)
        cams.pack(padx=8, pady=8)
        self._photo: dict[str, ImageTk.PhotoImage | None] = {WRIST_CAM: None, SIDE_CAM: None}
        self._labels: dict[str, tk.Label] = {}
        for i, name in enumerate((WRIST_CAM, SIDE_CAM)):
            col = tk.Frame(cams)
            col.grid(row=0, column=i, padx=6)
            tk.Label(col, text=name, font=("Helvetica", 11, "bold")).pack()
            lbl = tk.Label(col)
            lbl.pack()
            self._labels[name] = lbl

        self._worker = threading.Thread(target=self._worker_loop, name="calib-worker", daemon=True)
        self._worker.start()
        self.root.after(33, self._update_gui)

    def _set_status(self, text: str) -> None:
        with self._lock:
            self._status = text

    def _load_saved_result(self) -> None:
        if not OUTPUT_PATH.exists():
            return
        try:
            data = np.load(OUTPUT_PATH, allow_pickle=True)
            methods = {
                name.removeprefix("T_ee_cam_wrist_"): data[name]
                for name in data.files
                if name.startswith("T_ee_cam_wrist_")
            }
            self._result = CalibrationResult(
                T_ee_cam_wrist=np.asarray(data["T_ee_cam_wrist"], dtype=np.float64),
                T_base_board=np.asarray(data["T_base_board"], dtype=np.float64),
                T_base_cam_side=np.asarray(data["T_base_cam_side"], dtype=np.float64),
                methods_wrist=methods,
            )
            self._enable_guiding()
            self._set_status(
                f"Loaded {OUTPUT_PATH.name} — guiding on, 'robot base' overlay active. "
                "Start Calibration to re-run."
            )
            logger.info("Loaded previous calibration from %s", OUTPUT_PATH)
        except Exception:
            logger.exception("Failed to load %s", OUTPUT_PATH)

    def _enable_guiding(self) -> None:
        try:
            self.robot.stop_control_thread()
        except Exception:
            logger.exception("stop_control_thread before guiding failed")
        try:
            self.robot.set_guiding_mode(True, True, True, True, True, True, True)
            self.robot.zero_torque_guiding()
            self._guiding = True
            logger.info("Zero-torque guiding active — drag the arm to check the origin overlay.")
        except Exception:
            self._guiding = False
            logger.exception("Failed to enable zero-torque guiding")
            self._set_status("Guiding failed — see logs. FCI unlocked?")

    def _disable_guiding(self) -> None:
        if not self._guiding:
            return
        try:
            self.robot.stop_control_thread()
        except Exception:
            logger.exception("stop_control_thread while leaving guiding failed")
        self._guiding = False

    def _ee_tquat(self) -> np.ndarray:
        pose = self.robot.get_cartesian_position()
        return np.concatenate(
            [np.asarray(pose.translation(), dtype=np.float64), np.asarray(pose.rotation_q(), dtype=np.float64)]
        )

    def _read_camera_frames(self) -> dict[str, dict[str, np.ndarray]] | None:
        frameset = self.camera_set.get_latest_frames()
        if frameset is None:
            return None
        out: dict[str, dict[str, np.ndarray]] = {}
        for cam_name in (WRIST_CAM, SIDE_CAM):
            frame = frameset.frames.get(cam_name)
            if frame is None:
                return None
            out[cam_name] = {
                "rgb": np.asarray(frame.camera.color.data),
                "intrinsics": np.asarray(frame.camera.color.intrinsics, dtype=np.float64),
            }
        return out

    def _request_start(self) -> None:
        if self._calibrating:
            return
        self._start_requested.set()
        self.start_btn.config(state=tk.DISABLED)
        self._set_status("Calibration starting...")

    def _on_close(self) -> None:
        self._shutdown.set()
        self.root.after(200, self.root.destroy)

    def _annotate_frame(
        self,
        cam_name: str,
        rgb: np.ndarray,
        intrinsics: np.ndarray,
        ee_tquat: np.ndarray | None,
    ) -> np.ndarray:
        _ok, _rvec, _tvec, viz, _info = detect_board(
            rgb, intrinsics, self.board, self.detector, cam_name=cam_name
        )
        result = self._result
        if result is not None and ee_tquat is not None:
            T_cam_base = self._robot_base_in_camera(cam_name, ee_tquat, result)
            if T_cam_base is not None:
                draw_frame_on_image(
                    viz,
                    intrinsics,
                    T_cam_base,
                    axis_len=0.12,
                    label="robot base",
                )
        return viz

    def _robot_base_in_camera(
        self,
        cam_name: str,
        ee_tquat: np.ndarray,
        result: CalibrationResult,
    ) -> np.ndarray | None:
        """Pose of the robot base frame expressed in the given camera frame."""
        if cam_name == SIDE_CAM:
            # Stored T_base_cam maps camera → base; invert for OpenCV drawFrameAxes.
            return invert_pose(result.T_base_cam_side)
        if cam_name == WRIST_CAM:
            T_base_gripper = pose_from_rt(R.from_quat(ee_tquat[3:]).as_matrix(), ee_tquat[:3])
            # T_ee_cam is cam→TCP; compose to cam→base, then invert.
            T_base_cam = T_base_gripper @ result.T_ee_cam_wrist
            return invert_pose(T_base_cam)
        return None

    def _preview_step(self) -> None:
        if self._guiding:
            frames = self._read_camera_frames()
            if frames is None:
                sleep(0.02)
                return
            ee_tquat = self._ee_tquat()
            viz_map = {
                cam_name: self._annotate_frame(cam_name, data["rgb"], data["intrinsics"], ee_tquat)
                for cam_name, data in frames.items()
            }
            with self._lock:
                self._latest_viz = viz_map
            sleep(0.01)
            return

        obs, _, _, _, _ = self.env.step(hold_action())
        ee_tquat = np.asarray(obs[ROBOT_KEY]["tquat"], dtype=np.float64)
        viz_map: dict[str, np.ndarray] = {}
        for cam_name in (WRIST_CAM, SIDE_CAM):
            frame = obs["frames"][cam_name]
            rgb = frame["rgb"]["data"]
            intrinsics = np.asarray(frame["rgb"]["intrinsics"], dtype=np.float64)
            viz_map[cam_name] = self._annotate_frame(cam_name, rgb, intrinsics, ee_tquat)
        with self._lock:
            self._latest_viz = viz_map

    def _capture_pose_sample(self, pose_idx: int) -> dict[str, Any] | None:
        ee_pos_samples: list[np.ndarray] = []
        ee_quat_samples: list[np.ndarray] = []
        wrist_rvecs: list[np.ndarray] = []
        wrist_tvecs: list[np.ndarray] = []
        side_rvecs: list[np.ndarray] = []
        side_tvecs: list[np.ndarray] = []
        last_viz: dict[str, np.ndarray] = {}
        wrist_infos: list[dict[str, Any]] = []
        saved_debug = False

        for frame_i in range(CAPTURE_FRAMES):
            obs, _, _, _, _ = self.env.step(hold_action())
            ee_tquat = np.asarray(obs[ROBOT_KEY]["tquat"], dtype=np.float64)

            wrist_ok = False
            for cam_name in (WRIST_CAM, SIDE_CAM):
                frame = obs["frames"][cam_name]
                rgb = frame["rgb"]["data"]
                intrinsics = np.asarray(frame["rgb"]["intrinsics"], dtype=np.float64)
                ok, rvec, tvec, viz, info = detect_board(
                    rgb,
                    intrinsics,
                    self.board,
                    self.detector,
                    cam_name=cam_name,
                    log=True,
                    tag=f"pose{pose_idx}/f{frame_i}",
                )
                last_viz[cam_name] = viz
                if cam_name == WRIST_CAM:
                    wrist_infos.append(info)
                    # Save first failing wrist frame for offline inspection.
                    if not ok and not saved_debug:
                        save_debug_capture(pose_idx, cam_name, np.asarray(rgb), viz, info)
                        saved_debug = True
                if not ok or rvec is None or tvec is None:
                    continue
                if cam_name == WRIST_CAM:
                    wrist_ok = True
                    wrist_rvecs.append(rvec.ravel())
                    wrist_tvecs.append(tvec.ravel())
                else:
                    side_rvecs.append(rvec.ravel())
                    side_tvecs.append(tvec.ravel())

            if wrist_ok:
                ee_pos_samples.append(ee_tquat[:3])
                ee_quat_samples.append(ee_tquat[3:])

            with self._lock:
                self._latest_viz = dict(last_viz)

        if len(wrist_tvecs) < MIN_VALID_FRAMES:
            reasons = {}
            for info in wrist_infos:
                r = info.get("fail_reason") or "ok"
                reasons[r] = reasons.get(r, 0) + 1
            marker_counts = [info["n_markers"] for info in wrist_infos]
            corner_counts = [info["n_corners"] for info in wrist_infos]
            logger.warning(
                "Pose %d capture failed: %d/%d valid wrist frames (need >= %d). "
                "marker_counts=%s corner_counts=%s fail_reasons=%s",
                pose_idx,
                len(wrist_tvecs),
                CAPTURE_FRAMES,
                MIN_VALID_FRAMES,
                marker_counts,
                corner_counts,
                reasons,
            )
            return None

        sample: dict[str, Any] = {
            "ee_pos": np.mean(ee_pos_samples, axis=0),
            "ee_rot": R.from_quat(ee_quat_samples).mean().as_matrix(),
            "wrist_rot": R.from_rotvec(wrist_rvecs).mean().as_matrix(),
            "wrist_t": np.mean(wrist_tvecs, axis=0).reshape(3, 1),
            "n_wrist": len(wrist_tvecs),
            "n_side": len(side_tvecs),
        }
        if len(side_tvecs) >= MIN_VALID_FRAMES:
            sample["side_rot"] = R.from_rotvec(side_rvecs).mean().as_matrix()
            sample["side_t"] = np.mean(side_tvecs, axis=0).reshape(3, 1)
        return sample

    def _run_calibration(self) -> None:
        self._calibrating = True
        self._result = None
        self._disable_guiding()
        self._set_status("Starting pose sequence...")

        samples: list[dict[str, Any]] = []
        moves = generate_calibration_moves()

        self.env.step(hold_action())
        for _ in range(SETTLE_FRAMES):
            self.env.step(hold_action())

        for i, action_tquat in enumerate(moves):
            if self._shutdown.is_set():
                break
            self._set_status(f"Pose {i + 1}/{len(moves)}: moving...")
            self.env.step(move_action(action_tquat))
            for _ in range(SETTLE_FRAMES):
                self.env.step(hold_action())

            self._set_status(f"Pose {i + 1}/{len(moves)}: capturing...")
            sample = self._capture_pose_sample(pose_idx=i + 1)
            if sample is None:
                logger.warning("Pose %d: failed (not enough wrist detections)", i + 1)
                self._set_status(f"Pose {i + 1}: failed — board not seen by wrist cam")
                continue
            samples.append(sample)
            logger.info(
                "Pose %d: ok (wrist=%d side=%d frames)",
                i + 1,
                sample["n_wrist"],
                sample["n_side"],
            )
            self._set_status(f"Pose {i + 1}: ok ({len(samples)} valid so far)")

        if len(samples) < 5:
            self._set_status(f"Not enough poses ({len(samples)}); need >= 5. Ready to retry.")
            self._calibrating = False
            self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL))
            return

        self._set_status(f"Solving hand-eye with {len(samples)} poses...")
        R_gripper2base = [s["ee_rot"] for s in samples]
        t_gripper2base = [s["ee_pos"] for s in samples]
        R_target2cam = [s["wrist_rot"] for s in samples]
        t_target2cam = [s["wrist_t"] for s in samples]

        methods = {
            "Tsai": cv2.CALIB_HAND_EYE_TSAI,
            "Park": cv2.CALIB_HAND_EYE_PARK,
            "Horaud": cv2.CALIB_HAND_EYE_HORAUD,
            "Andreff": cv2.CALIB_HAND_EYE_ANDREFF,
            "Daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
        }
        methods_wrist: dict[str, np.ndarray] = {}
        best_R, best_t = None, None
        for name, flag in methods.items():
            try:
                R_cg, t_cg = cv2.calibrateHandEye(
                    R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, method=flag
                )
                methods_wrist[name] = pose_from_rt(R_cg, t_cg)
                if flag == cv2.CALIB_HAND_EYE_DANIILIDIS:
                    best_R, best_t = R_cg, t_cg
            except Exception as exc:
                logger.warning("%s hand-eye skipped (%s)", name, exc)

        if best_R is None or best_t is None:
            if not methods_wrist:
                self._set_status("All hand-eye methods failed. Ready to retry.")
                self._calibrating = False
                self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL))
                return
            T_fallback = next(iter(methods_wrist.values()))
            best_R, best_t = T_fallback[:3, :3], T_fallback[:3, 3]

        T_ee_cam = pose_from_rt(best_R, best_t)

        board_in_base = []
        for s in samples:
            T_base_gripper = pose_from_rt(s["ee_rot"], s["ee_pos"])
            T_cam_board = pose_from_rt(s["wrist_rot"], s["wrist_t"])
            board_in_base.append(T_base_gripper @ T_ee_cam @ T_cam_board)
        T_base_board = pose_from_rt(
            average_rotations([T[:3, :3] for T in board_in_base]),
            np.mean([T[:3, 3] for T in board_in_base], axis=0),
        )

        side_samples = [s for s in samples if "side_rot" in s]
        if not side_samples:
            logger.warning("Wrist OK but side camera never saw the board.")
            T_base_cam_side = np.eye(4)
        else:
            T_cam_board_side = pose_from_rt(
                average_rotations([s["side_rot"] for s in side_samples]),
                np.mean([s["side_t"].ravel() for s in side_samples], axis=0),
            )
            T_base_cam_side = T_base_board @ invert_pose(T_cam_board_side)

        result = CalibrationResult(
            T_ee_cam_wrist=T_ee_cam,
            T_base_board=T_base_board,
            T_base_cam_side=T_base_cam_side,
            methods_wrist=methods_wrist,
        )
        self._result = result
        self._save_result(result)

        np.set_printoptions(suppress=True, precision=6)
        print("\n========== Calibration results ==========")
        print("Wrist camera → TCP (cam→TCP, T_ee_cam):")
        print(T_ee_cam)
        print("\nSide camera → robot base (cam→base, T_base_cam):")
        print(T_base_cam_side)
        print("\nRobot base → board (T_base_board):")
        print(T_base_board)
        print("=========================================\n")

        self._enable_guiding()
        self._set_status(
            f"Done — {len(samples)} poses. Guiding on — check 'robot base' overlay. "
            f"Results → {OUTPUT_PATH.name}"
        )
        self._calibrating = False
        self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL))

    def _save_result(self, result: CalibrationResult) -> None:
        payload = {
            "T_ee_cam_wrist": result.T_ee_cam_wrist,
            "T_base_board": result.T_base_board,
            "T_base_cam_side": result.T_base_cam_side,
            "squares_x": SQUARES_X,
            "squares_y": SQUARES_Y,
            "square_length": SQUARE_LENGTH,
            "marker_length": MARKER_LENGTH,
            "wrist_cam": WRIST_CAM,
            "side_cam": SIDE_CAM,
            "cam_width": CAM_WIDTH,
            "cam_height": CAM_HEIGHT,
        }
        for name, T in result.methods_wrist.items():
            payload[f"T_ee_cam_wrist_{name}"] = T
        np.savez(OUTPUT_PATH, **payload)
        logger.info("Wrote %s", OUTPUT_PATH)

    def _worker_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                if self._start_requested.is_set():
                    self._start_requested.clear()
                    self._run_calibration()
                    continue
                if not self._calibrating:
                    self._preview_step()
                else:
                    sleep(0.01)
            except Exception:
                logger.exception("Worker loop error")
                self._set_status("Error in worker — see logs. Ready to retry.")
                self._calibrating = False
                self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL))
                sleep(0.5)
        try:
            self._disable_guiding()
            self.env.close()
        except Exception:
            logger.exception("Failed to close env")

    def _update_gui(self) -> None:
        with self._lock:
            viz = dict(self._latest_viz)
            status = self._status
        self.status_var.set(status)
        for cam_name, frame_bgr in viz.items():
            if cam_name not in self._labels:
                continue
            h, w = frame_bgr.shape[:2]
            scale = GUI_PREVIEW_WIDTH / max(w, 1)
            if scale < 1.0:
                frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            photo = ImageTk.PhotoImage(image=image)
            self._photo[cam_name] = photo
            self._labels[cam_name].configure(image=photo)
        if not self._shutdown.is_set():
            self.root.after(33, self._update_gui)

    def run(self) -> None:
        self.root.mainloop()
        self._shutdown.set()
        self._worker.join(timeout=5.0)


def main() -> None:
    if ROBOT_INSTANCE != RobotPlatform.HARDWARE:
        raise SystemExit("This calibration GUI expects hardware + ZED cameras.")
    _setup_debug_logging()
    logger.info("Building env (robot + ZED cameras)...")
    app = CalibrationApp()
    app.run()


if __name__ == "__main__":
    main()
