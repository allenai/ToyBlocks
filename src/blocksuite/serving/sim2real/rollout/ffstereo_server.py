"""Fast-FoundationStereo depth server for the vlagent RCS sim-to-real client.

Mirrors :mod:`vlagents_server`: a :class:`vlagents.Agent` subclass served through vlagents' rpyc
``AgentService`` (``register_agent`` + ``start_server``). Instead of a policy it turns a rectified
stereo pair into metric depth. The model code and weights live in the stand-alone ffstereo folder
(``FFSTEREO_ROOT``); the runtime dependencies (torch cu128, timm, omegaconf, ...) are already part
of this venv, so the server runs with the same interpreter as the client.

Wire contract (one ``SingleObs`` per camera, robot key = camera name, e.g. ``front_rgb``)::

    Obs(obs={cam: SingleObs(cameras={"left": HxWx3 uint8, "right": HxWx3 uint8},
                            info={"intrinsics": 3x4 (or 3x3), "serial": str | None,
                                  "baseline_m": float | None, "clip_m": float})})
    -> Act(acts=[{cam: SingleAct(action=zeros(1), info={"depth": HxW uint16 mm (0 = invalid),
                                                       "intrinsics": 3x4 float32, "fx", "baseline_m",
                                                       "fxB", "valid_frac", "disp_median_px",
                                                       "clip_m", "infer_ms"})}])

Images travel through POSIX shared memory when client and server run on the same machine
(``RemoteAgent(on_same_machine=True, image_size=None)``), depth comes back as json_numpy.

Example::

    python -m blocksuite.serving.sim2real.rollout.ffstereo_server --port 8081
    python -m blocksuite.serving.sim2real.rollout.ffstereo_server --port 8081 --scale 0.75 --valid-iters 8

``vlagent_rcs_client --ffstereo`` launches this server automatically (with ``--parent-pid`` so the
server exits when the client is gone) and ``FFStereoClient`` talks to it from
``ModelInference.obs_rcs2agents``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from vlagents import register_agent
from vlagents.__main__ import start_server
from vlagents.client import RemoteAgent
from vlagents.policies.interface import Act, Agent, Obs, SingleAct, SingleObs

from blocksuite.serving.sim2real.rollout.hardware_config import ZED_CAMERA_DICT
from blocksuite.serving.sim2real.rollout.remote_agent_utils import (
    ensure_server_ready,
    remote_agent_act,
)

logger = logging.getLogger(__name__)

AGENT_NAME = "ffstereo"
FFSTEREO_ROOT = Path(os.environ.get("FFSTEREO_ROOT", "/home/yejin/code/ffstereo"))
DEFAULT_WEIGHTS = FFSTEREO_ROOT / "weights" / "c-ffs" / "model_best_bp2_serialize.pth"
DEFAULT_PORT = 8081  # 8080 is the ManiFlow vlagents server
RIGHT_SUFFIX = "_right"
LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "::1")


# --------------------------------------------------------------------------------------------------
# server side
# --------------------------------------------------------------------------------------------------


class FFStereoAgent(Agent):
    """vlagents Agent that returns Fast-FoundationStereo depth instead of actions."""

    def __init__(
        self,
        default_checkpoint_path: str = str(DEFAULT_WEIGHTS),
        device: str = "cuda:0",
        scale: float = 0.5,
        valid_iters: int = 8,
        max_disp: int = 416,
        z_far_m: float = 30.0,
        remove_invisible: bool = False,
        warmup: bool = True,
        warmup_hw: tuple[int, int] = (720, 1280),
        **kwargs: Any,
    ) -> None:
        super().__init__(default_checkpoint_path=default_checkpoint_path, **kwargs)
        self.device = device
        self.scale = float(scale)
        self.valid_iters = int(valid_iters)
        self.max_disp = int(max_disp)
        self.z_far_m = float(z_far_m)
        self.remove_invisible = bool(remove_invisible)
        self.warmup = bool(warmup)
        self.warmup_hw = tuple(warmup_hw)
        self.runner = None
        self._rt = None

    def initialize(self) -> None:
        """Heavy init (runs on AgentService's background thread): load + compile the network."""
        if str(FFSTEREO_ROOT) not in sys.path:
            sys.path.insert(0, str(FFSTEREO_ROOT))
        import ffs_runtime  # noqa: PLC0415  (lives in FFSTEREO_ROOT)

        self._rt = ffs_runtime
        self.runner = ffs_runtime.FFStereoRunner(
            repo=FFSTEREO_ROOT / "Fast-FoundationStereo",
            weights=self.default_checkpoint_path,
            device=self.device,
            valid_iters=self.valid_iters,
            max_disp=self.max_disp,
            scale=self.scale,
        )
        t0 = time.perf_counter()
        self.runner.load()
        logger.info(
            "ffstereo model loaded in %.1fs (device=%s scale=%.2f iters=%d max_disp=%d)",
            time.perf_counter() - t0,
            self.device,
            self.scale,
            self.valid_iters,
            self.max_disp,
        )
        if self.warmup:
            # torch.compile specialises on the batch size: warm up for one camera and for a pair
            for batch in (1, 2):
                secs = self.runner.warmup(*self.warmup_hw, batch=batch)
                logger.info(
                    "ffstereo warm-up forward at %dx%d, batch %d took %.1fs", *self.warmup_hw, batch, secs
                )

    def act(self, obs: Obs) -> Act:
        super().act(obs)  # decodes shared-memory / JPEG camera payloads into numpy arrays
        assert self.runner is not None and self._rt is not None, "FFStereoAgent not initialized"
        names: list[str] = []
        lefts: list[np.ndarray] = []
        rights: list[np.ndarray] = []
        for cam_name, single_obs in obs.obs.items():
            cams = single_obs.cameras
            if "left" not in cams or "right" not in cams:
                raise ValueError(f"camera {cam_name!r}: expected cameras['left'] and ['right']")
            names.append(cam_name)
            lefts.append(np.asarray(cams["left"]))
            rights.append(np.asarray(cams["right"]))
        if not names:
            return Act(acts=[])

        t0 = time.perf_counter()
        disps = self.runner.predict(lefts, rights)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        acts: dict[str, SingleAct] = {}
        for cam_name, disp in zip(names, disps, strict=True):
            info_in = obs.obs[cam_name].info or {}
            K = np.asarray(info_in.get("intrinsics"), dtype=np.float64).reshape(-1)
            if K.size == 9:
                K = np.concatenate([K.reshape(3, 3), np.zeros((3, 1))], axis=1).reshape(-1)
            if K.size != 12:
                raise ValueError(f"camera {cam_name!r}: intrinsics must be 3x3 or 3x4")
            K34 = K.reshape(3, 4).astype(np.float32)
            fx = float(K34[0, 0])
            baseline = info_in.get("baseline_m")
            if baseline is None:
                baseline = self._rt.baseline_for(info_in.get("serial"), cam_name)
            fxB = fx * float(baseline)
            clip_m = float(info_in.get("clip_m") or 0.0)
            clip_min_m = float(info_in.get("clip_min_m") or 0.0)
            depth_mm = self._rt.disp_to_depth_mm(
                disp,
                fxB,
                z_far_m=self.z_far_m,
                remove_invisible=self.remove_invisible,
                clip_m=clip_m,
                clip_min_m=clip_min_m,
            )
            valid = depth_mm > 0
            acts[cam_name] = SingleAct(
                action=np.zeros(1, dtype=np.float32),
                info={
                    "depth": depth_mm,
                    "intrinsics": K34,
                    "fx": fx,
                    "baseline_m": float(baseline),
                    "fxB": fxB,
                    "valid_frac": float(valid.mean()),
                    "disp_median_px": float(np.median(disp[valid])) if valid.any() else 0.0,
                    "z_min_supported_m": self.runner.z_min_supported(fxB),
                    "clip_m": clip_m,
                    "clip_min_m": clip_min_m,
                    "infer_ms": infer_ms,
                },
            )
        return Act(acts=[acts])


register_agent(AGENT_NAME, FFStereoAgent)


def _start_parent_watchdog(parent_pid: int, poll_s: float = 1.0) -> None:
    """Exit this process once the launching client (parent) is gone."""

    def _watch() -> None:
        while True:
            if os.getppid() != parent_pid:
                logging.info("ffstereo server: parent %d is gone, exiting", parent_pid)
                os._exit(0)
            time.sleep(poll_s)

    threading.Thread(target=_watch, name="parent-watchdog", daemon=True).start()


# --------------------------------------------------------------------------------------------------
# client side (used by inference.py)
# --------------------------------------------------------------------------------------------------


def _is_local_host(host: str) -> bool:
    return host in LOCAL_HOSTS or host == socket.gethostname()


def _port_open(host: str, port: int, timeout_s: float = 0.5) -> bool:
    try:
        with socket.create_connection((host if host != "0.0.0.0" else "127.0.0.1", port), timeout=timeout_s):
            return True
    except OSError:
        return False


class FFStereoClient:
    """Synchronous client: env ``frames`` in, ``{<cam>_depth: uint16 mm}`` + intrinsics out."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = DEFAULT_PORT,
        camera_serials: dict[str, str] | None = None,
        ready_timeout_s: float = 600.0,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.camera_serials = dict(camera_serials or {})
        self.remote_agent = self._connect_with_retry(ready_timeout_s)
        ensure_server_ready(self.remote_agent, timeout_s=ready_timeout_s)
        logger.info(
            "ffstereo depth client ready (%s:%d, shared memory=%s, cameras=%s)",
            self.host,
            self.port,
            self.remote_agent.on_same_machine,
            sorted(self.camera_serials) or "any with a *_right frame",
        )

    @classmethod
    def from_config(cls, cfg: Any) -> FFStereoClient:
        """Build from :class:`InferenceConfig` (host/port) and ``hardware_config.ZED_CAMERA_DICT``."""
        return cls(cfg.ffstereo_host, cfg.ffstereo_port, camera_serials=ZED_CAMERA_DICT or {})

    def _connect_with_retry(self, timeout_s: float) -> RemoteAgent:
        deadline = time.monotonic() + timeout_s
        last_exc: Exception | None = None
        while time.monotonic() < deadline:
            try:
                return RemoteAgent(
                    self.host,
                    self.port,
                    AGENT_NAME,
                    on_same_machine=_is_local_host(self.host),
                    jpeg_encoding=False,
                    image_size=None,  # stereo matching needs native resolution
                )
            except (ConnectionRefusedError, EOFError, OSError) as exc:
                last_exc = exc
                time.sleep(0.5)
        raise TimeoutError(
            f"ffstereo server not reachable at {self.host}:{self.port} after {timeout_s:.0f}s"
        ) from last_exc

    @staticmethod
    def stereo_pairs(frames: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
        """``{cam: (left_frame, right_frame)}`` for every camera that has a ``<cam>_right`` twin."""
        pairs = {}
        for name, frame in frames.items():
            if name.endswith(RIGHT_SUFFIX):
                continue
            right = frames.get(f"{name}{RIGHT_SUFFIX}")
            if right is not None:
                pairs[name] = (frame, right)
        return pairs

    def build_obs(
        self,
        frames: dict[str, Any],
        clip_m: float = 0.0,
        clip_min_m: float = 0.0,
    ) -> Obs:
        pairs = self.stereo_pairs(frames)
        if not pairs:
            raise RuntimeError(
                "use_ffstereo=true but no '<camera>_right' frames are present; set "
                "INCLUDE_RIGHT = True (and INCLUDE_DEPTH = False) in hardware_config.py"
            )
        obs: dict[str, SingleObs] = {}
        for cam_name, (left, right) in pairs.items():
            K = left["rgb"].get("intrinsics")
            if K is None or not np.any(np.asarray(K)):
                raise ValueError(f"camera {cam_name!r} has no rgb.intrinsics; cannot convert disparity")
            obs[cam_name] = SingleObs(
                cameras={
                    "left": np.ascontiguousarray(left["rgb"]["data"]),
                    "right": np.ascontiguousarray(right["rgb"]["data"]),
                },
                info={
                    "intrinsics": np.asarray(K, dtype=np.float64),
                    "serial": self.camera_serials.get(cam_name),
                    "baseline_m": None,
                    "clip_m": float(clip_m or 0.0),
                    "clip_min_m": float(clip_min_m or 0.0),
                },
            )
        return Obs(obs=obs, language_instruction=None)

    @staticmethod
    def unpack(act: Act) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, dict]]:
        depths: dict[str, np.ndarray] = {}
        intrinsics: dict[str, np.ndarray] = {}
        meta: dict[str, dict] = {}
        if not act.acts:
            return depths, intrinsics, meta
        for cam_name, single_act in act.acts[0].items():
            info = single_act.info
            key = f"{cam_name.removesuffix('_rgb')}_depth"
            depths[key] = np.asarray(info["depth"], dtype=np.uint16)
            intrinsics[key] = np.asarray(info["intrinsics"], dtype=np.float32).reshape(3, 4)
            meta[key] = {k: v for k, v in info.items() if k not in ("depth", "intrinsics")}
        return depths, intrinsics, meta

    def depths_from_frames(
        self,
        frames: dict[str, Any],
        clip_m: float = 0.0,
        clip_min_m: float = 0.0,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """One synchronous round trip: ``obs["frames"]`` -> (``depths``, ``intrinsics``) like the ZED path."""
        depths, intrinsics, meta = self.unpack(
            remote_agent_act(self.remote_agent, self.build_obs(frames, clip_m, clip_min_m))
        )
        self.last_meta = meta
        return depths, intrinsics

    def close(self) -> None:
        try:
            self.remote_agent.close()
        except Exception:
            logger.exception("failed to close ffstereo client")


# --------------------------------------------------------------------------------------------------
# launching (used by vlagent_rcs_client.main and the end-to-end test)
# --------------------------------------------------------------------------------------------------


def launch_ffstereo_server(
    cfg: Any,
    *,
    port: int | None = None,
    extra_args: tuple[str, ...] | list[str] = (),
    parent_pid: int | None = None,
) -> subprocess.Popen | None:
    """Start the depth server as a child of this process with the same interpreter.

    Returns ``None`` when the configured host is remote or a server already listens on the port.
    The child gets ``--parent-pid`` and exits by itself once this process is gone.
    """
    host = cfg.ffstereo_host
    port = int(port if port is not None else cfg.ffstereo_port)
    if not _is_local_host(host):
        logger.info("ffstereo_host=%s is remote; not launching a local depth server", host)
        return None
    if _port_open(host, port):
        logger.info("ffstereo server already listening on %s:%d; not launching another", host, port)
        return None
    # ``max_depth_m`` on InferenceConfig is the client-facing knob for the
    # server's ``z_far_m`` (max trusted depth beyond which disparities are
    # marked invalid). Forward it so editing vlagent_rcs_client.json takes
    # effect without also having to pass --z-far-m manually.
    z_far_m = getattr(cfg, "max_depth_m", None)
    cmd = [
        sys.executable,
        "-m",
        "blocksuite.serving.sim2real.rollout.ffstereo_server",
        "--host",
        host,
        "--port",
        str(port),
        "--parent-pid",
        str(parent_pid if parent_pid is not None else os.getpid()),
    ]
    if z_far_m is not None:
        cmd.extend(["--z-far-m", str(float(z_far_m))])
    cmd.extend(extra_args)
    proc = subprocess.Popen(cmd)
    logger.info("launched ffstereo depth server (pid %d): %s", proc.pid, " ".join(cmd))
    return proc


def stop_ffstereo_server(proc: subprocess.Popen | None, timeout_s: float = 10.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=timeout_s)


# --------------------------------------------------------------------------------------------------
# saving depth images + point clouds (used by inference.py once per run and by the e2e test)
# --------------------------------------------------------------------------------------------------

# .../robot-control-stack/blocksuite/src/blocksuite/serving/sim2real/<this file> -> .../robot-control-stack/outputs/ffstereo_e2e
DEPTH_OUTPUT_DIR = Path(__file__).resolve().parents[5] / "outputs" / "ffstereo_e2e"
_depth_outputs_saved: set[str] = set()


def unproject_mm(depth_mm: np.ndarray, K: np.ndarray) -> np.ndarray:
    """uint16 mm depth + 3x4 (or 3x3) K -> (H, W, 3) float32 metres in the camera frame; 0 depth -> NaN."""
    K = np.asarray(K, dtype=np.float64).reshape(3, -1)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    H, W = depth_mm.shape
    z = depth_mm.astype(np.float32) / 1000.0
    z[depth_mm == 0] = np.nan
    vs, us = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")
    x = (us - np.float32(cx)) * z / np.float32(fx)
    y = (vs - np.float32(cy)) * z / np.float32(fy)
    return np.stack([x, y, z], axis=-1)


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> int:
    """Binary little-endian PLY (xyz float32 + rgb uint8), skipping invalid (NaN) points; returns #points."""
    pts = xyz.reshape(-1, 3)
    cols = rgb.reshape(-1, 3)
    keep = np.isfinite(pts).all(axis=1)
    pts, cols = pts[keep].astype(np.float32), cols[keep].astype(np.uint8)
    rec = np.empty(
        len(pts),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")],
    )
    rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    rec["r"], rec["g"], rec["b"] = cols[:, 0], cols[:, 1], cols[:, 2]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(rec)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(rec.tobytes())
    return int(len(rec))


def depth_color_range(depth_mm: np.ndarray, clip_m: float = 0.0, vmax_cap_m: float = 3.0) -> tuple[float, float]:
    """Display range in metres for a depth map: 2nd-98th percentile of valid depth, upper end capped at
    ``vmax_cap_m`` (ZED far-range outliers would otherwise flatten the colours); ``0..clip_m`` when clipping."""
    valid = depth_mm[depth_mm > 0].astype(np.float32) / 1000.0
    if clip_m and clip_m > 0:
        return 0.0, float(clip_m)
    if valid.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(valid, [2, 98])
    hi = min(float(hi), vmax_cap_m)
    lo = min(float(lo), hi - 1e-3)
    return float(lo), float(hi)


def colorize_depth_mm(depth_mm: np.ndarray, vmin_m: float, vmax_m: float, label: str = "") -> np.ndarray:
    """uint16 mm depth -> HxWx3 RGB uint8 (viridis, near = dark, far = yellow; invalid = black) with a range label."""
    import cv2

    depth_m = depth_mm.astype(np.float32) / 1000.0
    t = np.clip((depth_m - vmin_m) / max(vmax_m - vmin_m, 1e-6), 0.0, 1.0)
    img = cv2.applyColorMap((t * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)[..., ::-1].copy()
    img[depth_mm == 0] = 0
    text = f"{label} {vmin_m:.2f} m (dark) .. {vmax_m:.2f} m (yellow); black = invalid".strip()
    cv2.putText(img, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def save_depth_outputs(
    depths: dict[str, np.ndarray],
    intrinsics: dict[str, np.ndarray],
    frames: dict[str, Any],
    out_dir: Path,
    source: str,
    prefix: str = "",
    clip_m: float = 0.0,
) -> Path:
    """Write per camera ``<prefix><cam>_<source>_depth.tiff`` (uint16 mm), ``..._depth_color.png`` (colourised for
    humans), ``..._depth.ply`` (coloured with the left image), ``<prefix><cam>_rgb.png`` and ``<prefix>meta.json``.

    ``depths``/``intrinsics`` are keyed like the policy observation (``front_depth``); ``frames`` is the
    env ``obs["frames"]`` dict used for the colours.
    """
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {"source": source, "saved": datetime.now().isoformat(timespec="seconds"), "cameras": {}}
    for key, depth in depths.items():
        cam = key.removesuffix("_depth")
        depth = np.asarray(depth).astype(np.uint16)
        K = np.asarray(intrinsics[key], dtype=np.float64)
        rgb_entry = frames.get(f"{cam}_rgb", frames.get(cam, {})).get("rgb", {})
        rgb = np.asarray(rgb_entry.get("data")) if rgb_entry.get("data") is not None else None
        if rgb is None or rgb.shape[:2] != depth.shape:
            rgb = np.full(depth.shape + (3,), 200, dtype=np.uint8)
        vmin, vmax = depth_color_range(depth, clip_m)
        Image.fromarray(depth).save(out_dir / f"{prefix}{cam}_{source}_depth.tiff")
        Image.fromarray(colorize_depth_mm(depth, vmin, vmax, f"{cam} {source}")).save(
            out_dir / f"{prefix}{cam}_{source}_depth_color.png"
        )
        Image.fromarray(rgb).save(out_dir / f"{prefix}{cam}_rgb.png")
        n_pts = write_ply(out_dir / f"{prefix}{cam}_{source}_depth.ply", unproject_mm(depth, K), rgb)
        meta["cameras"][cam] = {
            "intrinsics": K.reshape(3, -1).tolist(),
            "valid_frac": float((depth > 0).mean()),
            "max_mm": int(depth.max()),
            "color_range_m": [vmin, vmax],
            "points": n_pts,
        }
    (out_dir / f"{prefix}meta.json").write_text(json.dumps(meta, indent=2))
    return out_dir


def maybe_save_depth_outputs(
    depths: dict[str, np.ndarray] | None,
    intrinsics: dict[str, np.ndarray] | None,
    frames: dict[str, Any],
    source: str,
    out_dir: Path = DEPTH_OUTPUT_DIR,
) -> Path | None:
    """Save the depth image + point cloud the policy receives, once per process (first observation that
    carries depth, ZED or FFS), as ``live_<cam>_<source>_depth.{tiff,ply}`` in ``<repo>/outputs/ffstereo_e2e``."""
    if not depths or not intrinsics or source in _depth_outputs_saved:
        return None
    _depth_outputs_saved.add(source)
    save_depth_outputs(depths, intrinsics, frames, out_dir, source, prefix="live_")
    logger.info("saved %s depth images and point clouds for %s to %s (live_*)", source, sorted(depths), out_dir)
    return Path(out_dir)


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scale", type=float, default=0.5, help="inference resize factor (authors: width < 1000 px)")
    parser.add_argument("--valid-iters", type=int, default=8, help="GRU refinement iterations")
    parser.add_argument("--max-disp", type=int, default=416, help="cost-volume disparity range (<= 416)")
    parser.add_argument("--z-far-m", type=float, default=30.0, help="depth beyond this is invalid (tiny disparities)")
    parser.add_argument("--remove-invisible", action="store_true", help="keep matches falling outside the right image")
    parser.add_argument("--no-warmup", action="store_true", help="skip the compile/warm-up forward at startup")
    parser.add_argument("--parent-pid", type=int, default=0, help="exit when this process disappears (0 = never)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("serving %s on %s:%d (weights=%s, scale=%.2f)", AGENT_NAME, args.host, args.port, args.weights, args.scale)

    if args.parent_pid > 0:
        _start_parent_watchdog(args.parent_pid)

    kwargs = dict(
        default_checkpoint_path=args.weights,
        device=args.device,
        scale=args.scale,
        valid_iters=args.valid_iters,
        max_disp=args.max_disp,
        z_far_m=args.z_far_m,
        remove_invisible=args.remove_invisible,
        warmup=not args.no_warmup,
    )
    start_server(AGENT_NAME, json.dumps(kwargs), port=args.port, host=args.host)


if __name__ == "__main__":
    main()
