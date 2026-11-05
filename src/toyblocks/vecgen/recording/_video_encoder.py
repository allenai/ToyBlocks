# ruff: noqa  # verbatim port from TAMP tmp3 (kept diffable against upstream)
"""Standalone video-encoder subprocess for IsaacDataRecorderLeRobot.

Kept deliberately FREE of isaaclab imports: the encoder runs under a
``forkserver`` context (see ``ENCODER_CTX``). The old plain-``fork`` default made
the encoder copy-on-write inherit the parent's (num_envs-scaled) Isaac/CUDA
footprint -> ~50GB RSS and OOM at high num_envs. ``spawn`` can't be used because
render.py launches Isaac at module import, so a spawned child would re-launch
Isaac. ``forkserver`` (with the server started BEFORE Isaac, and this lean module
preloaded) forks the encoder from a clean ~2-3GB (torch+lerobot+av, no isaaclab)
server instead of the ~26GB Isaac parent -> bounded memory regardless of num_envs.
This module must therefore stay isaaclab-free so the forkserver preload is light.
"""
import contextlib
import ctypes
import importlib.machinery
import json
import multiprocessing as mp
import os
import pickle
import queue
import shutil
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

# SVT-AV1 prints a ~16-line config banner to STDERR from its own logger every time an encoder is
# created -- and a wave holds num_envs*num_cams encoders open at once (48*6 = 288), so it buries
# the render log. It bypasses av_log, so ffmpeg's -loglevel and PyAV's logging both miss it, and
# the `log-level` svtav1-param is silently ignored. SVT_LOG is the only lever, it is read per
# encoder init, and it must be set before the first one is constructed. Measured with PyAV 18.1:
# 21 banner lines -> 0, output bit-identical (md5 97f3de75..., av1 640x360 25 packets).
os.environ.setdefault("SVT_LOG", "0")

import av
import numpy as np
from lerobot.datasets import lerobot_dataset as _lerobot_dataset_module
from lerobot.datasets.lerobot_dataset import LeRobotDataset

_lerobot_embed_images = _lerobot_dataset_module.embed_images


def _embed_images_if_any(ds):
    """``LeRobotDataset._save_episode_data`` runs ``embed_images`` -- a row-by-row
    ``Dataset.map`` -- on every episode even when no feature is an image.  For this
    recorder's ~70 tabular columns that was ~1.2 s per episode, 90% of ``save_episode``,
    and the single dataset-writer thread capped the whole run at <1 episode/s.  Skip it
    unless an image feature exists (camera frames never enter the LeRobot tables here;
    they go to the video streams)."""
    import datasets as _hf_datasets

    if any(isinstance(f, _hf_datasets.Image) for f in ds.features.values()):
        return _lerobot_embed_images(ds)
    return ds


_lerobot_dataset_module.embed_images = _embed_images_if_any

try:                                        # stdlib-only sibling (keeps this module isaaclab-free)
    from toyblocks.vecgen.recording._video_validate import has_moov
except Exception:                           # forkserver worker may load us as __mp_main__
    from _video_validate import has_moov

# Forkserver context: the encoder forks from a clean, pre-Isaac server (started early by
# render.py with this module preloaded), NOT from the Isaac-heavy parent.
ENCODER_CTX = mp.get_context("forkserver")
SHM_KEY = "__shm__"  # == toyblocks.vecgen.recording._shm_slots.SHM_KEY (a frame's slot descriptor)
# The forkserver preloads ``__main__`` by default, and spawn's main-module fixup re-imports the
# caller's script in every child. Datagen and eval scripts launch Isaac at import, so each encoder
# booted its own Isaac app (4.4-4.7 GB RSS per encoder, measured). Preload only this module, and
# start encoders with __main__ masked (_VideoEncoderProcess.start) so children never import it.
ENCODER_CTX.set_forkserver_preload([__name__])


@contextlib.contextmanager
def _main_not_reimported() -> "Iterator[None]":
    """Let a process started inside this block skip re-importing the caller's ``__main__``.

    ``multiprocessing.spawn`` re-imports a main *script* by path in the child, but skips a main
    whose spec is named ``__main__``. Nothing this package sends to the child lives in __main__.
    """
    main = sys.modules.get("__main__")
    if main is None or getattr(main, "__spec__", None) is not None:
        yield
        return
    main.__spec__ = importlib.machinery.ModuleSpec("__main__", None)
    try:
        yield
    finally:
        main.__spec__ = None


class _PrintLogger:
    """Tiny stdout logger so the spawned child needs no logging config to be visible
    (its stdout is inherited by the parent and captured into the render log)."""
    @staticmethod
    def info(msg):
        print(f"[VideoEncoder] {msg}", flush=True)

    @staticmethod
    def error(msg):
        print(f"[VideoEncoder] ERROR: {msg}", flush=True)


logger = _PrintLogger()


def payload_nbytes(obj: Any) -> int:
    """Bytes of numpy payload inside a queue command (tuples/lists/dicts walked recursively).

    Both ends of the writer queue compute this on the SAME arrays -- the recorder when it enqueues a
    command and the encoder when it retires it -- so ``sent - consumed`` is an exact count of the
    frame bytes in flight without any change to the command protocol. A small per-container
    constant keeps a flood of tiny commands bounded too.
    """
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if isinstance(obj, dict):
        return 256 + sum(payload_nbytes(v) for v in obj.values())
    if isinstance(obj, (tuple, list)):
        return 256 + sum(payload_nbytes(v) for v in obj)
    return 0


def _mem_total_bytes() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


def resolve_inflight_cap_bytes() -> int:
    """How many bytes of un-encoded frames may be in flight before the recorder blocks the exec loop.

    ``TMP3_RECORDER_INFLIGHT_GB`` overrides; the default is 4 GiB or a quarter of RAM, whichever is
    smaller. This is the ONLY point at which recording applies backpressure to the simulation: the
    writer queue itself is unbounded, so ``put`` never blocks on a count, and the cap exists purely so
    an encoder that is slower than the simulator cannot grow the backlog until the node OOMs.
    """
    raw = os.environ.get("TMP3_RECORDER_INFLIGHT_GB", "").strip()
    if raw:
        return max(64 << 20, int(float(raw) * (1 << 30)))
    total = _mem_total_bytes()
    cap = 4 << 30
    if total:
        cap = min(cap, total // 4)
    return max(64 << 20, cap)


def resolve_encode_threads() -> int:
    """Size of the encoder's shared encode pool. ``TMP3_ENCODER_THREADS`` overrides; default = cores.

    Every stream is pinned to one codec thread (``lp=1`` / ``pools=1`` / ``threads=1``, see the
    thread-cap note in ``_VideoEncoderProcess``), so parallelism comes from encoding many streams at
    once. PyAV releases the GIL inside ``encode``/``mux``: measured 7.5x on 8 threads (640x360,
    libsvtav1 / libx265 lossless / libx264rgb), so plain threads are enough here.
    """
    raw = os.environ.get("TMP3_ENCODER_THREADS", "").strip()
    if raw:
        return max(1, int(raw))
    try:
        n = len(os.sched_getaffinity(0))
    except Exception:
        n = os.cpu_count() or 4
    return max(2, n)


class EncoderProcessError(RuntimeError):
    """The video-encoder subprocess died unexpectedly.

    Raised by the main process when it detects the encoder child is no longer
    alive while it still has work to enqueue. Historically a dead encoder left
    the main loop parked forever in ``Queue.put`` (the queue fills, nothing
    drains it) -> a silent, GPU-holding deadlock. We now surface this as a
    hard, catchable error so the run aborts cleanly (and a ``--resume`` relaunch
    can continue from the last completed problem) instead of hanging.
    """


class _VideoEncoderProcess(ENCODER_CTX.Process):
    """Subprocess that encodes video frames via PyAV and writes tabular data via LeRobot.

    Forked (via forkserver) from a lean pre-Isaac server so it does not inherit the
    parent's Isaac/CUDA memory. All constructor args are picklable (queue from
    ENCODER_CTX, plain dicts/strings/ints), as forkserver requires.

    Queue protocol
    --------------
    ("start_episode", env_idx, cam_configs)
        Open PyAV containers for each camera in a staging directory.
    ("frame", env_idx, image_dict)
        Encode one frame per camera into the open containers.
    ("end_episode", env_idx, success, non_image_frames, goal_images)
        Flush videos, write non-image data to LeRobot, move videos to final paths.
    ("discard_episode", env_idx)
        Drop the env's open episode: its buffered rows and its staged videos.
    ("spilled", path)
        Any of the above, pickled to ``path`` by the recorder; loaded and deleted here.
    ("attach_shm", name, n_slots)
        Map the recorder's shared-memory frame arena; a later "frame" whose image_dict is a slot
        descriptor reads its images there and frees the slot once they are encoded.
    ("finalize",)
        Finalize both datasets, patch metadata, clean up staging.
    None
        Shutdown sentinel.
    """

    _VCODEC = "libsvtav1"
    #: Main-RGB codec choices (``rgb_codec``) and their per-stream options. Every stream is pinned
    #: to one codec thread (see the thread-cap note below). Measured native memory per open
    #: stream (16 streams, 30 frames): SVT-AV1 27 MiB at 320x240 / 57 MiB at 640x480, x264
    #: 5 / 14 MiB -- x264 is the one for recorders with hundreds of concurrent streams (eval).
    _RGB_CODECS = {
        "libsvtav1": {"svtav1-params": "lp=1"},
        "libx264": {"preset": "veryfast", "threads": "1"},
    }
    _PIX_FMT = "yuv420p"
    _CRF = "30"
    _GOP = "2"
    #: Instance-overridable main-RGB keyframe interval (--rgb_GOP). Declared at CLASS level
    #: so a _VideoEncoderProcess built without __init__ -- which the encoder tests do -- still
    #: resolves it; it was _GOP directly before --rgb_GOP existed.
    _rgb_gop = _GOP
    _PRESET = "12"

    # Plain dicts (no lambdas) so the Process instance stays picklable for forkserver/spawn.
    # NOTE thread caps: a wave keeps num_envs*num_cams encoder contexts open AT ONCE (e.g. 48*6=288).
    # If each context spawns a full thread pool (ffmpeg/x265/aom default = all cores), 288 contexts
    # explode into thousands of threads + per-pool native buffers -> RAM OOM (SIGKILL/-9) at wave 1.
    # Pinning each context to a single thread keeps total threads ~= #contexts and slashes native
    # memory; throughput is unaffected because the parallelism is across the many concurrent streams.
    _DEPTH_CODECS = {
        "av1": {
            "vcodec": "libaom-av1",
            "pix_fmt": "gray12le",
            "options": {
                "crf": "0", "g": "2", "cpu-used": "6", "row-mt": "0",
                "color_range": "2", "threads": "1",
            },
        },
        # preset ultrafast: lossless output does not depend on the preset (decoded frames are
        # identical), but speed and memory do. Measured on noisy 640x360 gray12 depth, GOP 2, one
        # thread: medium (x265's default) 52 fps / ~85 MiB per open stream, ultrafast 273 fps /
        # <1 MiB, files +5%.
        "hevc": {
            "vcodec": "libx265",
            "pix_fmt": "gray12le",
            "options": {
                "x265-params": "lossless=1:log-level=none:range=full:pools=1:frame-threads=1",
                "g": "2",
                "preset": "ultrafast",
            },
        },
        # g12: the uint12 value stored as what it IS -- a lossless gray12le HEVC stream with
        # temporal prediction and a SHORT gop. Matches maniflow-tb's
        # scripts/depth_conversion/reencode_to_g12_batch.py byte for byte in decoded values.
        # Why it exists: rg8 is exact and fast to decode but every frame is an intra frame of a
        # byte-split image (~16 KB/frame front at 224), ~2.2 TB on a 37k-episode dataset, which
        # never fits the page cache -- a cold rg8 file costs torchcodec 0.1-6 s/frame against
        # g12's 3-80 ms. bframes=0 makes decode order equal display order, so a reader seeks to
        # the keyframe at or before the target and decodes at most `gop` frames.
        # keyint/min-keyint are injected by _depth_options from the per-camera gop.
        "g12": {
            "vcodec": "libx265",
            "pix_fmt": "gray12le",
            "options": {
                "preset": "medium",
                "x265-params": ("lossless=1:bframes=0:open-gop=0:scenecut=0:"
                                "pools=1:frame-threads=1:log-level=none"),
            },
        },
        # RG8: the SAME uint12 value packed into an ordinary 8-bit RGB frame --
        # R = depth & 0xFF, G = depth >> 8, B = 0  ->  depth = R + (G << 8).
        # Lossless (qp=0) with GOP=1 so every frame is an intra keyframe, which lets
        # torchcodec random-access it instead of PyAV keyframe-seeking a gray12 stream.
        # NOT a precision reduction: the full 12-bit value survives bit-for-bit.
        "rg8": {
            "vcodec": "libx264rgb",
            "pix_fmt": "rgb24",
            "options": {"qp": "0", "g": "1", "threads": "1"},
        },
    }

    # depth_format -> depth_codec key. Named for what it IS (rg8 = the RG8 packing) rather than
    # its bit depth: "8bit" was actively misleading, since RG8 carries the full uint12 value
    # bit-for-bit across two 8-bit channels and reduces nothing.
    _DEPTH_FORMATS = {"rg8": "rg8", "12bit": None}    # None => use the caller's depth_codec

    # --extra_depth variants: an ADDITIONAL depth video written alongside the main one, into
    # videos/observation.depth.<cam>_<variant>/. Matches the layout and encode of the reference
    # dataset (yejink aug07_color_pnp_rendering_textrand_only_updated_N1), whose files carry
    # x264 settings `qp=0 keyint=1` at 224x224 -- i.e. the SAME rg8 codec config as the main
    # rg8 path, only downscaled. Kept lossless + GOP 1 on purpose: every frame is an intra
    # keyframe so a loader can random-access any episode index without keyframe seeking.
    # g12_224 carries the reference's PER-CAMERA gop: 8 for a static scene camera, 4 for the
    # moving gripper camera, which sees far more change per frame. gop_by_cam matches on a
    # SUBSTRING of the camera name so it works for 'camera:gripper_camera' and 'gripper_camera'
    # alike. Everything else falls back to "gop".
    _EXTRA_DEPTH = {
        "rg8_224": {"codec": "rg8", "size": (224, 224)},
        # 5+5: R = mm >> 5, G = mm & 31, so depth = (R << 5) | G -- 10 bits, CLIPPED to 1023 mm.
        # Lossy by construction (anything past 1.023 m saturates) and that is the point: the
        # near field is all a wrist policy needs, and halving the value range is what lets x264
        # squeeze it. Mirrors reencode_to_rg8_batch.py --pack 5+5 --out-size 224, whose output
        # this must stay byte-identical to.
        "rg8_55_224": {"codec": "rg8", "size": (224, 224), "pack": "5+5", "max_mm": 1023},
        "g12_224": {"codec": "g12", "size": (224, 224),
                    "gop": 8, "gop_by_cam": {"gripper_camera": 4}},
    }
    # --extra_rgb: a second, 224x224 copy of each RGB camera, byte-compatible with the sidecars
    # maniflow-tb/scripts/rgb_conversion/reencode_to_224_batch.py produces. Its crf/preset are
    # NOT the main RGB stream's (30/12): the training sidecars were encoded at 26/8 and this has to
    # match them, so the values live here rather than reusing _CRF/_PRESET.
    # No "g": the reference encoder passes no -g, so SVT-AV1 picks its own keyframe interval.
    # Forcing this recorder's g=2 instead made the sidecar 5.8x larger than the stored one
    # (1,781,024 B vs 306,410 B on a 622-frame clip) at identical crf/preset -- a keyframe every
    # 2 frames. Measured with no g: 289,374 B, i.e. 0.94x the reference.
    _EXTRA_RGB = {"224": {"size": (224, 224), "crf": "26", "preset": "8", "g": None}}

    def __init__(
        self,
        cmd_queue,
        data_dir: str,
        dataset_name: str,
        fps: int,
        features_tabular: dict,
        features_video: dict,
        robot_type: str = "franka",
        depth_codec: str = "av1",
        depth_gop: int = 1,
        extra_depth: "str | None" = None,
        extra_rgb: "str | None" = None,
        extra_rgb_gop: int = 0,
        rgb_gop: int = None,
        heartbeat=None,
        staging_root: "str | None" = None,
        defects=None,
        consumed=None,
        encode_threads: "int | None" = None,
        rgb_codec: str = "libsvtav1",
    ):
        super().__init__(daemon=True, name="VideoEncoderProcess")
        if rgb_codec not in self._RGB_CODECS:
            raise ValueError(f"Unknown rgb_codec {rgb_codec!r}, choose from {list(self._RGB_CODECS)}")
        self._VCODEC = rgb_codec
        self._cmd_queue = cmd_queue
        # mp.Value("Q"): bytes of queued payload this process has fully retired (frames encoded,
        # episodes saved). The recorder compares it with what it has sent to bound the backlog
        # (see resolve_inflight_cap_bytes) instead of blocking on a fixed queue length.
        self._consumed = consumed
        self._encode_threads = int(encode_threads) if encode_threads else resolve_encode_threads()
        # Progress heartbeat (mp.Value) bumped after each processed command + each video closed, so
        # the parent's finalize() can tell "slow but working" from "hung" and never SIGTERM a
        # still-flushing encoder (which truncates tail videos -> missing moov).
        self._heartbeat = heartbeat
        # Videos are staged here before publishing to _data_dir. Defaults to node-local scratch;
        # see the staging_dir comment in run() for why this must not live on weka.
        self._staging_root = staging_root or os.environ.get("RENDER_VIDEO_STAGING") \
            or tempfile.gettempdir()
        # mp.Value the parent reads after finalize: number of (episode, feature) videos that could
        # NOT be published. Non-zero means the dataset has holes and MUST NOT be marked complete.
        self._defects = defects
        self._data_dir = data_dir
        self._dataset_name = dataset_name
        self._fps = fps
        self._features_tabular = features_tabular
        self._features_video = features_video
        self._robot_type = robot_type
        if depth_codec not in self._DEPTH_CODECS:
            raise ValueError(f"Unknown depth_codec {depth_codec!r}, choose from {list(self._DEPTH_CODECS)}")
        self._depth_codec = depth_codec
        self._depth_cfg = self._DEPTH_CODECS[depth_codec]
        self._depth_gop = max(1, int(depth_gop))
        # Optional SECOND depth video per camera (see _EXTRA_DEPTH). The main depth keeps whatever
        # --depth_format asked for; this one is always the rg8 packing at the variant's size.
        if extra_depth and extra_depth not in self._EXTRA_DEPTH:
            raise ValueError(f"Unknown extra_depth {extra_depth!r}, "
                             f"choose from {list(self._EXTRA_DEPTH)}")
        self._extra_depth = extra_depth or None
        self._extra_suffix = f"_{extra_depth}" if extra_depth else None
        if extra_rgb and extra_rgb not in self._EXTRA_RGB:
            raise ValueError(f"Unknown extra_rgb {extra_rgb!r}, "
                             f"choose from {list(self._EXTRA_RGB)}")
        self._extra_rgb = extra_rgb or None
        self._extra_rgb_suffix = f"_{extra_rgb}" if extra_rgb else None
        # --extra_rgb_GOP overrides the sidecar keyframe interval. 0 keeps the TABLE's value,
        # which is g=None = "let SVT-AV1 choose" (it picks 161; that is what the PARTIAL5k
        # sidecars carry). Any other value is passed through as -g.
        # MAIN rgb keyframe interval. _GOP ("2") stays the default so existing datasets keep
        # reproducing; 0 drops -g entirely and lets SVT-AV1 choose, the same convention
        # --extra_rgb_GOP uses. Until this existed the main RGB was pinned at 2 with no way to
        # change it, while depth had --depth_gop -- GOP is the same size lever for both.
        self._rgb_gop = (None if rgb_gop == 0
                         else (str(int(rgb_gop)) if rgb_gop else self._GOP))
        self._extra_rgb_cfg = (
            {**self._EXTRA_RGB[extra_rgb],
             **({} if not extra_rgb_gop else {"g": str(int(extra_rgb_gop))})}
            # str, not int: PyAV option values must be strings (the table's own "crf"/
            # "preset" are). An int here raised "expected str, got int" only at stream
            # CLOSE, so every sidecar was dropped while the run still reported success and
            # info.json still declared the feature.
            if extra_rgb else None)
        self._extra_rgb_size = self._EXTRA_RGB[extra_rgb]["size"] if extra_rgb else None
        self._extra_spec = self._EXTRA_DEPTH[extra_depth] if extra_depth else {}
        self._extra_cfg = (self._DEPTH_CODECS[self._EXTRA_DEPTH[extra_depth]["codec"]]
                           if extra_depth else None)
        self._extra_size = self._EXTRA_DEPTH[extra_depth]["size"] if extra_depth else None

    def _bump(self):
        """Advance the progress heartbeat (best-effort; written from several encoder threads)."""
        hb = self._heartbeat
        if hb is not None:
            try:
                with hb.get_lock():
                    hb.value += 1
            except Exception:
                pass

    def _retire(self, nbytes: int) -> None:
        """Credit ``nbytes`` of queued payload back to the recorder's in-flight budget."""
        c = self._consumed
        if c is not None and nbytes:
            try:
                with c.get_lock():
                    c.value += int(nbytes)
            except Exception:
                pass

    @staticmethod
    def _resize_rgb_pil(img, size):
        """uint8 HWC RGB -> (size, size, 3), the way the HVLA dataloader resizes.

        This is deliberately Pillow and not ffmpeg `scale` / cv2 INTER_LINEAR: on these 640x360
        clips ffmpeg's bilinear disagrees with Pillow on ~70% of the 224x224 pixels (max abs 35),
        so a sidecar built any other way is not the uint8 the model sees. Mirrors
        maniflow-tb/scripts/rgb_conversion/pil_bilinear.py:resize_rgb_uint8; keep them in step.
        """
        from PIL import Image
        import numpy as _np
        a = _np.asarray(img, dtype=_np.uint8)
        return _np.asarray(Image.fromarray(a).resize((size, size), Image.BILINEAR),
                           dtype=_np.uint8)

    def _open_video_stream(self, path: str, width: int, height: int,
                           is_depth: bool = False, depth_cfg: dict = None,
                           rgb_cfg: dict = None, extra_gop: int = None):
        # +faststart puts the moov atom at the FRONT. The reference g12 encoder passes it, and it
        # is what makes a cold file cheap to open: otherwise the index sits at the end and a
        # random-access reader pays many small reads before its first frame.
        _copts = {"movflags": "+faststart"} if (depth_cfg or {}).get("vcodec") == "libx265" else {}
        container = av.open(path, "w", options=_copts) if _copts else av.open(path, "w")
        if is_depth:
            # depth_cfg is passed explicitly for an --extra_depth stream: it must be encoded with
            # the rg8 config even when the MAIN depth is gray12le/hevc.
            is_extra = depth_cfg is not None
            cfg = depth_cfg or self._depth_cfg
            # The extra stream ignores --depth_gop and uses its VARIANT's gop: rg8_224 stays at 1
            # (every frame intra, which is why it seeks fast), g12_224 uses the reference's short
            # per-camera gop passed in as extra_gop.
            _g = (extra_gop if extra_gop is not None else 1) if is_extra else self._depth_gop
            stream = container.add_stream(
                cfg["vcodec"], self._fps,
                options=self._depth_options(cfg, _g),
            )
            stream.pix_fmt = cfg["pix_fmt"]
        else:
            # rgb_cfg is passed for an --extra_rgb sidecar: same codec and pix_fmt as the main RGB
            # stream, but the training sidecars' crf/preset rather than this recorder's.
            _ropts = {"crf": (rgb_cfg or {}).get("crf", self._CRF)}
            if self._VCODEC == "libsvtav1":
                # lp=1: single-threaded SVT-AV1 per stream (see _DEPTH_CODECS thread-cap note) —
                # avoids 144 concurrent RGB encoders each grabbing a full core pool -> OOM.
                _ropts["preset"] = (rgb_cfg or {}).get("preset", self._PRESET)
            _ropts.update(self._RGB_CODECS[self._VCODEC])
            # A sidecar cfg may ask for the encoder's OWN keyframe interval by setting g=None;
            # the main stream keeps this recorder's _GOP.
            # A sidecar's own g wins; otherwise the main stream's. Either may be None, which
            # means "omit -g" -- SVT-AV1 then picks its own interval (measured 161).
            _g = (rgb_cfg or {}).get("g", "__unset__")
            if _g == "__unset__":
                _g = self._rgb_gop
            if _g is not None:
                _ropts["g"] = _g
            stream = container.add_stream(self._VCODEC, self._fps, options=_ropts)
            stream.pix_fmt = self._PIX_FMT
        stream.width = width
        stream.height = height
        return container, stream

    @classmethod
    def _extra_depth_gop(cls, spec: dict, fkey: str) -> int:
        """GOP for an --extra_depth stream on camera `fkey`.

        g12 uses a SHORT, per-camera gop (8 static / 4 gripper): it is the random-access budget,
        since a reader decodes at most `gop` frames from the preceding keyframe. rg8 has no "gop"
        key and keeps GOP 1 -- every frame intra -- which is the whole reason it decodes fast.
        """
        if "gop" not in spec:
            return 1
        for cam_sub, g in (spec.get("gop_by_cam") or {}).items():
            if cam_sub in fkey:
                return int(g)
        return int(spec["gop"])

    @staticmethod
    def _depth_options(cfg: dict, gop: int) -> dict:
        """`cfg`'s encoder options with the GOP applied.

        GOP is a real size/latency lever for lossless depth and belongs to the CALLER, not this
        table. Measured on 245 real 640x360 gray12 frames: GOP 2 -> 4974 KiB in 4.90 s, GOP 1000 ->
        1210 KiB in 1.57 s. Bigger GOP is both smaller AND faster (fewer intra keyframes = less
        work); the cost is random access, since a reader must decode from the last keyframe.
        x265 ignores a bare `g`, so keyint has to go into x265-params too.
        """
        opts = dict(cfg["options"])
        opts["g"] = str(gop)
        if "x265-params" in opts:
            opts["x265-params"] = f"{opts['x265-params']}:keyint={gop}:min-keyint={gop}"
        return opts

    @staticmethod
    def _pack_rg8(d):
        """uint12 depth -> rgb24 with R=low byte, G=high byte, B=0, so depth = R + (G << 8)."""
        rgb = np.zeros(d.shape[:2] + (3,), dtype=np.uint8)
        rgb[..., 0] = (d & 0xFF).astype(np.uint8)
        rgb[..., 1] = ((d >> 8) & 0xFF).astype(np.uint8)
        return rgb

    #: 5+5 saturates here; 1023 == (1 << 10) - 1.
    _RG8_55_MAX_MM = 1023

    @classmethod
    def _pack_rg8_55(cls, d):
        """uint12 depth -> rgb24 with R = mm >> 5, G = mm & 31, B = 0; depth = (R << 5) | G.

        CLIPS at 1023 mm before packing, exactly as reencode_to_rg8_batch.py does -- clipping
        after the shift would alias far pixels onto near values instead of saturating them.
        """
        mm = np.minimum(d, cls._RG8_55_MAX_MM)
        rgb = np.zeros(d.shape[:2] + (3,), dtype=np.uint8)
        rgb[..., 0] = (mm >> 5).astype(np.uint8)
        rgb[..., 1] = (mm & np.uint16(31)).astype(np.uint8)
        return rgb

    @staticmethod
    def _resize_nearest(d, out_w: int, out_h: int):
        """Nearest-neighbour resize matching the dataloader's
        `Image.fromarray(depth_m).resize((S, S), Image.NEAREST)`, so an rg8_224 stream selects the
        same source pixels as a resized uint12 one.

        Calls Pillow rather than reimplementing its floor((i+0.5)*scale) rule: in float64 that
        product can land just under an integer and pick the previous column.
        """
        h, w = d.shape[:2]
        if (w, h) == (out_w, out_h):
            return d
        from PIL import Image
        return np.array(
            Image.fromarray(d.astype(np.float32)).resize((out_w, out_h), Image.NEAREST)
        ).astype(d.dtype)

    def _quarantine(self, src: str, ep_idx, fkey: str) -> "str | None":
        """Preserve a condemned staging file instead of deleting it.

        Staging is wiped at finalize, so a file we refuse to publish used to be destroyed -- which
        made every withhold unrecoverable AND unfalsifiable: there was nothing left to prove whether
        the file was really bad or whether has_moov had produced a false negative. Quarantine keeps
        it next to the dataset, where it can be inspected and, if it is actually intact, published
        directly with no render at all.
        """
        if not src or not os.path.exists(src):
            return None
        try:
            qdir = os.path.join(self._data_dir, "_video_quarantine")
            os.makedirs(qdir, exist_ok=True)
            dest = os.path.join(qdir, f"ep{int(ep_idx):06d}_{fkey.replace('.', '_')}.mp4")
            shutil.copyfile(src, dest)
            logger.error(f"[VideoEncoder] quarantined condemned video -> {dest} "
                         f"({os.path.getsize(dest)} bytes); NOT deleted")
            return dest
        except Exception as e:
            logger.error(f"[VideoEncoder] quarantine failed for {src}: {type(e).__name__}: {e}")
            return None

    def _record_defect(self, env_idx, ep_idx, fkey, why: str, src: "str | None" = None) -> None:
        """Account for a video that will NOT exist in the output. Loud, and counted.

        A hole used to be logged (at best) and then forgotten, so the problem was still marked
        completed=true and the dataset shipped with missing files that only turned up later in a
        manual audit. The counter is an mp.Value the parent reads at finalize.
        """
        logger.error(f"VIDEO DEFECT env {env_idx} ep {ep_idx} {fkey}: {why}")
        if self._defects is not None:
            with self._defects.get_lock():
                self._defects.value += 1
        # Also record WHICH (episode, feature) is missing, not just how many. The count alone
        # cannot drive a targeted patch -- reconstructing the list afterwards meant a full listing
        # of ~115k objects over the weka gateway. One JSONL line per defect, appended.
        quarantined = self._quarantine(src, ep_idx, fkey) if src else None
        try:
            with open(os.path.join(self._data_dir, "_video_defects.jsonl"), "a") as fh:
                fh.write(json.dumps({"env": int(env_idx), "episode": int(ep_idx),
                                     "feature": fkey, "why": why,
                                     "quarantined": quarantined}) + "\n")
        except Exception as e:                       # never let bookkeeping kill the encoder
            logger.error(f"could not record defect sidecar: {type(e).__name__}: {e}")

    def _publish(self, src: str, dest: str, tries: int = 3) -> bool:
        """Copy a validated staging file to its final path on weka and CONFIRM it landed intact.

        Staging is local, so `src` stays valid and a failed publish is retryable -- the whole point
        of moving staging off weka. Verify size + moov at the destination: a short write over a
        network filesystem does not necessarily raise.
        """
        for attempt in range(1, tries + 1):
            try:
                shutil.copyfile(src, dest)
                if os.path.getsize(dest) == os.path.getsize(src) and has_moov(dest):
                    os.remove(src)
                    return True
                logger.error(f"[VideoEncoder] publish verify FAILED (attempt {attempt}/{tries}) "
                             f"size={os.path.getsize(dest)}/{os.path.getsize(src)} -> {dest}")
            except Exception as e:
                logger.error(f"[VideoEncoder] publish error (attempt {attempt}/{tries}) "
                             f"{type(e).__name__}: {e} -> {dest}")
            time.sleep(2 * attempt)
        return False

    @staticmethod
    def _close_videos(vids: dict) -> set:
        """Flush + close each container (this is what writes the mp4 ``moov`` trailer) and VALIDATE
        the result. Returns the set of fkeys whose file closed to a valid, moov-bearing mp4.

        Callers MUST only publish (move to the final dataset) validated files. Previously a swallowed
        close error let a truncated, moov-less file get moved to the final path -> the ``moov atom not
        found`` corruption seen across a killed run. We now surface + withhold such files instead.
        """
        valid = set()
        for fkey, (container, stream, path) in vids.items():
            closed = False
            try:
                for packet in stream.encode():
                    container.mux(packet)
                container.close()
                closed = True
            except Exception as e:
                logger.error(f"Error closing {fkey}: {e}")
            if closed and path and has_moov(path):
                valid.add(fkey)
            elif path:
                logger.error(
                    f"INVALID video after close (no moov trailer) — NOT publishing: {path}"
                )
        return valid

    def _make_dataset(self, repo_id: str, root: Path):
        if root.exists():
            shutil.rmtree(root)
        return LeRobotDataset.create(
            repo_id=repo_id,
            root=str(root),
            robot_type=self._robot_type,
            fps=self._fps,
            features=self._features_tabular,
        )

    def start(self) -> None:
        with _main_not_reimported():
            super().start()

    def _patch_video_metadata(self, ds):
        """Add video feature definitions to info.json after finalization."""
        info_path = Path(ds.root) / "meta" / "info.json"
        if not info_path.exists():
            return
        try:
            with open(info_path, "r") as f:
                info = json.load(f)
            features = info.get("features", {})
            for key, feat in self._features_video.items():
                if key not in features:
                    features[key] = feat
            info["features"] = features
            with open(info_path, "w") as f:
                json.dump(info, f, indent=2)
        except Exception as e:
            logger.error(f"Error patching metadata: {e}")

    @staticmethod
    def _rss_gb() -> float:
        """Resident set size of this encoder process, in GiB (from /proc/self/statm)."""
        try:
            with open("/proc/self/statm") as f:
                pages = int(f.read().split()[1])  # resident pages
            return pages * (os.sysconf("SC_PAGE_SIZE") / (1024 ** 3))
        except Exception:
            return -1.0

    @staticmethod
    def _malloc_trim():
        """Force glibc to return freed pages to the OS.

        The multiprocessing Queue deserializes large numpy arrays, causing
        glibc heap fragmentation (pages freed by Python but not returned
        to the OS).  Periodic ``malloc_trim(0)`` reclaims these pages.
        """
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    def _open_episode_streams(self, env_staging: str, cam_configs: dict) -> Dict[str, Any]:
        """Open one staging container per camera key for a new episode."""
        if os.path.exists(env_staging):
            shutil.rmtree(env_staging)
        os.makedirs(env_staging, exist_ok=True)
        vids: Dict[str, Any] = {}
        for fkey, (w, h) in cam_configs.items():
            path = os.path.join(env_staging, f"{fkey.replace('.', '_')}.mp4")
            c, st = self._open_key_stream(path, fkey, w, h)
            vids[fkey] = (c, st, path)
        return vids

    def _open_key_stream(self, path: str, fkey: str, w: int, h: int) -> Tuple[Any, Any]:
        """Open the container + stream for camera key ``fkey`` with that key's codec settings."""
        is_depth = "observation.depth." in fkey
        # An --extra_depth key (observation.depth.<cam>_rg8_224) is still a depth
        # stream, but must use the rg8 config rather than the main depth codec.
        is_extra = bool(self._extra_suffix) and fkey.endswith(self._extra_suffix)
        is_extra_rgb = (not is_depth and bool(self._extra_rgb_suffix)
                        and fkey.endswith(self._extra_rgb_suffix))
        return self._open_video_stream(
            path, w, h, is_depth=is_depth,
            depth_cfg=self._extra_cfg if is_extra else None,
            rgb_cfg=self._extra_rgb_cfg if is_extra_rgb else None,
            extra_gop=(self._extra_depth_gop(self._extra_spec, fkey)
                       if is_extra else None))

    def _trim_video(self, fkey: str, path: str, dropped: Sequence[Tuple[int, int]]) -> bool:
        """Re-encode the closed staged video at ``path`` without the frames in ``dropped``
        (``[start, end)`` frame spans), with ``fkey``'s own codec settings, and replace it.

        This is how failed skill attempts leave an episode recorded with videos: their frames
        were encoded before the attempt failed. Depth streams are lossless, so their kept frames
        come back bit-exact; lossy RGB goes through one more encode at the same crf.
        Returns False (and leaves ``path`` as it was) if the video could not be trimmed.
        """
        drop = {i for a, b in dropped for i in range(a, b)}
        need = max(b for _, b in dropped)
        tmp = path[:-len(".mp4")] + "_trim.mp4"
        n = kept = 0
        frame_tb = Fraction(1, int(self._fps))
        with av.open(path) as src:
            sv = src.streams.video[0]
            container, stream = self._open_key_stream(
                tmp, fkey, sv.codec_context.width, sv.codec_context.height)
            for frame in src.decode(sv):
                if n not in drop:
                    # Renumber the kept frames back to back (decoded frames carry the source's
                    # container timestamps, which the encoder would otherwise reuse).
                    frame.pts, frame.time_base = kept, frame_tb
                    kept += 1
                    for packet in stream.encode(frame):
                        container.mux(packet)
                n += 1
        ok = fkey in self._close_videos({fkey: (container, stream, tmp)})
        if ok and n < need:
            logger.error(f"{path}: {n} frames but frames up to {need} were to be dropped")
            ok = False
        if ok:
            os.replace(tmp, path)
        elif os.path.exists(tmp):
            os.unlink(tmp)
        return ok

    def _trim_videos(self, vids: Dict[str, Any], valid: set,
                     dropped: Sequence[Tuple[int, int]], fut: Future) -> None:
        """Trim every valid video of one episode (``_trim_video``), then resolve ``fut`` with
        ``(vids, valid)``; a stream that fails to trim leaves ``valid`` so it is not published
        out of step with the trimmed rows."""
        try:
            for fkey in sorted(valid):
                try:
                    trimmed = self._trim_video(fkey, vids[fkey][2], dropped)
                except Exception as e:
                    logger.error(f"trim {fkey}: {type(e).__name__}: {e}")
                    trimmed = False
                if not trimmed:
                    valid.discard(fkey)
        finally:
            fut.set_result((vids, valid))

    def _encode_stream_frame(self, vids: Dict[str, Any], fkey: str, img: np.ndarray) -> None:
        """Encode one camera frame into its main stream and any --extra sidecar derived from it.

        Runs on the shared encode pool; each call touches only ``fkey``'s containers, and the env
        worker never has two frames of the same stream in the pool at once, so no locking is needed.
        """
        is_depth = "observation.depth." in fkey
        entry = vids.get(fkey)
        if entry is not None:
            container, stream, _ = entry
            if is_depth:
                if self._depth_cfg["pix_fmt"] == "rgb24":
                    # RG8: split the uint12 value across R (low byte) and G (high byte); B stays 0.
                    # Recovered as depth = R + (G << 8). `img` is gray16le; >>4 gives the uint12
                    # millimetre both depth streams store.
                    frame = av.VideoFrame.from_ndarray(
                        self._pack_rg8(np.asarray(img) >> 4), format="rgb24")
                else:
                    # Feed gray12le directly; letting swscale down-convert a gray16le frame
                    # dithers it by one LSB.
                    frame = av.VideoFrame.from_ndarray(
                        (np.asarray(img) >> 4).astype(np.uint16), format="gray12le")
            else:
                frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

        # --extra_rgb: a second, Pillow-downscaled copy of this same RGB frame. Derived from the
        # SAME uint8 array as the main stream so the two cannot drift; the extra key never appears
        # in image_dict, only in cam_configs.
        if (not is_depth) and self._extra_rgb_suffix:
            xr = vids.get(fkey + self._extra_rgb_suffix)
            if xr is not None:
                xrc, xrs, _ = xr
                small_rgb = self._resize_rgb_pil(img, self._extra_rgb_size[0])
                rframe = av.VideoFrame.from_ndarray(small_rgb, format="rgb24")
                for packet in xrs.encode(rframe):
                    xrc.mux(packet)

        # --extra_depth: a second, downscaled copy (rg8 or g12) of this same depth frame, from
        # the SAME uint12 array as the main stream.
        if is_depth and self._extra_suffix:
            x = vids.get(fkey + self._extra_suffix)
            if x is not None:
                xcontainer, xstream, _ = x
                ow, oh = self._extra_size
                # SAME resize as rg8_224, deliberately: the reference g12 encoder transcodes
                # FROM rg8_224 and never resizes, so the two variants can only agree if this
                # picks the same source pixels. Do not "improve" one without the other.
                small = self._resize_nearest(np.asarray(img) >> 4, ow, oh)
                if self._extra_cfg["pix_fmt"] == "gray12le":
                    # g12: store the uint12 value as a gray12 sample. The reference feeds
                    # ffmpeg raw gray12le after unpacking rg8; this is the same array one step
                    # earlier, never packed.
                    xframe = av.VideoFrame.from_ndarray(
                        small.astype(np.uint16), format="gray12le")
                elif self._extra_spec.get("pack") == "5+5":
                    xframe = av.VideoFrame.from_ndarray(
                        self._pack_rg8_55(small), format="rgb24")
                else:
                    xframe = av.VideoFrame.from_ndarray(self._pack_rg8(small), format="rgb24")
                for packet in xstream.encode(xframe):
                    xcontainer.mux(packet)

    def _encode_goal_images(self, env_staging: str, goal_images: list) -> Dict[str, Any]:
        """Encode per-frame goal images (annotated path) into their own staging containers."""
        os.makedirs(env_staging, exist_ok=True)
        goal_vids: Dict[str, Any] = {}
        for gi in goal_images:
            for fkey, img in gi.items():
                if fkey not in goal_vids:
                    h, w = img.shape[:2]
                    path = os.path.join(env_staging, f"{fkey.replace('.', '_')}.mp4")
                    c, st = self._open_video_stream(path, w, h)
                    goal_vids[fkey] = (c, st, path)
                container, stream, _ = goal_vids[fkey]
                frame = av.VideoFrame.from_ndarray(img, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
        return goal_vids

    def run(self):
        """Dispatch loop.

        Threads inside this process (all created here, never in ``__init__``, so the Process object
        stays picklable for forkserver):

        * one ``_EnvEncodeWorker`` per env: owns that env's open containers, encodes its frames in
          order, and fans the per-camera encodes of each frame out to the shared pool;
        * a shared ``ThreadPoolExecutor`` of ``encode_threads`` (default: cores) doing the actual
          libsvtav1 / libx265 / libx264rgb work -- PyAV releases the GIL there;
        * one ``_DatasetWriter``: appends tabular rows, ``save_episode``, publishes the finished
          videos. It processes episodes strictly in the order the recorder ended them, so the
          episode index it assigns matches ``_predict_episode_video_path`` in the parent.

        This loop itself only routes commands, so a slow parquet write or a slow camera can never
        stall the intake of other envs' frames; the recorder's exec loop is throttled solely by the
        byte budget in ``resolve_inflight_cap_bytes``.
        """
        import datasets
        from queue import Empty

        datasets.disable_progress_bar()

        base = Path(self._data_dir).absolute()
        # Stage on LOCAL disk, never on weka. container.close() is what writes the mp4 `moov`
        # trailer; doing that over weka lets a silent short write produce a moov-less file with NO
        # exception (observed 2026-08-24: `INVALID video after close (no moov trailer)` while
        # close() itself raised nothing, 35 videos lost across one 12,800-episode run). Because the
        # withheld file was ALSO the only copy, the loss was unrecoverable in-process.
        # Local staging makes the trailer write cheap and reliable, and -- crucially -- leaves a
        # good source file behind so publishing to weka can be verified and RETRIED.
        staging_dir = os.path.join(self._staging_root, f"_video_staging_{os.getpid()}")
        shutil.rmtree(staging_dir, ignore_errors=True)
        os.makedirs(staging_dir, exist_ok=True)
        logger.info(f"[VideoEncoder] staging videos under {staging_dir} (local), publishing to {base}")

        ds_success = self._make_dataset(
            f"{self._dataset_name}_success",
            base / f"{self._dataset_name}_success",
        )
        ds_failure = self._make_dataset(
            f"{self._dataset_name}_failure",
            base / f"{self._dataset_name}_failure",
        )

        self._patch_video_metadata(ds_success)
        self._patch_video_metadata(ds_failure)

        logger.info(f"Datasets created")
        logger.info(f"  Success: {ds_success.root}")
        logger.info(f"  Failure: {ds_failure.root}")

        pool = ThreadPoolExecutor(max_workers=self._encode_threads, thread_name_prefix="encode")
        logger.info(f"[VideoEncoder] encode pool: {self._encode_threads} threads")
        writer = _DatasetWriter(self, ds_success, ds_failure)
        writer.start()
        workers: Dict[int, _EnvEncodeWorker] = {}

        def worker_for(env_idx: int) -> "_EnvEncodeWorker":
            w = workers.get(env_idx)
            if w is None:
                w = _EnvEncodeWorker(self, env_idx, os.path.join(staging_dir, f"env_{env_idx}"), pool)
                w.start()
                workers[env_idx] = w
            return w

        env_frame_buffers: Dict[int, List[Dict]] = {}
        # Tabular bytes buffered per env between "frame" and "end_episode"; retired at save.
        self._pending_tabular_bytes: Dict[int, int] = {}
        _frame_count = 0
        _trim_interval = 100

        while True:
            try:
                cmd = self._cmd_queue.get(timeout=2.0)
            except Empty:
                continue

            if cmd is None:
                break

            try:
                if cmd[0] == "spilled":
                    # The recorder wrote this item to local disk (its RAM budget was full); read it
                    # back here, in queue order, and process it like any other.
                    with open(cmd[1], "rb") as f:
                        cmd = pickle.load(f)
                    os.unlink(f.name)
                op = cmd[0]

                if op == "start_episode":
                    env_idx, cam_configs = cmd[1], cmd[2]
                    env_frame_buffers.pop(env_idx, None)
                    worker_for(env_idx).submit(("start", cam_configs))
                    self._retire(payload_nbytes(cmd))
                    n_streams = sum(len(w.cam_configs or {}) for w in workers.values())
                    logger.info(
                        f"open_episodes={sum(1 for w in workers.values() if w.cam_configs)} "
                        f"(streams={n_streams}) RSS={self._rss_gb():.1f}G"
                    )

                elif op == "attach_shm":
                    from toyblocks.vecgen.recording._shm_slots import SlotReader
                    self._slots = SlotReader(cmd[1], int(cmd[2]))

                elif op == "frame":
                    env_idx, image_dict = cmd[1], cmd[2]
                    tabular_frame = cmd[3] if len(cmd) > 3 else None
                    total = payload_nbytes(cmd)
                    img_bytes = payload_nbytes(image_dict)
                    slot = None
                    if SHM_KEY in image_dict:
                        # Images in the shared arena: nothing of them was counted in flight.
                        slot, image_dict = self._slots.views(image_dict)
                    if tabular_frame is not None:
                        env_frame_buffers.setdefault(env_idx, []).append(tabular_frame)
                        # Tabular bytes are retired when the episode is saved.
                        self._pending_tabular_bytes[env_idx] = (
                            self._pending_tabular_bytes.get(env_idx, 0) + (total - img_bytes))
                    else:
                        self._retire(total - img_bytes)
                    worker_for(env_idx).submit(("frame", image_dict, img_bytes, slot))
                    _frame_count += 1
                    if _frame_count % _trim_interval == 0:
                        self._malloc_trim()

                elif op == "end_episode":
                    env_idx = cmd[1]
                    success = cmd[2]
                    non_image_frames = cmd[3]
                    goal_images = cmd[4]
                    error_log = cmd[5] if len(cmd) > 5 else None
                    dropped = cmd[6] if len(cmd) > 6 else []
                    if non_image_frames is None:
                        non_image_frames = env_frame_buffers.pop(env_idx, [])
                        retire_bytes = self._pending_tabular_bytes.pop(env_idx, 0) + payload_nbytes(cmd)
                    else:
                        env_frame_buffers.pop(env_idx, None)
                        retire_bytes = self._pending_tabular_bytes.pop(env_idx, 0) + payload_nbytes(cmd)
                    closed: Future = Future()
                    worker_for(env_idx).submit(("end", goal_images, closed, dropped))
                    writer.submit((env_idx, success, non_image_frames, error_log, closed, retire_bytes,
                                   dropped))

                elif op == "discard_episode":
                    env_idx = cmd[1]
                    env_frame_buffers.pop(env_idx, None)
                    self._retire(self._pending_tabular_bytes.pop(env_idx, 0) + payload_nbytes(cmd))
                    worker_for(env_idx).submit(("discard",))

                elif op == "finalize":
                    self._retire(payload_nbytes(cmd))
                    # Drain in dependency order: env workers (close every open video), then the
                    # dataset writer (append + publish everything that was queued), then finalize.
                    for w in workers.values():
                        w.submit(None)
                    for w in workers.values():
                        w.join()
                        self._bump()
                    writer.submit(None)
                    writer.join()
                    pool.shutdown(wait=True)

                    for label, ds in [
                        ("success", ds_success), ("failure", ds_failure),
                    ]:
                        try:
                            if ds.num_episodes > 0:
                                ds.finalize()
                                self._bump()
                                self._patch_video_metadata(ds)
                                logger.info(f"Dataset finalized ({label}): {ds.root}")
                                logger.info(f"  Episodes: {ds.num_episodes}")
                                logger.info(f"  Frames: {ds.num_frames}")
                            else:
                                logger.info(f"Dataset ({label}): no episodes, removing")
                                shutil.rmtree(ds.root, ignore_errors=True)
                        except Exception as e:
                            logger.error(f"Error finalizing {label}: {e}")
                            traceback.print_exc()

                    shutil.rmtree(staging_dir, ignore_errors=True)
                    break

            except Exception as e:
                op_name = cmd[0] if isinstance(cmd, tuple) else str(cmd)
                logger.error(f"Error processing {op_name}: {e}")
                traceback.print_exc()

            # Progress signal for the parent's finalize() wait (workers bump it too, per frame and
            # per episode): as long as it advances the parent keeps waiting instead of terminating
            # (which would truncate tail videos).
            self._bump()


class _EnvEncodeWorker(threading.Thread):
    """Encodes one environment's frames, in order, using the shared encode pool.

    Commands (a ``queue.Queue``, processed FIFO):
      ("start", cam_configs)           open this env's staging containers (closing any left open)
      ("frame", image_dict, nbytes, slot)  encode one frame into every open stream, then free its
                                       shared-memory slot (if any) and retire nbytes
      ("end", goal_images, future, dropped)  close + validate the streams (and encode goal
                                       images); with ``dropped`` frame spans, re-encode the videos
                                       without them on the pool; resolve ``future`` with
                                       ``(vids, valid_fkeys)``
      ("discard",)                     close the streams and delete the episode's staged files
      None                             close anything still open and exit
    """

    def __init__(self, proc: _VideoEncoderProcess, env_idx: int, staging: str,
                 pool: ThreadPoolExecutor):
        super().__init__(name=f"encode-env-{env_idx}", daemon=True)
        self._proc = proc
        self.env_idx = env_idx
        self._staging = staging
        self._pool = pool
        self._q: "queue.Queue" = queue.Queue()
        self.cam_configs: Optional[dict] = None
        self._vids: Dict[str, Any] = {}
        # Each episode stages into its own directory. The dataset writer publishes an episode's
        # files AFTER this worker has moved on to the env's next episode, so a shared per-env
        # directory would be wiped / overwritten by the next "start" before it is published.
        self._episode = 0
        self._ep_staging: str = os.path.join(staging, "ep_000000")

    def submit(self, cmd) -> None:
        self._q.put(cmd)

    def _encode_frame(self, image_dict: Dict[str, np.ndarray]) -> None:
        if not self._vids or not image_dict:
            return
        futs = [self._pool.submit(self._proc._encode_stream_frame, self._vids, fkey, img)
                for fkey, img in image_dict.items()]
        # Wait for every camera of THIS frame before the next frame of this env is submitted, which
        # is what keeps each stream's encode order intact without per-stream locks.
        for f in futs:
            try:
                f.result()
            except Exception as e:
                logger.error(f"env {self.env_idx}: encode error: {type(e).__name__}: {e}")

    def run(self) -> None:
        proc = self._proc
        while True:
            cmd = self._q.get()
            if cmd is None:
                proc._close_videos(self._vids)
                self._vids = {}
                self.cam_configs = None
                return
            op = cmd[0]
            try:
                if op == "start":
                    proc._close_videos(self._vids)
                    self.cam_configs = cmd[1]
                    self._episode += 1
                    self._ep_staging = os.path.join(self._staging, f"ep_{self._episode:06d}")
                    self._vids = proc._open_episode_streams(self._ep_staging, cmd[1])
                elif op == "frame":
                    try:
                        self._encode_frame(cmd[1])
                    finally:
                        if cmd[3] is not None:
                            proc._slots.free(cmd[3])
                    proc._retire(cmd[2])
                    proc._bump()
                elif op == "discard":
                    # Not _close_videos: nothing here is published, so skip the flush and the
                    # moov validation (a short stream would only log a spurious INVALID).
                    for container, _stream, _path in self._vids.values():
                        try:
                            container.close()
                        except Exception:
                            pass
                    self._vids = {}
                    self.cam_configs = None
                    shutil.rmtree(self._ep_staging, ignore_errors=True)
                elif op == "end":
                    goal_images, fut = cmd[1], cmd[2]
                    dropped = cmd[3] if len(cmd) > 3 else ()
                    vids = self._vids
                    self._vids = {}
                    self.cam_configs = None
                    valid = proc._close_videos(vids)
                    if goal_images:
                        goal_vids = proc._encode_goal_images(self._ep_staging, goal_images)
                        valid |= proc._close_videos(goal_vids)
                        vids.update(goal_vids)
                    if dropped and valid:
                        # Off this worker: the env's next episode keeps encoding meanwhile (each
                        # episode stages into its own directory).
                        self._pool.submit(proc._trim_videos, vids, valid, dropped, fut)
                    else:
                        fut.set_result((vids, valid))
            except Exception as e:
                logger.error(f"env {self.env_idx}: error in '{op}': {type(e).__name__}: {e}")
                traceback.print_exc()
                if op == "end":
                    cmd[2].set_exception(e)


def keep_frames(frames: List[Dict], dropped: Sequence[Tuple[int, int]]) -> List[Dict]:
    """``frames`` without the raw index spans ``[start, end)`` in ``dropped``."""
    drop = set()
    for a, b in dropped:
        drop.update(range(a, b))
    return [f for i, f in enumerate(frames) if i not in drop]


class _DatasetWriter(threading.Thread):
    """Appends episodes to the LeRobot datasets and publishes their videos, one at a time, in the
    order the recorder ended them (that order is what the parent's episode-index prediction relies
    on). Waits on each episode's ``closed`` future from its env worker; that wait costs the exec
    loop nothing because nothing upstream waits on this thread except ``finalize``.
    """

    def __init__(self, proc: _VideoEncoderProcess, ds_success, ds_failure):
        super().__init__(name="dataset-writer", daemon=True)
        self._proc = proc
        self._ds_success = ds_success
        self._ds_failure = ds_failure
        self._q: "queue.Queue" = queue.Queue()

    def submit(self, item) -> None:
        self._q.put(item)

    def run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            try:
                self._save(*item)
            except Exception as e:
                logger.error(f"dataset writer error: {type(e).__name__}: {e}")
                traceback.print_exc()
            self._proc._bump()

    def _save(self, env_idx: int, success: bool, non_image_frames: list,
              error_log: Optional[str], closed: Future, retire_bytes: int,
              dropped: Sequence[Tuple[int, int]] = ()) -> None:
        proc = self._proc
        try:
            vids, valid_fkeys = closed.result()
        except Exception as e:
            logger.error(f"env {env_idx}: videos unavailable for this episode: {e}")
            vids, valid_fkeys = {}, set()

        if dropped:  # the env worker re-encoded this episode's videos without them too
            non_image_frames = keep_frames(non_image_frames, dropped)

        if error_log:
            for f in non_image_frames:
                f["error"] = error_log
        if non_image_frames:
            non_image_frames[-1]["is_last"] = np.array([True], dtype=np.bool_)
            non_image_frames[-1]["is_episode_successful"] = np.array([success], dtype=np.bool_)

        ds = self._ds_success if success else self._ds_failure
        t_add = time.monotonic()
        for f in non_image_frames:
            ds.add_frame(f)
        t_save = time.monotonic()
        ds.save_episode()
        t_done = time.monotonic()
        ds.episode_buffer = None

        ep_idx = ds.num_episodes - 1
        chunk = ep_idx // 1000
        file_num = ep_idx % 1000
        for fkey, (_, _, path) in vids.items():
            if not os.path.exists(path):
                # A video that never reached staging must not vanish without a trace.
                proc._record_defect(env_idx, ep_idx, fkey, "staging file absent")
                continue
            if fkey not in valid_fkeys:
                proc._record_defect(env_idx, ep_idx, fkey,
                                    f"invalid after close (no moov): {path}", src=path)
                continue
            dest_dir = os.path.join(str(ds.root), "videos", fkey, f"chunk-{chunk:03d}")
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, f"file-{file_num:03d}.mp4")
            if not proc._publish(path, dest):
                proc._record_defect(env_idx, ep_idx, fkey,
                                    f"publish to weka failed: {dest}", src=path)

        # Every file of this episode has been published or quarantined; drop its staging directory.
        for ep_dir in {os.path.dirname(path) for _, _, path in vids.values() if path}:
            shutil.rmtree(ep_dir, ignore_errors=True)

        n_frames_saved = len(non_image_frames)
        del non_image_frames
        proc._retire(retire_bytes)
        proc._malloc_trim()

        label = "SUCCESS" if success else "FAILURE"
        logger.info(
            f"Episode saved env {env_idx} ({n_frames_saved} frames, {label}; "
            f"add_frame {t_save - t_add:.2f}s, save_episode {t_done - t_save:.2f}s)"
        )
