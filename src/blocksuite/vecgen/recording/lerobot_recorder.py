# ruff: noqa  # ported from TAMP tmp3 (kept diffable against upstream)
"""The LeRobot recorder: the one ``IsaacDataRecorderLeRobot`` datagen (vecgen) and eval record with.

Ported from TAMP ``tmp3.DataRecorder.IsaacDataRecorderLeRobot``: verbatim except imports, plus
two cuts -- goal annotation (TAMP ``GoalAnnotator`` / ``goal_spec``) raises if enabled, and the
default PDDL goal no longer comes from TAMP's global ConfigLoader (callers set every env's goal
with ``set_pddl_goal``).

Added on top, all off by default so a datagen recording is unchanged:

- ``num_writers`` parallel encoder subprocesses (env ``i`` -> writer ``i // ceil(N / K)``), each
  writing a private dataset pair that :meth:`finalize` stitches into the published
  ``<name>_success`` / ``<name>_failure`` pair.
- Auto-reset envs (eval): ``record_step_batched(reset_env_ids=...)`` and
  :meth:`capture_reset_frames` hold the next episode's s_0 as its first row, and
  :meth:`discard_episode` drops an episode the caller does not want kept.
- Per-step extras staged by the caller and moved with the action block: the raw policy action
  (``action.policy``, ``action_dim``), a pd_mode override, and the learned HL's output
  (``record_hl_model``). ``query_physx_properties`` reads live masses and materials per episode.
- :meth:`merged_episode_index_map`: where each env's n-th episode landed.
"""
import collections
import ctypes
import hashlib
import json
import math
import multiprocessing as mp
import os
import pickle
import tempfile
import time
import shutil
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, Future
from queue import Empty
from typing import Optional, List, Dict, Any, Tuple, Sequence, Callable
from pathlib import Path

import av
import torch
import numpy as np

from isaaclab.scene.interactive_scene import InteractiveScene
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors.camera import Camera
from isaaclab.sensors import ContactSensor
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from blocksuite.utils.stiffness_profiles import pd_mode_for_stiffness
from blocksuite.vecgen.recording._logger import logger
from blocksuite.vecgen.recording._video_encoder import (  # forkserver-based encoder (see module docstring)
    _VideoEncoderProcess, EncoderProcessError, ENCODER_CTX,
    payload_nbytes, resolve_encode_threads, resolve_inflight_cap_bytes,
)

# Top-level key in each dataset's ``meta/info.json`` naming the row layout (see the class
# docstring). Readers use its absence to recognise datasets recorded before each row paired s_t
# with the action taken in s_t and the PD gains moved alongside that action.
ROW_LAYOUT_INFO_KEY = "blocksuite_row_layout"
ROW_LAYOUT: Dict[str, Any] = {"layout": "state_then_action", "version": 2}

# With num_writers > 1, writer k stages its dataset pair under <data_dir>/.writers/w<k>/ until
# finalize stitches them into <data_dir>/<name>_success and <name>_failure.
WRITERS_DIRNAME = ".writers"
_TABLE_OBJECT_NAMES = frozenset(("workspace_table", "table"))
# Items smaller than this go straight to the queue even over budget (episode control messages).
_SPILL_MIN_ITEM_BYTES = 64 << 10


def resolve_spill_dir(staging_root: Optional[str] = None) -> Path:
    """Where over-budget frames are spilled: ``BLOCKSUITE_RECORDER_SPILL_DIR``, else the video
    staging root, else the system temp dir. Must be node-local disk (not tmpfs, not weka)."""
    raw = os.environ.get("BLOCKSUITE_RECORDER_SPILL_DIR", "").strip()
    return Path(raw or staging_root or os.environ.get("RERENDER_VIDEO_STAGING") or tempfile.gettempdir())


def _spill_min_free_bytes() -> int:
    """Free space the spill disk must keep (``BLOCKSUITE_RECORDER_SPILL_MIN_FREE_GB``, default 20)."""
    raw = os.environ.get("BLOCKSUITE_RECORDER_SPILL_MIN_FREE_GB", "").strip()
    return int(float(raw or 20) * (1 << 30))


class _Writer:
    """One encoder subprocess and the parent-side bookkeeping for it."""

    def __init__(self, index: int, data_dir: str, dataset_name: str) -> None:
        self.index = index
        self.data_dir = data_dir
        base = Path(data_dir).absolute()
        self.success_root = str(base / f"{dataset_name}_success")
        self.failure_root = str(base / f"{dataset_name}_failure")
        # Mirrors the per-dataset episode counters the encoder subprocess maintains, so the parent
        # can name an episode's video without waiting for the write.
        self.queued_episode_counts: Dict[str, int] = {"success": 0, "failure": 0}
        # Queue MUST come from the forkserver context (same context as the encoder Process),
        # so the child can inherit it. It is UNBOUNDED on purpose: a put must never park the exec
        # loop behind the encoder. Memory is bounded instead by the byte budget -- the encoder
        # credits bytes back through ``consumed`` as it retires frames, and _queue_put waits only
        # when sent-minus-consumed would exceed the cap.
        self.queue = ENCODER_CTX.Queue()
        self.consumed = ENCODER_CTX.Value("Q", 0)
        # Liveness/progress heartbeat: the encoder bumps this after every processed command (and
        # every video closed) so finalize() can distinguish a SLOW-but-working encoder (keep
        # waiting) from a genuinely HUNG one (terminate). A blind join-timeout+terminate used to
        # SIGTERM the encoder mid-flush -> truncated tail videos ("moov atom not found").
        self.heartbeat = ENCODER_CTX.Value("L", 0)
        # Count of videos the encoder could not publish. Any non-zero value means this dataset has
        # holes; video_defects() lets the caller refuse to mark the problem complete instead of
        # silently shipping missing files (35 were lost that way on 2026-08-24).
        self.defects = ENCODER_CTX.Value("L", 0)
        self.outbox_sent = 0
        self.backpressure_s = 0.0
        # In-flight byte budget of the queued (pickled) path; halved once `arena` takes the rest.
        self.cap = 0
        # Shared-memory slots for camera frames (see _shm_slots); None until the first frame
        # with images, or for good when /dev/shm is too small (`arena_failed`).
        self.arena: Optional[Any] = None
        self.arena_failed = False
        # Items written to disk because the RAM budget was full (see _spill).
        self.spill_dir: Optional[Path] = None
        self.spilled_items = 0
        self.spilled_bytes = 0
        self.dead = False
        self.force_terminated = False
        self.process: Optional[_VideoEncoderProcess] = None


_IMAGE_KEY_PREFIXES = (
    "observation.images.",
    "observation.depth.",
    "observation.goal.",
)

SOURCE_PROBLEM_KEY = "source_problem"
INSTANCE_ID_KEY = "instance_id"
SOURCE_PROBLEM_FEATURE = {"dtype": "string", "shape": (1,), "names": None}
INSTANCE_ID_FEATURE = {"dtype": "int64", "shape": (1,), "names": None}
PROVENANCE_DATA_KEYS = (SOURCE_PROBLEM_KEY, INSTANCE_ID_KEY)

# Preemption-resume bookkeeping. A "shard" is one finalized LeRobot dataset pair
# (``blocks_success`` / ``blocks_failure``) under a problem output directory.
SHARD_DONE_MARKER = ".shard_done"
RECORDED_INSTANCES_FILE = "recorded_instances.json"


def shard_dir_for(base_dir: str, index: int) -> Path:
    """Directory that holds shard ``index`` for a problem output ``base_dir``.

    Shard 0 is ``base_dir`` itself, so a run that is never preempted keeps the
    historical ``<data_dir>/blocks_success`` layout. Later shards are siblings
    ``<data_dir>__sNNN`` — still ``problem_*``-prefixed, so merge/annotate, which
    glob ``problem_*`` directories, pick them up with no changes.
    """
    base = Path(base_dir)
    if index == 0:
        return base
    return base.parent / f"{base.name}__s{index:03d}"


def _shard_dir_free(candidate: Path) -> bool:
    """A shard dir is free when it is missing, or exists but holds nothing yet.

    Callers routinely ``mkdir`` the problem dir before constructing the recorder
    (execute_plans / the datagen pipeline / rerender.py). Counting that empty dir
    as taken pushed shard 0 to ``<dir>__s001``; under rerender.py, which does no
    resume scan, every attempt then landed in a fresh sibling that its output
    verifier and completion oracle never looked at, so the item failed forever.
    A dir with any content is still a shard that must not be overwritten.
    """
    if not candidate.exists():
        return True
    return candidate.is_dir() and not any(candidate.iterdir())


def next_free_shard(base_dir: str, start: int = 0) -> Tuple[Path, int]:
    """First shard dir at or after ``start`` that is missing or still empty on disk."""
    index = start
    while True:
        candidate = shard_dir_for(base_dir, index)
        if _shard_dir_free(candidate):
            return candidate, index
        index += 1


def scan_shards(base_dir: str) -> Tuple[set, List[Path], List[Path]]:
    """Inspect existing shards of a problem output dir for resume.

    Returns ``(done_instance_ids, incomplete_dirs, complete_dirs)``. A shard is
    complete iff it carries a ``.shard_done`` marker, in which case the instance
    ids it recorded are read from ``recorded_instances.json``. Incomplete shards
    were interrupted (e.g. by preemption) and must be discarded before rerunning.
    """
    base = Path(base_dir)
    candidates: List[Path] = []
    if base.exists():
        candidates.append(base)
    candidates.extend(sorted(base.parent.glob(f"{base.name}__s*")))

    done_ids: set = set()
    incomplete: List[Path] = []
    complete: List[Path] = []
    for d in candidates:
        if not d.is_dir():
            continue
        if (d / SHARD_DONE_MARKER).is_file():
            complete.append(d)
            manifest = d / RECORDED_INSTANCES_FILE
            try:
                with open(manifest) as f:
                    done_ids.update(int(x) for x in json.load(f))
            except Exception as e:
                logger.warn(f"[DataRecorder] Could not read {manifest}: {e}")
        else:
            incomplete.append(d)
    return done_ids, incomplete, complete


def _problem_name_from_data_dir(data_dir: str) -> str:
    """Extract the problem directory name from a recording output path."""
    for part in reversed(Path(data_dir).absolute().parts):
        if part.startswith("problem_"):
            return part
    return Path(data_dir).name


def _resolve_instance_ids(
    num_envs: int,
    instance_ids: Any = None,
    recording_provenance: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Resolve per-env instance ids from explicit args or recording provenance."""
    raw = instance_ids
    if recording_provenance and recording_provenance.get("instance_json_ids"):
        raw = recording_provenance["instance_json_ids"]

    resolved = np.zeros(num_envs, dtype=np.int64)
    if isinstance(raw, list):
        for env_idx in range(num_envs):
            resolved[env_idx] = int(
                raw[env_idx] if env_idx < len(raw) else raw[0]
            )
    elif raw is not None:
        resolved[:] = int(raw)
    else:
        resolved[:] = np.arange(num_envs, dtype=np.int64)
    return resolved


class StdoutCapture:
    """Wraps sys.stdout to tee all output into a rolling line buffer."""

    def __init__(self, original_stdout, max_lines: int = 100):
        """Wrap ``original_stdout`` and keep the last ``max_lines`` written lines."""
        self._original = original_stdout
        self._buffer = collections.deque(maxlen=max_lines)
        self._partial = ""

    def write(self, text):
        """Forward ``text`` to the underlying stream while accumulating completed lines."""
        result = self._original.write(text)
        self._partial += text
        while "\n" in self._partial:
            line, self._partial = self._partial.split("\n", 1)
            self._buffer.append(line)
        return result

    def flush(self):
        """Flush the underlying stream."""
        return self._original.flush()

    def get_last_lines(self) -> str:
        """Return the buffered tail of stdout (including any partial trailing line)."""
        lines = list(self._buffer)
        if self._partial:
            lines.append(self._partial)
        return "\n".join(lines)

    def __getattr__(self, name):
        """Delegate any unknown attribute access to the underlying stream."""
        return getattr(self._original, name)


class IsaacDataRecorderLeRobot:
    """LeRobot data recorder with comprehensive feature extraction.

    Supports vectorized (multi-env) operation with rich per-frame features
    including goal annotations, error logging, tactile cameras, contact
    sensors, object poses, and PDDL goals.

    State/action alignment
    ----------------------
    Each row holds an observed state and the action taken *in* that state::

        row t:  observation.* = s_t,  action.* = a_{t+1}  (drives s_t -> s_{t+1})
        row N:  observation.* = s_N,  action.* = 0, action.is_valid = False

    The final row is the terminal state the episode reached and carries no
    action; ``action.is_valid`` is the mask to train on, since a zero in
    ``action.joint_target`` is itself a reachable pose.

    This needs care because recording happens *after* a physics step, so a
    single sample yields s_k alongside the command that produced it — the
    latched ``joint_pos_target`` read at call k drove s_{k-1} -> s_k. Writing
    both to one row gives ``(observation after action_t, action_t)``, which is
    off by one and was what earlier datasets stored. Emission is therefore
    deferred one record call: the observation from call k is held, the action
    block from call k+1 is moved onto it, and the pair is emitted together.

    The delta fields need no rescaling under the shift — ``relative_joint_target``,
    ``joint_position`` and ``ee_position`` are already differences against the
    previous recorded frame, i.e. against exactly the state they now sit beside.
    """

    # ------------------------------------------------------------------
    # Construction & setup
    # ------------------------------------------------------------------

    def __init__(self,
            scene: InteractiveScene,
            data_dir: Optional[str] = None,
            dataset_name: Optional[str] = None,
            record_every_n_steps: int = 4,
            fps: Optional[int] = None,
            wrist_body_name: str = "panda_link7",
            ee_body_name: Optional[str] = None,
            num_envs: int = 1,
            enable_goal_annotation: bool = False,
            gripper_binarize_threshold: float = 0.3,
            recording_provenance: Optional[Dict[str, Any]] = None,
            gripper_open_command: Optional[Sequence[float]] = None,
            gripper_close_command: Optional[Sequence[float]] = None,
            robotiq: bool = False,
            instance_ids: Any = None,
            depth_codec: str = "hevc",
            depth_gop: int = 1,
            extra_depth: str = None,
            extra_rgb: str = None,
        extra_rgb_gop: int = 0,
        rgb_gop: int = None,
            record_depth: bool = True,
            video_staging_root: str = None,
            record_videos: bool = True,
            ee_target_fk: Optional[Callable[[np.ndarray], np.ndarray]] = None,
            enable_sharding: bool = True,
            num_writers: int = 1,
            action_dim: Optional[int] = None,
            record_hl_model: bool = False,
            query_physx_properties: bool = False,
            rgb_codec: str = "libsvtav1",
            hl_overlay_camera: Optional[str] = None,
            hl_overlay_side_by_side: bool = False,
        ):
        """Configure the recorder, discover scene sensors, and create the LeRobot dataset.

        Args:
            scene: Isaac Lab scene containing robot, cameras, and rigid objects.
            data_dir: Output directory where the dataset will be written.
            dataset_name: Optional explicit dataset name; auto-generated otherwise.
            record_every_n_steps: Subsample factor applied to simulation steps.
            wrist_body_name: Body name whose incoming joint wrench is recorded.
            ee_body_name: Body name of the end-effector link whose pose is recorded.
                If ``None``, defaults to ``"robotiq_arg2f_base_link"`` when ``robotiq``
                is set, otherwise ``"panda_hand"``. Resolved to a body index at init;
                falls back to the last body in the chain if the named link is absent.
            enable_goal_annotation: If True, load goal-spec deps and emit goal fields.
            gripper_binarize_threshold: Threshold on raw gripper command for binarized
                ``action.gripper_position`` (recorded in ``meta/recording_config.json``).
            recording_provenance: Optional paths for ``meta/extra`` copies and
                ``meta/provenance.json`` (keys: ``domain_pddl``, ``problem_pddl``,
                ``ll_env_directory``, ``instance_json_ids`` list[int], optional).
            gripper_open_command: Finger joint position target(s) for fully open (metadata).
            gripper_close_command: Finger joint position target(s) for fully closed (metadata).
            robotiq: Whether the Robotiq gripper is used (stored in recording metadata).
            instance_ids: Per-env instance/placement ids (``int`` or ``list[int]``). Falls
                back to ``recording_provenance["instance_json_ids"]`` or ``0..num_envs-1``.
            depth_codec: Depth video codec — ``"av1"`` (libaom, smaller files) or
                ``"hevc"`` (libx265, ~15x faster encode). Both use lossless gray12le.
            record_videos: If False, omit all camera features and write only Parquet
                data and dataset metadata.
            ee_target_fk: Optional batched forward-kinematics callable mapping
                ``(N, 7)`` arm joint targets to ``(N, 7)`` EE poses
                ``[x, y, z, qw, qx, qy, qz]`` in the robot base frame. When
                provided, ``action.ee_target`` / ``action.relative_ee_target``
                are recorded; otherwise those columns are zero-filled.
            enable_sharding: Write each run to the first free preemption-resume shard
                (``<data_dir>__sNNN``); False writes into ``data_dir`` itself.
            num_writers: Encoder subprocesses recording in parallel. Env ``i`` goes to writer
                ``i // ceil(num_envs / num_writers)``; with more than one, each stages its own
                dataset pair and :meth:`finalize` stitches them into ``<name>_success`` /
                ``<name>_failure``, so the published layout does not depend on this.
            action_dim: Record the raw policy action staged by
                :meth:`set_pending_actions_batched` as ``action.policy`` (this width).
            record_hl_model: Record the learned HL's per-step output staged by
                :meth:`set_hl_model_outputs_batched` (``action.hl_model_output``,
                ``action.hl_prev_action``, ``action.hl_input_prompt``, ``action.hl_done``).
            query_physx_properties: At each episode's first frame, read the live PhysX
                materials and masses: block/table friction override
                :meth:`set_friction_properties` / ``set_scene_params(table_friction=...)``, and
                ``observation.state.object_mass`` / ``table_mass`` are recorded.
            rgb_codec: Main RGB video codec, ``"libsvtav1"`` (datagen) or ``"libx264"`` (about a
                quarter of the native memory per open stream, for many concurrent episodes).
            hl_overlay_camera: Record ``observation.images.<cam>_hl_overlay``: this camera with the
                HL heatmap and text staged by :meth:`set_hl_overlay_batched` drawn on it (plus the
                Molmo2 previous action / prompt when staged). ``None`` records no overlay.
            hl_overlay_side_by_side: Make the overlay twice as wide, with the staged oracle HL
                drawn on a right-hand copy of the frame (``--oracle_comparison``).
        """
        self.scene = scene
        # The problem output directory. Its provenance name (``source_problem``)
        # comes from the base, but the actual dataset is written into the first
        # collision-free shard dir so a resumed run never overwrites a shard a
        # previous (preempted) run already finalized.
        self._shard_base_dir = data_dir if data_dir else "/home/jason/TMP3/datasets_test"
        self._source_problem = _problem_name_from_data_dir(self._shard_base_dir)
        self._enable_sharding = enable_sharding
        if enable_sharding:
            shard_dir, self._shard_index = next_free_shard(self._shard_base_dir)
        else:
            # Callers that are not the datagen orchestrator write ONE dataset per problem dir and
            # have no resume-from-preemption story. rerender is the case in point: its output
            # verifier, --resume, the orchestrator's completion oracle, merge and annotate all read
            # the problem dir itself, so a sibling __sNNN renders fully and then fails verification
            # forever. Such a caller says so here instead of having to keep the dir empty to coax
            # next_free_shard into returning shard 0.
            shard_dir, self._shard_index = Path(self._shard_base_dir), 0
        self.data_dir = str(shard_dir)
        self.record_every_n_steps = record_every_n_steps
        self.fps = fps if fps else int(100 / record_every_n_steps)
        self.wrist_body_name = wrist_body_name
        self.wrist_body_idx = None
        self.ee_body_name = ee_body_name
        self.ee_body_idx = None
        self.num_envs = num_envs
        self._depth_codec = depth_codec
        # GOP for the MAIN depth stream (see _VideoEncoderProcess._depth_options). The extra
        # stream is always GOP 1.
        self._depth_gop = int(depth_gop)
        # Optional additional depth video per camera, written to
        # videos/observation.depth.<cam>_<variant>/ alongside the main one. See
        # _VideoEncoderProcess._EXTRA_DEPTH.
        self._extra_depth = extra_depth or None
        # False: no depth feature and no depth frames (main or --extra_depth) are recorded.
        self._record_depth = bool(record_depth)
        if not self._record_depth and self._extra_depth:
            raise ValueError("record_depth=False cannot write extra_depth (it is derived from depth)")
        self._extra_rgb = extra_rgb or None
        self._extra_rgb_gop = extra_rgb_gop
        self._rgb_gop = rgb_gop
        # Where the encoder stages videos before publishing. None -> node-local scratch. It must
        # NOT be on weka: the mp4 `moov` trailer is written at close(), and a silent short write
        # over weka produces a moov-less file with no exception raised.
        self._video_staging_root = video_staging_root or None
        self._record_videos = bool(record_videos)
        if not self._record_videos and enable_goal_annotation:
            raise ValueError("goal annotation requires record_videos=True")
        self._gripper_binarize_threshold = float(gripper_binarize_threshold)
        self._recording_provenance = dict(recording_provenance) if recording_provenance else {}
        self._instance_ids = _resolve_instance_ids(
            num_envs,
            instance_ids=instance_ids,
            recording_provenance=self._recording_provenance,
        )
        # per-env episode provenance (set_episode_provenance): the problem the env's current
        # episode belongs to, and the scene objects that are not part of it (parked bodies of a
        # multi-problem package), recorded as NaN poses like an object the scene lacks
        self._source_problems: List[str] = [self._source_problem] * num_envs
        self._inactive_objects: List[frozenset] = [frozenset()] * num_envs
        self._robotiq = bool(robotiq)
        if self.ee_body_name is None:
            self.ee_body_name = (
                "robotiq_arg2f_base_link" if self._robotiq else "panda_hand"
            )
        self._gripper_open_command = (
            None if gripper_open_command is None
            else np.asarray(list(gripper_open_command), dtype=np.float32)
        )
        self._gripper_close_command = (
            None if gripper_close_command is None
            else np.asarray(list(gripper_close_command), dtype=np.float32)
        )
        self._joint_velocity_limits: Optional[list] = None
        self._ee_target_fk = ee_target_fk
        self._num_writers = max(1, min(int(num_writers), int(num_envs)))
        self._writer_block = -(-int(num_envs) // self._num_writers)
        self._action_dim = int(action_dim) if action_dim else None
        self.record_hl_model = bool(record_hl_model)
        self._query_physx_properties = bool(query_physx_properties)
        self._rgb_codec = rgb_codec

        # Per-environment state tracking — numpy arrays for vectorized ops
        self.step_counters = np.zeros(num_envs, dtype=np.int64)
        self.success_flags = np.zeros(num_envs, dtype=np.bool_)
        # Set by eval's --collision: the episode is a failure whatever the goal check says.
        self.success_vetoed = np.zeros(num_envs, dtype=np.bool_)
        self.active_episodes = np.zeros(num_envs, dtype=np.bool_)

        # One built-but-unemitted frame per env.  Frames are held back a single
        # record call so the action block sampled on call k+1 can be moved onto
        # the observation sampled on call k — see the state/action alignment
        # note in the class docstring.
        self._pending_frames: Dict[int, Optional[Dict[str, Any]]] = {
            i: None for i in range(num_envs)
        }

        # Per-step extras staged by the caller for the step about to run, consumed by the next
        # record call and moved with the action block (they describe the command, like the
        # latched joint target): the raw policy action, a pd_mode override (None = derive it
        # from the measured stiffness, as datagen does) and the learned HL's output.
        self._pending_actions: List[Optional[np.ndarray]] = [None] * num_envs
        self._pending_pd_modes: List[Optional[float]] = [None] * num_envs
        self._hl_model_outputs: List[str] = [""] * num_envs
        self._hl_prev_actions: List[str] = [""] * num_envs
        self._hl_input_prompts: List[str] = [""] * num_envs
        self._hl_dones: List[bool] = [False] * num_envs
        # HL overlay video inputs (hl_overlay_camera), staged per step like the HL outputs above:
        # (text, heatmap, prev_action, prompt, oracle_text, oracle_heatmap).
        self._hl_overlay_camera = hl_overlay_camera
        # The recorded image the overlay draws on; resolved against the scene's camera
        # keys ("camera:<name>") when the features schema is built.
        self._hl_overlay_src = f"observation.images.{hl_overlay_camera}"
        self._hl_overlay_side_by_side = bool(hl_overlay_side_by_side)
        self._hl_overlays: List[Optional[tuple]] = [None] * num_envs

        # Auto-reset envs: the next episode's s_0, sampled on the step that reset the env (or by
        # capture_reset_frames), held until that episode's first record call promotes it to row
        # 0, plus the previous-state snapshot its first action's deltas are taken against.
        self._reset_frames: List[Optional[Dict[str, Any]]] = [None] * num_envs
        self._reset_prev: List[Optional[Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]]] = (
            [None] * num_envs
        )

        # Per-env episode ordinal (one per save_episode / discard_episode of an episode) and,
        # per saved episode, (writer, "success"|"failure", index in that writer's dataset,
        # env, ordinal) -- what merged_episode_index_map() maps to published indices.
        self._episode_ordinals = np.zeros(num_envs, dtype=np.int64)
        self._saved_episodes: List[Tuple[int, str, int, int, int]] = []

        # Live PhysX masses per env (query_physx_properties), read at each episode's first frame.
        self._object_masses = np.full(num_envs, np.nan, dtype=np.float32)
        self._table_masses = np.full(num_envs, np.nan, dtype=np.float32)

        # Per-environment action context (kept as dicts for GoalAnnotator compat)
        self.hl_actions: Dict[int, str] = {i: "" for i in range(num_envs)}
        self.hl_args_dict: Dict[int, Any] = {i: {} for i in range(num_envs)}
        self.ll_actions: Dict[int, str] = {i: "" for i in range(num_envs)}
        self.ll_args_dict: Dict[int, Dict[str, Any]] = {i: {} for i in range(num_envs)}
        self.hl_state_strs: List[str] = [""] * num_envs

        # Cached JSON serializations — updated in update_symbolic_actions,
        # reused every frame without re-serializing.
        self._cached_hl_args_json: List[str] = [""] * num_envs
        self._cached_ll_args_json: List[str] = [""] * num_envs
        self._cached_task_descriptions: List[str] = ["manipulation"] * num_envs

        # Per-environment previous state for delta computations — (num_envs, 7)
        self.previous_joint_states = np.zeros((num_envs, 7), dtype=np.float32)
        self.previous_joint_targets = np.zeros((num_envs, 7), dtype=np.float32)
        self.previous_ee_poses: List[Dict[str, np.ndarray]] = [{} for _ in range(num_envs)]

        # Per-environment frame buffers — only populated when goal annotation
        # is enabled (otherwise frames are streamed to the subprocess).
        self.frame_buffers: Dict[int, List[Dict]] = {i: [] for i in range(num_envs)}
        self._frame_counts = np.zeros(num_envs, dtype=np.int64)
        # Raw frame-index spans [start, end) of the open episode that are left out at save
        # (vecgen's failed skill attempts undone by a snapshot restore; see drop_frames_since).
        self._dropped_spans: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(num_envs)}
        # Rows of the open episode, counted when each record call is made: the rows themselves
        # are built on a background thread, so _frame_counts lags by up to one call.
        self._rows_queued = np.zeros(num_envs, dtype=np.int64)
        self._finalized = False

        # Async frame building — overlaps frame construction with
        # the next simulation step.
        self._frame_build_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="framebuild")
        self._pending_build_future: Optional[Future] = None
        # What recording costs the caller's (sim) thread: record_step_batched wall time, and the
        # part of it spent waiting for the previous frame build. Logged when a shard closes.
        self._copy_stream: Optional[Any] = None  # torch.cuda.Stream for _materialize_images
        self._caller_record_s = 0.0
        self._caller_build_wait_s = 0.0
        self._caller_record_calls = 0

        # Per-environment error logs
        self.error_logs: List[str] = [""] * num_envs

        # Per-environment friction properties (set externally via set_friction_properties)
        self._friction_arrays = np.tile(
            np.array([0.5, 0.5, 0.0], dtype=np.float32), (num_envs, 1),
        )

        # Per-environment scale properties (set externally via set_scale_properties)
        self._scale_arrays = np.ones((num_envs, 3), dtype=np.float32)
        self._scale_values: Dict[int, Dict[str, Tuple[float, float, float]]] = {
            i: {} for i in range(num_envs)
        }

        # Per-environment scene randomization params (from N.json)
        self._cube_scale = np.ones(num_envs, dtype=np.float32)
        self._table_height_delta = np.zeros(num_envs, dtype=np.float32)
        self._table_friction = np.full(num_envs, 0.5, dtype=np.float32)
        # Scene profile baked into each env slot; -1 until the executor reports it.
        self._profile_ids = np.full(num_envs, -1, dtype=np.int64)

        # Per-environment identifiers (typically constant across envs but stored per-env for flexibility)
        self._scene_ids: List[str] = [""] * num_envs
        self._robot_ids: List[str] = [""] * num_envs

        # Install stdout capture for error context
        self._stdout_capture = StdoutCapture(sys.stdout, max_lines=100)
        sys.stdout = self._stdout_capture

        # Resolve dataset name early so GoalAnnotator can use it
        self._resolved_dataset_name = dataset_name if dataset_name else "blocks"

        # Goal annotation (optional, self-contained in GoalAnnotator)
        # Not ported from TAMP: GoalAnnotator needs TAMP's goal_spec package.
        self._goal_annotator: Optional[Any] = None
        if enable_goal_annotation:
            raise NotImplementedError(
                "goal annotation (TAMP GoalAnnotator / goal_spec) is not ported to vecgen"
            )

        # Create output directory
        os.makedirs(self.data_dir, exist_ok=True)

        # Discover rigid objects in the scene
        self._object_names = list(self.scene.rigid_objects.keys()) if hasattr(self.scene, 'rigid_objects') else []

        # Discover robot links. Poses are recorded in the robot-base frame.
        robot_name = next(iter(self.scene.articulations.keys()), None)
        robot = self.scene.articulations[robot_name] if robot_name else None
        self._robot_link_names = list(getattr(robot, "body_names", ()))

        # Discover tactile cameras and contact sensors
        self._tactile_camera_names = (
            self._get_tactile_camera_names() if self._record_videos else []
        )
        self._contact_sensor_names = self._get_contact_sensor_names()

        # Load PDDL goal from problem file
        self._pddl_goal = self._load_pddl_goal()
        # Per-environment goal override. Runs where every episode shares one
        # problem file keep the default; play-data collection plans a fresh goal
        # per episode and pushes it in via ``set_pddl_goal``.
        self._pddl_goals: List[str] = [self._pddl_goal] * num_envs
        # Per-env block glyph swap (flags.yaml randomize_task_letters). Empty = no swap, which
        # is the default and leaves every string below byte-identical to before.
        self._glyph_maps: List[dict] = [{} for _ in range(num_envs)]

        # Per-env cuboid colour swap (flags.yaml randomize_block_colors): {block: colour name},
        # plus the palette, so the unified block_variable_map can be built at frame time from this
        # and the glyph map together. Empty = no swap.
        self._cuboid_color_maps: List[dict] = [{} for _ in range(num_envs)]
        self._cuboid_palette = None

        # Find wrist body index for wrench extraction
        self._init_wrench_sensor()

        # Resolve end-effector body index for EE pose extraction
        self._init_ee_body()

        # Create the LeRobot dataset immediately
        self._create_dataset(dataset_name)

        logger.info(f"IsaacDataRecorderLeRobot initialized (vectorized)")
        logger.info(f"  Dataset: {self.dataset_name}")
        logger.info(f"  Success dir: {self._success_root}")
        logger.info(f"  Failure dir: {self._failure_root}")
        logger.info(f"  num_envs: {self.num_envs}")
        logger.info(f"  Record videos: {self._record_videos}")
        logger.info(f"  Robot links ({len(self._robot_link_names)}): {self._robot_link_names}")
        logger.info(f"  Tracked objects ({len(self._object_names)}): {self._object_names}")
        logger.info(f"  Tactile cameras ({len(self._tactile_camera_names)}): {self._tactile_camera_names}")
        logger.info(f"  Contact sensors ({len(self._contact_sensor_names)}): {self._contact_sensor_names}")
        logger.info(f"  PDDL goal loaded: {bool(self._pddl_goal)}")
        logger.info(f"  Goal annotation enabled: {self._goal_annotator is not None}")

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _init_wrench_sensor(self):
        """Initialize wrist body index for wrench extraction."""
        try:
            robot_name = next(iter(self.scene.articulations.keys()), None)
            if robot_name:
                robot = self.scene.articulations[robot_name]
                if hasattr(robot, 'body_names'):
                    self.wrist_body_idx = robot.body_names.index(self.wrist_body_name)
                    logger.info(f"  Wrench sensor: {self.wrist_body_name} (index {self.wrist_body_idx})")
        except (ValueError, StopIteration) as e:
            logger.warn(f" Could not find wrist body '{self.wrist_body_name}': {e}")
            self.wrist_body_idx = None

    def _init_ee_body(self):
        """Resolve the end-effector body index from ``self.ee_body_name``.

        Falls back to the last body in the kinematic chain if the named link is
        not present on the articulation (preserves legacy behavior).
        """
        try:
            robot_name = next(iter(self.scene.articulations.keys()), None)
            if robot_name:
                robot = self.scene.articulations[robot_name]
                if hasattr(robot, 'body_names') and robot.body_names:
                    if self.ee_body_name in robot.body_names:
                        self.ee_body_idx = robot.body_names.index(self.ee_body_name)
                    else:
                        self.ee_body_idx = len(robot.body_names) - 1
                        logger.warn(
                            f" End-effector body '{self.ee_body_name}' not found; "
                            f"falling back to last body '{robot.body_names[-1]}'"
                        )
                    logger.info(f"  End-effector body: {self.ee_body_name} (index {self.ee_body_idx})")
        except (ValueError, StopIteration) as e:
            logger.warn(f" Could not resolve end-effector body '{self.ee_body_name}': {e}")
            self.ee_body_idx = None

    @staticmethod
    def _is_image_key(key: str) -> bool:
        return any(key.startswith(p) for p in _IMAGE_KEY_PREFIXES)

    def _queue_put(self, item):
        """Hand an env's *item* (``(op, env_idx, ...)``) to that env's writer; see :meth:`_writer_put`."""
        self._writer_put(self._writer_for(int(item[1])), item)

    def _writer_for(self, env_idx: int) -> "_Writer":
        """The writer that owns env ``env_idx`` (contiguous blocks of ``ceil(N / K)`` envs)."""
        return self._writers[min(int(env_idx) // self._writer_block, len(self._writers) - 1)]

    def _writer_put(self, writer: "_Writer", item):
        """Hand *item* to *writer*'s encoder process without blocking the exec loop.

        The writer queue is unbounded, so the put itself never waits on the encoder. RAM is
        bounded by the in-flight byte budget (``TMP3_RECORDER_INFLIGHT_GB``, see
        :func:`resolve_inflight_cap_bytes`, split across the writers): once more than that many
        bytes are queued but not yet encoded/saved, an item is SPILLED -- written to local disk
        (:func:`resolve_spill_dir`) with only a marker on the queue, which the encoder reads back
        in order -- so neither RAM nor the exec loop pays for a slow encoder. Only when the spill
        disk is short of space (``BLOCKSUITE_RECORDER_SPILL_MIN_FREE_GB``) does the exec loop wait
        for the encoder; that wait is accounted in ``backpressure_s`` (reported at shard close).

        Never waits forever: if the encoder subprocess has died we raise
        :class:`EncoderProcessError` instead of deadlocking on a budget that will never be
        credited again (the historical silent, GPU-holding hang).

        An encoder that is alive but cannot drain the budget is reported, not waited on silently:
        a warning every minute, and after ``_queue_full_abort_s`` (``RERENDER_QUEUE_FULL_ABORT_S``,
        default 900 s; 0 disables) an :class:`EncoderProcessError`, so a stalled encoder fails the
        item instead of holding the GPU until the render watchdog (the 2026-09-15 rerender stall).
        """
        import time as _time
        proc = writer.process
        nbytes = payload_nbytes(item)
        t0 = _time.monotonic()
        waited, next_warn = 0.0, 15.0
        while True:
            # Check liveness BEFORE committing to a (potentially blocking) wait.
            if proc is None or not proc.is_alive():
                writer.dead = True
                code = None if proc is None else proc.exitcode
                raise EncoderProcessError(
                    f"video encoder subprocess is not running (exitcode={code}); "
                    "aborting to avoid a deadlock on the writer queue"
                )
            inflight = writer.outbox_sent - int(writer.consumed.value)
            if inflight <= 0 or inflight + nbytes <= writer.cap:
                break
            if nbytes < _SPILL_MIN_ITEM_BYTES:
                # Episode control messages: tiny, and must never wait behind frames.
                break
            if self._spill(writer, item, nbytes):
                return
            if waited == 0.0:
                logger.debug(
                    f"[DataRecorder] encoder backlog {inflight / 2**20:.0f} MiB >= cap "
                    f"{writer.cap / 2**20:.0f} MiB; exec loop waits for the encoder")
            _time.sleep(0.005)
            waited = _time.monotonic() - t0
            # ALIVE BUT BEHIND. Say so, and give up eventually: from outside, a render parked
            # here is indistinguishable from a wedged one (busy-looking GPU, no progress).
            if waited >= self._queue_full_abort_s > 0:
                writer.dead = True
                raise EncoderProcessError(
                    f"video encoder has not drained its in-flight budget for {waited:.0f}s "
                    f"({inflight / 2**20:.0f} MiB queued, cap {writer.cap / 2**20:.0f} MiB, "
                    f"still alive). The encoder cannot keep up with the render at this "
                    f"configuration -- reduce cameras, raise --depth_gop (GOP 1 is ~3x the CPU "
                    f"of GOP 1000 per depth frame), or lower --num_envs/--procs_per_gpu."
                )
            if waited >= next_warn:
                print(f"[recorder] encoder BACKPRESSURE: in-flight budget full for {waited:.0f}s "
                      f"({inflight / 2**20:.0f} MiB >= cap {writer.cap / 2**20:.0f} MiB); "
                      f"the render is blocked waiting on the encoder, NOT hung. Aborting at "
                      f"{self._queue_full_abort_s:.0f}s.", flush=True)
                next_warn = 60.0 if next_warn < 60.0 else next_warn + 60.0
        if waited:
            writer.backpressure_s += waited
            if waited >= 15.0:
                print(f"[recorder] encoder backpressure CLEARED after {waited:.0f}s", flush=True)
        writer.outbox_sent += nbytes
        writer.queue.put(item)

    def _spill(self, writer: "_Writer", item: tuple, nbytes: int) -> bool:
        """Write *item* to the spill dir and queue a marker for it; False when the disk is short.

        The bytes still count as in flight until the encoder retires them, so while a spilled
        backlog drains new items keep spilling and the queue order is preserved.
        """
        try:
            if writer.spill_dir is None:
                writer.spill_dir = resolve_spill_dir(self._video_staging_root) / (
                    f"blocksuite_recorder_spill_{os.getpid()}_{id(self):x}"
                ) / f"w{writer.index:03d}"
                writer.spill_dir.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(writer.spill_dir).free < _spill_min_free_bytes():
                return False
            path = writer.spill_dir / f"{writer.spilled_items:09d}.pkl"
            with open(path, "wb") as f:
                pickle.dump(item, f, protocol=5)
        except OSError as e:
            logger.error(f"[DataRecorder] spill to {writer.spill_dir} failed ({e}); waiting on the encoder")
            return False
        if writer.spilled_items == 0:
            print(f"[recorder] writer {writer.index}: encoder behind, spilling frames to "
                  f"{writer.spill_dir} (RAM budget {writer.cap / 2**20:.0f} MiB full)", flush=True)
        writer.spilled_items += 1
        writer.spilled_bytes += nbytes
        writer.outbox_sent += nbytes
        writer.queue.put(("spilled", str(path)))
        return True

    # ------------------------------------------------------------------
    # Goal annotation loading
    # ------------------------------------------------------------------

    def _load_pddl_goal(self) -> str:
        """Default PDDL goal for every env.

        TAMP read it from its global ConfigLoader problem file; vecgen has no such config
        and pushes each env's goal in through ``set_pddl_goal``, so the default is empty
        (what TAMP returned when no problem file was configured).
        """
        return ""

    # ------------------------------------------------------------------
    # Dataset creation
    # ------------------------------------------------------------------

    def _create_dataset(self, dataset_name: Optional[str] = None):
        """Build feature schemas and start the video encoder subprocess."""
        dataset_name = self._resolved_dataset_name
        self.dataset_name = dataset_name

        camera_names = self._get_camera_names() if self._record_videos else []
        height, width = self._get_camera_resolution()
        features = self._build_features_schema(camera_names, height, width)

        features_tabular = {
            k: v for k, v in features.items() if v.get("dtype") != "video"
        }
        features_video = {
            k: v for k, v in features.items() if v.get("dtype") == "video"
        }

        self._cam_configs: Dict[str, Tuple[int, int]] = {}
        for key, feat in features_video.items():
            info = feat.get("info", {})
            w = info.get("video.width", width)
            h = info.get("video.height", height)
            self._cam_configs[key] = (w, h)

        base = Path(self.data_dir).absolute()
        self._success_root = str(base / f"{dataset_name}_success")
        self._failure_root = str(base / f"{dataset_name}_failure")

        # The in-flight byte budget is split across the writers so the total stays what one
        # writer used to get.
        self._outbox_cap = max(1, resolve_inflight_cap_bytes() // self._num_writers)
        # Fail an item after this long with the budget full and a LIVE encoder. Legitimate
        # backpressure is seconds; minutes means the encoder cannot keep up at this configuration
        # and will not recover. 0 disables (wait forever).
        self._queue_full_abort_s = float(os.environ.get("RERENDER_QUEUE_FULL_ABORT_S", 900))
        # Each writer gets its share of the cores for its encode pool (TMP3_ENCODER_THREADS still
        # sizes a single writer).
        encode_threads = (
            None if self._num_writers == 1
            else max(2, resolve_encode_threads() // self._num_writers)
        )
        self._writers: List[_Writer] = []
        for k in range(self._num_writers):
            wdir = (
                self.data_dir if self._num_writers == 1
                else str(base / WRITERS_DIRNAME / f"w{k:03d}")
            )
            os.makedirs(wdir, exist_ok=True)
            writer = _Writer(k, wdir, dataset_name)
            writer.cap = self._outbox_cap
            writer.process = _VideoEncoderProcess(
                cmd_queue=writer.queue,
                data_dir=wdir,
                dataset_name=dataset_name,
                fps=self.fps,
                features_tabular=features_tabular,
                features_video=features_video,
                depth_codec=self._depth_codec,
                depth_gop=self._depth_gop,
                extra_depth=self._extra_depth,
                extra_rgb=self._extra_rgb,
                extra_rgb_gop=self._extra_rgb_gop,
                rgb_gop=self._rgb_gop,
                heartbeat=writer.heartbeat,
                defects=writer.defects,
                staging_root=self._video_staging_root,
                consumed=writer.consumed,
                encode_threads=encode_threads,
                rgb_codec=self._rgb_codec,
            )
            writer.process.start()
            self._writers.append(writer)
        # Fail fast if an encoder can't even come up (bad env / import error /
        # immediate native crash) rather than discovering it later mid-run.
        import time as _time
        _time.sleep(0.2)
        for writer in self._writers:
            if not writer.process.is_alive():
                raise EncoderProcessError(
                    f"video encoder subprocess {writer.index} exited immediately on start "
                    f"(exitcode={writer.process.exitcode})"
                )

        self._camera_names = camera_names
        self._episode_started = np.zeros(self.num_envs, dtype=np.bool_)

    def video_defects(self) -> int:
        """Videos the encoder could not publish. Non-zero => the output has holes and the problem
        MUST NOT be recorded as completed."""
        try:
            return sum(int(w.defects.value) for w in self._writers)
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Camera / sensor introspection
    # ------------------------------------------------------------------

    def _get_camera_names(self) -> List[str]:
        """Get list of camera names from scene, excluding tactile sensors."""
        camera_names = []
        for name, sensor in self.scene.sensors.items():
            if isinstance(sensor, Camera) and 'tactile' not in name.lower():
                camera_names.append(name)
        return camera_names

    def _get_camera_resolution(self) -> Tuple[int, int]:
        """Get camera resolution (height, width) from first camera."""
        for name, sensor in self.scene.sensors.items():
            if isinstance(sensor, Camera) and 'tactile' not in name.lower() and hasattr(sensor, 'cfg'):
                return (sensor.cfg.height, sensor.cfg.width)
        return (480, 640)

    def _get_tactile_camera_names(self) -> List[str]:
        names = []
        for name, sensor in self.scene.sensors.items():
            if isinstance(sensor, Camera) and 'tactile' in name.lower():
                names.append(name)
        return names

    def _get_tactile_camera_resolution(self) -> Tuple[int, int]:
        """Get tactile camera resolution (height, width) from first tactile camera."""
        for name, sensor in self.scene.sensors.items():
            if isinstance(sensor, Camera) and 'tactile' in name.lower() and hasattr(sensor, 'cfg'):
                return (sensor.cfg.height, sensor.cfg.width)
        return (480, 640)

    def _get_contact_sensor_names(self) -> List[str]:
        names = []
        for name, sensor in self.scene.sensors.items():
            if isinstance(sensor, ContactSensor):
                names.append(name)
        return names

    # ------------------------------------------------------------------
    # Delta computation helpers (vectorized per-env)
    # ------------------------------------------------------------------

    def _compute_delta_joint_action(self, current_joint_position: np.ndarray, env_idx: int = 0) -> np.ndarray:
        """Compute delta joint action w.r.t. previous recorded joint position."""
        prev = self.previous_joint_states[env_idx]
        if not prev.any():
            return np.zeros(7, dtype=np.float32)
        return (current_joint_position - prev).astype(np.float32)

    def _compute_relative_joint_target(self, current_joint_target: np.ndarray, current_joint_position: np.ndarray, env_idx: int = 0) -> np.ndarray:
        """Compute relative joint target w.r.t. previous recorded joint position.

        Computes target[t] - joint_pos[t-1] to match replay timing.
        On the first frame, uses current_joint_position as there is no previous state.
        """
        prev = self.previous_joint_states[env_idx]
        if not prev.any():
            return (current_joint_target - current_joint_position).astype(np.float32)
        return (current_joint_target - prev).astype(np.float32)

    # ------------------------------------------------------------------
    # Feature schema
    # ------------------------------------------------------------------

    def _build_features_schema(self, camera_names: List[str], height: int, width: int) -> Dict[str, Dict]:
        """Build LeRobot features schema with all observation and action fields."""
        features = {
            "is_first": {"dtype": "bool", "shape": (1,), "names": None},
            "is_last": {"dtype": "bool", "shape": (1,), "names": None},
            "observation.state.gripper_position": {"dtype": "float32", "shape": (1,), "names": {"axes": ["gripper"]}},
            "observation.state.ee_position": {"dtype": "float32", "shape": (6,), "names": {"axes": ["x", "y", "z", "roll", "pitch", "yaw"]}},
            "observation.state.joint_position": {"dtype": "float32", "shape": (7,), "names": {"axes": [f"joint_{i}" for i in range(7)]}},
            "observation.state.joint_velocity": {"dtype": "float32", "shape": (7,), "names": {"axes": [f"joint_{i}" for i in range(7)]}},
            "observation.state.joint_acceleration": {"dtype": "float32", "shape": (7,), "names": {"axes": [f"joint_{i}" for i in range(7)]}},
            "observation.state.joint_effort": {"dtype": "float32", "shape": (7,), "names": {"axes": [f"joint_{i}" for i in range(7)]}},
            "observation.state": {"dtype": "float32", "shape": (8,), "names": {"axes": [f"joint_{i}" for i in range(7)] + ["gripper"]}},
            "observation.state.current_stiffness": {"dtype": "float32", "shape": (13,), "names": {"axes": [f"joint_{i}" for i in range(7)]}},
            "observation.state.current_damping": {"dtype": "float32", "shape": (13,), "names": {"axes": [f"joint_{i}" for i in range(7)]}},
            "observation.state.current_effort_limit": {"dtype": "float32", "shape": (13,), "names": {"axes": [f"joint_{i}" for i in range(7)]}},
            "observation.state.current_velocity_limit": {"dtype": "float32", "shape": (13,), "names": {"axes": [f"joint_{i}" for i in range(7)]}},
            "observation.state.current_torque": {"dtype": "float32", "shape": (13,), "names": {"axes": [f"joint_{i}" for i in range(7)]}},
            "observation.state.wrench": {"dtype": "float32", "shape": (6,), "names": {"axes": ["fx", "fy", "fz", "tx", "ty", "tz"]}},
            "action.gripper_position": {"dtype": "float32", "shape": (1,), "names": {"axes": ["gripper"]}},
            "action.gripper_raw": {"dtype": "float32", "shape": (1,), "names": {"axes": ["gripper_raw"]}},
            "action.ee_position": {"dtype": "float32", "shape": (6,), "names": {"axes": ["x", "y", "z", "roll", "pitch", "yaw"]}},
            "action.joint_position": {"dtype": "float32", "shape": (7,), "names": {"axes": [f"joint_{i}" for i in range(7)]}},
            "action.joint_target": {"dtype": "float32", "shape": (7,), "names": {"axes": [f"joint_{i}" for i in range(7)]}},
            "action.relative_joint_target": {"dtype": "float32", "shape": (7,), "names": {"axes": [f"joint_{i}" for i in range(7)]}},
            "action.ee_target": {"dtype": "float32", "shape": (7,), "names": {"axes": ["x", "y", "z", "qw", "qx", "qy", "qz"]}},
            "action.relative_ee_target": {"dtype": "float32", "shape": (7,), "names": {"axes": ["x", "y", "z", "qw", "qx", "qy", "qz"]}},
            "action.pd_mode": {"dtype": "float32", "shape": (1,), "names": None},
            "action": {"dtype": "float32", "shape": (9,), "names": {"axes": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper", "hard_config"]}},
            "action.is_valid": {"dtype": "bool", "shape": (1,), "names": None},
            "action.high_level_action": {"dtype": "string", "shape": (1,), "names": None},
            "action.high_level_args": {"dtype": "string", "shape": (1,), "names": None},
            "action.low_level_action": {"dtype": "string", "shape": (1,), "names": None},
            "action.low_level_args": {"dtype": "string", "shape": (1,), "names": None},
            "pddl_goal": {"dtype": "string", "shape": (1,), "names": None},
            # {original_block_name: glyph} for this episode, JSON. The pose COLUMNS keep the
            # original names (parquet schema is per file, not per episode), while pddl_goal and
            # task carry the swapped glyphs -- this column is what reconciles the two.
            "glyph_map": {"dtype": "string", "shape": (1,), "names": None},
            # {rewritten pddl name: provenance} for this episode, JSON, covering glyph blocks
            # AND colour cuboids in one place. pddl_goal/task carry rewritten names while the
            # pose COLUMNS keep the originals (parquet schema is per file, the mapping is per
            # episode), so this is what reconciles them. Keyed by the rewritten name because that
            # is the direction consumers read, and the only unambiguous one: a rewritten name can
            # collide with another block's authored name. glyph_map keeps its own forward shape
            # for the datasets that already ship it.
            "block_variable_map": {"dtype": "string", "shape": (1,), "names": None},
            "observation.state.high_level_state": {"dtype": "string", "shape": (1,), "names": None},
            "is_episode_successful": {"dtype": "bool", "shape": (1,), "names": None},
            "error": {"dtype": "string", "shape": (1,), "names": None},
            "observation.state.friction_properties": {"dtype": "float32", "shape": (3,), "names": {"axes": ["static_friction", "dynamic_friction", "restitution"]}},
            "observation.state.scale_properties": {"dtype": "float32", "shape": (3,), "names": {"axes": ["sx", "sy", "sz"]}},
            "observation.state.cube_scale": {"dtype": "float32", "shape": (1,), "names": None},
            "observation.state.table_height_delta": {"dtype": "float32", "shape": (1,), "names": None},
            "observation.state.table_friction": {"dtype": "float32", "shape": (1,), "names": None},
            "observation.state.profile_id": {"dtype": "int64", "shape": (1,), "names": None},
            "scene_id": {"dtype": "string", "shape": (1,), "names": None},
            "robot_id": {"dtype": "string", "shape": (1,), "names": None},
            SOURCE_PROBLEM_KEY: SOURCE_PROBLEM_FEATURE,
            INSTANCE_ID_KEY: INSTANCE_ID_FEATURE,
        }

        for cam_name in camera_names:
            features[f"observation.images.{cam_name}"] = {
                "dtype": "video", "shape": [height, width, 3], "names": ["height", "width", "channels"],
                "info": {"video.fps": self.fps, "video.height": height, "video.width": width, "video.channels": 3, "video.is_depth_map": False, "has_audio": False},
            }
            depth_cfg = _VideoEncoderProcess._DEPTH_CODECS[self._depth_codec]
            # RG8 stores the same uint12 value in an 8-bit RGB frame (R=low byte, G=high byte,
            # B=0), so it is genuinely 3-channel rgb24 and must be described as such -- a reader
            # that trusts channels=1 / gray12le would decode it as garbage.
            _is_rg8 = depth_cfg["pix_fmt"] == "rgb24"
            _ch = 3 if _is_rg8 else 1
            features[f"observation.depth.{cam_name}"] = {
                "dtype": "video", "shape": [height, width, _ch], "names": ["height", "width", "channels"],
                "info": {
                    "video.fps": self.fps, "video.height": height, "video.width": width,
                    "video.channels": _ch, "video.is_depth_map": True,
                    "video.codec": self._depth_codec,
                    "video.pix_fmt": depth_cfg["pix_fmt"],
                    "video.depth_encoding": "rg8_mm" if _is_rg8 else "gray12le_mm",
                    "video.depth_scale": 1000.0,
                    "video.depth_max_mm": 4095,
                    "has_audio": False,
                },
            }
            if not self._record_depth:
                features.pop(f"observation.depth.{cam_name}", None)
            # --extra_depth: declared as its own video feature so _cam_configs picks up its
            # (smaller) size and the encoder's staging->final move lands it in
            # videos/observation.depth.<cam>_<variant>/. Always rg8/rgb24 regardless of the main
            # depth format, and always its own resolution -- consumers must read width/height from
            # here rather than assuming the main depth's.
            # --extra_rgb: a 224x224 sidecar of this RGB camera, named the way the training
            # sidecars are (observation.images.<cam>_224 -- no codec prefix, unlike depth's
            # _rg8_224, because rg8 is a DEPTH packing and means nothing for colour).
            if self._extra_rgb:
                _rspec = _VideoEncoderProcess._EXTRA_RGB[self._extra_rgb]
                _rw, _rh = _rspec["size"]
                features[f"observation.images.{cam_name}_{self._extra_rgb}"] = {
                    "dtype": "video", "shape": [_rh, _rw, 3],
                    "names": ["height", "width", "channels"],
                    "info": {
                        "video.fps": self.fps, "video.height": _rh, "video.width": _rw,
                        "video.channels": 3, "video.is_depth_map": False,
                        "video.codec": "av1", "video.pix_fmt": "yuv420p",
                        "video.resize": "pillow_bilinear",
                        "has_audio": False,
                    },
                }
            if self._extra_depth:
                _spec = _VideoEncoderProcess._EXTRA_DEPTH[self._extra_depth]
                _ew, _eh = _spec["size"]
                # Derived from the variant's codec, NOT hardcoded: rg8_224 is a 3-channel rgb24
                # byte-split, g12_224 a 1-channel gray12le stream. Declaring g12 as rgb24/rg8_mm
                # would tell a consumer to unpack R+(G<<8) out of a grayscale plane.
                _ecfg = _VideoEncoderProcess._DEPTH_CODECS[_spec["codec"]]
                _egray = _ecfg["pix_fmt"] == "gray12le"
                _ech = 1 if _egray else 3
                _einfo = {
                    "video.fps": self.fps, "video.height": _eh, "video.width": _ew,
                    "video.channels": _ech, "video.is_depth_map": True,
                    "video.codec": _spec["codec"],
                    "video.pix_fmt": _ecfg["pix_fmt"],
                    "video.depth_encoding": "gray12le_mm" if _egray else "rg8_mm",
                    "video.depth_scale": 1000.0,
                    # The 5+5 packing SATURATES at 1023 mm, so declaring 4095 would tell a consumer
                    # the far field is present when it is clipped.
                    "video.depth_max_mm": int(_spec.get("max_mm", 4095)),
                    "has_audio": False,
                }
                if "gop" in _spec:
                    # The random-access budget: a reader decodes at most this many frames from the
                    # preceding keyframe. Per camera, so it belongs on the feature, not the run.
                    _einfo["video.gop"] = _VideoEncoderProcess._extra_depth_gop(
                        _spec, f"observation.depth.{cam_name}")
                features[f"observation.depth.{cam_name}_{self._extra_depth}"] = {
                    "dtype": "video", "shape": [_eh, _ew, _ech],
                    "names": ["height", "width", "channels"],
                    "info": _einfo,
                }

        for obj_name in self._object_names:
            features[f"observation.state.object_pose.{obj_name}"] = {"dtype": "float32", "shape": (7,), "names": {"axes": ["x", "y", "z", "qw", "qx", "qy", "qz"]}}

        for link_name in self._robot_link_names:
            features[f"robot.poses.{link_name}"] = {
                "dtype": "float32",
                "shape": (7,),
                "names": {"axes": ["x", "y", "z", "qw", "qx", "qy", "qz"]},
            }

        if self._tactile_camera_names:
            t_height, t_width = self._get_tactile_camera_resolution()
            for cam_name in self._tactile_camera_names:
                features[f"observation.images.{cam_name}"] = {
                    "dtype": "video", "shape": [t_height, t_width, 3], "names": ["height", "width", "channels"],
                    "info": {"video.fps": self.fps, "video.height": t_height, "video.width": t_width, "video.channels": 3, "video.is_depth_map": False, "has_audio": False},
                }

        all_cam_names = list(camera_names) + list(self._tactile_camera_names)
        for cam_name in all_cam_names:
            features[f"observation.camera_pose.{cam_name}"] = {"dtype": "float32", "shape": (7,), "names": {"axes": ["x", "y", "z", "qw", "qx", "qy", "qz"]}}
            features[f"observation.camera_intrinsics.{cam_name}"] = {"dtype": "float32", "shape": (9,), "names": {"axes": ["fx", "s", "cx", "zero1", "fy", "cy", "zero2", "zero3", "one"]}}

        for sensor_name in self._contact_sensor_names:
            features[f"observation.state.contact_force.{sensor_name}"] = {"dtype": "float32", "shape": (3,), "names": {"axes": ["fx", "fy", "fz"]}}

        if self._action_dim is not None:
            features["action.policy"] = {
                "dtype": "float32", "shape": (self._action_dim,),
                "names": {"axes": [f"d{i}" for i in range(self._action_dim)]},
            }
        if self.record_hl_model:
            for key in self._HL_MODEL_STRING_FIELDS:
                features[key] = {"dtype": "string", "shape": (1,), "names": None}
            features["action.hl_done"] = {"dtype": "bool", "shape": (1,), "names": None}
        if self._query_physx_properties:
            # Live PhysX masses (NaN when the query fails or the scene has no such body).
            features["observation.state.object_mass"] = {"dtype": "float32", "shape": (1,), "names": None}
            features["observation.state.table_mass"] = {"dtype": "float32", "shape": (1,), "names": None}

        if self._hl_overlay_camera is not None:
            # Scene cameras register as "camera:<name>"; accept the bare name too. The
            # overlay stream keeps the configured name (observation.images.<cam>_hl_overlay).
            sources = [
                f"observation.images.{cam}"
                for cam in (self._hl_overlay_camera, f"camera:{self._hl_overlay_camera}")
            ]
            src = next((key for key in sources if key in features), None)
            if src is None:
                raise ValueError(
                    f"hl_overlay_camera={self._hl_overlay_camera!r} is not a recorded camera "
                    f"(available: {camera_names})."
                )
            self._hl_overlay_src = src
            overlay_w = width * 2 if self._hl_overlay_side_by_side else width
            features[f"observation.images.{self._hl_overlay_camera}_hl_overlay"] = {
                "dtype": "video", "shape": [height, overlay_w, 3], "names": ["height", "width", "channels"],
                "info": {"video.fps": self.fps, "video.height": height, "video.width": overlay_w, "video.channels": 3, "video.is_depth_map": False, "has_audio": False},
            }

        if self._goal_annotator:
            features.update(self._goal_annotator.feature_schema(height, width))

        return features

    # ------------------------------------------------------------------
    # Tensor extraction and utilities
    # ------------------------------------------------------------------

    def _extract_tensor(self, tensor: torch.Tensor, env_idx: int = 0) -> np.ndarray:
        """Extract numpy array from torch tensor for a specific environment.

        Handles both vectorized (num_envs, ...) and single-env tensors.
        """
        arr = tensor.cpu().numpy()
        if arr.ndim > 0 and arr.shape[0] >= self.num_envs and self.num_envs > 1:
            return np.array(arr[env_idx], dtype=np.float32, copy=True)
        if arr.shape[0] == 1:
            return np.array(np.squeeze(arr, axis=0), dtype=np.float32, copy=True)
        return np.array(arr, dtype=np.float32, copy=True)

    def _quaternion_to_euler(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion [w, x, y, z] to euler angles [roll, pitch, yaw]."""
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (w * y - z * x)
        pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return np.array([roll, pitch, yaw], dtype=np.float32)

    @staticmethod
    def _quaternion_to_euler_batched(quats: np.ndarray) -> np.ndarray:
        """Convert (N, 4) quaternions [w, x, y, z] to (N, 3) euler [roll, pitch, yaw]."""
        w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (w * y - z * x)
        pitch = np.where(
            np.abs(sinp) >= 1.0,
            np.copysign(np.pi / 2.0, sinp),
            np.arcsin(np.clip(sinp, -1.0, 1.0)),
        )
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return np.stack([roll, pitch, yaw], axis=-1).astype(np.float32)

    def _extract_wrench(self, env_idx: int = 0) -> np.ndarray:
        """Extract force-torque wrench from wrist for a specific environment."""
        if self.wrist_body_idx is None:
            return np.zeros(6, dtype=np.float32)
        try:
            robot_name = next(iter(self.scene.articulations.keys()), None)
            if robot_name:
                robot = self.scene.articulations[robot_name]
                body_wrenches = robot.data.body_incoming_joint_wrench_b
                wrench = body_wrenches[env_idx, self.wrist_body_idx]
                return wrench.cpu().numpy().astype(np.float32)
        except Exception:
            pass
        return np.zeros(6, dtype=np.float32)

    # ------------------------------------------------------------------
    # End-effector pose math
    # ------------------------------------------------------------------

    def _compute_delta_ee_pose(self, current_ee_pose: np.ndarray, robot_name: str, env_idx: int = 0) -> np.ndarray:
        """Compute delta EE pose w.r.t. previous EE pose for a specific environment."""
        if robot_name in self.previous_ee_poses[env_idx]:
            prev_pose = self.previous_ee_poses[env_idx][robot_name]
            pos_delta = current_ee_pose[:3] - prev_pose[:3]
            quat_delta = self._quaternion_difference(prev_pose[3:7], current_ee_pose[3:7])
            return np.concatenate([pos_delta, quat_delta])
        return np.zeros(7, dtype=np.float32)

    def _quaternion_difference(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Compute quaternion difference q2 * inv(q1)."""
        q1_inv = np.array([q1[0], -q1[1], -q1[2], -q1[3]])
        w = q2[0] * q1_inv[0] - q2[1] * q1_inv[1] - q2[2] * q1_inv[2] - q2[3] * q1_inv[3]
        x = q2[0] * q1_inv[1] + q2[1] * q1_inv[0] + q2[2] * q1_inv[3] - q2[3] * q1_inv[2]
        y = q2[0] * q1_inv[2] - q2[1] * q1_inv[3] + q2[2] * q1_inv[0] + q2[3] * q1_inv[1]
        z = q2[0] * q1_inv[3] + q2[1] * q1_inv[2] - q2[2] * q1_inv[1] + q2[3] * q1_inv[0]
        return np.array([w, x, y, z], dtype=np.float32)

    # ------------------------------------------------------------------
    # Batched numpy quaternion ops (used by prefetch path)
    # ------------------------------------------------------------------

    @staticmethod
    def _quat_rotate_vec_np_batched(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Rotate 3D vectors by quaternions using numpy.

        Args:
            q: (..., 4) quaternions [qw, qx, qy, qz]
            v: (..., 3) vectors
        Returns:
            (..., 3) rotated vectors as float32
        """
        q_xyz = q[..., 1:]
        qw = q[..., 0:1]
        t = 2.0 * np.cross(q_xyz, v)
        return (v + qw * t + np.cross(q_xyz, t)).astype(np.float32)

    @staticmethod
    def _world_to_local_np_batched(
        poses_w: np.ndarray,
        base_pos: np.ndarray,
        base_quat_inv: np.ndarray,
    ) -> np.ndarray:
        """Transform world poses to local (robot base) frame using batched numpy.

        Args:
            poses_w: (N, 7) world poses [x, y, z, qw, qx, qy, qz]
            base_pos: (N, 3) robot base positions
            base_quat_inv: (N, 4) inverse base quaternions [qw, qx, qy, qz]
        Returns:
            (N, 7) local poses as float32
        """
        world_pos = poses_w[:, :3]
        world_quat = poses_w[:, 3:]
        rel_pos = world_pos - base_pos

        local_pos = IsaacDataRecorderLeRobot._quat_rotate_vec_np_batched(
            base_quat_inv, rel_pos
        )

        w1, x1, y1, z1 = (
            base_quat_inv[:, 0], base_quat_inv[:, 1],
            base_quat_inv[:, 2], base_quat_inv[:, 3],
        )
        w2, x2, y2, z2 = (
            world_quat[:, 0], world_quat[:, 1],
            world_quat[:, 2], world_quat[:, 3],
        )
        local_quat = np.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], axis=-1)

        return np.concatenate([local_pos, local_quat], axis=-1).astype(np.float32)

    # ------------------------------------------------------------------
    # Image / serialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_image(img: np.ndarray) -> np.ndarray:
        """Prepare image for LeRobot: ensure uint8, 3-channel, HWC format."""
        if img is None:
            return None
        if img.dtype != np.uint8:
            if img.dtype == np.float32 or img.dtype == np.float64:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[-1] == 1:
            img = np.concatenate([img, img, img], axis=-1)
        elif img.shape[-1] == 4:
            img = img[:, :, :3].copy()
        return img

    @staticmethod
    def _prepare_depth_image(depth: np.ndarray) -> np.ndarray:
        """Encode depth as a gray12le-compatible uint16 frame for video storage.

        Isaac Lab outputs ``distance_to_image_plane`` in **metres** (float32).
        We convert to uint16 millimetres, clamp to 4095 (12-bit max), and
        left-shift by 4 to fill the upper 12 bits of uint16.  This is
        necessary because PyAV's gray16le→gray12le conversion right-shifts
        by 4; the pre-shift ensures round-trip fidelity.

        Decoding (at training / loading time)::

            depth_mm = np.array(frame, dtype=np.uint16)   # (H, W) uint16, 0-4095
            depth_m  = depth_mm.astype(np.float32) / 1000.0
        """
        if depth is None:
            return None

        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[:, :, 0]

        depth_m = np.nan_to_num(depth, nan=0.0).astype(np.float32)
        depth_mm = np.clip(depth_m * 1000.0, 0, 4095).astype(np.uint16)
        depth_mm <<= 4
        return depth_mm

    def _convert_to_serializable(self, value: Any) -> Any:
        """Convert various data types to JSON-serializable format."""
        if isinstance(value, torch.Tensor):
            return value.cpu().numpy().tolist()
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, (list, tuple)):
            return [self._convert_to_serializable(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._convert_to_serializable(v) for k, v in value.items()}
        else:
            try:
                json.dumps(value)
                return value
            except (TypeError, OverflowError):
                return str(value) if value is not None else None

    # ------------------------------------------------------------------
    # Error logging
    # ------------------------------------------------------------------

    def set_friction_properties(self, friction_vals: Dict[str, float], env_idx: int = 0):
        """Store sampled friction properties for a specific environment."""
        self._friction_arrays[env_idx] = [
            friction_vals.get("static_friction", 0.5),
            friction_vals.get("dynamic_friction", 0.5),
            friction_vals.get("restitution", 0.0),
        ]

    def set_scale_properties(self, scale_vals: Dict[str, Tuple[float, float, float]], env_idx: int = 0):
        """Store sampled scale properties for a specific environment."""
        self._scale_values[env_idx] = scale_vals
        if scale_vals:
            self._scale_arrays[env_idx] = np.array(
                next(iter(scale_vals.values())), dtype=np.float32,
            )
        else:
            self._scale_arrays[env_idx] = [1.0, 1.0, 1.0]

    def set_scene_params(self, cube_scale: float = 1.0,
                         table_height_delta: float = 0.0,
                         table_friction: float = 0.5,
                         env_idx: int = 0,
                         scene_id: Optional[str] = None,
                         robot_id: Optional[str] = None,
                         profile_id: Optional[int] = None):
        """Store per-environment scene parameters from the instance JSON."""
        self._cube_scale[env_idx] = cube_scale
        self._table_height_delta[env_idx] = table_height_delta
        self._table_friction[env_idx] = table_friction
        if profile_id is not None:
            self._profile_ids[env_idx] = profile_id
        if scene_id is not None:
            self._scene_ids[env_idx] = scene_id
        if robot_id is not None:
            self._robot_ids[env_idx] = robot_id

    def set_table_height_delta(self, table_height_delta: float, env_idx: int = 0) -> None:
        """Override this env's ``table_height_delta`` (e.g. per episode, from the placed table)."""
        self._table_height_delta[env_idx] = table_height_delta

    def set_episode_provenance(
        self,
        env_idx: int,
        source_problem: Optional[str] = None,
        instance_id: Optional[int] = None,
        inactive_objects: Optional[Sequence[str]] = None,
    ) -> None:
        """Per-episode provenance of env ``env_idx``: its problem (``source_problem``), its
        reset-state id within that problem (``instance_id``) and the scene objects outside
        the problem, whose poses are recorded as NaN. ``None`` keeps the current value."""
        if source_problem is not None:
            self._source_problems[env_idx] = source_problem
        if instance_id is not None:
            self._instance_ids[env_idx] = int(instance_id)
        if inactive_objects is not None:
            self._inactive_objects[env_idx] = frozenset(inactive_objects)

    def set_pddl_goal(self, goal: str, env_idx: int = 0):
        """Override the recorded PDDL goal for one environment.

        Play-data collection plans a new goal for every episode, so the goal is
        not a run-wide constant read from the problem file. Falls back to the
        problem file's goal when ``goal`` is empty.
        """
        self._pddl_goals[env_idx] = goal if goal else self._pddl_goal

    def set_glyph_map(self, mapping: dict, env_idx: int = 0):
        """Set this env's block->glyph swap; {} disables rewriting for that env.

        Applied at FRAME BUILD time rather than by mutating the cached strings, because the task
        description is rebuilt from the action on every ``set_..._action`` call and would
        overwrite anything written here.
        """
        self._glyph_maps[env_idx] = dict(mapping or {})

    def set_cuboid_color_map(self, mapping: dict, env_idx: int = 0, palette=None):
        """Set this env's {block: colour name} swap; {} disables rewriting for that env.

        Applied at FRAME BUILD time for the same reason as the glyph map. The palette is stored
        once (it is per run, not per env) so the rgb can be carried in block_variable_map.
        """
        self._cuboid_color_maps[env_idx] = dict(mapping or {})
        if palette is not None:
            self._cuboid_palette = palette

    def set_error(self, error_str: str, env_idx: int = 0):
        """Append an error description to the per-env error log."""
        if self.error_logs[env_idx]:
            self.error_logs[env_idx] += "\n"
        self.error_logs[env_idx] += error_str

        if hasattr(self, '_stdout_capture'):
            stdout_snapshot = self._stdout_capture.get_last_lines()
            if stdout_snapshot:
                self.error_logs[env_idx] += "\n--- STDOUT (last 100 lines) ---\n" + stdout_snapshot

        logger.error(f"[DataRecorder] Error logged (env {env_idx}): {error_str}")

    def mark_action_failed(self, action_type: str = None, error: str = None, env_idx: int = 0, **kwargs):
        """Mark the current action as failed and log the error."""
        error_msg = f"ACTION_FAILED({action_type or 'unknown'}): {error or 'no details'}"
        if kwargs:
            metadata_parts = [f"{k}={v}" for k, v in kwargs.items() if v is not None]
            if metadata_parts:
                error_msg += f" [{', '.join(metadata_parts)}]"
        self.set_error(error_msg, env_idx=env_idx)

    def mark_trajectory_failed(self, error_message: str, failed_node: str = None, failed_action: str = None, env_idx: int = 0):
        """Mark the current trajectory as failed and log the error."""
        parts = [f"TRAJECTORY_FAILED: {error_message}"]
        if failed_node:
            parts.append(f"node={failed_node}")
        if failed_action:
            parts.append(f"action={failed_action}")
        self.set_error(", ".join(parts), env_idx=env_idx)

    # ------------------------------------------------------------------
    # Public API: recording, episodes, and finalization
    # ------------------------------------------------------------------

    def update_symbolic_actions(self,
            hl_action: str, hl_args: List[str],
            ll_action: str, ll_args: Dict[str, Any],
            env_idx: int = 0,
        ):
        """Update the symbolic action context for a specific environment."""
        self.step_counters[env_idx] = 0
        self.hl_actions[env_idx] = hl_action
        self.hl_args_dict[env_idx] = hl_args
        self.ll_actions[env_idx] = ll_action

        hl_state = ll_args.get("current_hl_state")
        if hl_state is not None and hasattr(hl_state, 'getTrueProps'):
            props = sorted(hl_state.getTrueProps())
            self.hl_state_strs[env_idx] = "\n".join(props)

        serialized = {}
        for key, value in ll_args.items():
            if key == "current_hl_state":  # its own column, observation.state.high_level_state
                continue
            serialized[key] = self._convert_to_serializable(value)
        self.ll_args_dict[env_idx] = serialized

        # Pre-serialize JSON for use in every frame — avoids repeated json.dumps
        self._cached_hl_args_json[env_idx] = json.dumps(hl_args) if hl_args else ""
        self._cached_ll_args_json[env_idx] = json.dumps(serialized) if serialized else ""
        task = f"{hl_action}"
        if hl_args:
            task += f" {hl_args}"
        self._cached_task_descriptions[env_idx] = task if task.strip() else "manipulation"

    def update_last_action_success(self, status: str, env_idx: int = 0):
        """Track action success status for a specific environment."""
        pass

    def record_step(self, env_idx: int = 0):
        """Record current frame to the per-environment buffer.

        Delegates to ``record_step_batched`` for a single environment.
        """
        self.record_step_batched([env_idx])

    # ------------------------------------------------------------------
    # Batched recording — async pipeline
    # ------------------------------------------------------------------

    def _prefetch_scene_data(self) -> dict:
        """Extract all sensor data from GPU into numpy with batched transfers.

        Concatenates small robot tensors into one GPU tensor before a single
        ``.cpu()`` call, and converts camera images to uint8 on the GPU to
        avoid large float32 intermediate copies.
        """
        cache: Dict[str, Any] = {}

        robot_name = next(iter(self.scene.articulations.keys()), None)
        cache['robot_name'] = robot_name

        if robot_name:
            art = self.scene.articulations[robot_name]
            d = art.data

            # Batch small robot tensors into one transfer:
            # [joint_pos | joint_vel | joint_acc | applied_torque |
            #  joint_stiffness | joint_damping | joint_effort_limits |
            #  joint_vel_limits | joint_pos_target]
            n_joints = d.joint_pos.shape[-1]
            jpt = d.joint_pos_target if hasattr(d, 'joint_pos_target') else d.joint_pos
            robot_block_gpu = torch.cat([
                d.joint_pos, d.joint_vel, d.joint_acc, d.applied_torque,
                d.joint_stiffness, d.joint_damping, d.joint_effort_limits,
                d.joint_vel_limits,
                jpt,
            ], dim=-1)
            robot_block = robot_block_gpu.cpu().numpy().astype(np.float32)

            # Slice the concatenated block
            off = 0
            cache['joint_pos'] = robot_block[:, off:off+n_joints]; off += n_joints
            cache['joint_vel'] = robot_block[:, off:off+n_joints]; off += n_joints
            cache['joint_acc'] = robot_block[:, off:off+n_joints]; off += n_joints
            cache['applied_torque'] = robot_block[:, off:off+n_joints]; off += n_joints
            cache['joint_stiffness'] = robot_block[:, off:off+n_joints]; off += n_joints
            cache['joint_damping'] = robot_block[:, off:off+n_joints]; off += n_joints
            cache['joint_effort_limits'] = robot_block[:, off:off+n_joints]; off += n_joints
            cache['joint_vel_limits'] = robot_block[:, off:off+n_joints]; off += n_joints
            cache['joint_pos_target'] = robot_block[:, off:off+n_joints]

            # Absolute EE target pose (base frame) via FK on the commanded arm
            # joint targets — mirrors how the joint target is recorded.
            if self._ee_target_fk is not None:
                try:
                    arm_targets = cache['joint_pos_target'][:, :7]
                    cache['ee_target_local'] = np.asarray(
                        self._ee_target_fk(arm_targets), dtype=np.float32,
                    )
                except Exception as e:
                    logger.warn(f"[DataRecorder] EE target FK failed: {e}")

            if self._joint_velocity_limits is None:
                self._joint_velocity_limits = (
                    d.joint_vel_limits[0].cpu().numpy().astype(np.float32).tolist()
                )

            root_state = d.root_state_w[:, :7].cpu().numpy().astype(np.float32)

            # Per-env base frame
            base_pos = root_state[:, :3]
            base_quat = root_state[:, 3:7]
            base_quat_inv = base_quat.copy()
            base_quat_inv[:, 1:] *= -1

            # All robot link poses in the robot-base frame.
            link_poses_w = d.body_link_pose_w[:, :, :7].cpu().numpy().astype(np.float32)
            if link_poses_w.shape[1] != len(self._robot_link_names):
                raise RuntimeError(
                    "robot body pose count does not match the discovered link names"
                )
            link_poses_local = np.empty_like(link_poses_w)
            for link_idx in range(link_poses_w.shape[1]):
                link_poses_local[:, link_idx] = self._world_to_local_np_batched(
                    link_poses_w[:, link_idx],
                    base_pos,
                    base_quat_inv,
                )
            cache['robot_link_poses_local'] = link_poses_local

            # EE pose → local frame
            if hasattr(art, 'body_names') and art.body_names:
                ee_idx = (
                    self.ee_body_idx if self.ee_body_idx is not None
                    else len(art.body_names) - 1
                )
                ee_pose_w = d.body_link_pose_w[:, ee_idx, :].cpu().numpy().astype(np.float32)
                cache['ee_pose_local'] = self._world_to_local_np_batched(
                    ee_pose_w, base_pos, base_quat_inv,
                )

            # Wrench
            if self.wrist_body_idx is not None:
                try:
                    cache['wrenches'] = (
                        d.body_incoming_joint_wrench_b[:, self.wrist_body_idx]
                        .cpu().numpy().astype(np.float32)
                    )
                except Exception:
                    pass

            # Object poses → local frame (batched)
            obj_poses_local: Dict[str, np.ndarray] = {}
            for obj_name in self._object_names:
                if obj_name in self.scene.rigid_objects:
                    obj = self.scene.rigid_objects[obj_name]
                    rpw = obj.data.root_pose_w
                    if rpw.dim() > 1:
                        obj_w = rpw[:, :7].cpu().numpy().astype(np.float32)
                    else:
                        obj_w = rpw[:7].cpu().numpy().astype(np.float32).reshape(1, 7)
                    obj_poses_local[obj_name] = self._world_to_local_np_batched(
                        obj_w, base_pos[:obj_w.shape[0]], base_quat_inv[:obj_w.shape[0]],
                    )
            cache['obj_poses_local'] = obj_poses_local

            # Contact forces → local frame (batched rotation)
            contact_forces_local: Dict[str, np.ndarray] = {}
            for sensor_name in self._contact_sensor_names:
                if sensor_name in self.scene.sensors:
                    try:
                        sensor = self.scene.sensors[sensor_name]
                        forces_w = sensor.data.net_forces_w[:, 0].cpu().numpy().astype(np.float32)
                        contact_forces_local[sensor_name] = self._quat_rotate_vec_np_batched(
                            base_quat_inv[:forces_w.shape[0]], forces_w,
                        )
                    except Exception:
                        pass
            cache['contact_forces_local'] = contact_forces_local

            cache['_base_pos'] = base_pos
            cache['_base_quat_inv'] = base_quat_inv

        # Camera data: converted to their recorded dtypes on the GPU (RGB uint8, depth the uint16
        # millimetres _prepare_depth_image writes) and snapshotted there; the host copy is made by
        # the frame-build thread (_materialize_images), so this thread never waits on it.
        images: Dict[str, torch.Tensor] = {}
        cache['cam_resolution'] = self._get_camera_resolution()
        cam_poses_local: Dict[str, np.ndarray] = {}

        for cam_name in self._camera_names:
            if cam_name not in self.scene.sensors:
                continue
            sensor = self.scene.sensors[cam_name]
            if not (hasattr(sensor, 'data') and hasattr(sensor.data, 'output')):
                continue
            cam_data = sensor.data
            output = cam_data.output

            if 'rgb' in output:
                images[f'cam_rgb_{cam_name}'] = self._rgb_uint8(output['rgb'])

            depth = output.get('depth', output.get('distance_to_image_plane'))
            if depth is not None and self._record_depth:
                images[f'cam_depth_{cam_name}'] = self._depth_mm16(depth)

            try:
                pos_w = cam_data.pos_w.cpu().numpy().astype(np.float32)
                quat_w = cam_data.quat_w_world.cpu().numpy().astype(np.float32)
                cam_pose_w = np.concatenate([pos_w, quat_w], axis=-1)
                if robot_name:
                    bp = cache['_base_pos'][:cam_pose_w.shape[0]]
                    bqi = cache['_base_quat_inv'][:cam_pose_w.shape[0]]
                    cam_poses_local[cam_name] = self._world_to_local_np_batched(
                        cam_pose_w, bp, bqi,
                    )
                else:
                    cam_poses_local[cam_name] = cam_pose_w
                cache[f'cam_intrinsics_{cam_name}'] = (
                    cam_data.intrinsic_matrices.cpu().numpy().astype(np.float32)
                )
            except Exception:
                pass

        for cam_name in self._tactile_camera_names:
            if cam_name not in self.scene.sensors:
                continue
            sensor = self.scene.sensors[cam_name]
            if not (hasattr(sensor, 'data') and hasattr(sensor.data, 'output')):
                continue
            cam_data = sensor.data
            output = cam_data.output
            if 'rgb' in output:
                images[f'cam_rgb_{cam_name}'] = self._rgb_uint8(output['rgb'])
            try:
                pos_w = cam_data.pos_w.cpu().numpy().astype(np.float32)
                quat_w = cam_data.quat_w_world.cpu().numpy().astype(np.float32)
                cam_pose_w = np.concatenate([pos_w, quat_w], axis=-1)
                if robot_name:
                    bp = cache['_base_pos'][:cam_pose_w.shape[0]]
                    bqi = cache['_base_quat_inv'][:cam_pose_w.shape[0]]
                    cam_poses_local[cam_name] = self._world_to_local_np_batched(
                        cam_pose_w, bp, bqi,
                    )
                else:
                    cam_poses_local[cam_name] = cam_pose_w
                cache[f'cam_intrinsics_{cam_name}'] = (
                    cam_data.intrinsic_matrices.cpu().numpy().astype(np.float32)
                )
            except Exception:
                pass

        cache['cam_poses_local'] = cam_poses_local
        if images:
            event = None
            if next(iter(images.values())).is_cuda:
                event = torch.cuda.Event()
                event.record()
            cache['_images'] = (images, event)
        return cache

    @staticmethod
    def _rgb_uint8(rgb: torch.Tensor) -> torch.Tensor:
        """A uint8 RGB snapshot of a camera output (a copy: the sensor reuses its buffer)."""
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        if rgb.dtype == torch.uint8:
            return rgb.clone()
        if rgb.is_floating_point():
            # [0, 1] floats are scaled; decided on the GPU, without a host sync.
            rgb = torch.where(rgb.amax() <= 1.0, rgb * 255, rgb)
        return rgb.to(torch.uint8)

    @staticmethod
    def _depth_mm16(depth: torch.Tensor) -> torch.Tensor:
        """:meth:`_prepare_depth_image` on the GPU, for a batch of ``[N, H, W(, 1)]`` metres."""
        if depth.dim() == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        mm = (torch.nan_to_num(depth.to(torch.float32), nan=0.0) * 1000.0).clamp_(0, 4095)
        return (mm.to(torch.int32) << 4).to(torch.uint16)

    def _materialize_images(self, cache: dict) -> None:
        """Copy the camera snapshot of :meth:`_prefetch_scene_data` to host numpy arrays.

        CUDA tensors are copied on a side stream that waits only for the snapshot, so the copy
        neither waits for nor delays the simulation's later GPU work.
        """
        snapshot = cache.pop('_images', None)
        if snapshot is None:
            return
        images, event = snapshot
        if event is None:
            for key, tensor in images.items():
                cache[key] = tensor.numpy()
            return
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream(device=next(iter(images.values())).device)
        with torch.cuda.stream(self._copy_stream):
            self._copy_stream.wait_event(event)
            for key, tensor in images.items():
                cache[key] = tensor.cpu().numpy()

    def record_step_batched(self, env_indices: List[int], reset_env_ids: Sequence[int] = ()):
        """Record frames for multiple environments with a single GPU prefetch.

        ``reset_env_ids`` are envs that auto-reset inside the step just taken (eval): their
        sample is already the next episode's s_0, so it is held as that episode's first row
        instead of being appended to the episode that just ended. That episode stays active
        until the caller saves or discards it; its last held frame becomes its terminal row.

        Pipeline:
        1. Wait for any previous async frame-build to finish.
        2. Increment step counters; filter to envs due for recording.
        3. Prefetch ALL sensor data once (single batched GPU→CPU transfer).
        4. Snapshot per-env previous state and compute deltas (cheap).
        5. Submit frame building + image spilling to background thread.
        """
        t_call = time.perf_counter()
        try:
            self._record_step_batched(env_indices, reset_env_ids)
        finally:
            self._caller_record_s += time.perf_counter() - t_call
            self._caller_record_calls += 1

    def _record_step_batched(self, env_indices: List[int], reset_env_ids: Sequence[int]):
        # Barrier: ensure previous async build is done so frame_buffers
        # and previous_* arrays are consistent.
        if self._pending_build_future is not None:
            t_wait = time.perf_counter()
            self._pending_build_future.result()
            self._caller_build_wait_s += time.perf_counter() - t_wait
            self._pending_build_future = None

        recording_envs: List[int] = []
        for env_idx in env_indices:
            if env_idx < 0 or env_idx >= self.num_envs:
                continue
            self.step_counters[env_idx] += 1
            if self.step_counters[env_idx] % self.record_every_n_steps == 0:
                recording_envs.append(env_idx)

        if not recording_envs:
            return

        cache = self._prefetch_scene_data()

        reset_envs = {int(i) for i in reset_env_ids} & set(recording_envs)
        # Snapshot per-env "is_first" flags and mark episodes active
        is_first_flags = {}
        for env_idx in recording_envs:
            if env_idx in reset_envs:
                # Becomes the next episode's row 0; the finished episode stays active until the
                # caller saves (or discards) it.
                is_first_flags[env_idx] = True
                continue
            # A held reset frame is this episode's row 0, not this sample.
            is_first = not self.active_episodes[env_idx] and self._reset_frames[env_idx] is None
            is_first_flags[env_idx] = is_first
            self.active_episodes[env_idx] = True
            self._rows_queued[env_idx] += 1

        if self._query_physx_properties:
            starting = [e for e in recording_envs if is_first_flags[e]]
            if starting:
                self._read_physx_properties(starting)

        # Snapshot previous states needed for delta computations,
        # then update them eagerly so the next call sees fresh values.
        prev_joint_snap = self.previous_joint_states[recording_envs].copy()
        prev_ee_snap = {i: dict(self.previous_ee_poses[i]) for i in recording_envs}

        robot_name = cache.get('robot_name')
        if robot_name:
            for ri, env_idx in enumerate(recording_envs):
                jp = cache['joint_pos'][env_idx][:7]
                jpt = cache['joint_pos_target'][env_idx][:7]
                ee = cache.get('ee_pose_local', np.zeros((self.num_envs, 7), dtype=np.float32))[env_idx]
                self.previous_joint_states[env_idx] = jp.copy()
                self.previous_joint_targets[env_idx] = jpt.copy()
                self.previous_ee_poses[env_idx][robot_name] = ee.copy()
                if env_idx in reset_envs:
                    self._reset_prev[env_idx] = (
                        jp.copy(), jpt.copy(), {robot_name: ee.copy()},
                    )

        # Consume the staged extras (move-from): they describe the command of the step just
        # taken, so they ride on this call's action block like the latched joint target.
        extras_snap = {
            env_idx: (
                self._pending_actions[env_idx],
                self._pending_pd_modes[env_idx],
                (
                    self._hl_model_outputs[env_idx], self._hl_prev_actions[env_idx],
                    self._hl_input_prompts[env_idx], self._hl_dones[env_idx],
                ),
                self._hl_overlays[env_idx],
            )
            for env_idx in recording_envs
        }
        for env_idx in recording_envs:
            self._pending_actions[env_idx] = None
            self._pending_pd_modes[env_idx] = None

        # Submit frame building to background thread
        self._pending_build_future = self._frame_build_pool.submit(
            self._build_and_buffer_frames,
            cache, recording_envs, is_first_flags,
            prev_joint_snap, prev_ee_snap, extras_snap, frozenset(reset_envs),
        )

    def capture_reset_frames(self, env_indices: Sequence[int]) -> None:
        """Sample the current scene as each env's next-episode s_0 (call after ``env.reset()``).

        The frame becomes row 0 on that env's next record call; a later call overwrites it.
        """
        if self._pending_build_future is not None:
            self._pending_build_future.result()
            self._pending_build_future = None
        envs = [int(i) for i in env_indices if 0 <= int(i) < self.num_envs]
        if not envs:
            return
        cache = self._prefetch_scene_data()
        self._materialize_images(cache)
        if self._query_physx_properties:
            self._read_physx_properties(envs)
        robot_name = cache.get('robot_name')
        rows = (
            self._precompute_frame_rows(
                cache, envs, np.zeros((len(envs), 7), dtype=np.float32),
                {e: {} for e in envs}, robot_name,
            )
            if robot_name else None
        )
        for ri, env_idx in enumerate(envs):
            frame = self._build_frame_cached(
                env_idx, cache, is_first=True, is_last=False, is_successful=False,
                prev_joint_state=np.zeros(7, dtype=np.float32), prev_ee_poses={},
                robot_name_override=robot_name,
                precomputed=(rows, ri) if rows is not None else None,
            )
            self._apply_frame_extras(frame, env_idx, (None, None, ("", "", "", False)))
            frame["is_episode_successful"] = np.array([False], dtype=np.bool_)
            self._reset_frames[env_idx] = frame
            if robot_name:
                jp = cache['joint_pos'][env_idx][:7].copy()
                jpt = cache['joint_pos_target'][env_idx][:7].copy()
                ee = cache.get('ee_pose_local', np.zeros((self.num_envs, 7), dtype=np.float32))[env_idx]
                self._reset_prev[env_idx] = (jp, jpt, {robot_name: ee.copy()})
                self.previous_joint_states[env_idx] = jp
                self.previous_joint_targets[env_idx] = jpt
                self.previous_ee_poses[env_idx] = {robot_name: ee.copy()}

    def _apply_frame_extras(self, frame: Dict[str, Any], env_idx: int, extras: tuple) -> None:
        """Write the staged per-step extras (policy action, pd_mode override, HL output, HL overlay)."""
        action, pd_mode, hl = extras[:3]
        if self._hl_overlay_camera is not None:
            # Drawn in _emit_frame, once it has moved onto the row whose action it describes.
            frame["_hl_overlay"] = extras[3] if len(extras) > 3 else None
        if self._action_dim is not None:
            policy = np.zeros(self._action_dim, dtype=np.float32)
            if action is not None:
                arr = np.asarray(action, dtype=np.float32).reshape(-1)
                n = min(arr.shape[0], self._action_dim)
                policy[:n] = arr[:n]
            frame["action.policy"] = policy
        if pd_mode is not None and np.isfinite(pd_mode):
            frame["action.pd_mode"] = np.array([pd_mode], dtype=np.float32)
            if "action" in frame:
                act = np.array(frame["action"], dtype=np.float32, copy=True)
                act[-1] = pd_mode
                frame["action"] = act
        if self.record_hl_model:
            for key, value in zip(self._HL_MODEL_STRING_FIELDS, hl[:3]):
                frame[key] = str(value)
            frame["action.hl_done"] = np.array([bool(hl[3])], dtype=np.bool_)

    def _build_and_buffer_frames(
        self,
        cache: dict,
        recording_envs: List[int],
        is_first_flags: Dict[int, bool],
        prev_joint_snap: np.ndarray,
        prev_ee_snap: Dict[int, Dict[str, np.ndarray]],
        extras_snap: Optional[Dict[int, tuple]] = None,
        reset_envs: frozenset = frozenset(),
    ):
        """Build frames, send images to video subprocess, buffer non-image data."""
        self._materialize_images(cache)
        robot_name = cache.get('robot_name')
        rows = (
            self._precompute_frame_rows(cache, recording_envs, prev_joint_snap, prev_ee_snap, robot_name)
            if robot_name else None
        )

        for ri, env_idx in enumerate(recording_envs):
            frame = self._build_frame_cached(
                env_idx, cache,
                is_first=is_first_flags[env_idx], is_last=False,
                is_successful=bool(self.success_flags[env_idx]),
                prev_joint_state=prev_joint_snap[ri],
                prev_ee_poses=prev_ee_snap[env_idx],
                robot_name_override=robot_name,
                precomputed=(rows, ri) if rows is not None else None,
            )
            if extras_snap is not None:
                self._apply_frame_extras(frame, env_idx, extras_snap[env_idx])

            if env_idx in reset_envs:
                # The next episode's s_0. The command this sample carries drove the finished
                # episode into a state the reset discarded, so that episode's held frame stays
                # its terminal row (released when the caller saves it).
                frame["is_episode_successful"] = np.array([False], dtype=np.bool_)
                self._reset_frames[env_idx] = frame
                continue

            reset_frame = self._reset_frames[env_idx]
            if reset_frame is not None:
                self._reset_frames[env_idx] = None
                self._reset_prev[env_idx] = None
                if self._pending_frames[env_idx] is None:
                    self._pending_frames[env_idx] = reset_frame

            # Hold each frame back one call so this call's action block can be
            # moved onto the previous call's observation.  The held frame is
            # released by _flush_pending_frame at episode end.
            pending = self._pending_frames[env_idx]
            self._pending_frames[env_idx] = frame
            if pending is None:
                continue
            self._transfer_action_fields(src=frame, dst=pending)
            self._emit_frame(env_idx, pending)

    def _read_physx_properties(self, env_ids: List[int]) -> None:
        """Live block/table materials and masses of ``env_ids`` (query_physx_properties).

        Read once per episode -- PhysX view queries are not free, and the properties only change
        when a reset writes them.
        """
        import contextlib
        block_done = table_done = False
        for obj_name in self._object_names:
            obj = self.scene.rigid_objects.get(obj_name) if hasattr(self.scene, "rigid_objects") else None
            if obj is None:
                continue
            is_table = obj_name in _TABLE_OBJECT_NAMES
            if (is_table and table_done) or (not is_table and block_done):
                continue
            with contextlib.suppress(Exception):
                masses = obj.root_physx_view.get_masses()
                masses = np.asarray(
                    masses.detach().cpu().numpy() if isinstance(masses, torch.Tensor) else masses,
                    dtype=np.float32,
                ).reshape(-1 if np.ndim(masses) == 0 else len(masses), -1)
                target = self._table_masses if is_table else self._object_masses
                for e in env_ids:
                    if e < masses.shape[0]:
                        target[e] = float(masses[e, 0])
            with contextlib.suppress(Exception):
                mats = obj.root_physx_view.get_material_properties()
                mats = np.asarray(
                    mats.detach().cpu().numpy() if isinstance(mats, torch.Tensor) else mats,
                    dtype=np.float32,
                )
                if mats.ndim == 3 and mats.shape[-1] >= 3:
                    for e in env_ids:
                        if e >= mats.shape[0]:
                            continue
                        if is_table:
                            self._table_friction[e] = float(mats[e, 0, 0])
                        else:
                            self._friction_arrays[e] = mats[e, 0, :3]
            if is_table:
                table_done = True
            else:
                block_done = True

    def _emit_frame(self, env_idx: int, frame: Dict[str, Any]) -> None:
        """Split images off *frame* and hand it to the encoder or the local buffer."""
        if not self._episode_started[env_idx]:
            self._queue_put(("start_episode", env_idx, self._cam_configs))
            self._episode_started[env_idx] = True

        overlay = frame.pop("_hl_overlay", None)
        if self._hl_overlay_camera is not None:
            frame[f"observation.images.{self._hl_overlay_camera}_hl_overlay"] = (
                self._draw_hl_overlay(frame[self._hl_overlay_src], overlay)
            )

        image_dict: Dict[str, np.ndarray] = {}
        for key in list(frame.keys()):
            val = frame[key]
            if self._is_image_key(key) and isinstance(val, np.ndarray) and val.ndim >= 2:
                image_dict[key] = val
                if not (self._goal_annotator and key == "observation.images.front_camera"):
                    del frame[key]

        if self._goal_annotator:
            if image_dict:
                self._queue_put(("frame", env_idx, image_dict))
            self.frame_buffers[env_idx].append(frame)
        else:
            if image_dict:
                image_dict = self._shm_put(self._writer_for(env_idx), image_dict) or image_dict
            self._queue_put(("frame", env_idx, image_dict, frame))
        self._frame_counts[env_idx] += 1

    def _draw_hl_overlay(self, img: np.ndarray, overlay: Optional[tuple]) -> np.ndarray:
        """The HL overlay frame: heatmap + HL text (and Molmo2 prev action / prompt) on *img*."""
        from blocksuite.utils.visualization_utils import (
            overlay_heatmap_and_text,
            overlay_hl_input_debug,
        )

        text, heatmap, prev_action, prompt, oracle_text, oracle_heatmap = overlay or (None,) * 6
        if prev_action is None and prompt is None:
            left = overlay_heatmap_and_text(img, heatmap, text)
        else:
            left = overlay_hl_input_debug(
                overlay_heatmap_and_text(img, heatmap),
                prev_action=prev_action or "", prompt=prompt or "", hl_output=text or "",
            )
        if not self._hl_overlay_side_by_side:
            return left
        right = overlay_heatmap_and_text(img, oracle_heatmap, oracle_text)
        return np.concatenate([left, right], axis=1)

    def _shm_put(self, writer: "_Writer", images: Dict[str, np.ndarray]) -> Optional[Dict[str, Any]]:
        """Images of one frame into *writer*'s shared-memory arena; their descriptor, or None.

        The arena is made on the writer's first frame, sized to half its RAM budget (the queued
        path keeps the other half) and to at most half of the free /dev/shm.
        """
        if writer.arena is None:
            if writer.arena_failed:
                return None
            from blocksuite.vecgen.recording._shm_slots import SlotArena, frame_image_bytes, shm_free_bytes

            slot_bytes = frame_image_bytes(images)
            n_slots = min(writer.cap // 2, shm_free_bytes() // 2) // max(1, slot_bytes)
            if n_slots < 4:
                writer.arena_failed = True
                print(f"[recorder] writer {writer.index}: /dev/shm too small for a frame arena "
                      f"({shm_free_bytes() / 2**20:.0f} MiB free); frames go through the queue", flush=True)
                return None
            try:
                writer.arena = SlotArena(slot_bytes, n_slots)
            except OSError as e:
                writer.arena_failed = True
                print(f"[recorder] writer {writer.index}: no shared-memory arena ({e})", flush=True)
                return None
            writer.cap = max(1, writer.cap - n_slots * slot_bytes)
            self._writer_put(writer, ("attach_shm", writer.arena.name, n_slots))
        return writer.arena.put(images)

    def _flush_pending_frame(self, env_idx: int) -> None:
        """Release the held frame as the episode's terminal row.

        No further command follows it, so its action block is blanked and
        ``action.is_valid`` set False.  ``is_last`` is stamped downstream, by the
        encoder subprocess for the streaming path and by
        ``_save_episode_with_goal_annotation`` for the annotated path.
        """
        pending = self._pending_frames[env_idx]
        if pending is None:
            return
        self._pending_frames[env_idx] = None
        self._clear_action_fields(pending)
        self._emit_frame(env_idx, pending)

    # ------------------------------------------------------------------
    # Shared frame-building helpers
    # ------------------------------------------------------------------

    _ZERO_FILL_ROBOT = {
        "observation.state.joint_position": (7,),
        "observation.state.joint_velocity": (7,),
        "observation.state.joint_acceleration": (7,),
        "observation.state.joint_effort": (7,),
        "observation.state.gripper_position": (1,),
        "observation.state": (8,),
        "observation.state.ee_position": (6,),
        "observation.state.wrench": (6,),
        "observation.state.current_stiffness": (13,),
        "observation.state.current_damping": (13,),
        "observation.state.current_effort_limit": (13,),
        "observation.state.current_velocity_limit": (13,),
        "observation.state.current_torque": (13,),
        "action.joint_position": (7,),
        "action.joint_target": (7,),
        "action.relative_joint_target": (7,),
        "action.ee_target": (7,),
        "action.relative_ee_target": (7,),
        "action.gripper_position": (1,),
        "action.gripper_raw": (1,),
        "action.ee_position": (6,),
        "action.pd_mode": (1,),
        "action": (9,),
    }

    @staticmethod
    def _zero_fill_robot_data(frame: dict):
        """Write zero-valued robot observation and action fields into *frame*."""
        for key, shape in IsaacDataRecorderLeRobot._ZERO_FILL_ROBOT.items():
            frame[key] = np.zeros(shape, dtype=np.float32)

    # ------------------------------------------------------------------
    # State/action alignment
    # ------------------------------------------------------------------
    #
    # Every field below is sampled *after* a physics step, so on record call k
    # it describes the command that drove s_{k-1} -> s_k.  Emission is deferred
    # one call so these land on the row holding s_{k-1}, giving each row the
    # observed state paired with the action taken *in* that state.

    _ACTION_NUMERIC_FIELDS: Dict[str, Tuple[int, ...]] = {
        "action": (9,),
        "action.joint_target": (7,),
        "action.relative_joint_target": (7,),
        "action.ee_target": (7,),
        "action.relative_ee_target": (7,),
        "action.joint_position": (7,),
        "action.ee_position": (6,),
        "action.gripper_position": (1,),
        "action.gripper_raw": (1,),
        "action.pd_mode": (1,),
    }

    _ACTION_STRING_FIELDS: Tuple[str, ...] = (
        "action.high_level_action",
        "action.high_level_args",
        "action.low_level_action",
        "action.low_level_args",
    )

    # Gain columns are named ``observation.*`` but describe the profile latched
    # for the commanded transition, and ``action.pd_mode`` is derived from
    # stiffness.  They move with the action block so pd_mode and the gains it
    # was computed from never end up on different rows.
    _ACTION_ALIGNED_OBSERVATION_FIELDS: Tuple[str, ...] = (
        "observation.state.current_stiffness",
        "observation.state.current_damping",
        "observation.state.current_effort_limit",
        "observation.state.current_velocity_limit",
    )

    # The learned HL's output for the step (record_hl_model); action-aligned like the rest.
    _HL_MODEL_STRING_FIELDS: Tuple[str, ...] = (
        "action.hl_model_output",
        "action.hl_prev_action",
        "action.hl_input_prompt",
    )

    def _extra_action_keys(self) -> Tuple[str, ...]:
        """Action-aligned columns this recorder adds beyond the datagen schema."""
        keys: Tuple[str, ...] = ()
        if self._action_dim is not None:
            keys += ("action.policy",)
        if self.record_hl_model:
            keys += self._HL_MODEL_STRING_FIELDS + ("action.hl_done",)
        if self._hl_overlay_camera is not None:
            keys += ("_hl_overlay",)
        return keys

    def _transfer_action_fields(self, src: dict, dst: dict) -> None:
        """Move the action block of *src* onto *dst*, leaving observations alone."""
        keys = (
            tuple(self._ACTION_NUMERIC_FIELDS)
            + self._ACTION_STRING_FIELDS
            + self._ACTION_ALIGNED_OBSERVATION_FIELDS
            + self._extra_action_keys()
        )
        for key in keys:
            if key in src:
                dst[key] = src[key]

    def _clear_action_fields(self, frame: dict) -> None:
        """Blank the action block of a terminal frame — no action was taken in it.

        Gain columns keep their sampled values: they still describe the robot at
        the recorded state, and there is no following command to source them from.
        """
        for key, shape in self._ACTION_NUMERIC_FIELDS.items():
            frame[key] = np.zeros(shape, dtype=np.float32)
        for key in self._ACTION_STRING_FIELDS:
            frame[key] = ""
        if self._action_dim is not None:
            frame["action.policy"] = np.zeros(self._action_dim, dtype=np.float32)
        if self.record_hl_model:
            for key in self._HL_MODEL_STRING_FIELDS:
                frame[key] = ""
            frame["action.hl_done"] = np.array([False], dtype=np.bool_)
        frame["action.is_valid"] = np.array([False], dtype=np.bool_)

    def _populate_frame_metadata(
        self, frame: dict, env_idx: int, is_first: bool, is_last: bool,
    ):
        """Fill shared per-frame metadata using pre-cached strings."""
        frame["observation.state.friction_properties"] = self._friction_arrays[env_idx].copy()
        frame["observation.state.scale_properties"] = self._scale_arrays[env_idx].copy()
        frame["observation.state.cube_scale"] = np.array([self._cube_scale[env_idx]], dtype=np.float32)
        frame["observation.state.table_height_delta"] = np.array([self._table_height_delta[env_idx]], dtype=np.float32)
        frame["observation.state.table_friction"] = np.array([self._table_friction[env_idx]], dtype=np.float32)
        frame["observation.state.profile_id"] = np.array([self._profile_ids[env_idx]], dtype=np.int64)
        if self._query_physx_properties:
            frame["observation.state.object_mass"] = np.array([self._object_masses[env_idx]], dtype=np.float32)
            frame["observation.state.table_mass"] = np.array([self._table_masses[env_idx]], dtype=np.float32)
        frame["scene_id"] = self._scene_ids[env_idx]
        frame["robot_id"] = self._robot_ids[env_idx]

        frame["action.high_level_action"] = self.hl_actions[env_idx]
        frame["action.high_level_args"] = self._cached_hl_args_json[env_idx]
        frame["action.low_level_action"] = self.ll_actions[env_idx]
        frame["action.low_level_args"] = self._cached_ll_args_json[env_idx]
        frame["pddl_goal"] = self._pddl_goals[env_idx]
        frame["observation.state.high_level_state"] = self.hl_state_strs[env_idx]

        frame["is_first"] = np.array([is_first], dtype=np.bool_)
        frame["is_last"] = np.array([is_last], dtype=np.bool_)
        # Cleared by _clear_action_fields on the terminal row.
        frame["action.is_valid"] = np.array([True], dtype=np.bool_)
        frame["is_episode_successful"] = np.array(
            [bool(self.success_flags[env_idx])], dtype=np.bool_,
        )
        frame["error"] = self.error_logs[env_idx]
        _gm = self._glyph_maps[env_idx]
        if _gm:
            from blocksuite.domain_randomization.glyphs import glyph_map_json, rewrite_tokens
            frame["pddl_goal"] = rewrite_tokens(frame["pddl_goal"], _gm)
            frame["task"] = rewrite_tokens(self._cached_task_descriptions[env_idx], _gm)
            frame["glyph_map"] = glyph_map_json(_gm)
        else:
            frame["task"] = self._cached_task_descriptions[env_idx]
            frame["glyph_map"] = ""
        # Cuboids, after glyphs. The two rewrites target DISJOINT names (glyphs own type=="usd"
        # blocks, colours own type=="cuboid"), so order does not matter -- but each must run even
        # when the other is off, which is why this is its own block and not an elif.
        _cm = self._cuboid_color_maps[env_idx]
        if _cm:
            from blocksuite.domain_randomization.block_colors import rewrite_tokens as _rw_colors
            frame["pddl_goal"] = _rw_colors(frame["pddl_goal"], _cm)
            frame["task"] = _rw_colors(frame["task"], _cm)
        # One unified lookup for BOTH families, written whenever either randomizer ran.
        if _gm or _cm:
            from blocksuite.domain_randomization.block_variables import block_variable_map_json
            frame["block_variable_map"] = block_variable_map_json(
                _gm or None, _cm or None, self._cuboid_palette)
        else:
            frame["block_variable_map"] = ""
        frame[SOURCE_PROBLEM_KEY] = self._source_problems[env_idx]
        frame[INSTANCE_ID_KEY] = np.array([int(self._instance_ids[env_idx])], dtype=np.int64)

        if self._goal_annotator:
            self._goal_annotator.initialize_fields(frame)
            self._goal_annotator.populate_frame_fields(frame, env_idx)

    # ------------------------------------------------------------------
    # Frame builder (cached / batched path)
    # ------------------------------------------------------------------

    @staticmethod
    def _quaternion_difference_batched(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Row-wise ``q2 * inv(q1)`` for ``(N, 4)`` [w, x, y, z] arrays (same op order as
        :meth:`_quaternion_difference`, so the float32 results match it)."""
        q1w, q1x, q1y, q1z = q1[:, 0], -q1[:, 1], -q1[:, 2], -q1[:, 3]
        w = q2[:, 0] * q1w - q2[:, 1] * q1x - q2[:, 2] * q1y - q2[:, 3] * q1z
        x = q2[:, 0] * q1x + q2[:, 1] * q1w + q2[:, 2] * q1z - q2[:, 3] * q1y
        y = q2[:, 0] * q1y - q2[:, 1] * q1z + q2[:, 2] * q1w + q2[:, 3] * q1x
        z = q2[:, 0] * q1z + q2[:, 1] * q1y - q2[:, 2] * q1x + q2[:, 3] * q1w
        return np.stack([w, x, y, z], axis=-1).astype(np.float32)

    def _precompute_frame_rows(
        self,
        cache: dict,
        recording_envs: List[int],
        prev_joint_snap: np.ndarray,
        prev_ee_snap: Dict[int, Dict[str, np.ndarray]],
        robot_name: str,
    ) -> Dict[str, np.ndarray]:
        """Robot-state and action columns for all ``recording_envs`` at once.

        Row ``ri`` holds env ``recording_envs[ri]``; :meth:`_build_frame_cached` slices
        these instead of redoing the per-env quaternion / delta math in Python.  The
        arithmetic mirrors the scalar path (float64 euler conversion, float32 deltas).
        """
        n = len(recording_envs)
        idx = np.asarray(recording_envs, dtype=np.int64)
        jp = cache['joint_pos'][idx]
        jpt = cache['joint_pos_target'][idx]
        ee = cache.get('ee_pose_local', np.zeros((self.num_envs, 7), dtype=np.float32))[idx]
        if ee.shape[1] < 7:
            ee = np.concatenate(
                [ee, np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (n, 1))], axis=1
            )
        rows: Dict[str, np.ndarray] = {}
        rows['ee_euler'] = self._quaternion_to_euler_batched(ee[:, 3:7].astype(np.float64))

        has_prev_js = prev_joint_snap.any(axis=1)
        jpos7, jtgt7 = jp[:, :7], jpt[:, :7]
        rows['joint_action_delta'] = np.where(
            has_prev_js[:, None], jpos7 - prev_joint_snap, 0.0
        ).astype(np.float32)
        rows['relative_jt'] = np.where(
            has_prev_js[:, None], jtgt7 - prev_joint_snap, jtgt7 - jpos7
        ).astype(np.float32)

        prev_ee = np.zeros((n, 7), dtype=np.float32)
        has_prev_ee = np.zeros(n, dtype=np.bool_)
        for ri, env_idx in enumerate(recording_envs):
            p = prev_ee_snap[env_idx].get(robot_name)
            if p is not None:
                prev_ee[ri] = p[:7]
                has_prev_ee[ri] = True
        delta_pos = ee[:, :3] - prev_ee[:, :3]
        delta_quat = self._quaternion_difference_batched(prev_ee[:, 3:7], ee[:, 3:7])
        delta_quat[~has_prev_ee] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        delta_pos[~has_prev_ee] = 0.0
        rows['action_ee_position'] = np.concatenate(
            [delta_pos, self._quaternion_to_euler_batched(delta_quat.astype(np.float64))], axis=1
        ).astype(np.float32)

        ee_target_cache = cache.get('ee_target_local')
        if ee_target_cache is not None:
            valid = idx < ee_target_cache.shape[0]
            tgt = np.zeros((n, 7), dtype=np.float32)
            tgt[valid] = ee_target_cache[idx[valid]].astype(np.float32)
            base_pose = np.where(has_prev_ee[:, None], prev_ee, ee)
            rel_pos = tgt[:, :3] - base_pose[:, :3]
            rel_quat = self._quaternion_difference_batched(base_pose[:, 3:7], tgt[:, 3:7])
            neg = rel_quat[:, 0] < 0
            rel_quat[neg] = -rel_quat[neg]
            rel = np.concatenate([rel_pos, rel_quat], axis=1).astype(np.float32)
            tgt[~valid] = 0.0
            rel[~valid] = 0.0
            rows['ee_target'] = tgt
            rows['relative_ee_target'] = rel
        else:
            rows['ee_target'] = np.zeros((n, 7), dtype=np.float32)
            rows['relative_ee_target'] = np.zeros((n, 7), dtype=np.float32)
        return rows

    def _build_frame_cached(
        self, env_idx: int, cache: dict,
        is_first: bool = False, is_last: bool = False, is_successful: bool = True,
        prev_joint_state: Optional[np.ndarray] = None,
        prev_ee_poses: Optional[Dict[str, np.ndarray]] = None,
        robot_name_override: Optional[str] = None,
        precomputed: Optional[Tuple[Dict[str, np.ndarray], int]] = None,
    ) -> Dict[str, Any]:
        """Build a frame dict using prefetched numpy data (zero GPU operations).

        ``precomputed`` is ``(rows, ri)`` from :meth:`_precompute_frame_rows`: the derived
        robot/action columns of this env are then taken from row ``ri`` instead of being
        recomputed here.
        """
        frame: Dict[str, Any] = {}
        robot_name = robot_name_override if robot_name_override is not None else cache['robot_name']

        if robot_name and precomputed is not None:
            rows, ri = precomputed
            joint_pos = cache['joint_pos'][env_idx]
            joint_pos_target = cache['joint_pos_target'][env_idx]
            ee_pose = cache.get(
                'ee_pose_local', np.zeros((self.num_envs, 7), dtype=np.float32),
            )[env_idx]
            applied_torque = cache['applied_torque'][env_idx]
            frame["observation.state.joint_position"] = joint_pos[:7].copy()
            frame["observation.state.joint_velocity"] = cache['joint_vel'][env_idx][:7].copy()
            frame["observation.state.joint_acceleration"] = cache['joint_acc'][env_idx][:7].copy()
            frame["observation.state.joint_effort"] = applied_torque[:7].copy()
            gripper_val = float(joint_pos[7]) if len(joint_pos) > 7 else 0.0
            frame["observation.state.gripper_position"] = np.array([gripper_val], dtype=np.float32)
            state_vals = np.empty(8, dtype=np.float32)
            state_vals[:7] = joint_pos[:7]
            state_vals[7] = gripper_val
            frame["observation.state"] = state_vals
            frame["observation.state.ee_position"] = np.concatenate(
                [ee_pose[:3], rows['ee_euler'][ri]],
            ).astype(np.float32)
            frame["observation.state.current_torque"] = applied_torque.copy()
            wrenches = cache.get('wrenches')
            frame["observation.state.wrench"] = (
                wrenches[env_idx].copy() if wrenches is not None
                else np.zeros(6, dtype=np.float32)
            )
            stiffness = cache['joint_stiffness'][env_idx]
            frame["observation.state.current_stiffness"] = stiffness.copy()
            frame["observation.state.current_damping"] = cache['joint_damping'][env_idx].copy()
            frame["observation.state.current_effort_limit"] = cache['joint_effort_limits'][env_idx].copy()
            frame["observation.state.current_velocity_limit"] = cache['joint_vel_limits'][env_idx].copy()
            frame["action.pd_mode"] = np.array(
                [pd_mode_for_stiffness(float(stiffness[0]))], dtype=np.float32,
            )
            frame["action.joint_position"] = rows['joint_action_delta'][ri].copy()
            frame["action.joint_target"] = joint_pos_target[:7].copy()
            frame["action.relative_joint_target"] = rows['relative_jt'][ri].copy()
            raw_gripper_target = float(joint_pos_target[7]) if len(joint_pos_target) > 7 else 0.0
            gripper_action_val = 1.0 if raw_gripper_target > self._gripper_binarize_threshold else 0.0
            frame["action.gripper_raw"] = np.array([raw_gripper_target], dtype=np.float32)
            frame["action.gripper_position"] = np.array([gripper_action_val], dtype=np.float32)
            frame["action.ee_position"] = rows['action_ee_position'][ri].copy()
            frame["action.ee_target"] = rows['ee_target'][ri].copy()
            frame["action.relative_ee_target"] = rows['relative_ee_target'][ri].copy()
            frame["action"] = np.concatenate([
                frame["action.relative_joint_target"],
                frame["action.gripper_position"],
                frame["action.pd_mode"],
            ])
        elif robot_name:
            joint_pos = cache['joint_pos'][env_idx]
            joint_pos_target = cache['joint_pos_target'][env_idx]
            ee_pose = cache.get(
                'ee_pose_local', np.zeros((self.num_envs, 7), dtype=np.float32),
            )[env_idx]

            frame["observation.state.joint_position"] = joint_pos[:7].copy()

            joint_vel = cache['joint_vel'][env_idx]
            frame["observation.state.joint_velocity"] = joint_vel[:7].copy()

            joint_acc = cache['joint_acc'][env_idx]
            frame["observation.state.joint_acceleration"] = joint_acc[:7].copy()

            applied_torque = cache['applied_torque'][env_idx]
            frame["observation.state.joint_effort"] = applied_torque[:7].copy()

            gripper_val = float(joint_pos[7]) if len(joint_pos) > 7 else 0.0
            frame["observation.state.gripper_position"] = np.array(
                [gripper_val], dtype=np.float32,
            )

            state_vals = np.empty(8, dtype=np.float32)
            state_vals[:7] = joint_pos[:7]
            state_vals[7] = gripper_val
            frame["observation.state"] = state_vals

            ee_pos = ee_pose[:3]
            ee_quat = ee_pose[3:7] if len(ee_pose) >= 7 else np.array(
                [1.0, 0.0, 0.0, 0.0], dtype=np.float32,
            )
            ee_euler = self._quaternion_to_euler(ee_quat)
            frame["observation.state.ee_position"] = np.concatenate(
                [ee_pos, ee_euler],
            ).astype(np.float32)

            frame["observation.state.current_torque"] = applied_torque.copy()
            wrenches = cache.get('wrenches')
            frame["observation.state.wrench"] = (
                wrenches[env_idx].copy() if wrenches is not None
                else np.zeros(6, dtype=np.float32)
            )
            frame["observation.state.current_stiffness"] = cache['joint_stiffness'][env_idx].copy()
            frame["observation.state.current_damping"] = cache['joint_damping'][env_idx].copy()
            frame["observation.state.current_effort_limit"] = cache['joint_effort_limits'][env_idx].copy()
            frame["observation.state.current_velocity_limit"] = cache['joint_vel_limits'][env_idx].copy()

            stiffness = cache['joint_stiffness'][env_idx]
            frame["action.pd_mode"] = np.array(
                [pd_mode_for_stiffness(float(stiffness[0]))], dtype=np.float32,
            )

            joint_action_target = joint_pos_target[:7].copy()
            joint_position_abs = joint_pos[:7].copy()

            # Delta computations using snapshotted previous state
            prev_js = prev_joint_state if prev_joint_state is not None else self.previous_joint_states[env_idx]
            if prev_js.any():
                joint_action_delta = (joint_position_abs - prev_js).astype(np.float32)
                relative_jt = (joint_action_target - prev_js).astype(np.float32)
            else:
                joint_action_delta = np.zeros(7, dtype=np.float32)
                relative_jt = (joint_action_target - joint_position_abs).astype(np.float32)

            frame["action.joint_position"] = joint_action_delta
            frame["action.joint_target"] = joint_action_target
            frame["action.relative_joint_target"] = relative_jt

            raw_gripper_target = float(joint_pos_target[7]) if len(joint_pos_target) > 7 else 0.0
            thr = self._gripper_binarize_threshold
            gripper_action_val = 1.0 if raw_gripper_target > thr else 0.0
            frame["action.gripper_raw"] = np.array(
                [raw_gripper_target], dtype=np.float32,
            )
            frame["action.gripper_position"] = np.array(
                [gripper_action_val], dtype=np.float32,
            )

            prev_ee = prev_ee_poses if prev_ee_poses is not None else self.previous_ee_poses[env_idx]
            if robot_name in prev_ee:
                prev_pose = prev_ee[robot_name]
                pos_delta = ee_pose[:3] - prev_pose[:3]
                quat_delta = self._quaternion_difference(prev_pose[3:7], ee_pose[3:7])
                delta_ee = np.concatenate([pos_delta, quat_delta])
            else:
                delta_ee = np.zeros(7, dtype=np.float32)

            ee_pos_action = delta_ee[:3]
            ee_quat_action = delta_ee[3:7] if len(delta_ee) >= 7 else np.array(
                [1.0, 0.0, 0.0, 0.0], dtype=np.float32,
            )
            ee_euler_action = self._quaternion_to_euler(ee_quat_action)
            frame["action.ee_position"] = np.concatenate(
                [ee_pos_action, ee_euler_action],
            ).astype(np.float32)

            # Absolute and relative EE target — stored the same way as the
            # joint target. Absolute is FK(joint_target) in the base frame;
            # relative is that target expressed w.r.t. the previous EE pose
            # (or the current EE pose on the first frame), mirroring how
            # action.relative_joint_target is computed against joint position.
            # After the one-call emission deferral, row t therefore holds
            # s_t alongside ee_target(a_{t+1}) and relative_ee_target =
            # [p_target - p_t, q_target * inv(q_t)] with q_rel.w >= 0, so the
            # absolute target is recovered as p_t + dp and q_rel * q_t.
            ee_target_cache = cache.get('ee_target_local')
            if ee_target_cache is not None and env_idx < ee_target_cache.shape[0]:
                ee_target_pose = ee_target_cache[env_idx].astype(np.float32).copy()
                base_pose = prev_ee[robot_name] if robot_name in prev_ee else ee_pose
                rel_pos = ee_target_pose[:3] - base_pose[:3]
                rel_quat = self._quaternion_difference(base_pose[3:7], ee_target_pose[3:7])
                # Canonicalize to the shortest arc (w >= 0): the observed pose
                # (Isaac) and the target (CuRobo FK) do not share a quaternion
                # sign convention, so q and -q would otherwise alternate.
                if rel_quat[0] < 0:
                    rel_quat = -rel_quat
                relative_ee_target = np.concatenate([rel_pos, rel_quat]).astype(np.float32)
            else:
                ee_target_pose = np.zeros(7, dtype=np.float32)
                relative_ee_target = np.zeros(7, dtype=np.float32)
            frame["action.ee_target"] = ee_target_pose
            frame["action.relative_ee_target"] = relative_ee_target

            frame["action"] = np.concatenate([
                frame["action.relative_joint_target"],
                frame["action.gripper_position"],
                frame["action.pd_mode"],
            ])
        else:
            self._zero_fill_robot_data(frame)

        # Robot link poses (already transformed to the robot-base frame).
        link_poses = cache.get('robot_link_poses_local')
        for link_idx, link_name in enumerate(self._robot_link_names):
            if (
                link_poses is not None
                and env_idx < link_poses.shape[0]
                and link_idx < link_poses.shape[1]
            ):
                frame[f"robot.poses.{link_name}"] = link_poses[
                    env_idx, link_idx
                ].copy()
            else:
                frame[f"robot.poses.{link_name}"] = np.zeros(
                    7, dtype=np.float32,
                )

        # Object poses (already transformed to local frame in prefetch)
        obj_poses = cache.get('obj_poses_local', {})
        inactive = self._inactive_objects[env_idx]
        for obj_name in self._object_names:
            if obj_name in inactive:
                frame[f"observation.state.object_pose.{obj_name}"] = np.full(
                    7, np.nan, dtype=np.float32,
                )
            elif obj_name in obj_poses and env_idx < obj_poses[obj_name].shape[0]:
                frame[f"observation.state.object_pose.{obj_name}"] = (
                    obj_poses[obj_name][env_idx].copy()
                )
            else:
                frame[f"observation.state.object_pose.{obj_name}"] = np.zeros(
                    7, dtype=np.float32,
                )

        # Contact forces (already rotated to local frame in prefetch)
        contact_forces = cache.get('contact_forces_local', {})
        for sensor_name in self._contact_sensor_names:
            if sensor_name in contact_forces and env_idx < contact_forces[sensor_name].shape[0]:
                frame[f"observation.state.contact_force.{sensor_name}"] = (
                    contact_forces[sensor_name][env_idx].copy()
                )
            else:
                frame[f"observation.state.contact_force.{sensor_name}"] = np.zeros(
                    3, dtype=np.float32,
                )

        self._populate_frame_metadata(frame, env_idx, is_first, is_last)

        # Camera images — RGB already converted to uint8 in prefetch
        height, width = cache['cam_resolution']
        for cam_name in self._camera_names:
            rgb_key = f'cam_rgb_{cam_name}'
            if rgb_key in cache:
                img = cache[rgb_key][env_idx]
                if img.ndim == 2:
                    img = np.stack([img, img, img], axis=-1)
                elif img.shape[-1] == 1:
                    img = np.concatenate([img, img, img], axis=-1)
                frame[f"observation.images.{cam_name}"] = img

            depth_key = f'cam_depth_{cam_name}'
            if not self._record_depth:
                pass
            elif depth_key in cache:
                depth_raw = cache[depth_key][env_idx]
                frame[f"observation.depth.{cam_name}"] = (
                    depth_raw if depth_raw.dtype == np.uint16  # converted in the prefetch
                    else self._prepare_depth_image(depth_raw.astype(np.float32))
                )
            else:
                frame[f"observation.depth.{cam_name}"] = np.zeros(
                    (height, width), dtype=np.uint16,
                )

        # Tactile camera images — also pre-converted to uint8
        for cam_name in self._tactile_camera_names:
            rgb_key = f'cam_rgb_{cam_name}'
            if rgb_key in cache:
                img = cache[rgb_key][env_idx]
                if img.ndim == 2:
                    img = np.stack([img, img, img], axis=-1)
                elif img.shape[-1] == 1:
                    img = np.concatenate([img, img, img], axis=-1)
                frame[f"observation.images.{cam_name}"] = img

        # Camera poses and intrinsics (already in local frame)
        all_cam_names = list(self._camera_names) + list(self._tactile_camera_names)
        cam_poses = cache.get('cam_poses_local', {})
        for cam_name in all_cam_names:
            if cam_name in cam_poses and env_idx < cam_poses[cam_name].shape[0]:
                frame[f"observation.camera_pose.{cam_name}"] = (
                    cam_poses[cam_name][env_idx].copy()
                )
            else:
                frame[f"observation.camera_pose.{cam_name}"] = np.zeros(
                    7, dtype=np.float32,
                )

            intr_key = f'cam_intrinsics_{cam_name}'
            if intr_key in cache:
                frame[f"observation.camera_intrinsics.{cam_name}"] = (
                    cache[intr_key][env_idx].flatten().copy()
                )
            else:
                frame[f"observation.camera_intrinsics.{cam_name}"] = np.zeros(
                    9, dtype=np.float32,
                )

        return frame

    @staticmethod
    def _sha256_file(path: Path) -> Optional[str]:
        if not path.is_file():
            return None
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _collect_git_provenance(repo_root: Optional[Path] = None) -> Dict[str, Any]:
        """Best-effort git remote and commit for the repo containing this file."""
        if repo_root is None:
            repo_root = Path(__file__).resolve().parents[4]  # src/blocksuite/vecgen/recording/
        out: Dict[str, Any] = {"repo_root": str(repo_root)}
        try:
            commit = subprocess.run(
                ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            if commit.returncode == 0:
                out["git_commit"] = commit.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
        try:
            remote = subprocess.run(
                ["git", "-C", str(repo_root), "config", "--get", "remote.origin.url"],
                capture_output=True, text=True, timeout=5,
            )
            if remote.returncode == 0:
                out["git_remote_origin_url"] = remote.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
        return out

    def _write_post_finalize_metadata(self) -> None:
        """After LeRobot finalize, write recording_config, provenance, and meta/extra copies."""
        recording_cfg = {
            "gripper_binarize_threshold": self._gripper_binarize_threshold,
            "robotiq": self._robotiq,
            "record_videos": self._record_videos,
            "robot_link_pose_frame": "robot_base",
            "gripper_open_command": (
                self._gripper_open_command.tolist()
                if self._gripper_open_command is not None else None
            ),
            "gripper_close_command": (
                self._gripper_close_command.tolist()
                if self._gripper_close_command is not None else None
            ),
            "joint_velocity_limits": self._joint_velocity_limits,
        }

        prov_sources: Dict[str, Any] = dict(self._recording_provenance)
        ll_env = prov_sources.get("ll_env_directory")
        domain_p = prov_sources.get("domain_pddl")
        problem_p = prov_sources.get("problem_pddl")
        instance_ids = prov_sources.get("instance_json_ids") or []

        for root_str in (self._success_root, self._failure_root):
            root = Path(root_str)
            if not root.is_dir():
                continue
            # Backfill videos/<key>/{chunk_index,file_index,from_timestamp,to_timestamp} into
            # meta/episodes. Our encoder writes+moves the mp4s itself (bypassing lerobot's video path),
            # so these per-episode video pointers are otherwise absent and a modern lerobot raises
            # KeyError('videos/<key>/chunk_index') on load. Idempotent; correct for -N>1 (each output
            # episode has its own file-{episode_index}.mp4).
            try:
                from blocksuite.vecgen.recording._video_meta_backfill import backfill_video_meta
                res = backfill_video_meta(str(root))
                if res.get("patched_files"):
                    logger.info(f"backfilled video meta ({res['added_columns']} cols across "
                                f"{res['episodes']} eps) in {root}")
            except Exception as e:
                logger.error(f"video-meta backfill failed for {root}: {e}")
            meta = root / "meta"
            meta.mkdir(parents=True, exist_ok=True)
            extra = meta / "extra"
            extra.mkdir(parents=True, exist_ok=True)

            with open(meta / "recording_config.json", "w") as f:
                json.dump(recording_cfg, f, indent=2)

            file_records: List[Dict[str, Any]] = []
            git_meta = self._collect_git_provenance()

            def _copy_logged(src: Path, dest_name: str) -> None:
                if not src.is_file():
                    return
                dst = extra / dest_name
                shutil.copy2(src, dst)
                digest = self._sha256_file(dst)
                file_records.append({
                    "dest": f"meta/extra/{dest_name}",
                    "source": str(src.resolve()),
                    "sha256": digest,
                })

            if domain_p:
                _copy_logged(Path(domain_p), "domain.pddl")
            if problem_p:
                _copy_logged(Path(problem_p), Path(problem_p).name)

            if ll_env:
                ll_path = Path(ll_env)
                for name in ("block_config.json", "init.py"):
                    p = ll_path / name
                    _copy_logged(p, name)
                if instance_ids:
                    for iid in instance_ids:
                        p = ll_path / f"{int(iid)}.json"
                        _copy_logged(p, f"{int(iid)}.json")
                else:
                    for p in sorted(ll_path.glob("[0-9]*.json")):
                        _copy_logged(p, p.name)

            provenance_payload = {
                "git": git_meta,
                "recording_provenance_sources": prov_sources,
                "files": file_records,
            }
            with open(meta / "provenance.json", "w") as f:
                json.dump(provenance_payload, f, indent=2)

            info_path = meta / "info.json"
            if info_path.is_file():
                try:
                    with open(info_path, "r") as f:
                        info = json.load(f)
                except Exception:
                    info = {}
                info[ROW_LAYOUT_INFO_KEY] = ROW_LAYOUT
                info["tmp3_recording"] = {
                    "gripper_binarize_threshold": self._gripper_binarize_threshold,
                    "recording_config": "meta/recording_config.json",
                    "provenance": "meta/provenance.json",
                    "extra_dir": "meta/extra",
                }
                try:
                    with open(info_path, "w") as f:
                        json.dump(info, f, indent=2)
                except Exception as e:
                    logger.error(f"[DataRecorder] Could not patch info.json: {e}")

    def _validate_parquet_integrity(self) -> None:
        """Check that all written parquet files have valid magic bytes.

        Parquet files that were truncated mid-write (e.g. by a process kill)
        will have the ``PAR1`` header but a missing footer, making them
        unreadable.  This logs a clear warning so operators know which files
        need to be discarded or regenerated.
        """
        MAGIC = b"PAR1"
        for root_str in (self._success_root, self._failure_root):
            root = Path(root_str)
            if not root.is_dir():
                continue
            for pq_file in root.rglob("*.parquet"):
                try:
                    with open(pq_file, "rb") as f:
                        header = f.read(4)
                        f.seek(-4, 2)
                        footer = f.read(4)
                    if header != MAGIC or footer != MAGIC:
                        logger.error(
                            f"CORRUPT PARQUET: {pq_file} "
                            f"(header={header!r}, footer={footer!r}) — "
                            f"file was likely truncated by a process kill"
                        )
                except Exception as e:
                    logger.error(f"CORRUPT PARQUET: {pq_file} — cannot read: {e}")

    def write_success(self, success: bool, env_idx: int = 0):
        """Mark an environment's episode as successful."""
        if success:
            self.success_flags[env_idx] = True

    def set_success_flags_batched(self, success: Any) -> None:
        """Per-env success for the step just taken; a True sticks for the rest of the episode."""
        if success is None:
            return
        if isinstance(success, torch.Tensor):
            success = success.detach().cpu().numpy()
        arr = np.asarray(success).astype(bool).reshape(-1)
        n = min(self.num_envs, arr.shape[0])
        self.success_flags[:n] |= arr[:n] & ~self.success_vetoed[:n]

    def set_pending_actions_batched(self, actions: Any) -> None:
        """Stage every env's raw policy action for the step about to run (``action.policy``)."""
        if actions is None or self._action_dim is None:
            return
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        arr = np.asarray(actions, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        for i in range(min(self.num_envs, arr.shape[0])):
            self._pending_actions[i] = arr[i].copy()

    def set_pending_pd_modes_batched(self, pd_modes: Any) -> None:
        """Stage the pd_mode in effect in every env for the step about to run.

        Overrides the mode derived from the measured stiffness for that step: a caller switching
        gains that do not match a named profile knows which mode they stand for. NaN entries
        keep the derived value.
        """
        if pd_modes is None:
            return
        if isinstance(pd_modes, torch.Tensor):
            pd_modes = pd_modes.detach().cpu().numpy()
        arr = np.asarray(pd_modes, dtype=np.float32).reshape(-1)
        for i in range(min(self.num_envs, arr.shape[0])):
            self._pending_pd_modes[i] = float(arr[i])

    def set_hl_model_outputs_batched(
        self,
        outputs: Sequence[str],
        prev_actions: Sequence[str],
        prompts: Sequence[str],
        done: Sequence[bool],
    ) -> None:
        """Stage every env's learned-HL output for the step about to run (``record_hl_model``).

        ``outputs`` is the raw HL text (instruction and target coordinates), ``prev_actions`` /
        ``prompts`` the ``Previous action`` string and full question it was asked with, ``done``
        its done flag. Kept until restaged, so steps between HL calls carry the last output.
        """
        if not self.record_hl_model:
            return
        n = min(self.num_envs, len(outputs), len(prev_actions), len(prompts), len(done))
        for i in range(n):
            self._hl_model_outputs[i] = str(outputs[i])
            self._hl_prev_actions[i] = str(prev_actions[i])
            self._hl_input_prompts[i] = str(prompts[i])
            self._hl_dones[i] = bool(done[i])

    def set_hl_overlay_batched(
        self,
        texts: Sequence[str],
        heatmaps: Optional[np.ndarray] = None,
        prev_actions: Optional[Sequence[str]] = None,
        prompts: Optional[Sequence[str]] = None,
        oracle_texts: Optional[Sequence[str]] = None,
        oracle_heatmaps: Optional[np.ndarray] = None,
    ) -> None:
        """Stage every env's HL overlay inputs for the step about to run (``hl_overlay_camera``).

        ``heatmaps`` / ``oracle_heatmaps`` are ``(num_envs, H, W)``. ``prev_actions`` / ``prompts``
        are the learned HL's Molmo2 inputs (``None`` for the oracle HL); ``oracle_*`` the
        comparison oracle's HL for the right-hand panel. Kept until restaged.
        """
        if self._hl_overlay_camera is None:
            return

        def at(seq: Optional[Sequence[Any]], i: int) -> Any:
            return seq[i] if seq is not None and i < len(seq) else None

        for i in range(min(self.num_envs, len(texts))):
            self._hl_overlays[i] = (
                str(texts[i]), at(heatmaps, i), at(prev_actions, i), at(prompts, i),
                at(oracle_texts, i), at(oracle_heatmaps, i),
            )

    def _predict_episode_video_path(self, writer: "_Writer", success: bool, ep_idx: int) -> str:
        """Path the encoder subprocess will move this episode's primary camera video to.

        The subprocess names the file from ``ds.num_episodes - 1`` after appending the episode, and it
        drains its queue in FIFO order, so the writer's mirrored per-dataset counter yields the same
        index without having to wait for the encode to land. Returns "" when the run records no video,
        and with several writers, whose episodes are renumbered when finalize stitches them.
        """
        if not self._record_videos or self._num_writers > 1:
            return ""
        cameras = self._get_camera_names()
        if not cameras:
            return ""
        # Prefer the front camera — it is the view the dashboard is most useful showing.
        cam = next((c for c in cameras if "front" in c.lower()), cameras[0])
        root = writer.success_root if success else writer.failure_root
        return os.path.join(
            root,
            "videos",
            f"observation.images.{cam}",
            f"chunk-{ep_idx // 1000:03d}",
            f"file-{ep_idx % 1000:03d}.mp4",
        )

    def frames_recorded(self, env_idx: int) -> int:
        """Frames recorded so far in env ``env_idx``'s open episode, the held frame included."""
        return int(self._rows_queued[env_idx])

    def drop_frames_since(self, env_idx: int, start: int) -> None:
        """Leave the frames recorded since ``start`` (a :meth:`frames_recorded` value) out of
        the saved episode; LeRobot numbers frames as they are added, so the kept ones are
        renumbered without gaps. Streaming save only; the encoder re-encodes the episode's
        videos without these frames.
        """
        end = self.frames_recorded(env_idx)
        if end > start:
            self._dropped_spans[env_idx].append((int(start), end))

    def save_episode(self, env_idx: int = 0) -> str:
        """Send buffered frames to the video encoder subprocess for saving.

        When goal annotation is disabled, frames were already streamed to
        the subprocess — only a lightweight ``end_episode`` with metadata
        is sent.  When goal annotation is enabled, frames are buffered
        locally for ``finalize_annotations`` and sent in bulk.

        Returns:
            Path the episode's primary camera video will be written to, or "" when nothing was saved
            or the run records no video. The file appears asynchronously once the encoder drains.
        """
        if self._pending_build_future is not None:
            self._pending_build_future.result()
            self._pending_build_future = None

        if not self.active_episodes[env_idx]:
            logger.warn(f"No active episode for env {env_idx}, skipping save")
            return ""
        # One finished episode, whether or not it turns out to hold frames: this is what keeps
        # the ordinal aligned with a caller's own per-env episode counter.
        ordinal = int(self._episode_ordinals[env_idx])
        self._episode_ordinals[env_idx] += 1

        # Release the held frame as the terminal row before counting frames —
        # a one-record-call episode has its only frame still pending here.
        self._flush_pending_frame(env_idx)

        has_frames = (
            self._frame_counts[env_idx] > 0
            or bool(self.frame_buffers.get(env_idx))
        )
        if not has_frames:
            self.active_episodes[env_idx] = False
            logger.warn(f"No frames buffered for env {env_idx}, skipping save")
            return ""

        success = bool(self.success_flags[env_idx])
        error_log = self.error_logs[env_idx] or None

        if self._goal_annotator:
            self._save_episode_with_goal_annotation(env_idx, success, error_log)
        else:
            self._save_episode_streaming(env_idx, success, error_log)

        writer = self._writer_for(env_idx)
        label = "success" if success else "failure"
        ep_idx = writer.queued_episode_counts[label]
        writer.queued_episode_counts[label] += 1
        self._saved_episodes.append((writer.index, label, ep_idx, int(env_idx), ordinal))
        video_path = self._predict_episode_video_path(writer, success, ep_idx)

        self.frame_buffers[env_idx] = []
        self._frame_counts[env_idx] = 0
        self.active_episodes[env_idx] = False
        self.reset_env_state(env_idx)
        return video_path

    def _save_episode_streaming(self, env_idx: int, success: bool, error_log: Optional[str]):
        """Lightweight save — frames were already streamed to the subprocess."""
        if not self._episode_started[env_idx]:
            self._queue_put(("start_episode", env_idx, self._cam_configs))
            self._episode_started[env_idx] = True

        num_frames = int(self._frame_counts[env_idx])
        label = "SUCCESS" if success else "FAILURE"
        logger.debug(f"Episode queued for env {env_idx} ({num_frames} frames, {label})")

        self._queue_put((
            "end_episode", env_idx, success, None, None, error_log,
            list(self._dropped_spans[env_idx]),
        ))

    def _save_episode_with_goal_annotation(self, env_idx: int, success: bool, error_log: Optional[str]):
        """Full save path — frames are buffered locally for goal annotation."""
        frames = self.frame_buffers[env_idx]
        if not frames:
            logger.warn(f"No frames buffered for env {env_idx}, skipping save")
            return

        was_recording = bool(self._episode_started[env_idx])

        if not self._episode_started[env_idx]:
            self._queue_put(("start_episode", env_idx, self._cam_configs))
            self._episode_started[env_idx] = True

        if not was_recording:
            for frame in frames:
                remaining: Dict[str, np.ndarray] = {}
                for key in list(frame.keys()):
                    val = frame[key]
                    if (self._is_image_key(key)
                            and isinstance(val, np.ndarray) and val.ndim >= 2
                            and not key.startswith("observation.goal.")):
                        remaining[key] = val
                        del frame[key]
                if remaining:
                    self._queue_put(("frame", env_idx, remaining))

        try:
            self._goal_annotator.finalize_annotations(frames, env_idx)
        except Exception as e:
            self.set_error(f"GOAL_ANNOTATION_FINALIZE_FAILED: {e}", env_idx=env_idx)

        goal_images = []
        for frame in frames:
            gi: Dict[str, np.ndarray] = {}
            for key in list(frame.keys()):
                val = frame[key]
                if (self._is_image_key(key)
                        and isinstance(val, np.ndarray) and val.ndim >= 2):
                    if key.startswith("observation.goal."):
                        gi[key] = val
                    del frame[key]
            goal_images.append(gi)

        if error_log:
            for frame in frames:
                frame["error"] = error_log

        frames[-1]["is_last"] = np.array([True], dtype=np.bool_)
        frames[-1]["is_episode_successful"] = np.array(
            [success], dtype=np.bool_,
        )

        num_frames = len(frames)
        label = "SUCCESS" if success else "FAILURE"
        logger.debug(f"Episode queued for env {env_idx} ({num_frames} frames, {label})")

        self._queue_put((
            "end_episode", env_idx, success, frames, goal_images,
        ))

    def discard_episode(self, env_idx: int) -> None:
        """Drop env ``env_idx``'s episode without saving it (e.g. an eval env that ran with no task).

        Frames already streamed are dropped by the encoder. A held reset frame belongs to the
        next episode and survives. Counts as a finished episode for the ordinal if one was active.
        """
        if self._pending_build_future is not None:
            self._pending_build_future.result()
            self._pending_build_future = None
        if self.active_episodes[env_idx]:
            self._episode_ordinals[env_idx] += 1
        if self._episode_started[env_idx]:
            self._queue_put(("discard_episode", int(env_idx)))
        self.active_episodes[env_idx] = False
        self.reset_env_state(env_idx)

    def merged_episode_index_map(self) -> Dict[Tuple[int, int], Tuple[str, int]]:
        """``(env_idx, that env's n-th episode)`` -> ``(dataset root, episode index)`` of every
        saved episode. Call after :meth:`finalize`.

        Writers are stitched in ascending index order, each in the order it saved its episodes,
        which is the numbering reproduced here; an env that finishes early can therefore take a
        lower index than lower-numbered envs, so the index cannot be derived from ``env_idx``.
        """
        counts: Dict[Tuple[int, str], int] = {}
        for writer_idx, label, _, _, _ in self._saved_episodes:
            counts[(writer_idx, label)] = counts.get((writer_idx, label), 0) + 1
        out: Dict[Tuple[int, int], Tuple[str, int]] = {}
        for writer_idx, label, ep_idx, env_idx, ordinal in self._saved_episodes:
            offset = sum(counts.get((k, label), 0) for k in range(writer_idx))
            root = self._success_root if label == "success" else self._failure_root
            out[(env_idx, ordinal)] = (root, offset + ep_idx)
        return out

    def reset_env_state(self, env_idx: int):
        """Reset the state tracking for a specific environment (call when env resets).

        A held reset frame belongs to the next episode and survives, and the deltas of that
        episode's first action are taken against it.
        """
        self.step_counters[env_idx] = 0
        self._episode_started[env_idx] = False
        self._pending_frames[env_idx] = None
        self._dropped_spans[env_idx] = []
        self._rows_queued[env_idx] = int(self._reset_frames[env_idx] is not None)  # a held s_0
        self.success_flags[env_idx] = False
        self.success_vetoed[env_idx] = False
        reset_prev = self._reset_prev[env_idx] if self._reset_frames[env_idx] is not None else None
        if reset_prev is not None:
            self.previous_joint_states[env_idx] = reset_prev[0]
            self.previous_joint_targets[env_idx] = reset_prev[1]
            self.previous_ee_poses[env_idx] = dict(reset_prev[2])
        else:
            self.previous_ee_poses[env_idx] = {}
            self.previous_joint_states[env_idx] = 0.0
            self.previous_joint_targets[env_idx] = 0.0
        self._pending_actions[env_idx] = None
        self._pending_pd_modes[env_idx] = None
        self.hl_actions[env_idx] = ""
        self.hl_args_dict[env_idx] = {}
        self.ll_actions[env_idx] = ""
        self.ll_args_dict[env_idx] = {}
        self.hl_state_strs[env_idx] = ""
        self._cached_hl_args_json[env_idx] = ""
        self._cached_ll_args_json[env_idx] = ""
        self._cached_task_descriptions[env_idx] = "manipulation"
        self.error_logs[env_idx] = ""
        if self._goal_annotator:
            self._goal_annotator.reset(env_idx)
        self.frame_buffers[env_idx] = []
        self._frame_counts[env_idx] = 0

    def _build_minimal_error_frame(self, env_idx: int = 0) -> Dict[str, Any]:
        """Build a minimal frame with zero-valued observations for error-only datasets."""
        frame: Dict[str, Any] = {}

        self._zero_fill_robot_data(frame)

        for obj_name in self._object_names:
            frame[f"observation.state.object_pose.{obj_name}"] = np.zeros(7, dtype=np.float32)
        for sensor_name in self._contact_sensor_names:
            frame[f"observation.state.contact_force.{sensor_name}"] = np.zeros(3, dtype=np.float32)

        height, width = self._get_camera_resolution()
        for cam_name in self._camera_names:
            frame[f"observation.images.{cam_name}"] = np.zeros((height, width, 3), dtype=np.uint8)
            if self._record_depth:
                frame[f"observation.depth.{cam_name}"] = np.zeros((height, width), dtype=np.uint16)
        if self._hl_overlay_camera is not None:
            frame[f"observation.images.{self._hl_overlay_camera}_hl_overlay"] = self._draw_hl_overlay(
                frame[f"observation.images.{self._hl_overlay_camera}"], None)
        if self._tactile_camera_names:
            t_height, t_width = self._get_tactile_camera_resolution()
            for cam_name in self._tactile_camera_names:
                frame[f"observation.images.{cam_name}"] = np.zeros((t_height, t_width, 3), dtype=np.uint8)
        for cam_name in list(self._camera_names) + list(self._tactile_camera_names):
            frame[f"observation.camera_pose.{cam_name}"] = np.zeros(7, dtype=np.float32)
            frame[f"observation.camera_intrinsics.{cam_name}"] = np.zeros(9, dtype=np.float32)

        self._populate_frame_metadata(frame, env_idx, is_first=True, is_last=True)
        self._apply_frame_extras(frame, env_idx, (None, None, ("", "", "", False)))
        frame.pop("_hl_overlay", None)  # drawn above; this frame skips _emit_frame
        frame["task"] = "failed"
        # Zero-filled placeholder for an episode that produced no trajectory —
        # there is no action here either.
        frame["action.is_valid"] = np.array([False], dtype=np.bool_)

        return frame

    def _flush_active_episodes(self) -> None:
        """Save any still-active episodes, materializing an error-only episode
        for envs that ended without recording a trajectory. Shared by the final
        :meth:`finalize` and by :meth:`roll_shard`.
        """
        for env_idx in range(self.num_envs):
            if self.active_episodes[env_idx]:
                has_frames = (
                    self._frame_counts[env_idx] > 0
                    or bool(self.frame_buffers.get(env_idx))
                    # save_episode below turns a still-held frame into the
                    # terminal row, so this episode is not frameless.
                    or self._pending_frames[env_idx] is not None
                )
                if not has_frames:
                    if not self.error_logs[env_idx]:
                        self.set_error("UNKNOWN: No frames recorded and no error was logged", env_idx=env_idx)
                    logger.warn(f"No trajectory frames for env {env_idx}. Creating error-only dataset.")
                    error_frame = self._build_minimal_error_frame(env_idx)
                    if self._goal_annotator:
                        self.frame_buffers[env_idx] = [error_frame]
                    else:
                        if not self._episode_started[env_idx]:
                            self._queue_put(("start_episode", env_idx, self._cam_configs))
                            self._episode_started[env_idx] = True
                        image_dict: Dict[str, np.ndarray] = {}
                        for key in list(error_frame.keys()):
                            val = error_frame[key]
                            if self._is_image_key(key) and isinstance(val, np.ndarray) and val.ndim >= 2:
                                image_dict[key] = val
                                del error_frame[key]
                        self._queue_put(("frame", env_idx, image_dict, error_frame))
                        self._frame_counts[env_idx] = 1
                    self.active_episodes[env_idx] = True
                self.save_episode(env_idx)

    def _write_shard_manifest(self, shard_dir: Path, instance_ids: List[int]) -> None:
        """Mark a finalized shard done and list the instance ids it recorded.

        Resume reads these to skip already-collected instances (see
        :func:`scan_shards`).
        """
        try:
            shard_dir.mkdir(parents=True, exist_ok=True)
            with open(shard_dir / RECORDED_INSTANCES_FILE, "w") as f:
                json.dump([int(x) for x in (instance_ids or [])], f)
            (shard_dir / SHARD_DONE_MARKER).write_text("")
        except Exception as e:
            logger.error(f"[DataRecorder] Could not write shard manifest in {shard_dir}: {e}")

    def _send_writer_finalize(self, writer: "_Writer") -> None:
        """Ask *writer*'s encoder to finalize its datasets (no-op if it already died)."""
        writer.force_terminated = False
        if writer.dead or not writer.process.is_alive():
            writer.force_terminated = True
            logger.error(
                f"video encoder subprocess {writer.index} already dead before finalize "
                f"(exitcode={writer.process.exitcode}); skipping finalize send. "
                f"This dataset is INCOMPLETE (completed flag stays false) and will be "
                f"re-rendered on a --resume relaunch."
            )
            return
        logger.info(
            f"Sending finalize to video encoder subprocess {writer.index} ... "
            f"(exec loop spent {writer.backpressure_s:.1f}s waiting on the encoder budget; "
            f"backlog now {(writer.outbox_sent - int(writer.consumed.value)) / 2**20:.0f} MiB)")
        try:
            self._writer_put(writer, ("finalize",))
        except EncoderProcessError as e:
            logger.error(f"encoder {writer.index} died before finalize could be sent: {e}")

    def _await_writer(self, writer: "_Writer") -> None:
        """Wait for *writer*'s encoder to drain its backlog and close every video.

        Do NOT kill a still-working encoder on a blind timeout — that was the root cause of the
        truncated tail videos ("moov atom not found"): a large encode backlog at finalize took
        >600s, the old code SIGTERM'd the encoder mid-close, and every unflushed video lost its
        moov trailer. Instead watch the heartbeat the encoder bumps per processed command: as long
        as it keeps making progress keep waiting (no matter how big the backlog); terminate ONLY if
        it is genuinely HUNG (no progress for STALL_GRACE_S). Encoding is finite, so a healthy
        drain always finishes; a real hang (deadlock) still gets bounded.
        """
        import time as _time
        STALL_GRACE_S = 600            # no-progress window that means "hung", not "slow"
        proc = writer.process
        last_hb, last_progress_t = -1, _time.monotonic()
        while proc.is_alive():
            proc.join(timeout=15)
            if not proc.is_alive():
                break
            hb = int(writer.heartbeat.value)
            if hb != last_hb:
                last_hb = hb
                last_progress_t = _time.monotonic()
                logger.info(f"video encoder {writer.index} still flushing (heartbeat={hb}); waiting ...")
            elif _time.monotonic() - last_progress_t > STALL_GRACE_S:
                logger.error(
                    f"video encoder {writer.index} HUNG: no progress for {STALL_GRACE_S}s "
                    f"(heartbeat={hb}); terminating. Tail videos may be incomplete — the post-write / "
                    f"upload validator will flag them; --resume re-renders this problem."
                )
                writer.force_terminated = True
                proc.terminate()
                proc.join(timeout=10)
                break

    def _stitch_writers(self) -> None:
        """Stitch the writers' dataset pairs into the published ``_success_root`` / ``_failure_root``."""
        from blocksuite.vecgen.recording._stitch import stitch_datasets

        for label, final_root in (("success", self._success_root), ("failure", self._failure_root)):
            sources = [
                Path(w.success_root if label == "success" else w.failure_root)
                for w in self._writers
            ]
            sources = [src for src in sources if (src / "meta" / "info.json").is_file()]
            if not sources:
                continue
            try:
                stitch_datasets(sources, Path(final_root))
            except Exception as e:
                logger.error(
                    f"stitching the {label} datasets of {len(sources)} writers failed: {e} "
                    f"(they remain under {Path(self.data_dir) / WRITERS_DIRNAME})"
                )
                traceback.print_exc()
                return
        shutil.rmtree(Path(self.data_dir) / WRITERS_DIRNAME, ignore_errors=True)

    def _close_shard(self, instance_ids: Optional[List[int]] = None) -> Path:
        """Finalize the current shard's datasets, write its metadata + manifest.

        Returns the finalized shard directory. Does not tear down the recorder,
        so :meth:`roll_shard` can open a fresh shard afterwards.

        The shard is marked done (``.shard_done`` + ``recorded_instances.json``)
        only when ``instance_ids`` is an explicit list — the caller vouches for
        exactly which instances landed here. ``None`` means an unclean close
        (e.g. the simulator tearing down after an exception): the shard is
        finalized but left unmarked so resume discards and re-runs it rather
        than trusting an unknown instance set (which would risk duplicates).
        """
        if self._pending_build_future is not None:
            self._pending_build_future.result()
            self._pending_build_future = None

        self._flush_active_episodes()

        calls = max(1, self._caller_record_calls)
        print(
            f"[recorder] caller thread: {self._caller_record_calls} record calls, "
            f"{1000 * self._caller_record_s / calls:.1f} ms/call "
            f"({1000 * self._caller_build_wait_s / calls:.1f} ms waiting on the frame build); "
            f"encoder budget waits {sum(w.backpressure_s for w in self._writers):.1f}s",
            flush=True,
        )
        # Send every writer its finalize first so they drain in parallel, then wait on each.
        for writer in self._writers:
            self._send_writer_finalize(writer)
        for writer in self._writers:
            self._await_writer(writer)
        # True if any encoder did NOT finish cleanly (died, or had to be force-terminated after a
        # hang). The caller must then withhold render_params.completed=true so the incomplete problem
        # is re-rendered on a --resume relaunch instead of being uploaded with truncated tail videos.
        self._writer_force_terminated = any(w.force_terminated for w in self._writers)

        for writer in self._writers:
            if writer.arena is not None:
                writer.arena.close()
                writer.arena = None

        if self._num_writers > 1:
            self._stitch_writers()

        self._write_post_finalize_metadata()
        self._validate_parquet_integrity()
        shard_dir = Path(self.data_dir)
        if instance_ids is not None:
            self._write_shard_manifest(shard_dir, instance_ids)
        return shard_dir

    def roll_shard(self, instance_ids: Optional[List[int]] = None) -> Path:
        """Finalize the current shard and open a fresh one for continued recording.

        Raises when the recorder was built with ``enable_sharding=False``: that caller declared
        one dataset per problem dir, and silently rolling to a sibling would put the data where
        nothing it owns will look.

        Called by the orchestrator once it has drained all envs at a shard
        boundary, so no episode is mid-write. ``instance_ids`` are the instances
        this (now-finalized) shard recorded. Returns the finalized shard dir.
        """
        if not getattr(self, "_enable_sharding", True):
            raise RuntimeError(
                "roll_shard() called on a recorder built with enable_sharding=False. That caller "
                "writes one dataset per problem dir; rolling would move the data to a sibling "
                "__sNNN it never reads.")
        if self._finalized:
            return Path(self.data_dir)

        finished = self._close_shard(instance_ids)

        shard_dir, self._shard_index = next_free_shard(
            self._shard_base_dir, self._shard_index + 1
        )
        self.data_dir = str(shard_dir)
        os.makedirs(self.data_dir, exist_ok=True)
        # Recreates _success_root/_failure_root, resets per-dataset episode
        # counters and _episode_started, and spawns a fresh encoder subprocess.
        self._create_dataset(self.dataset_name)
        logger.info(f"[DataRecorder] Rolled to new shard: {self.data_dir}")
        return finished

    def finalize(self, instance_ids: Optional[List[int]] = None):
        """Finalize the dataset - call this when all environments are done.

        Saves any remaining active episodes, finalizes the current shard, and
        marks it done. ``instance_ids`` are the instances recorded in the final
        (still-open) shard.
        """
        if self._finalized:
            return

        if hasattr(self, '_stdout_capture'):
            sys.stdout = self._stdout_capture._original

        self._close_shard(instance_ids)

        self._frame_build_pool.shutdown(wait=False)
        self._finalized = True
