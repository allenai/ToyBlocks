"""Model inference loops for vlagent RCS sim-to-real client."""

from __future__ import annotations

import copy
import datetime
import logging
import threading
from collections import deque
from dataclasses import dataclass
from queue import Empty, Queue
from time import monotonic, sleep
from typing import Any, Protocol, runtime_checkable

import cv2
import gymnasium as gym
import numpy as np
from rcs._core.common import RobotPlatform
from rcs.envs.base import RelativeTo
from rcs.envs.storage_wrapper import StorageWrapper
from rcs.utils import SimpleFrameRate
from vlagents.client import RemoteAgent
from vlagents.policies.interface import Obs, SingleAct

from blocksuite.serving.sim2real.rollout.camera_utils import (
    INTERACTIVE_GUI_CAMERAS,
    POINT_PICK_CAMERA,
    camera_color_to_rgb,
    display_available,
    maybe_save_first_frame,
    mock_camera_frame,
    read_camera_frames,
    try_open_hardware_camera_set,
)
from blocksuite.serving.sim2real.rollout.eval_log import (
    EvalLogger,
    eval_output_dir,
    run_metadata,
)
from blocksuite.serving.sim2real.rollout.ffstereo_server import FFStereoClient, maybe_save_depth_outputs
from blocksuite.serving.sim2real.rollout.goal_utils import (
    allowed_goal_actions_display,
    format_language_goal,
    instruction_from_goal,
    is_valid_goal_action,
    policy_obs_target_heatmap,
    policy_space_uv,
    src_frame_size_for_point,
    TARGET_CROP_KEY,
    target_front_crop_from_point,
    target_heatmap_from_point,
)
from blocksuite.serving.sim2real.rollout.molmo2_server import Molmo2Client
from blocksuite.serving.sim2real.rollout.hardware_config import RELATIVETO, ROBOT_INSTANCE
from blocksuite.serving.sim2real.rollout.inference_config import (
    CONFIG_PATH,
    InferenceConfig,
    load_inference_config,
    sync_frame_conditioning,
    sync_point_goal_format,
)
from blocksuite.serving.sim2real.rollout.remote_agent_utils import (
    ensure_server_ready,
    remote_agent_act,
    remote_checkpoint_path,
)
from blocksuite.serving.sim2real.rollout.tam_utils import (
    check_ideal_model_alignment,
    hardware_robot,
    init_tam_runtime,
    launch_history_encoder_thread,
    tam_supported,
    warmup_history_encoder,
)
from blocksuite.serving.sim2real.rollout.vlagent_obs import (
    ChunkActionSource,
    TemporalEnsembleActionSource,
    build_dry_run_obs,
    build_vlagent_obs,
    refresh_policy_obs,
    unpack_single_act,
)

logger = logging.getLogger(__name__)

GoalQueueItem = tuple[str, tuple[str, float, float] | None, bool, bool]

# Tries per robot for the "go home" command, each after an error recovery.
_HOME_ATTEMPTS = 3
# Consecutive failed policy queries before the loop stops the episode instead
# of retrying (a dead server would otherwise be retried forever).
_ACT_FAILURE_LIMIT = 5

# Extra camera carrying the front view frozen at the moment the goal point was
# clicked, which checkpoints trained with ``framework.use_previous_front_frame``
# consume on every inference.
PREV_FRAME_KEY = "prev_front_rgb"


def add_prev_front_rgb(
    cameras: dict[str, Any], prev_frame: np.ndarray | None
) -> dict[str, Any]:
    """Add :data:`PREV_FRAME_KEY` to ``cameras`` (in place), live frame if unset."""
    if POINT_PICK_CAMERA in cameras:
        cameras[PREV_FRAME_KEY] = (
            cameras[POINT_PICK_CAMERA] if prev_frame is None else prev_frame
        )
    return cameras


@dataclass(frozen=True)
class _ObsBuildSnapshot:
    """Immutable snapshot of every controller field ``obs_rcs2agents`` reads.

    Built on the main thread at launch time and handed to any worker thread
    that will run the observation-building pipeline (Molmo2 HL, ffstereo
    depth, pose packing, ...). All fields are captured atomically here so
    cross-field coherence is guaranteed even if the main thread later
    hot-reloads the config or a GUI callback updates the target heatmap /
    Molmo prompt.

    Notes:
    - ``cfg`` is captured by reference; a later ``self._cfg = new`` rebind
      on the main thread does not affect this snapshot (Python pointer
      swap; snapshots hold the old object).
    - ``target_heatmap``/``prev_front_rgb``/``ffstereo``/``molmo`` are
      captured by reference too.
      The arrays/clients themselves are treated as read-only by the worker.
    - ``molmo_reset_prev_action`` is the *consumed* value: the flag is
      read-and-cleared on the main thread when the snapshot is built, so
      exactly one worker call after each :meth:`_mark_molmo_reset` sees
      ``True`` and instructs the Molmo2 server to clear its rolling
      ``Previous action: ...`` prefix.
    - ``_latest_frames`` is *not* in the snapshot: it is written under
      ``self._frames_lock`` on both threads (already race-free).
    """

    cfg: InferenceConfig
    target_heatmap: np.ndarray | None
    target_point: tuple[str, float, float] | None
    prev_front_rgb: np.ndarray | None
    prev_play_mode: float
    ffstereo: FFStereoClient | None
    molmo: Molmo2Client | None
    molmo_prompt: str
    molmo_reset_prev_action: bool
    # ``async_molmo`` only: False means "reuse ``molmo_cached``, do not call
    # the HL this build". Like ``molmo_reset_prev_action`` it is the
    # *consumed* value -- the pending flag is read-and-cleared on the main
    # thread when the snapshot is built, so exactly one worker call after
    # each :meth:`_mark_molmo_refresh` runs the HL.
    molmo_run: bool
    # ``(target_heatmap, instruction, target_point)`` from the last HL call,
    # used when ``molmo_run`` is False. ``instruction is None`` means no HL
    # output exists yet (nothing cached), and the caller falls back to
    # ``cfg.instruction``.
    molmo_cached: tuple[np.ndarray | None, str | None, tuple[str, float, float] | None]

# Graded commands kept for the GUI's ticker tape. Bounded: this is an operator
# aid, the eval log on disk is the record.
VERDICT_HISTORY = 25


@dataclass
class _GradedCommand:
    """The command a later success/failure verdict attaches to.

    Captured when the goal is submitted rather than when it is graded: by then
    the episode may already have been reset, which mints a new uuid and zeroes
    the step counter, losing the only handles on the recorded data.

    ``started_monotonic`` is stamped here too, so the logged duration measures
    how long the robot actually ran the command -- not how long the operator
    took to grade it afterwards. ``duration_s`` is filled in when the episode
    stops; a grade arriving before that measures up to the moment of grading.
    """

    goal: str
    instruction: str
    point: tuple[str, float, float] | None
    episode_uuid: str
    step_start: int
    recording: bool
    submitted_at: str
    started_monotonic: float
    duration_s: float | None = None


_cv2_display_warned = False


def cv2_camera_windows_enabled(cfg: InferenceConfig) -> bool:
    global _cv2_display_warned
    if not cfg.show_camera_windows or cfg.interactive_gui:
        return False
    if display_available():
        return True
    if not _cv2_display_warned:
        logger.warning(
            "show_camera_windows=true but no DISPLAY/WAYLAND_DISPLAY; "
            "skipping OpenCV camera windows (use --interactive-gui for browser GUI)"
        )
        _cv2_display_warned = True
    return False


@runtime_checkable
class InferenceController(Protocol):
    """GUI-facing controller API implemented by full and dry-run inference."""

    _cfg: InferenceConfig
    _shutdown_requested: threading.Event

    def submit_command(self, command: str) -> None: ...
    def submit_verdict(self, success: bool) -> None: ...
    def claim_verdict(self, success: bool) -> str: ...
    def claim_amendment(self, success: bool) -> str: ...
    def submit_language_goal(
        self,
        goal: str,
        point: tuple[str, float, float] | None = None,
        *,
        auto_start: bool = True,
        record: bool = False,
    ) -> None: ...
    def submit_molmo_prompt(self, prompt: str, *, auto_start: bool = True) -> None: ...
    def clear_molmo_prompt(self) -> None: ...
    def request_shutdown(self) -> None: ...
    def get_latest_frames(self) -> dict[str, np.ndarray]: ...
    def capture_prev_front_rgb(self) -> None: ...
    @property
    def last_submitted_goal(self) -> str: ...
    @property
    def policy_checkpoint(self) -> str: ...
    @property
    def eval_log_path(self) -> str: ...
    @property
    def target_heatmap_preview(self) -> np.ndarray | None: ...
    @property
    def pending_verdict_command(self) -> str: ...
    @property
    def command_runtime(self) -> tuple[float, bool]: ...
    @property
    def command_history(self) -> list[dict[str, Any]]: ...
    @property
    def verdict_tally(self) -> tuple[int, int]: ...
    @property
    def current_skill(self) -> str: ...
    @property
    def episode_running(self) -> bool: ...
    @property
    def molmo_prompt(self) -> str: ...
    def get_molmo_state(self) -> dict[str, Any]: ...


class GoalQueueMixin:
    """Shared queues and GUI-facing methods for inference controllers."""

    def _init_goal_queue_state(self, cfg: InferenceConfig) -> None:
        self._cfg = cfg
        self._command_queue: Queue[str] = Queue()
        self._goal_queue: Queue[GoalQueueItem] = Queue()
        self._verdict_queue: Queue[bool] = Queue()
        self._shutdown_requested = threading.Event()
        self._frames_lock = threading.Lock()
        self._latest_frames: dict[str, np.ndarray] = {}
        self._last_submitted_goal = cfg.instruction
        self._target_point: tuple[str, float, float] | None = None
        self._target_heatmap: np.ndarray | None = None
        self._target_heatmap_preview: np.ndarray | None = None
        # Front view frozen when the goal point was clicked, sent as
        # ``prev_front_rgb``. Guarded by ``_frames_lock``: the GUI thread
        # captures it, the inference thread reads it while building an obs.
        self._prev_front_rgb: np.ndarray | None = None
        self._policy_checkpoint = ""
        self._point_goal_format_checked = False
        # Run timer for the command currently submitted: started when the goal
        # is applied, stopped when the episode resets. ``_command_ended`` stays
        # None while the command is still running.
        self._command_started: float | None = None
        self._command_ended: float | None = None
        self._pending_verdict_goal = ""
        # The goal the run timer refers to. Unlike ``_pending_verdict_goal``
        # this survives grading, so a frozen time stays labelled with the skill
        # that produced it.
        self._timed_goal = ""
        self._verdict_history: deque[dict[str, Any]] = deque(maxlen=VERDICT_HISTORY)
        # Running totals for every grade this run. Kept apart from
        # ``_verdict_history``, which is a bounded deque and so cannot answer
        # for commands older than :data:`VERDICT_HISTORY`.
        self._verdict_successes = 0
        self._verdict_total = 0
        # One grade per command. Claimed on the caller's thread so a second
        # grade is refused at the source rather than silently dropped a tick
        # later, when the inference loop gets round to the queue.
        self._verdict_lock = threading.Lock()
        self._verdict_claimed = False
        self._amendment_queue: Queue[bool] = Queue()
        # Molmo2 HL state. ``_molmo_lock`` guards every field below so the
        # obs-build thread (worker in temporal_agg mode, main thread
        # otherwise) can publish the latest inference output while the GUI
        # thread reads it — without either side seeing torn state or
        # racing on the one-shot ``reset_prev_action`` flag consumed by
        # ``_snapshot_for_build``. Empty defaults are safe: ``obs_rcs2agents``
        # only touches this state when ``cfg.use_molmo`` is set.
        self._molmo_lock = threading.Lock()
        self._molmo_prompt: str = cfg.molmo_prompt
        # One-shot flag: tell the Molmo2 server to clear its rolling
        # ``prev_action_str`` on the next inference call. Set to True at
        # startup, on every ``env.reset``, and on HL-prompt updates so the
        # ``Previous action:`` prefix never leaks across episodes / re-tasks.
        # ``_snapshot_for_build`` atomically read-and-clears this flag under
        # ``_molmo_lock`` so exactly one worker call consumes it.
        self._molmo_reset_prev_action: bool = True
        # Latest Molmo2 output mirrored for the GUI + recording (all under
        # ``_molmo_lock``).
        self._molmo_last_raw: str = ""
        self._molmo_last_language: str = ""
        self._molmo_last_target_xy: tuple[float, float] | None = None
        self._molmo_last_target_point: tuple[str, float, float] | None = None
        self._molmo_last_prev_action_str: str = ""
        self._molmo_last_model_prompt: str = ""
        # ``async_molmo`` state (also under ``_molmo_lock``): one-shot
        # "the HL is stale, run it on the next obs build" flag plus the
        # cached derivation of the last HL call that every build in between
        # reuses. Pending starts True so the first build always queries.
        self._molmo_refresh_pending: bool = True
        self._molmo_cached_heatmap: np.ndarray | None = None
        self._molmo_cached_instruction: str | None = None
        self._molmo_cached_target_point: tuple[str, float, float] | None = None
        # Set when HL emits ``done``; consumed by the main loop to end the episode.
        self._molmo_done_pending: bool = False

    def _start_command_timer(self, goal: str) -> float:
        """Begin timing ``goal``; returns the monotonic start stamp."""
        self._command_started = monotonic()
        self._command_ended = None
        self._pending_verdict_goal = goal
        self._timed_goal = goal
        # A new command is gradeable again, whatever happened to the last one.
        self._release_verdict_claim()
        return self._command_started

    def _stop_command_timer(self) -> None:
        """Freeze the timer at the moment the episode stopped, once."""
        if self._command_started is not None and self._command_ended is None:
            self._command_ended = monotonic()

    @property
    def command_runtime(self) -> tuple[float, bool]:
        """``(seconds, still running)`` for the command last submitted.

        ``(0.0, False)`` before anything has been submitted. The GUI polls this
        to drive its timer, so it keeps reporting the final duration after the
        episode stops instead of resetting to zero.
        """
        start = self._command_started
        if start is None:
            return 0.0, False
        if self._command_ended is None:
            return monotonic() - start, True
        return self._command_ended - start, False

    @property
    def pending_verdict_command(self) -> str:
        """Goal awaiting a success/failure grade; ``""`` when there is none.

        Empty before the first command of a run, and again after each grade is
        written -- the GUI uses it to say whether a verdict will actually land.
        """
        return self._pending_verdict_goal

    def _record_verdict(self, goal: str, success: bool, duration_s: float) -> None:
        """Remember a grade for the GUI tape, oldest first, and count it.

        The single choke point for "a grade actually landed" -- reached only
        after the eval log line is written -- so the tally never counts a
        verdict that was dropped or refused.
        """
        self._verdict_total += 1
        self._verdict_successes += int(bool(success))
        self._verdict_history.append(
            {
                "goal": goal,
                "success": bool(success),
                "duration_s": round(duration_s, 3),
                "time": datetime.datetime.now()
                .astimezone()
                .isoformat(timespec="seconds"),
            }
        )

    @property
    def command_history(self) -> list[dict[str, Any]]:
        """Grades given so far this run, oldest first, capped at
        :data:`VERDICT_HISTORY`. Drives the GUI tape; the eval log is the record."""
        return list(self._verdict_history)

    @property
    def episode_running(self) -> bool:
        """Whether a command is executing on the robot right now."""
        return False

    @property
    def current_skill(self) -> str:
        """Skill the run timer is showing: the one running, or the last to run.

        ``""`` only before the first command of a run, where the GUI falls back
        to a generic label.
        """
        return self._timed_goal

    @property
    def verdict_tally(self) -> tuple[int, int]:
        """``(successes, graded)`` over every command graded this run.

        Uncapped, unlike :attr:`command_history`, and reset only by restarting
        the client -- a reload keeps counting, since it grades the same robot.
        """
        return self._verdict_successes, self._verdict_total

    def verify_point_goal_format(self) -> None:
        """Check point_goal_format against the loaded checkpoint, once.

        Uses the checkpoint path the server reports, so a server on this
        machine needs no configuration; a remote one resolves to nothing and
        leaves the configured format alone. Re-armed by a config reload, since
        that can point at a different checkpoint.
        """
        if self._point_goal_format_checked:
            return
        checkpoint = self.policy_checkpoint
        if not checkpoint and not self._cfg.model_config_path:
            return  # server has not answered yet; retry on the next call
        sync_point_goal_format(self._cfg, checkpoint)
        # The checkpoint the server actually loaded also decides the extra
        # cameras. Like the call above, it wins here (loudly): by this point a
        # CLI flag and a JSON value are indistinguishable, and the checkpoint
        # is the only thing that knows what the encoder expects.
        sync_frame_conditioning(self._cfg, checkpoint, sync_depth=False)
        self._point_goal_format_checked = True

    def _set_target_point(
        self,
        point: tuple[str, float, float] | None,
        *,
        src_frame_size: tuple[float, float] | None = None,
    ) -> None:
        self._target_point = point
        if point is None or not self._cfg.include_heatmap:
            self._target_heatmap = None
            self._target_heatmap_preview = None
            return
        preview = target_heatmap_from_point(
            point,
            src_frame_size,
            target_sigma=self._cfg.target_sigma,
        )
        self._target_heatmap_preview = preview
        self._target_heatmap = (
            policy_obs_target_heatmap(preview) if preview is not None else None
        )
        if self._target_heatmap is not None:
            logger.info(
                "target_heatmap ready for policy obs: shape=%s",
                self._target_heatmap.shape,
            )
        elif point is not None and self._cfg.include_heatmap:
            logger.warning("include_heatmap=true but failed to build target heatmap")
        elif self._cfg.include_heatmap:
            logger.debug("include_heatmap=true but no point selected yet")

    @property
    def target_heatmap_preview(self) -> np.ndarray | None:
        return self._target_heatmap_preview

    def capture_prev_front_rgb(self) -> None:
        """Freeze the current front frame as the policy's ``prev_front_rgb``.

        Called by the GUI when a goal point is clicked. ``_latest_frames`` holds
        the camera image exactly as the policy sees it -- the click marker and
        the heatmap overlay are drawn on copies at render time and never reach
        it. Falls back to ``None`` (i.e. the live frame) when nothing is there.
        """
        with self._frames_lock:
            frame = self._latest_frames.get(POINT_PICK_CAMERA)
            self._prev_front_rgb = None if frame is None else np.array(frame, copy=True)
        logger.info("captured %s at click", PREV_FRAME_KEY)

    @property
    def prev_front_rgb(self) -> np.ndarray | None:
        with self._frames_lock:
            return self._prev_front_rgb

    def submit_command(self, command: str) -> None:
        self._command_queue.put(command)

    def submit_verdict(self, success: bool) -> None:
        """Grade the command that ran last; applied on the inference thread."""
        self._verdict_queue.put(bool(success))

    def claim_verdict(self, success: bool) -> str:
        """Queue a grade for the pending command, at most once per command.

        Returns the goal the grade was queued against, or ``""`` when there is
        nothing to grade: no command has run yet, or this one was already
        graded -- a "Stop (success/failed)" followed by a "Submit
        (success/failed)" must not write a second line for the same skill.

        The claim is taken here, on the caller's thread, rather than where the
        inference loop drains the queue. The loop can be a tick behind, and a
        GUI that has to answer "was this graded?" immediately would otherwise
        promise a verdict that is later dropped.
        """
        with self._verdict_lock:
            goal = self.pending_verdict_command
            if not goal or self._verdict_claimed:
                return ""
            self._verdict_claimed = True
        self.submit_verdict(success)
        return goal

    def claim_amendment(self, success: bool) -> str:
        """Queue a correction of the last grade, for a mis-graded command.

        Only when nothing is running and nothing is awaiting a grade -- that is
        the state a "Stop (success/failed)" lands in once it has already graded
        the command, so pressing the other one straight after reads as "no,
        that one was the other way round". Returns the goal whose grade will be
        corrected, or ``""`` when this is not that situation: a running command
        or a pending grade belongs to :meth:`claim_verdict`, and a run that has
        graded nothing yet has nothing to correct.
        """
        with self._verdict_lock:
            if self.episode_running or self.pending_verdict_command:
                return ""
            if not self._verdict_history:
                return ""
            goal = str(self._verdict_history[-1]["goal"])
        self._amendment_queue.put(bool(success))
        return goal

    def _amend_history(self, success: bool) -> tuple[str, bool] | None:
        """Correct the last grade in the tape and the tally.

        Returns ``(goal, changed)``, or None when there is nothing to correct.
        ``changed`` is False when the grade already had that value, so callers
        can skip rewriting the log for a no-op.
        """
        if not self._verdict_history:
            return None
        last = self._verdict_history[-1]
        goal = str(last["goal"])
        if bool(last["success"]) == bool(success):
            return goal, False
        last["success"] = bool(success)
        # The command was counted once already; only which way it counted moves.
        self._verdict_successes += 1 if success else -1
        return goal, True

    def _release_verdict_claim(self) -> None:
        """Make the pending command gradeable again (new command, or a claimed
        grade that could not be written)."""
        with self._verdict_lock:
            self._verdict_claimed = False

    def submit_language_goal(
        self,
        goal: str,
        point: tuple[str, float, float] | None = None,
        *,
        auto_start: bool = True,
        record: bool = False,
    ) -> None:
        self._goal_queue.put((goal, point, auto_start, record))

    def request_shutdown(self) -> None:
        self._shutdown_requested.set()

    def get_latest_frames(self) -> dict[str, np.ndarray]:
        with self._frames_lock:
            return copy.deepcopy(self._latest_frames)

    @property
    def last_submitted_goal(self) -> str:
        return self._last_submitted_goal

    @property
    def policy_checkpoint(self) -> str:
        """Checkpoint the vlagents server loaded; ``""`` until it answers.

        Asked of the server once and cached, then retried on every read while
        the answer is still empty (server not up yet, or started from a
        ``vlagents_server`` without ``exposed_checkpoint_path``). Only an
        already-connected agent is used, so a dry run without a server never
        dials one just to fill in the GUI footer.
        """
        if not self._policy_checkpoint:
            agent = getattr(self, "remote_agent", None)
            if agent is None:
                agent = getattr(self, "_remote_agent", None)
            if agent is not None:
                self._policy_checkpoint = remote_checkpoint_path(agent)
        return self._policy_checkpoint

    @property
    def eval_log_path(self) -> str:
        """Eval log this controller appends to; ``""`` when it writes none."""
        return ""

    # ---- Molmo2 HL state (used by the interactive GUI + inference loop) -----

    @property
    def molmo_prompt(self) -> str:
        with self._molmo_lock:
            return self._molmo_prompt

    def submit_molmo_prompt(self, prompt: str, *, auto_start: bool = True) -> None:
        """Update the HL prompt fed to the Molmo2 server (called from the GUI).

        Piggy-backs on the goal queue so the inference loop consumes it on
        the next iteration like a regular language-goal submission. The
        LL instruction + heatmap are re-derived from the Molmo2 output
        every step, so no click/point is needed.
        """
        prompt = str(prompt or "").strip()
        with self._molmo_lock:
            self._molmo_prompt = prompt
        # Mutating cfg from the caller thread is safe: it's a scalar rebind.
        self._cfg.molmo_prompt = prompt
        self._goal_queue.put((f"__molmo__:{prompt}", None, auto_start, False))

    def clear_molmo_prompt(self) -> None:
        """Forget the HL prompt, so the GUI stops showing a stale one.

        Deliberately does *not* go through the goal queue the way
        :meth:`submit_molmo_prompt` does: this runs when the episode is being
        stopped, and queueing an empty ``__molmo__:`` goal would hand the
        inference loop one more thing to consume on its way down.
        """
        with self._molmo_lock:
            self._molmo_prompt = ""
        self._cfg.molmo_prompt = ""

    def _record_molmo_output(
        self,
        *,
        raw: str,
        language: str,
        target_xy: tuple[float, float] | None,
        target_point: tuple[str, float, float] | None,
        prev_action_str: str = "",
        model_prompt: str = "",
    ) -> None:
        """Publish the latest Molmo2 output under ``_molmo_lock`` (thread-safe).

        Called from ``obs_rcs2agents`` on whichever thread is currently
        building the observation (worker thread in temporal_agg mode).
        """
        with self._molmo_lock:
            self._molmo_last_raw = raw
            self._molmo_last_language = language
            self._molmo_last_target_xy = target_xy
            self._molmo_last_target_point = target_point
            self._molmo_last_prev_action_str = prev_action_str
            self._molmo_last_model_prompt = model_prompt

    def get_molmo_state(self) -> dict[str, Any]:
        """Snapshot of the latest Molmo2 HL output, for the interactive GUI."""
        with self._molmo_lock:
            return {
                "prompt": self._molmo_prompt,
                "raw": self._molmo_last_raw,
                "language": self._molmo_last_language,
                "target_xy": self._molmo_last_target_xy,
                "target_point": self._molmo_last_target_point,
                "prev_action_str": self._molmo_last_prev_action_str,
                "model_prompt": self._molmo_last_model_prompt,
            }

    def _mark_molmo_reset(self) -> None:
        """Flag the Molmo2 server to clear its ``prev_action_str`` on the
        next inference call. Called on every ``env.reset`` and HL-prompt
        change so the ``Previous action:`` prefix never crosses episodes.

        The mirrored prediction is dropped at the same time. It describes the
        episode that just ended, and the GUI paints ``target_point`` onto the
        camera panel unconditionally -- left in place it would show the last
        episode's marker and skill as if they were current, until the next
        inference happens to overwrite them. The operator's HL prompt is
        deliberately kept: that is an input, not a prediction.
        """
        with self._molmo_lock:
            self._molmo_reset_prev_action = True
            self._molmo_last_raw = ""
            self._molmo_last_language = ""
            self._molmo_last_target_xy = None
            self._molmo_last_target_point = None
            self._molmo_last_prev_action_str = ""
            self._molmo_last_model_prompt = ""
            self._molmo_refresh_pending = True
            self._molmo_cached_heatmap = None
            self._molmo_cached_instruction = None
            self._molmo_cached_target_point = None
            self._molmo_done_pending = False

    def _mark_molmo_refresh(self) -> None:
        """Arm one fresh Molmo2 HL call on the next observation build.

        Only consulted when ``cfg.async_molmo`` is set; with the flag off
        every build queries the HL anyway. Called at the low-level skill
        boundary (execution status -1), which is the point at which the
        cached skill + target pixel stop describing what the robot should
        do next. The mirrored prediction and ``prev_action_str`` are
        deliberately left alone: the finished skill is the HL's context for
        choosing the next one (unlike :meth:`_mark_molmo_reset`).
        """
        with self._molmo_lock:
            self._molmo_refresh_pending = True

    def _consume_molmo_done(self) -> bool:
        with self._molmo_lock:
            done = self._molmo_done_pending
            self._molmo_done_pending = False
            return done


class ModelInference(GoalQueueMixin):
    def __init__(self, env: gym.Env, cfg: InferenceConfig):
        self._init_goal_queue_state(cfg)
        self.env = env
        self.gripper_state = 1
        self._episode_running = False
        # Consecutive ``act`` failures; reset on every successful query.
        self._act_failures = 0
        # Eval log + recordings live in the checkpoint's folder, which is only
        # known once the vlagents server names it -> resolved lazily in the loop.
        self._eval_logger: EvalLogger | None = None
        self._eval_dir_resolved = False
        self._eval_dir_next_try = 0.0
        self._graded_command: _GradedCommand | None = None
        # (uuid, final step count) of the episode reset last, so a verdict that
        # arrives after the reset can still name the step range it graded.
        self._last_episode_end: tuple[str, int] | None = None
        self.remote_agent = RemoteAgent(
            cfg.vlagents_host,
            cfg.vlagents_port,
            cfg.vlagents_model,
            cfg.on_same_machine,
            cfg.jpeg_encoding,
        )
        self.frame_rate = SimpleFrameRate(self._cfg.fps)
        self._camera_windows_enabled = False
        # Execution status fed back as the state's 9th channel (was pd_mode):
        # 1 at episode start, then the previous chunk's last predicted value.
        self._prev_play_mode = 1.0
        self._first_frame_saved = False
        # Fast-FoundationStereo depth server client (built from cfg like the RemoteAgent above)
        self._ffstereo: FFStereoClient | None = (
            FFStereoClient.from_config(cfg) if cfg.use_ffstereo else None
        )
        # Molmo2 HL server client (mirrors the FFStereo pattern above)
        #  JSON edit + ``o`` (reload) can flip ``use_molmo`` on mid-run.
        self._molmo: Molmo2Client | None = (
            Molmo2Client.from_config(cfg) if cfg.use_molmo else None
        )
        # Pluggable per-step action producer. Kept as an attribute (not a
        # per-tick branch on cfg) so future strategies drop in without
        # touching act() / reset sites.
        self._action_source: ChunkActionSource | TemporalEnsembleActionSource = (
            self._build_action_source(cfg)
        )
        # TAM history-encoder runtime (simadaptor) + its 5 Hz latent thread.
        self.tam_runtime: Any | None = None
        self._tam_thread: threading.Thread | None = None
        self._tam_stop = threading.Event()
        if cfg.tam:
            self._init_tam()

    def _build_action_source(
        self, cfg: InferenceConfig
    ) -> ChunkActionSource | TemporalEnsembleActionSource:
        if cfg.temporal_agg:
            if cfg.n_action_steps is None or cfg.n_action_steps <= 0:
                raise ValueError(
                    "temporal_agg requires n_action_steps > 0 (chunk horizon H)."
                )
            if cfg.stop_after_action_chunk:
                raise ValueError(
                    "temporal_agg is incompatible with stop_after_action_chunk: "
                )
            if RELATIVETO != RelativeTo.NONE:
                raise ValueError(
                    "temporal_agg currently only supports "
                    "RELATIVETO=RelativeTo.NONE (absolute joint/pose targets); "
                )
            stride = cfg.n_steps_per_prediction or 1
            if stride < 1 or stride > cfg.n_action_steps:
                raise ValueError(
                    f"n_steps_per_prediction must be in [1, n_action_steps]="
                    f"[1, {cfg.n_action_steps}]; got {stride}."
                )
            return TemporalEnsembleActionSource(
                remote_agent=self.remote_agent,
                build_obs=self.obs_rcs2agents,
                robot_keys=cfg.robot_keys,
                chunk_size=cfg.n_action_steps,
                stride=stride,
                k=cfg.temporal_agg_k,
                snapshot_fn=self._snapshot_for_build,
            )
        return ChunkActionSource(
            remote_agent=self.remote_agent,
            build_obs=self.obs_rcs2agents,
            n_action_steps=cfg.n_action_steps,
            on_refill=self._maybe_reset_configured_origin,
        )

    def _snapshot_for_build(self) -> _ObsBuildSnapshot:
        """Atomically capture the controller state ``obs_rcs2agents`` needs.

        Called on the main thread from both the ``ChunkActionSource``
        obs-build path (when the caller passes ``snapshot=None``) and the
        ``TemporalEnsembleActionSource._launch`` path (before submitting to
        the worker). The Molmo ``reset_prev_action`` flag is read-and-cleared
        here under ``_molmo_lock`` so exactly one downstream ``obs_rcs2agents``
        call after each :meth:`_mark_molmo_reset` sees ``True`` and asks the
        Molmo2 server to clear its rolling ``Previous action:`` prefix.
        """
        if self._cfg.use_ffstereo and self._ffstereo is None:
            self._ffstereo = FFStereoClient.from_config(self._cfg)
        if self._cfg.use_molmo and self._molmo is None:
            self._molmo = Molmo2Client.from_config(self._cfg)
        with self._molmo_lock:
            molmo_prompt = self._molmo_prompt
            # In async mode a build that reuses the cache must not consume the
            # one-shot reset flag -- it makes no HL call, so nothing would
            # carry the reset to the server and the stale ``Previous action:``
            # prefix would survive the episode boundary.
            molmo_run = (not self._cfg.async_molmo) or self._molmo_refresh_pending
            self._molmo_refresh_pending = False
            molmo_reset = self._molmo_reset_prev_action if molmo_run else False
            if molmo_run:
                self._molmo_reset_prev_action = False
            molmo_cached = (
                self._molmo_cached_heatmap,
                self._molmo_cached_instruction,
                self._molmo_cached_target_point,
            )
        return _ObsBuildSnapshot(
            cfg=self._cfg,
            target_heatmap=self._target_heatmap,
            target_point=self._target_point,
            prev_front_rgb=self.prev_front_rgb,
            prev_play_mode=self._prev_play_mode,
            ffstereo=self._ffstereo,
            molmo=self._molmo,
            molmo_prompt=molmo_prompt,
            molmo_reset_prev_action=molmo_reset,
            molmo_run=molmo_run,
            molmo_cached=molmo_cached,
        )

    def _maybe_reset_configured_origin(self) -> None:
        """Callback fired by ChunkActionSource right after a chunk refill."""
        if RELATIVETO != RelativeTo.CONFIGURED_ORIGIN:
            return
        for robot in self.env.get_wrapper_attr("envs"):
            self.env.get_wrapper_attr("envs")[robot].get_wrapper_attr(
                "set_origin_to_current"
            )()

    # ---- TAM (Torque Adaptation Module) ------------------------------------
    # See tam_utils.py. Lifecycle (driven by vlagent_rcs_client.main):
    #   __init__ -> _init_tam (load checkpoint, verify ideal model)
    #   warmup_history_encoder (compile JAX before the robot moves)
    #   start_history_encoder (5 Hz latent thread, exports MLP weights)
    #   stop_history_encoder (on shutdown)

    def _init_tam(self) -> None:
        """Load the TAM checkpoint and check that the controller's ideal-model
        gravity uses the same MJCF as the history encoder."""
        if not tam_supported():
            logger.warning(
                "tam=true but ROBOT_INSTANCE is not HARDWARE; TAM only exists in "
                "the Franka hardware controller, ignoring"
            )
            return
        self.tam_runtime = init_tam_runtime()
        check_ideal_model_alignment(hardware_robot(self.env), self.tam_runtime)

    def warmup_history_encoder(self) -> None:
        """Compile every JAX path the live encoder will hit (blocking)."""
        if self.tam_runtime is None:
            return
        warmup_history_encoder(self.tam_runtime, self._cfg)

    def start_history_encoder(self) -> None:
        """Start the ~5 Hz history-encoder thread that feeds latents into the
        controller. Idempotent; a no-op when TAM is disabled."""
        if self.tam_runtime is None:
            logger.info("TAM disabled; history encoder not started")
            return
        if self._tam_thread is not None and self._tam_thread.is_alive():
            return
        self._tam_stop.clear()
        self._tam_thread = launch_history_encoder_thread(
            hardware_robot(self.env), self.tam_runtime, self._tam_stop
        )

    def stop_history_encoder(self, timeout: float = 2.0) -> None:
        if self._tam_thread is None:
            return
        self._tam_stop.set()
        self._tam_thread.join(timeout)
        self._tam_thread = None

    # ---- evaluation logging ------------------------------------------------

    @property
    def eval_log_path(self) -> str:
        return str(self._eval_logger.path) if self._eval_logger is not None else ""

    @property
    def episode_running(self) -> bool:
        """Overrides the mixin: this controller actually drives a robot."""
        return self._episode_running

    @property
    def pending_verdict_command(self) -> str:
        """Overrides the mixin: ``_graded_command`` is the source of truth here,
        and it survives the episode reset that a grade usually follows."""
        return self._graded_command.goal if self._graded_command is not None else ""

    def _sync_eval_outputs(self) -> None:
        """Point recordings and the eval log at the checkpoint's own folder.

        The checkpoint path arrives only once the vlagents server answers, so
        this retries (throttled) until it resolves; a run whose checkpoint never
        resolves locally keeps recording into the configured ``record_path`` and
        writes no eval log. Never moves the dataset mid-episode.
        """
        if self._eval_dir_resolved:
            return
        now = monotonic()
        if now < self._eval_dir_next_try:
            return
        self._eval_dir_next_try = now + 2.0
        out_dir = eval_output_dir(self._cfg, self.policy_checkpoint)
        if out_dir is None:
            return
        # Open the log as soon as the folder is known, so a verdict for the
        # episode already running has somewhere to go.
        if self._eval_logger is None:
            self._eval_logger = EvalLogger(out_dir)
            self._eval_logger.append(
                run_metadata(
                    self._cfg,
                    self.env,
                    policy_checkpoint=self.policy_checkpoint,
                    record_dir=out_dir,
                )
            )
            logger.info("eval log: %s", self._eval_logger.path)
        if isinstance(self.env, StorageWrapper) and str(self.env.base_dir) != str(out_dir):
            if self._episode_running:
                # Never move the dataset out from under a running episode; the
                # next tick between episodes picks it up.
                return
            logger.info("recording into the checkpoint folder: %s", out_dir)
            self.env.base_dir = str(out_dir)
        self._eval_dir_resolved = True

    def _finish_command_timing(self) -> None:
        """Freeze the run timer: the command being timed has stopped running.

        Called at the moments a command ends -- an operator stop/reload, the
        agent issuing ``done``, or a finished action chunk -- rather than from
        ``_reset_episode``, which runs after the next goal may already have been
        drained in the same loop iteration and would then zero its timer.
        Idempotent: the first call wins until the next goal restarts the timer.
        """
        self._stop_command_timer()
        command = self._graded_command
        if command is not None and command.duration_s is None:
            command.duration_s = monotonic() - command.started_monotonic

    def _drain_verdicts(self) -> None:
        """Apply grades queued by the GUI.

        Drained before commands and goals so a verdict submitted alongside a
        stop, or alongside the next goal, still lands on the command that ran.
        """
        while True:
            try:
                success = self._verdict_queue.get_nowait()
            except Empty:
                break
            self._log_verdict(success)

    def _drain_amendments(self) -> None:
        """Apply corrections to the last grade queued by the GUI.

        Drained after verdicts, so a grade and a correction arriving in the
        same tick are applied in the order they were made. The eval log is
        rewritten first: it is the record, and the tape and success rate must
        not claim a correction that never reached disk.
        """
        while True:
            try:
                success = self._amendment_queue.get_nowait()
            except Empty:
                break
            verdict = "success" if success else "failure"
            target = self._amend_history_if_changed(success)
            if target is None:
                logger.warning("nothing to correct; ignoring %s amendment", verdict)
                continue
            goal, changed = target
            if not changed:
                logger.info("%r is already recorded as %s", goal, verdict)
                continue
            logger.info("corrected the grade for %r to %s -> %s", goal, verdict,
                        self._eval_logger.path if self._eval_logger else "(no log)")

    def _amend_history_if_changed(self, success: bool) -> tuple[str, bool] | None:
        """Rewrite the logged verdict, then mirror it into the tape and tally.

        Returns what :meth:`_amend_history` returns, or None when the log could
        not be rewritten -- in which case nothing in memory is touched either,
        so the GUI keeps agreeing with the file.
        """
        if not self._verdict_history:
            return None
        last = self._verdict_history[-1]
        if bool(last["success"]) == bool(success):
            return str(last["goal"]), False
        if self._eval_logger is None:
            logger.warning(
                "cannot correct a grade: no eval log is open (the checkpoint "
                "folder does not resolve on this machine)"
            )
            return None
        if self._eval_logger.amend_last_verdict(success) is None:
            return None
        return self._amend_history(success)

    def _log_verdict(self, success: bool) -> None:
        verdict = "success" if success else "failure"
        command = self._graded_command
        if command is None:
            # Nothing has run since the last grade -- the first command of a
            # run, or a double grade. There is no skill to attribute this to,
            # so it is dropped rather than misfiled against the next command.
            logger.info("nothing to grade; ignoring %s verdict", verdict)
            return
        duration_s = (
            command.duration_s
            if command.duration_s is not None
            else monotonic() - command.started_monotonic
        )
        # A grade is operator-triggered and rare: always make a fresh attempt at
        # resolving the log location rather than waiting out the retry throttle.
        self._eval_dir_next_try = 0.0
        self._sync_eval_outputs()
        if self._eval_logger is None:
            logger.warning(
                "verdict for %r not logged: the checkpoint folder does not "
                "resolve on this machine (set model_config_path in %s)",
                command.goal,
                CONFIG_PATH.name,
            )
            # Nothing was written, so let the operator grade this command again
            # once the path resolves.
            self._release_verdict_claim()
            return

        parquet_files: list[str] = []
        step_end = command.step_start
        dataset_path: str | None = None
        if isinstance(self.env, StorageWrapper):
            dataset_path = str(self.env.base_dir)
            try:
                # Land every fragment of this episode on disk before naming them.
                self.env.flush_and_wait()
                parquet_files = self.env.episode_files(command.episode_uuid)
            except Exception:
                logger.exception(
                    "could not flush the recording before grading %r", command.goal
                )
            if self.env.episode_uuid == command.episode_uuid:
                step_end = self.env.step_cnt
            elif (
                self._last_episode_end is not None
                and self._last_episode_end[0] == command.episode_uuid
            ):
                step_end = self._last_episode_end[1]

        self._eval_logger.append(
            {
                "kind": "verdict",
                "time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                "success": bool(success),
                "command": command.goal,
                "instruction": command.instruction,
                "point": list(command.point) if command.point is not None else None,
                "submitted_at": command.submitted_at,
                "duration_s": round(duration_s, 3),
                "recording": command.recording,
                "episode": {
                    "uuid": command.episode_uuid,
                    "dataset_path": dataset_path,
                    "parquet_files": parquet_files,
                    "step_range": [command.step_start, step_end],
                },
            }
        )
        self._record_verdict(command.goal, success, duration_s)
        logger.info(
            "graded %r as %s after %.1fs -> %s",
            command.goal,
            verdict,
            duration_s,
            self._eval_logger.path,
        )
        self._graded_command = None

    def _move_robots_home(self) -> bool:
        """Drive every robot to its home pose.

        This env is built with ``home_on_reset=False`` (see ``env_setup``), so
        the button is the only way home. It mirrors what ``RobotWrapper.reset``
        used to do -- including recovering from a controller fault and retrying
        -- but bounded, so a robot that cannot home never wedges the loop.

        Runs on the inference thread (the GUI only queues the command) and only
        between episodes: homing under a running policy would fight the action
        chunk still executing. Returns whether anything moved.
        """
        if self._episode_running:
            logger.warning(
                "ignoring home request while an episode is running; stop it first ('q')"
            )
            return False
        try:
            envs = self.env.get_wrapper_attr("envs")
        except Exception:
            logger.exception("cannot reach the robot envs to home them")
            return False
        moved = False
        for name in envs:
            robot = envs[name].get_wrapper_attr("robot")
            for attempt in range(_HOME_ATTEMPTS):
                try:
                    robot.move_home()
                except Exception:
                    logger.warning(
                        "move_home failed for robot %r (attempt %d/%d)",
                        name,
                        attempt + 1,
                        _HOME_ATTEMPTS,
                        exc_info=True,
                    )
                    recover = getattr(robot, "automatic_error_recovery", None)
                    if recover is None:
                        break
                    try:
                        recover()
                    except Exception:
                        logger.exception("error recovery failed for robot %r", name)
                        break
                    sleep(0.1)
                    continue
                logger.info("robot %r moved to its home pose", name)
                moved = True
                break
            else:
                logger.error("giving up homing robot %r", name)
        return moved

    def _reset_episode(self) -> dict[str, Any]:
        """Reset the env and clear per-episode state.

        Stamps the finished episode's final step count first: ``reset()`` mints
        a new uuid and zeroes the counter, and a verdict for the episode that
        just ended can still arrive afterwards.
        """
        if isinstance(self.env, StorageWrapper):
            self._last_episode_end = (self.env.episode_uuid, self.env.step_cnt)
        obs, _ = self.env.reset()
        self._update_latest_frames(obs)
        self._show_camera_windows(obs)
        self._action_source.reset()
        self._prev_play_mode = 1.0  # new episode: first state input is 1
        # Clear Molmo2's rolling ``prev_action_str`` so the "Previous action:"
        # prefix never leaks across episodes. Centralized here so every reset
        # path gets it, the way the inline reset sequences used to.
        self._mark_molmo_reset()
        self._episode_running = False
        return obs

    def _drain_commands(self) -> tuple[bool, bool, bool, bool, bool, bool]:
        start_requested = False
        record_requested = False
        success_requested = False
        stop_requested = False
        reload_requested = False
        home_requested = False

        while True:
            try:
                command = self._command_queue.get_nowait()
            except Empty:
                break

            if command == "e":
                start_requested = True
            elif command == "r":
                record_requested = True
            elif command == "s":
                success_requested = True
            elif command == "q":
                stop_requested = True
            elif command == "o":
                reload_requested = True
            elif command == "h":
                home_requested = True

            if command in {"q", "s", "o"}:
                self._finish_command_timing()

        return (
            start_requested,
            record_requested,
            success_requested,
            stop_requested,
            reload_requested,
            home_requested,
        )

    def _drain_goal_updates(self) -> tuple[bool, bool, bool]:
        goal_updated = False
        start_requested = False
        record_requested = False

        while True:
            try:
                goal, point, auto_start, record = self._goal_queue.get_nowait()
            except Empty:
                break

            # Molmo2 HL text-box submits from the GUI ride on the same queue
            # via the ``__molmo__:`` sentinel so the loop consumes them like
            # a regular goal update. The LL instruction + heatmap are
            # re-derived from the Molmo2 output every step; here we just
            # publish the new HL prompt and ask the loop to refill.
            if isinstance(goal, str) and goal.startswith("__molmo__:"):
                new_prompt = goal[len("__molmo__:") :].strip()
                # ``submit_molmo_prompt`` already wrote through to
                # ``_molmo_prompt`` + ``_cfg.molmo_prompt`` under the lock,
                # so just reset the action buffer and the HL prev-action tracker.
                self._action_source.reset()
                # A new HL instruction is a new task: don't carry the last
                # HL output over as ``Previous action:`` (it belonged to the
                # previous goal).
                self._mark_molmo_reset()
                # A Molmo run is graded as a whole, so the HL prompt is the
                # command a verdict attaches to -- the long-horizon sequence
                # the HL drove, not any one skill inside it. Without this the
                # branch returned before ``_GradedCommand`` was ever built and
                # every Molmo grade was dropped as "nothing to grade".
                # ``point`` is None: under Molmo the target comes from the HL,
                # not from a click.
                self._graded_command = _GradedCommand(
                    goal=new_prompt,
                    instruction=new_prompt,
                    point=None,
                    episode_uuid=(
                        self.env.episode_uuid
                        if isinstance(self.env, StorageWrapper)
                        else ""
                    ),
                    step_start=(
                        self.env.step_cnt if isinstance(self.env, StorageWrapper) else 0
                    ),
                    recording=record,
                    submitted_at=datetime.datetime.now()
                    .astimezone()
                    .isoformat(timespec="seconds"),
                    started_monotonic=self._start_command_timer(new_prompt),
                )
                goal_updated = True
                logger.info("updated Molmo2 HL prompt: %s", new_prompt)
                if auto_start:
                    start_requested = True
                continue

            if not is_valid_goal_action(goal):
                logger.warning(
                    "skipping invalid goal %r; allowed actions: %s",
                    goal,
                    allowed_goal_actions_display(),
                )
                continue

            instruction = instruction_from_goal(
                self._cfg,
                goal,
                point,
                src_frame_size=src_frame_size_for_point(
                    point, self.get_latest_frames()
                ),
            )
            self._set_target_point(
                point,
                src_frame_size=src_frame_size_for_point(
                    point, self.get_latest_frames()
                ),
            )
            self._cfg.instruction = instruction
            self._last_submitted_goal = instruction
            if isinstance(self.env, StorageWrapper):
                self.env.set_instruction(instruction)
            self._graded_command = _GradedCommand(
                goal=goal,
                instruction=instruction,
                point=point,
                episode_uuid=(
                    self.env.episode_uuid
                    if isinstance(self.env, StorageWrapper)
                    else ""
                ),
                step_start=(
                    self.env.step_cnt if isinstance(self.env, StorageWrapper) else 0
                ),
                recording=record,
                submitted_at=datetime.datetime.now()
                .astimezone()
                .isoformat(timespec="seconds"),
                started_monotonic=self._start_command_timer(goal),
            )
            self._action_source.reset()
            goal_updated = True
            logger.info("updated language goal: %s", instruction)
            if auto_start:
                start_requested = True
            if record:
                record_requested = True

        return goal_updated, start_requested, record_requested

    def _poll_camera_frames(self) -> None:
        try:
            camera_set = self.env.get_wrapper_attr("camera_set")
        except Exception:
            return
        latest = read_camera_frames(camera_set)
        if latest:
            with self._frames_lock:
                # Preserve any ffstereo depth already stashed by obs_rcs2agents.
                self._latest_frames.update(latest)

    def _run_molmo_from_snapshot(
        self, obs: dict[str, Any], snap: _ObsBuildSnapshot
    ) -> tuple[np.ndarray | None, str, dict[str, Any]]:
        """Run one Molmo2 HL step and derive ManiFlow's ``target_heatmap`` +
        LL instruction. Pure w.r.t. controller state — safe to call from the
        worker thread.

        Returns ``(target_heatmap, instruction, mirror)``, where ``mirror``
        holds the fields the GUI + recording surface expose
        (``raw``, ``language``, ``target_xy``, ``target_point``,
        ``prev_action_str``, ``model_prompt``). The caller publishes
        ``mirror`` into shared state under ``_molmo_lock`` via
        :meth:`_publish_molmo`.
        """
        assert snap.molmo is not None, (
            "cfg.use_molmo=True but molmo client is None; "
            "_snapshot_for_build should have eagerly initialised it."
        )
        front_entry = obs["frames"].get("front_rgb")
        if front_entry is None or front_entry.get("rgb", {}).get("data") is None:
            raise ValueError(
                "use_molmo=true but obs['frames'] has no 'front_rgb'; "
                "Molmo2 requires the front camera."
            )
        front_rgb = np.asarray(front_entry["rgb"]["data"])
        src_h, src_w = front_rgb.shape[:2]
        src_frame_size = (float(src_w), float(src_h))

        front_view_only = bool(snap.cfg.molmo_front_view_only)
        gripper_rgb: np.ndarray | None = None
        if not front_view_only:
            gripper_entry = obs["frames"].get("gripper_rgb")
            if gripper_entry is None or gripper_entry.get("rgb", {}).get("data") is None:
                raise ValueError(
                    "use_molmo=true and molmo_front_view_only=false but "
                    "obs['frames'] has no 'gripper_rgb'; set "
                    "molmo_front_view_only=true for a front-only checkpoint."
                )
            gripper_rgb = np.asarray(gripper_entry["rgb"]["data"])

        # Send the full front frame at Molmo2's training resolution
        # (``cfg.molmo_input_size``); the server crops it per the checkpoint's
        # training config, and checks ``molmo_input_size`` /
        # ``molmo_front_crop`` / ``molmo_front_view_only`` against that config.
        # The resize is a no-op when the live frame already matches.
        molmo_w, molmo_h = (int(v) for v in snap.cfg.molmo_input_size)
        crop = snap.cfg.molmo_front_crop
        if (src_w, src_h) != (molmo_w, molmo_h):
            front_rgb_molmo_full = cv2.resize(
                front_rgb,
                (molmo_w, molmo_h),
                interpolation=cv2.INTER_AREA,
            )
        else:
            front_rgb_molmo_full = front_rgb

        result = snap.molmo.infer(
            front_rgb_molmo_full,
            snap.molmo_prompt,
            gripper_rgb=gripper_rgb,
            reset_prev_action=snap.molmo_reset_prev_action,
            use_prev_action=snap.cfg.molmo_use_prev_action,
            image_size=(molmo_w, molmo_h),
            front_crop=tuple(int(v) for v in crop) if crop is not None else None,
            front_view_only=front_view_only,
        )
        raw = str(result.get("raw") or "")
        language = str(result.get("language") or "").strip()
        if language.lower().rstrip(".").strip() == "done":
            with self._molmo_lock:
                self._molmo_done_pending = True
        target_xy = result.get("target_xy")
        target_xy_tuple: tuple[float, float] | None
        target_point: tuple[str, float, float] | None
        if target_xy is not None:
            target_xy_tuple = (float(target_xy[0]), float(target_xy[1]))
            # ``target_xy`` is an absolute pixel coordinate in the full
            # Molmo-space frame (``molmo_input_size``) for both answer
            # formats, even when the input the model saw was cropped -- the
            # legacy ``<u,v>`` is labelled in full-frame coords, and the
            # server already lifts an html-v2 0-1000 point out of the crop
            # it took (from the checkpoint's training config). So no
            # (x0, y0) shift here. Lift once from Molmo-space to the live
            # front_rgb canvas (src_w x src_h, i.e. the actual ZED
            # resolution). The GUI marker is drawn on the live frame and the
            # heatmap builder / ``action_goal`` language convert down to the
            # 640x360 ManiFlow policy space themselves via
            # ``src_frame_size``, so staying in live coords here keeps every
            # consumer consistent across Molmo input resolutions. No-op when
            # the live frame already matches ``molmo_input_size``.
            # ``target_xy_tuple`` below stays as the raw Molmo output so the
            # GUI readout matches the raw string.
            target_u = target_xy_tuple[0] * src_w / molmo_w
            target_v = target_xy_tuple[1] * src_h / molmo_h
            target_point = ("front_rgb", target_u, target_v)
        else:
            target_xy_tuple = None
            target_point = None

        # Build the target heatmap locally (no ``self.*`` mutation) using the
        # same 640x360 rescale + Gaussian the click path uses, so ManiFlow
        # sees an identical contract whether the target came from a human
        # click or the HL model.
        target_heatmap: np.ndarray | None = None
        if snap.cfg.include_heatmap and target_point is not None:
            preview = target_heatmap_from_point(
                target_point,
                src_frame_size,
                target_sigma=snap.cfg.target_sigma,
            )
            if preview is not None:
                target_heatmap = policy_obs_target_heatmap(preview)

        # Build the LL instruction in the same format as ``instruction_from_goal``:
        # rescale the front-cam pixel into the 640x360 policy space when
        # appending it to the language; fall back to ``action_only`` when
        # Molmo emits no coord or ``point_goal_format=action_only`` is set.
        if language:
            if snap.cfg.point_goal_format == "action_goal" and target_xy_tuple is not None:
                policy_uv = policy_space_uv(target_point, src_frame_size)
                if policy_uv is not None:
                    u, v = policy_uv
                    instruction = format_language_goal(language, "action_goal", u=u, v=v)
                else:
                    instruction = format_language_goal(language, "action_only")
            else:
                instruction = format_language_goal(language, "action_only")
        else:
            # Model produced no parseable language (e.g. empty output): fall
            # back to the raw string so the operator sees what happened
            # instead of a silent no-op instruction.
            instruction = raw

        mirror = {
            "raw": raw,
            "language": language,
            "target_xy": target_xy_tuple,
            "target_point": target_point,
            "prev_action_str": str(result.get("prev_action_str") or ""),
            "model_prompt": str(result.get("model_prompt") or ""),
        }
        return target_heatmap, instruction, mirror

    def _publish_molmo(self, mirror: dict[str, Any], instruction: str) -> None:
        """Publish the derived Molmo state + LL instruction to shared surfaces.

        Called from ``obs_rcs2agents`` on the obs-build thread (worker in
        temporal_agg mode). Every write is guarded by ``_molmo_lock`` so
        the GUI / ``StorageWrapper`` never see torn state.
        """
        self._record_molmo_output(**mirror)
        # ``_cfg.instruction`` and ``_last_submitted_goal`` are scalar
        # rebinds; guarded by the same lock so the GUI reader in
        # ``get_molmo_state()`` sees them consistently with the mirror.
        with self._molmo_lock:
            self._cfg.instruction = instruction
            self._last_submitted_goal = instruction
        if isinstance(self.env, StorageWrapper):
            # StorageWrapper.set_instruction is a simple attribute write;
            # calling it outside the lock is safe (independent surface).
            self.env.set_instruction(instruction)

    def _cache_molmo(
        self,
        target_heatmap: np.ndarray | None,
        instruction: str,
        target_point: tuple[str, float, float] | None,
    ) -> None:
        """Stash the derived HL outputs so ``async_molmo`` builds in between
        skill boundaries can reuse them without calling the server."""
        with self._molmo_lock:
            self._molmo_cached_heatmap = target_heatmap
            self._molmo_cached_instruction = instruction
            self._molmo_cached_target_point = target_point

    def _update_latest_frames(self, obs: dict[str, Any]) -> None:
        if not self._cfg.interactive_gui:
            return
        frames = obs.get("frames", {})
        latest: dict[str, np.ndarray] = {}
        for cam_name in INTERACTIVE_GUI_CAMERAS:
            frame_data = frames.get(cam_name)
            if frame_data is None:
                continue
            rgb = frame_data.get("rgb", {}).get("data")
            if rgb is not None:
                latest[cam_name] = camera_color_to_rgb(rgb)
            depth_entry = frame_data.get("depth", {}) or {}
            depth_data = depth_entry.get("data")
            if depth_data is not None:
                latest[f"{cam_name.removesuffix('_rgb')}_depth"] = np.asarray(depth_data)
        if latest:
            with self._frames_lock:
                # Merge (don't replace) so ffstereo depth from the previous
                # obs_rcs2agents step stays available for the next GUI refresh.
                self._latest_frames.update(latest)
        else:
            self._poll_camera_frames()

    def obs_rcs2agents(
        self,
        obs: dict,
        info: dict | None = None,
        *,
        snapshot: _ObsBuildSnapshot | None = None,
    ) -> Obs:
        # RGB rides in SingleObs.cameras (JPEG/shared-mem path); depth rides in
        # SingleObs.info["depths"] to bypass JPEG (which cannot represent uint16
        # metric depth) and bilinear resize (which corrupts depth
        # discontinuities). Depth is only sent when the client is configured
        # for it and the env actually captured it. Depth is still recorded by
        # StorageWrapper from the env obs when INCLUDE_DEPTH is enabled,
        # independent of this flag.
        #
        # `snapshot` is a frozen bag of every ``self.*`` field this method
        # reads, taken atomically on the main thread. When passed in from a
        # worker thread (temporal_agg), it makes this function race-free
        # w.r.t. concurrent cfg hot-reload, goal updates, etc. Callers on
        # the main thread can pass ``snapshot=None`` (default) and one will
        # be built here.
        snap = snapshot if snapshot is not None else self._snapshot_for_build()
        cfg = snap.cfg

        cameras = {
            frame: obs["frames"][frame]["rgb"]["data"] for frame in obs["frames"]
        }
        if cfg.include_prev_frame:
            add_prev_front_rgb(cameras, snap.prev_front_rgb)

        # Molmo2 HL runs BEFORE ffstereo so the "Molmo2 -> ffstereo -> ManiFlow"
        # pipeline flows in one thread — main thread with ``ChunkActionSource``,
        # worker thread with ``TemporalEnsembleActionSource`` (parallel to
        # the main controller loop consuming ensembled actions). The parsed
        # ``(language, target_xy)`` replaces the GUI click's
        # ``target_heatmap`` + ``instruction`` so the downstream ManiFlow
        # contract is identical whether the target came from a click or the
        # HL model.
        molmo_target_heatmap: np.ndarray | None = None
        molmo_instruction: str | None = None
        molmo_target_point: tuple[str, float, float] | None = None
        if cfg.use_molmo and snap.molmo_run:
            molmo_target_heatmap, molmo_instruction, molmo_mirror = (
                self._run_molmo_from_snapshot(obs, snap)
            )
            molmo_target_point = molmo_mirror["target_point"]
            self._publish_molmo(molmo_mirror, molmo_instruction)
            self._cache_molmo(
                molmo_target_heatmap, molmo_instruction, molmo_target_point
            )
        elif cfg.use_molmo:
            # async_molmo: the current skill is still running, so reuse the
            # last HL derivation instead of paying for a VLM call per policy
            # query. ``molmo_instruction`` stays None when nothing is cached
            # yet, which falls back to ``cfg.instruction`` below.
            (
                molmo_target_heatmap,
                molmo_instruction,
                molmo_target_point,
            ) = snap.molmo_cached

        depths: dict[str, np.ndarray] | None = None
        intrinsics: dict[str, np.ndarray] | None = None
        if cfg.use_ffstereo:
            assert snap.ffstereo is not None, (
                "cfg.use_ffstereo=True but ffstereo client is None; "
                "_snapshot_for_build should have eagerly initialised it."
            )
            depths, intrinsics = snap.ffstereo.depths_from_frames(
                obs["frames"],
                clip_m=cfg.ffstereo_clip_m,
                clip_min_m=cfg.ffstereo_clip_min_m,
            )
        elif cfg.include_depth:
            depths = {}
            intrinsics = {}
            for frame_name, frame_data in obs["frames"].items():
                depth_entry = frame_data.get("depth", {})
                depth = depth_entry.get("data")
                if depth is None:
                    continue
                depth_key = f"{frame_name.removesuffix('_rgb')}_depth"
                depth_arr = np.asarray(depth)
                if depth_arr.ndim == 3 and depth_arr.shape[-1] == 1:
                    # RCS's SimCameraSet keeps the channel axis its observation space declares;
                    # the ZED driver drops it. The server and the depth snapshot want (H, W).
                    depth_arr = depth_arr[:, :, 0]
                depths[depth_key] = depth_arr
                K = depth_entry.get("intrinsics")
                if K is None:
                    raise ValueError(
                        f"include_depth=true but camera {frame_name!r} has no "
                        f"depth.intrinsics; back-projection requires the depth "
                        f"stream's own K matrix."
                    )
                K_arr = np.asarray(K, dtype=np.float32)
                if not np.any(K_arr):
                    raise ValueError(
                        f"include_depth=true but camera {frame_name!r} has "
                        f"all-zero depth.intrinsics; back-projection would "
                        f"produce a zero pointmap. Check camera calibration."
                    )
                intrinsics[depth_key] = K_arr
            if not depths:
                depths = None
                intrinsics = None
        maybe_save_depth_outputs(depths, intrinsics, obs["frames"], "ffstereo" if cfg.use_ffstereo else "zed")

        # Feed the interactive GUI's depth panels. Merge with existing frames so
        # RGB (written by _update_latest_frames on the same step) is preserved.
        # _latest_frames is protected by _frames_lock on both threads.
        if depths and cfg.interactive_gui:
            with self._frames_lock:
                for key, depth in depths.items():
                    self._latest_frames[key] = np.asarray(depth)

        # Prefer Molmo2's per-step heatmap + LL instruction when it is on;
        # fall back to the click-path values captured in the snapshot.
        if cfg.use_molmo:
            target_heatmap = molmo_target_heatmap
            target_instruction = molmo_instruction if molmo_instruction is not None else cfg.instruction
        else:
            target_heatmap = snap.target_heatmap
            target_instruction = cfg.instruction

        # Target-centered front crop (``--use-target-front-crop``), cropped from
        # the unresized frame (RemoteAgent squashes every camera to 224x224 on
        # the way out) around whichever target the heatmap above came from.
        if cfg.use_target_front_crop >= 0:
            cameras[TARGET_CROP_KEY] = target_front_crop_from_point(
                cameras.get(POINT_PICK_CAMERA),
                molmo_target_point if cfg.use_molmo else snap.target_point,
                bottom_pct=cfg.use_target_front_crop,
            )

        obs_by_robot = {}
        for robot in cfg.robot_keys:
            robot_obs = build_vlagent_obs(
                cameras=cameras,
                joints=obs[robot]["joints"],
                gripper=obs[robot]["gripper"][0],
                instruction=target_instruction,
                robot_key=robot,
                info=dict(
                    # velo=obs[robot]["robot_state"]["dq_d"],
                    ee_xyz=obs[robot]["tquat_flange"][:3],
                    play_mode=snap.prev_play_mode,
                ),
                target_heatmap=target_heatmap,
                depths=depths,
                intrinsics=intrinsics,
            )
            obs_by_robot.update(robot_obs.obs)

        return Obs(obs=obs_by_robot, language_instruction=cfg.instruction)

    def act(self, obs: dict) -> dict[str, SingleAct]:
        return self._action_source.act(obs)

    def action_agents2rcs(self, action: dict[str, SingleAct]) -> tuple[dict[str, Any], float]:
        act = {}
        play_mode = 1.0
        for robot in self._cfg.robot_keys:
            joints, gripper, play_mode = unpack_single_act(action[robot])
            act[robot] = {
                "joints": np.asarray(joints, dtype=np.float32),
                "gripper": np.asarray([gripper], dtype=np.float32),
            }
        return act, play_mode

    def _set_camera_windows_enabled(self, enabled: bool) -> None:
        if enabled == self._camera_windows_enabled:
            return
        if not enabled:
            cv2.destroyAllWindows()
        self._camera_windows_enabled = enabled

    def _show_camera_windows(self, obs: dict[str, Any]) -> None:
        if not self._camera_windows_enabled:
            return
        frames = obs.get("frames", {})
        for frame_name, frame_data in frames.items():
            rgb_frame = frame_data.get("rgb", {}).get("data")
            if rgb_frame is None:
                continue
            cv2.imshow(
                f"camera:{frame_name}", cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            )
        cv2.waitKey(1)

    def loop(self) -> None:
        obs, _ = self.env.reset()
        first_frame_camera_set = self.env.get_wrapper_attr("camera_set")
        self._first_frame_saved = maybe_save_first_frame(
            self._cfg,
            camera_set=first_frame_camera_set,
            first_frame_saved=self._first_frame_saved,
        )
        self._update_latest_frames(obs)
        use_cv2_windows = cv2_camera_windows_enabled(self._cfg)
        self._set_camera_windows_enabled(use_cv2_windows)
        self._show_camera_windows(obs)
        # first obs_dict build to update the interactive GUI
        _ = self.obs_rcs2agents(obs)
        self.verify_point_goal_format()
        if self._cfg.interactive_gui:
            logger.info(
                "interactive GUI active — click a point on %s, enter a goal, and submit",
                POINT_PICK_CAMERA,
            )
        else:
            logger.info(
                "waiting for input: 'e' to start, 'r' to start and record, 's' for success and reset, "
                "'q' to stop and reset, 'h' to go home, and 'o' to reload config"
            )

        while not self._shutdown_requested.is_set():
            self._sync_eval_outputs()
            self._drain_verdicts()
            self._drain_amendments()
            (
                start_requested,
                record_requested,
                success_requested,
                stop_requested,
                reload_requested,
                home_requested,
            ) = self._drain_commands()
            _goal_updated, goal_start_requested, goal_record_requested = (
                self._drain_goal_updates()
            )
            start_requested = start_requested or goal_start_requested
            record_requested = record_requested or goal_record_requested
            needs_fresh_obs = _goal_updated or (
                (start_requested or record_requested) and not self._episode_running
            )
            if needs_fresh_obs:
                refresh_policy_obs(self.env, obs, self._cfg.robot_keys)
                self._update_latest_frames(obs)
            if _goal_updated:
                # Goal-paced retry: the server may not have been up (or may
                # have been swapped by a reload) when the check last ran.
                self.verify_point_goal_format()

            if reload_requested:
                self._cfg = load_inference_config()
                self._policy_checkpoint = ""
                self._point_goal_format_checked = False
                use_cv2_windows = cv2_camera_windows_enabled(self._cfg)
                self._set_camera_windows_enabled(use_cv2_windows)
                try:
                    self.remote_agent.reconnect(
                        host=self._cfg.vlagents_host,
                        port=self._cfg.vlagents_port,
                        model=self._cfg.vlagents_model,
                        on_same_machine=self._cfg.on_same_machine,
                        jpeg_encoding=self._cfg.jpeg_encoding,
                    )
                    logger.info(
                        "reloaded config from %s with host=%s port=%s model=%s",
                        CONFIG_PATH,
                        self._cfg.vlagents_host,
                        self._cfg.vlagents_port,
                        self._cfg.vlagents_model,
                    )
                except Exception:
                    logger.exception(
                        "failed to reconnect after reloading %s", CONFIG_PATH
                    )
                if isinstance(self.env, StorageWrapper):
                    self.env.base_dir = self._cfg.record_path
                    self.env.set_instruction(self._cfg.instruction)
                # A reload can name a different checkpoint: re-resolve the
                # recording folder and start a fresh eval log for it.
                self._eval_dir_resolved = False
                self._eval_dir_next_try = 0.0
                self._eval_logger = None
                obs = self._reset_episode()

            if success_requested:
                if self._episode_running:
                    logger.info("marking episode successful and resetting environment")
                self.env.get_wrapper_attr("success")()
                obs = self._reset_episode()

            if stop_requested:
                if self._episode_running:
                    logger.info("stopping episode and resetting environment")
                obs = self._reset_episode()

            if home_requested and self._move_robots_home():
                # The arm moved outside the policy loop: re-read the state the
                # next chunk will be conditioned on, and refresh the GUI frames.
                refresh_policy_obs(self.env, obs, self._cfg.robot_keys)
                self._update_latest_frames(obs)
                self._show_camera_windows(obs)

            if not self._episode_running:
                try:
                    ensure_server_ready(self.remote_agent)
                except Exception:
                    if self._cfg.interactive_gui:
                        self._poll_camera_frames()
                    sleep(0.5)
                    continue
                if start_requested or record_requested:
                    if isinstance(self.env, StorageWrapper):
                        self.env.set_instruction(self._cfg.instruction)
                        if record_requested:
                            self.env.start_record()
                    logger.info(
                        "starting episode%s",
                        " with recording" if record_requested else "",
                    )
                    # async_molmo: the cache may hold an HL call made while
                    # the robot was idle (the startup build, or a scene the
                    # operator has since changed). Query once at episode start.
                    self._mark_molmo_refresh()
                    self._episode_running = True
                else:
                    if self._cfg.interactive_gui:
                        self._poll_camera_frames()
                    sleep(0.05)
                    continue

            try:
                action = self.act(obs)
            except Exception:
                # An inference error (HL or LL RPC) used to propagate out of
                # ``loop`` and kill the controller thread outright -- the GUI
                # stayed up but the episode was dead and no further command
                # did anything. These are recoverable: drop the half-built
                # chunk, re-query the HL, and retry on the next tick. The
                # skill boundary (execution status -1) is where the HL is
                # re-queried, so that is where this fires most often.
                self._act_failures += 1
                logger.exception(
                    "inference failed (%d/%d consecutive); dropping the chunk and retrying",
                    self._act_failures,
                    _ACT_FAILURE_LIMIT,
                )
                self._action_source.reset()
                self._prev_play_mode = 1.0
                if self._cfg.use_molmo:
                    self._mark_molmo_refresh()
                refresh_policy_obs(self.env, obs, self._cfg.robot_keys)
                self._update_latest_frames(obs)
                if self._act_failures >= _ACT_FAILURE_LIMIT:
                    logger.error(
                        "giving up after %d consecutive inference failures; "
                        "stopping the episode (press start to retry)",
                        self._act_failures,
                    )
                    self._finish_command_timing()
                    self._episode_running = False
                    self._act_failures = 0
                sleep(0.5)
                continue
            self._act_failures = 0
            # Molmo2 'done' ends the episode (exec_status -1 only ends skills under use_molmo).
            if self._cfg.use_molmo and self._consume_molmo_done():
                logger.info("Molmo2 HL predicted 'done'; stopping episode")
                self._action_source.reset()
                self._prev_play_mode = 1.0
                self._finish_command_timing()
                self._episode_running = False
                refresh_policy_obs(self.env, obs, self._cfg.robot_keys)
                self._update_latest_frames(obs)
                continue
            # Execution status: -1 stops here and drops the rest of the chunk;
            # 0/1 are fed back as the next inference's state input. It rides in
            # info["play_mode"] because ManiFlowPolicy strips index 8 off the
            # action vector before sending (see vlagents_server.act).
            exec_status = min(float(a.info.get("play_mode", 1.0)) for a in action.values())
            if exec_status < -0.5:
                self._action_source.reset()
                self._prev_play_mode = 1.0  # -1 is never fed back
                if self._cfg.use_molmo:
                    # Deliberately no ``_finish_command_timing`` here: under
                    # Molmo this is a *skill* boundary inside a long-horizon
                    # run, and freezing the timer would make the logged
                    # duration measure only up to the first skill. The run's
                    # command ends at ``done``, at an operator stop/grade, or
                    # when inference gives up -- each of which freezes it.
                    # Molmo2 HL drives the task: the execute flag ends the
                    # *skill*, not the episode. Keep ``_episode_running`` set so
                    # the loop keeps calling ``act`` -- that is the only caller
                    # of ``_run_molmo_from_snapshot``, so clearing it would stop
                    # querying the HL too and leave the operator to press Start.
                    # The env is not stepped here, so re-read the observation the
                    # HL is about to look at; otherwise it would see the frame
                    # that just produced the stop and emit the same skill again.
                    # ``prev_action_str`` is deliberately not reset: the finished
                    # skill is the HL's context for choosing the next one.
                    logger.info("execution status -1; skill done, re-querying Molmo2 HL")
                    # With async_molmo the HL is only queried here, at the LL
                    # skill boundary; arm the one-shot so the next obs build
                    # runs it instead of reusing the finished skill's cache.
                    self._mark_molmo_refresh()
                    refresh_policy_obs(self.env, obs, self._cfg.robot_keys)
                    self._update_latest_frames(obs)
                    continue
                self._finish_command_timing()
                logger.info("execution status -1; stopping without moving the robot")
                self._episode_running = False
                continue
            if any(robot_action.done for robot_action in action.values()):
                logger.info("done issued by agent, resetting environment")
                self._finish_command_timing()
                obs = self._reset_episode()
                continue
            self._prev_play_mode = exec_status  # chunk's last step feeds the next obs
            a, _play_mode = self.action_agents2rcs(action)
            obs, _, _, _, info = self.env.step(a)
            self._update_latest_frames(obs)
            self._show_camera_windows(obs)

            if self._cfg.stop_after_action_chunk and self._action_source.just_finished_chunk:
                chunk_size = self._cfg.n_action_steps if self._cfg.n_action_steps is not None else 1
                logger.info(
                    "action chunk complete (%d step(s)); waiting for next command",
                    chunk_size,
                )
                self._finish_command_timing()
                self._episode_running = False
                continue


        self._set_camera_windows_enabled(False)
        self._action_source.close()
        self.stop_history_encoder()

    def play_mode(self, mode: float) -> None:
        """No-op: play_mode from the policy is ignored (see vlagent_rcs_client)."""
        return


class DryRunInference(GoalQueueMixin):
    """Interactive GUI backend without robot motion; uses cameras when available."""

    def __init__(self, cfg: InferenceConfig) -> None:
        self._init_goal_queue_state(cfg)
        self._camera_set = try_open_hardware_camera_set()
        self._using_mock_cameras = self._camera_set is None
        self._live_camera_names: set[str] = set()
        self._latest_frames = self._fallback_frames()
        self._first_frame_saved = False
        # Lazy: dry-run should start without a vlagents server (connect on goal submit).
        self._remote_agent: RemoteAgent | None = None
        # Same story for the ffstereo depth server: only connect on the first
        # frame poll that actually has an L/R pair. If the connect fails once
        # we stop retrying so the poll loop does not spam the log; `o` reload
        # clears the flag to give a restarted server a chance.
        self._ffstereo: FFStereoClient | None = None
        self._ffstereo_init_failed = False
        self._ffstereo_runtime_warned = False
        if self._using_mock_cameras:
            logger.warning("dry-run: no hardware cameras; showing mock frames")
        else:
            self._live_camera_names = set(self._camera_set.camera_names)
            self._poll_camera_frames()
            missing = [
                cam
                for cam in INTERACTIVE_GUI_CAMERAS
                if cam not in self._live_camera_names
            ]
            if missing:
                logger.info(
                    "dry-run using live cameras [%s]; mock placeholders for [%s]",
                    ", ".join(sorted(self._live_camera_names)),
                    ", ".join(missing),
                )
        self._first_frame_saved = maybe_save_first_frame(
            self._cfg,
            camera_set=self._camera_set,
            first_frame_saved=self._first_frame_saved,
        )

    def _fallback_frames(self) -> dict[str, np.ndarray]:
        return {cam: mock_camera_frame(cam) for cam in INTERACTIVE_GUI_CAMERAS}

    def _get_remote_agent(self) -> RemoteAgent:
        if self._remote_agent is None:
            self._remote_agent = RemoteAgent(
                self._cfg.vlagents_host,
                self._cfg.vlagents_port,
                self._cfg.vlagents_model,
                self._cfg.on_same_machine,
                self._cfg.jpeg_encoding,
            )
        return self._remote_agent

    def _reset_remote_agent(self) -> None:
        if self._remote_agent is None:
            return
        try:
            self._remote_agent.close()
        except Exception:
            logger.exception("failed to close dry-run remote agent")
        self._remote_agent = None

    def _get_ffstereo(self) -> FFStereoClient | None:
        """Return the ffstereo client, connecting on first use.

        Returns ``None`` when ``cfg.use_ffstereo`` is off or the connect has
        already failed this session (see ``_ffstereo_init_failed``).
        """
        if not self._cfg.use_ffstereo or self._ffstereo_init_failed:
            return None
        if self._ffstereo is None:
            try:
                self._ffstereo = FFStereoClient.from_config(self._cfg)
            except Exception:
                logger.exception(
                    "dry-run: failed to connect to ffstereo server at %s:%d; "
                    "depth panels disabled until config reload (o)",
                    self._cfg.ffstereo_host,
                    self._cfg.ffstereo_port,
                )
                self._ffstereo_init_failed = True
                return None
        return self._ffstereo

    def _reset_ffstereo(self) -> None:
        if self._ffstereo is not None:
            try:
                self._ffstereo.close()
            except Exception:
                logger.exception("failed to close dry-run ffstereo client")
        self._ffstereo = None
        self._ffstereo_init_failed = False
        self._ffstereo_runtime_warned = False

    def _rcs_style_frames(self) -> dict[str, dict[str, Any]]:
        """Return frames in the RCS obs format (``{cam: {"rgb": {"data",
        "intrinsics"}}}``) so :class:`FFStereoClient` can pair L/R and
        rescale disparity. ``read_camera_frames`` drops intrinsics and the
        ``_right`` twins, both of which the depth server needs.
        """
        if self._camera_set is None:
            return {}
        frameset = self._camera_set.get_latest_frames()
        if frameset is None:
            return {}
        out: dict[str, dict[str, Any]] = {}
        for cam_name, frame in frameset.frames.items():
            color = getattr(frame.camera, "color", None)
            if color is None or getattr(color, "data", None) is None:
                continue
            K = getattr(color, "intrinsics", None)
            out[cam_name] = {
                "rgb": {
                    "data": np.asarray(color.data),
                    "intrinsics": (
                        np.asarray(K) if K is not None else None
                    ),
                }
            }
        return out

    def close(self) -> None:
        self._reset_remote_agent()
        self._reset_ffstereo()
        if self._camera_set is not None:
            try:
                self._camera_set.close()
            except Exception:
                logger.exception("failed to close dry-run cameras")

    def _drain_commands(self) -> None:
        while True:
            try:
                command = self._command_queue.get_nowait()
            except Empty:
                break
            labels = {
                "e": "start",
                "r": "start+record",
                "q": "stop/reset",
                "s": "success/reset",
                "o": "reload config",
                "h": "go home",
            }
            if command == "o":
                self._cfg = load_inference_config()
                self._cfg.dry_run = True
                self._cfg.interactive_gui = True
                self._policy_checkpoint = ""
                self._point_goal_format_checked = False
                self._reset_remote_agent()
                # Drop the depth client too so use_ffstereo / host / port
                # toggles in the JSON take effect on the next poll tick.
                self._reset_ffstereo()
                logger.info("dry-run reloaded config from %s", CONFIG_PATH)
                continue
            if command in {"q", "s"}:
                self._stop_command_timer()
            logger.info("dry-run command: %s", labels.get(command, command))

    def _drain_verdicts(self) -> None:
        """Report and discard grades. Dry runs write no eval log, but draining
        keeps the queue bounded and rehearses the GUI's grading flow."""
        while True:
            try:
                success = self._verdict_queue.get_nowait()
            except Empty:
                break
            verdict = "success" if success else "failure"
            if not self._pending_verdict_goal:
                logger.info("dry-run: nothing to grade; ignoring %s verdict", verdict)
                continue
            duration_s, _running = self.command_runtime
            logger.info(
                "dry-run: would grade %r as %s after %.1fs",
                self._pending_verdict_goal,
                verdict,
                duration_s,
            )
            self._record_verdict(self._pending_verdict_goal, success, duration_s)
            self._pending_verdict_goal = ""

    def _drain_amendments(self) -> None:
        """Correct the last grade in the tape and tally; dry runs keep no log."""
        while True:
            try:
                success = self._amendment_queue.get_nowait()
            except Empty:
                break
            verdict = "success" if success else "failure"
            target = self._amend_history(success)
            if target is None:
                logger.warning("nothing to correct; ignoring %s amendment", verdict)
                continue
            logger.info("dry-run: corrected the grade for %r to %s", target[0], verdict)

    def _drain_goal_updates(self) -> bool:
        goal_updated = False
        while True:
            try:
                goal, point, _auto_start, record = self._goal_queue.get_nowait()
            except Empty:
                break

            # Molmo2 HL prompt update (same sentinel as ModelInference). In
            # dry-run there is no policy loop to refill, but the GUI still
            # needs the mirrored prompt to keep the text box in sync.
            if isinstance(goal, str) and goal.startswith("__molmo__:"):
                new_prompt = goal[len("__molmo__:") :].strip()
                # ``submit_molmo_prompt`` already wrote through under the lock;
                # log + set the flag for GUI-visible reset marker.
                self._mark_molmo_reset()
                goal_updated = True
                logger.info("dry-run updated Molmo2 HL prompt: %s", new_prompt)
                continue

            if not is_valid_goal_action(goal):
                logger.warning(
                    "skipping invalid goal %r; allowed actions: %s",
                    goal,
                    allowed_goal_actions_display(),
                )
                continue

            instruction = instruction_from_goal(
                self._cfg,
                goal,
                point,
                src_frame_size=src_frame_size_for_point(
                    point, self.get_latest_frames()
                ),
            )
            self._set_target_point(
                point,
                src_frame_size=src_frame_size_for_point(
                    point, self.get_latest_frames()
                ),
            )
            self._cfg.instruction = instruction
            self._last_submitted_goal = instruction
            self._start_command_timer(goal)
            goal_updated = True
            suffix = " (record requested)" if record else ""
            logger.info("dry-run submitted language goal: %s%s", instruction, suffix)
            self._ping_server()

        return goal_updated

    def _ping_server(self) -> None:
        self.verify_point_goal_format()
        try:
            self._poll_camera_frames()
            remote_agent = self._get_remote_agent()
            cameras = self.get_latest_frames()
            if self._cfg.include_prev_frame:
                add_prev_front_rgb(cameras, self.prev_front_rgb)
            if self._cfg.use_target_front_crop >= 0:
                cameras[TARGET_CROP_KEY] = target_front_crop_from_point(
                    cameras.get(POINT_PICK_CAMERA),
                    self._target_point,
                    bottom_pct=self._cfg.use_target_front_crop,
                )
            act = remote_agent_act(
                remote_agent,
                build_dry_run_obs(
                    self._cfg,
                    cameras,
                    target_heatmap=self._target_heatmap,
                ),
            )
            logger.info(
                "dry-run server ping ok (%d action step(s) returned)",
                len(act.acts),
            )
        except Exception:
            logger.warning(
                "dry-run server ping skipped (is vlagents server running?)",
                exc_info=True,
            )

    def _poll_camera_frames(self) -> None:
        if self._camera_set is None:
            return
        latest = read_camera_frames(self._camera_set)
        merged = self._fallback_frames()
        merged.update(latest)

        # ModelInference.obs_rcs2agents runs ffstereo on the RCS obs; dry-run
        # never builds a full obs, so do the same step here off the raw
        # frameset. The GUI's depth panels read the same _latest_frames keys
        # (``<cam>_depth``) whether they came from the ZED driver, the real
        # inference loop, or this call.
        client = self._get_ffstereo()
        if client is not None:
            rcs_frames = self._rcs_style_frames()
            if rcs_frames:
                try:
                    depths, _intrinsics = client.depths_from_frames(
                        rcs_frames,
                        clip_m=self._cfg.ffstereo_clip_m,
                        clip_min_m=self._cfg.ffstereo_clip_min_m,
                    )
                except Exception:
                    # Log once per session: the poll loop runs at 20 Hz and a
                    # broken server would otherwise fill the log.
                    if not self._ffstereo_runtime_warned:
                        logger.exception(
                            "dry-run: ffstereo depths_from_frames failed; "
                            "further errors will be suppressed until config "
                            "reload (o). Common causes: INCLUDE_RIGHT=False "
                            "in hardware_config.py, or the depth server died."
                        )
                        self._ffstereo_runtime_warned = True
                else:
                    for key, depth in depths.items():
                        merged[key] = np.asarray(depth)

        with self._frames_lock:
            self._latest_frames = merged

    def loop(self) -> None:
        if self._using_mock_cameras:
            logger.info(
                "dry-run mode: mock cameras, no robot. "
                "Submit goals from the GUI to test prompt formatting / server connectivity."
            )
        else:
            logger.info(
                "dry-run mode: hardware cameras active, no robot motion. "
                "Submit goals from the GUI to test prompts / server connectivity."
            )
        self.verify_point_goal_format()
        while not self._shutdown_requested.is_set():
            self._drain_verdicts()
            self._drain_amendments()
            self._drain_commands()
            self._drain_goal_updates()
            self._poll_camera_frames()
            sleep(0.05)


def command_loop(controller: ModelInference) -> None:
    prompt = (
        "Command [e=start, r=record, s=success/reset, q=stop/reset, h=go home, "
        "o=reload, x=exit]: "
    )
    while True:
        try:
            command = input(prompt).strip().lower()
        except EOFError:
            command = "x"
        except KeyboardInterrupt:
            print()
            command = "x"

        if not command:
            continue
        if command == "x":
            controller.request_shutdown()
            return
        if command in {"e", "r", "s", "q", "o", "h"}:
            controller.submit_command(command)
            continue
        logger.info("unknown command %r", command)
