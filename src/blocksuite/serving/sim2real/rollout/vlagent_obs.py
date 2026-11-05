"""Build vlagents observations and unpack policy actions."""

from __future__ import annotations

import copy
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

import gymnasium as gym
import numpy as np

from vlagents.client import RemoteAgent
from vlagents.policies.interface import Obs, SingleAct, SingleObs

from blocksuite.serving.sim2real.rollout.camera_utils import INTERACTIVE_GUI_CAMERAS, mock_camera_frame
from blocksuite.serving.sim2real.rollout.hardware_config import HOME_POSE, INCLUDE_RIGHT
from blocksuite.serving.sim2real.rollout.inference_config import InferenceConfig
from blocksuite.serving.sim2real.rollout.remote_agent_utils import remote_agent_act

logger = logging.getLogger(__name__)


def _find_wrapper(env: gym.Env | gym.Wrapper | None, wrapper_type: type) -> gym.Wrapper | None:
    cur: gym.Env | gym.Wrapper | None = env
    while cur is not None:
        if isinstance(cur, wrapper_type):
            return cur
        cur = getattr(cur, "env", None)
    return None


def _sync_camera_frames(env: gym.Env, obs: dict[str, Any]) -> None:
    from rcs.envs.base import CameraSetWrapper

    wrapper = _find_wrapper(env, CameraSetWrapper)
    if wrapper is None:
        return
    updated_obs, _ = wrapper.observation(obs, {})
    camera_key = wrapper.camera_key
    if camera_key in updated_obs:
        obs[camera_key] = updated_obs[camera_key]


def _sync_robot_state(robot_env: gym.Env, robot_obs: dict[str, Any]) -> None:
    from rcs.envs.base import GripperWrapper, RobotWrapper

    try:
        from rcs_fr3.envs import FR3HW
    except ImportError:
        FR3HW = None  # type: ignore[misc, assignment]

    if FR3HW is not None:
        fr3 = _find_wrapper(robot_env, FR3HW)
        if fr3 is not None:
            fresh = fr3.get_obs(dict(fr3.get_wrapper_attr("get_robot_obs")()))
            robot_obs.update(fresh)
            gripper = _find_wrapper(robot_env, GripperWrapper)
            if gripper is not None:
                updated, _ = gripper.observation(robot_obs, {})
                robot_obs.update(updated)
            return

    robot = _find_wrapper(robot_env, RobotWrapper)
    if robot is not None:
        robot_obs.update(robot.get_robot_obs())
    gripper = _find_wrapper(robot_env, GripperWrapper)
    if gripper is not None:
        updated, _ = gripper.observation(robot_obs, {})
        robot_obs.update(updated)


def refresh_policy_obs(
    env: gym.Env, obs: dict[str, Any], robot_keys: list[str]
) -> dict[str, Any]:
    """Refresh camera and robot readings in a cached env observation."""
    _sync_camera_frames(env, obs)
    try:
        envs = env.get_wrapper_attr("envs")
    except Exception:
        envs = None
    if isinstance(envs, dict):
        for robot_key in robot_keys:
            if robot_key not in obs or robot_key not in envs:
                continue
            _sync_robot_state(envs[robot_key], obs[robot_key])
    return obs


def normalize_robot_keys(
    cfg: InferenceConfig, available_robots: set[str]
) -> InferenceConfig:
    cfg.robot_keys = [robot for robot in cfg.robot_keys if robot in available_robots]
    if not cfg.robot_keys and "right" in available_robots:
        cfg.robot_keys = ["right"]
    return cfg


def build_vlagent_obs(
    cameras: dict[str, np.ndarray],
    joints: np.ndarray,
    gripper: float,
    instruction: str,
    robot_key: str = "right",
    info: dict | None = None,
    target_heatmap: np.ndarray | None = None,
    depths: dict[str, np.ndarray] | None = None,
    intrinsics: dict[str, np.ndarray] | None = None,
) -> Obs:
    obs_info = dict(info) if info else {}
    if target_heatmap is not None:
        obs_info["target_heatmap"] = np.asarray(target_heatmap, dtype=np.float32)
    if depths:
        obs_info["depths"] = {name: np.asarray(depth) for name, depth in depths.items()}
    if intrinsics:
        obs_info["camera_intrinsics"] = {
            name: np.asarray(K, dtype=np.float32) for name, K in intrinsics.items()
        }
    single_obs = SingleObs(
        cameras=copy.deepcopy(cameras),
        joints=np.asarray(joints, dtype=np.float32),
        gripper=float(gripper),
        info=obs_info,
    )
    return Obs(obs={robot_key: single_obs}, language_instruction=instruction)


def build_dry_run_obs(
    cfg: InferenceConfig,
    cameras: dict[str, np.ndarray] | None = None,
    *,
    target_heatmap: np.ndarray | None = None,
) -> Obs:
    robot_key = cfg.robot_keys[0]
    if cameras is None:
        cameras = {cam: mock_camera_frame(cam) for cam in INTERACTIVE_GUI_CAMERAS}
        if INCLUDE_RIGHT:
            cameras.update(
                {
                    f"{cam}_right": mock_camera_frame(f"{cam}_right")
                    for cam in INTERACTIVE_GUI_CAMERAS
                }
            )
    return build_vlagent_obs(
        cameras=cameras,
        joints=np.asarray(HOME_POSE[:7], dtype=np.float32),
        gripper=1.0,
        instruction=cfg.instruction,
        robot_key=robot_key,
        info={
            "velo": np.zeros(7, dtype=np.float32),
            "ee_xyz": np.zeros(3, dtype=np.float32),
            "play_mode": 1.0,
        },
        target_heatmap=target_heatmap,
    )


class ChunkActionSource:
    """Execute the policy's chunk in order; re-query the server when empty.
    """

    def __init__(
        self,
        remote_agent: RemoteAgent,
        build_obs: Callable[[dict], Obs],
        n_action_steps: int | None,
        on_refill: Callable[[], None] | None = None,
    ) -> None:
        self._remote_agent = remote_agent
        self._build_obs = build_obs
        self._n_action_steps = n_action_steps
        self._on_refill = on_refill or (lambda: None)
        self._buffer: list[dict[str, SingleAct]] = []

    def act(self, obs: dict) -> dict[str, SingleAct]:
        if self._n_action_steps is None:
            chunk = remote_agent_act(self._remote_agent, self._build_obs(obs)).acts
            if not chunk:
                raise ValueError("Received empty action chunk from policy")
            return chunk[0]
        if len(self._buffer) == 0:
            chunk = remote_agent_act(
                self._remote_agent, self._build_obs(obs)
            ).acts[: self._n_action_steps]
            if not chunk:
                raise ValueError("Received empty action chunk from policy")
            self._buffer.extend(chunk)
            self._on_refill()
        return self._buffer.pop(0)

    def reset(self) -> None:
        self._buffer = []

    def close(self) -> None:
        """No async resources; provided for interface symmetry with other sources."""

    @property
    def just_finished_chunk(self) -> bool:
        """True after the last chunk step has been popped (chunk boundary)."""
        return len(self._buffer) == 0


class TemporalEnsembleActionSource:
    """ACT-style temporal action ensembling.

    Queries the policy every ``stride`` executed steps and combines overlapping
    predictions across the last ``H = chunk_size`` chunks with an exponentially
    decayed weighted average (oldest overlapping chunk gets the highest
    weight; ACT convention).

    Per-robot ring buffer layout 
        action_chunks[chunk_idx, chunk_offset]  -> (action_dim + 1,) vector
                                                    packed as [action || gripper]
        chunk_start_step[chunk_idx]             -> absolute step ``q`` at which
                                                    the chunk was queried (or -1)

    Gripper is a hard majority vote across overlapping chunks (older wins ties);
    ``done`` and ``info`` are copied from the freshest overlapping prediction.

    **Async inference.** Policy calls run on a single-worker background thread
    so that the main control loop never blocks waiting for ManiFlow — the
    consumer keeps executing ensembled actions at the loop rate while the
    producer thread fetches the next chunk in parallel.
    """

    @dataclass
    class _RobotBuffer:
        action_dim: int
        action_chunks: np.ndarray  # (H, H, action_dim + 1)
        chunk_start_step: np.ndarray  # (H,) int64, -1 = empty
        raw_single_acts: list[list[SingleAct | None]]  # H x H

    def __init__(
        self,
        remote_agent: RemoteAgent,
        build_obs: Callable[..., Obs],
        robot_keys: list[str],
        chunk_size: int,
        stride: int,
        k: float,
        snapshot_fn: Callable[[], Any],
    ) -> None:
        """
        Args:
            build_obs: pure(ish) obs builder invoked as
                ``build_obs(obs, snapshot=<snapshot>) -> Obs``. Runs on the
                worker thread; must not read controller state except
                through ``snapshot``.
            snapshot_fn: main-thread factory that atomically captures
                every controller field ``build_obs`` needs. Called at
                launch time so the worker sees a coherent view.
        """
        self._remote_agent = remote_agent
        self._build_obs = build_obs
        self._snapshot_fn = snapshot_fn
        self._robot_keys = list(robot_keys)
        self._H = int(chunk_size)
        self._stride = int(stride)
        self._k = float(k)
        # Per-robot ring buffers, built lazily on the first query.
        self._buffers: dict[str, TemporalEnsembleActionSource._RobotBuffer] | None = None
        # Absolute execution step, shared across robots 
        self._t = 0
        # Single background worker: RemoteAgent isn't safe for concurrent
        # .act() calls, so we serialize inference through one thread.
        self._executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="temporal-agg-inf"
        )
        # An in-flight inference call and the ``self._t`` when it was launched.
        self._pending: Future[list[dict[str, SingleAct]]] | None = None
        self._pending_t: int | None = None

    def act(self, obs: dict) -> dict[str, SingleAct]:
        # 1) Collect any completed prior inference (non-blocking).
        if self._pending is not None and self._pending.done():
            self._collect_pending()

        # 2) Kick off a new inference if it's time and no call is in flight.
        if self._t % self._stride == 0 and self._pending is None:
            self._launch(obs)

        # 3) Cold start: on the very first tick we have nothing to ensemble
        #    yet; block on the launched future exactly once so we can execute.
        if self._buffers is None:
            assert self._pending is not None, "cold start expected a launched inference"
            self._collect_pending()  # blocks on future.result()

        assert self._buffers is not None
        out = {r: self._pop_ensembled(buf) for r, buf in self._buffers.items()}
        self._t += 1
        return out

    def reset(self) -> None:
        self._t = 0
        if self._pending is not None:
            # If the worker hasn't started yet, cancel; otherwise let it run
            # to completion and drop the result on the floor. Either way we
            # drop our reference so a fresh act() may launch a new call.
            self._pending.cancel()
            self._pending = None
            self._pending_t = None
        # Drop the buffers dict entirely (not just clear its contents). This
        # is what re-arms the cold-start branch in act(): the next tick will
        # see ``self._buffers is None``, block on the freshly launched
        # inference exactly once, and lazily rebuild the per-robot buffers
        # from that first chunk via ``_store_chunk`` — same path used on
        # the very first startup.
        self._buffers = None

    def close(self) -> None:
        """Shut down the inference worker thread. Idempotent."""
        if self._executor is None:
            return
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._executor = None
        self._pending = None
        self._pending_t = None

    @property
    def just_finished_chunk(self) -> bool:
        # Temporal aggregation queries mid-chunk; ``stop_after_action_chunk``
        # is rejected up-front when temporal_agg is on.
        return False

    # ---- internals -----------------------------------------------------------

    def _launch(self, obs: dict) -> None:
        """Submit a fresh inference call to the worker thread (non-blocking).
        """
        assert self._executor is not None, "close() was called; source is unusable"
        snap = self._snapshot_fn()  # atomic capture on main thread
        build_obs = self._build_obs
        remote_agent = self._remote_agent
        H = self._H
        self._pending_t = self._t
        self._pending = self._executor.submit(
            lambda: remote_agent_act(
                remote_agent, build_obs(obs, snapshot=snap)
            ).acts[:H]
        )

    def _collect_pending(self) -> None:
        """Drain the in-flight future and store its chunk. May block on result()."""
        assert self._pending is not None
        query_t = self._pending_t
        try:
            chunk = self._pending.result()  # non-blocking if .done(), else blocks
        except Exception:
            logger.exception(
                "temporal_agg inference launched at t=%s failed", query_t
            )
            self._pending = None
            self._pending_t = None
            raise
        self._pending = None
        self._pending_t = None
        assert query_t is not None
        self._store_chunk(chunk, query_t)

    def _store_chunk(self, chunk: list[dict[str, SingleAct]], query_t: int) -> None:
        """Fan the chunk out across per-robot ring buffers."""
        if len(chunk) < self._H:
            raise ValueError(
                f"temporal_agg needs a chunk of length >= H={self._H}, "
                f"got {len(chunk)}"
            )
        if self._buffers is None:
            self._buffers = {
                r: self._make_buffer(int(np.asarray(chunk[0][r].action).shape[0]))
                for r in self._robot_keys
            }
        for r, buf in self._buffers.items():
            self._add_chunk(buf, [step[r] for step in chunk], query_t)

    def _make_buffer(self, action_dim: int) -> _RobotBuffer:
        return TemporalEnsembleActionSource._RobotBuffer(
            action_dim=action_dim,
            action_chunks=np.zeros(
                (self._H, self._H, action_dim + 1), dtype=np.float32
            ),
            chunk_start_step=np.full(self._H, -1, dtype=np.int64),
            raw_single_acts=[[None] * self._H for _ in range(self._H)],
        )

    def _add_chunk(
        self, buf: _RobotBuffer, chunk: list[SingleAct], query_t: int
    ) -> None:
        """Store one freshly predicted chunk for a single robot.

        ``query_t`` is the absolute step at which the inference call was
        launched (may lag ``self._t`` by the inference latency).
        """
        chunk_idx = query_t % self._H
        buf.chunk_start_step[chunk_idx] = query_t
        buf.action_chunks[chunk_idx].fill(0.0)
        for i in range(self._H):
            single = chunk[i]
            action_vec = np.asarray(single.action, dtype=np.float32)
            if action_vec.shape[0] != buf.action_dim:
                raise ValueError(
                    f"temporal_agg: expected action_dim={buf.action_dim}, "
                    f"got {action_vec.shape[0]}"
                )
            buf.action_chunks[chunk_idx, i, : buf.action_dim] = action_vec
            buf.action_chunks[chunk_idx, i, buf.action_dim] = float(
                single.gripper if single.gripper is not None else 0.0
            )
            buf.raw_single_acts[chunk_idx][i] = single

    def _pop_ensembled(self, buf: _RobotBuffer) -> SingleAct:
        """Weighted-average the overlapping predictions for ``self._t`` (one robot)."""
        offsets = self._t - buf.chunk_start_step
        valid = (
            (buf.chunk_start_step >= 0)
            & (offsets >= 0)
            & (offsets < self._H)
        )
        idx = np.nonzero(valid)[0]
        if idx.size == 0:
            raise RuntimeError(
                f"temporal_agg: no valid chunks at t={self._t}"
            )
        # Sort oldest overlapping chunk first (ACT convention: oldest = highest weight).
        idx = idx[np.argsort(buf.chunk_start_step[idx])]
        offs = offsets[idx]
        preds = buf.action_chunks[idx, offs]  # (N, action_dim + 1)

        weights = np.exp(-self._k * np.arange(len(preds), dtype=np.float32))
        weights /= weights.sum()
        ensembled = (preds * weights[:, None]).sum(axis=0)  # (action_dim + 1,)

        action_vec = ensembled[: buf.action_dim].astype(np.float32)

        # Gripper: hard majority vote (older wins ties: preds is oldest-first).
        gripper_votes = preds[:, buf.action_dim] >= 0.5
        open_votes = int(gripper_votes.sum())
        closed_votes = int(len(preds) - open_votes)
        if open_votes > closed_votes:
            gripper_out: float = 1.0
        elif closed_votes > open_votes:
            gripper_out = 0.0
        else:
            gripper_out = float(preds[0, buf.action_dim])

        # done / info from the freshest overlapping chunk.
        newest_single_act = buf.raw_single_acts[idx[-1]][int(offs[-1])]
        done_out = bool(newest_single_act.done) if newest_single_act is not None else False
        info_out = dict(newest_single_act.info) if newest_single_act is not None else {}

        return SingleAct(
            action=action_vec,
            gripper=gripper_out,
            done=done_out,
            info=info_out,
        )


def unpack_single_act(single_act: SingleAct) -> tuple[np.ndarray, float, float]:
    """Split one agent ``SingleAct`` into ``(joint targets, gripper, pd_mode)``."""
    action = np.asarray(single_act.action, dtype=np.float32)
    gripper = float(single_act.gripper)
    if action.shape[0] > 8:
        pd_mode = float(action[8])
    elif action.shape[0] > 7:
        pd_mode = float(action[7])
    else:
        pd_mode = 1.0
    return action, gripper, pd_mode
