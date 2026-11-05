import json
import os
import subprocess
import time as _time_mod
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from toyblocks.tasks.base_env import BaseEnv
from toyblocks.tasks.base_env_cfg import BaseEnvCfg
from toyblocks.wrappers.flatten import FlattenObservationWrapper
from toyblocks.wrappers.record_metrics import RecordMetrics

try:
    import h5py
except ImportError:
    h5py = None


def _stack_dict_arrays(data_list: list):
    """Stack a list of (possibly nested-dict-of) arrays along a new axis 0."""
    if isinstance(data_list[0], dict):
        return {k: _stack_dict_arrays([d[k] for d in data_list]) for k in data_list[0]}
    return np.stack(data_list, axis=0)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        try:
            return super().default(o)
        except TypeError:
            # TODO (rmh) - falls back to repr(o); hides type bugs in serialized metrics
            return repr(o)


class IsaacDataRecorderBase(ABC):
    """Abstract base for Isaac Lab trajectory / episode recorders.

    Defines the lifecycle that :class:`RecordEpisode` relies on:
    ``clear`` → ``append_obs`` / ``append_step`` → ``flush_env`` → ``close``.

    Subclasses that pull data from the scene directly (e.g. LeRobot) can
    leave ``append_obs`` and ``append_step`` as no-ops.
    """

    num_envs: int

    @abstractmethod
    def clear_env_buffer(self, env_idx: int):
        """Clear / reset trajectory buffers for one environment."""
        ...

    def append_obs(self, env_idx: int, obs):  # noqa: B027
        """Append a single observation to the buffer for one environment.

        Default no-op — override in recorders that accept external observations.
        """

    def append_step(  # noqa: B027
        self,
        env_idx: int,
        action,
        reward,
        terminated,
        truncated,
        success=None,
        fail=None,
    ):
        """Append one transition (action, reward, done flags) to the buffer.

        Default no-op — override in recorders that accept external step data.
        """

    @property
    def num_steps(self) -> list[int]:
        """Number of buffered action steps per environment."""
        return [0] * self.num_envs

    @abstractmethod
    def flush_env(self, env_idx: int, save: bool = True):
        """Flush one environment's buffer to storage."""
        ...

    def flush(
        self,
        ignore_empty: bool = True,
        env_idxs=None,
        save: bool = True,
    ):
        """Flush trajectory buffers for the given (or all) environments."""
        if env_idxs is None:
            env_idxs = range(self.num_envs)
        for i in env_idxs:
            if ignore_empty and self.num_steps[i] == 0:
                continue
            self.flush_env(i, save=save)

    @abstractmethod
    def close(self, clean: bool = True):
        """Close the recorder and release any resources."""
        ...


class IsaacDataRecorderHDF5(IsaacDataRecorderBase):
    """HDF5 trajectory recorder for Isaac Lab environments.

    Manages per-environment trajectory buffers and writes completed episodes
    to an HDF5 file with a JSON metadata sidecar.  Each trajectory group
    ``traj_{id}`` contains:

    - ``obs``:  observations with shape ``[T+1, ...]`` (includes initial obs)
    - ``actions``:  shape ``[T, action_dim]``
    - ``rewards``:  shape ``[T]`` (optional)
    - ``terminated`` / ``truncated``:  boolean, shape ``[T]``
    - ``success`` / ``fail`` (optional):  boolean, shape ``[T]``

    Args:
        output_dir: Directory for output files.
        num_envs: Number of parallel environments.
        trajectory_name: Base filename (no extension). Defaults to a timestamp.
        record_reward: Include per-step rewards in trajectory data.
        env_info: Optional dict of environment metadata for the JSON sidecar.
        source_type: Short label (e.g. ``"rl"``, ``"human"``).
        source_desc: Longer description of data provenance.
    """

    def __init__(
        self,
        output_dir,
        num_envs: int,
        trajectory_name: str | None = None,
        record_reward: bool = True,
        env_info: dict | None = None,
        source_type: str | None = None,
        source_desc: str | None = None,
    ):
        if h5py is None:
            raise ImportError("h5py is required for trajectory saving: pip install h5py")

        self.num_envs = num_envs
        self.record_reward = record_reward
        self._episode_id = -1

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not trajectory_name:
            trajectory_name = _time_mod.strftime("%Y%m%d_%H%M%S")

        self._h5_file = h5py.File(output_dir / f"{trajectory_name}.h5", "w")
        self._json_path = str(self._h5_file.filename).replace(".h5", ".json")
        self._json_data: dict = dict(
            env_info=env_info,
            episodes=[],
        )
        if source_type is not None:
            self._json_data["source_type"] = source_type
        if source_desc is not None:
            self._json_data["source_desc"] = source_desc

        self._traj_obs: list[list] = [[] for _ in range(num_envs)]
        self._traj_actions: list[list] = [[] for _ in range(num_envs)]
        self._traj_rewards: list[list] = [[] for _ in range(num_envs)]
        self._traj_terminated: list[list] = [[] for _ in range(num_envs)]
        self._traj_truncated: list[list] = [[] for _ in range(num_envs)]
        self._traj_success: list[list] = [[] for _ in range(num_envs)]
        self._traj_fail: list[list] = [[] for _ in range(num_envs)]
        self.recent_save_times: list[float] = []

    def clear_env_buffer(self, env_idx: int):
        """Clear trajectory buffers for one environment."""
        self._traj_obs[env_idx] = []
        self._traj_actions[env_idx] = []
        self._traj_rewards[env_idx] = []
        self._traj_terminated[env_idx] = []
        self._traj_truncated[env_idx] = []
        self._traj_success[env_idx] = []
        self._traj_fail[env_idx] = []

    def append_obs(self, env_idx: int, obs):
        """Append a single observation to the buffer for one environment."""
        self._traj_obs[env_idx].append(obs)

    def append_step(
        self,
        env_idx: int,
        action,
        reward,
        terminated,
        truncated,
        success=None,
        fail=None,
    ):
        """Append one transition (action, reward, done flags) to the buffer."""
        self._traj_actions[env_idx].append(action)
        self._traj_rewards[env_idx].append(reward)
        self._traj_terminated[env_idx].append(terminated)
        self._traj_truncated[env_idx].append(truncated)
        if success is not None:
            self._traj_success[env_idx].append(success)
        if fail is not None:
            self._traj_fail[env_idx].append(fail)

    @property
    def num_steps(self) -> list[int]:
        """Number of buffered action steps per environment."""
        return [len(a) for a in self._traj_actions]

    def flush_env(self, env_idx: int, save: bool = True):
        """Flush one environment's buffer to the HDF5 file."""
        t0_save = _time_mod.perf_counter()
        n_steps = len(self._traj_actions[env_idx])
        if n_steps == 0 or not save:
            self.clear_env_buffer(env_idx)
            return

        self._episode_id += 1
        traj_id = f"traj_{self._episode_id}"
        group = self._h5_file.create_group(traj_id, track_order=True)

        obs_stacked = _stack_dict_arrays(self._traj_obs[env_idx])
        self._write_dict_to_hdf5(group, "obs", obs_stacked)

        actions_stacked = _stack_dict_arrays(self._traj_actions[env_idx])
        if isinstance(actions_stacked, dict):
            self._write_dict_to_hdf5(group, "actions", actions_stacked)
        else:
            group.create_dataset("actions", data=actions_stacked, dtype=np.float32)

        if self.record_reward:
            group.create_dataset(
                "rewards",
                data=np.array(self._traj_rewards[env_idx], dtype=np.float32),
            )

        terminated_arr = np.array(self._traj_terminated[env_idx], dtype=bool)
        truncated_arr = np.array(self._traj_truncated[env_idx], dtype=bool)
        group.create_dataset("terminated", data=terminated_arr, dtype=bool)
        group.create_dataset("truncated", data=truncated_arr, dtype=bool)

        episode_info: dict = dict(
            episode_id=self._episode_id,
            elapsed_steps=n_steps,
            env_idx=int(env_idx),
        )

        if self._traj_success[env_idx]:
            success_arr = np.array(self._traj_success[env_idx], dtype=bool)
            group.create_dataset("success", data=success_arr, dtype=bool)
            episode_info["success"] = bool(success_arr[-1])

        if self._traj_fail[env_idx]:
            fail_arr = np.array(self._traj_fail[env_idx], dtype=bool)
            group.create_dataset("fail", data=fail_arr, dtype=bool)
            episode_info["fail"] = bool(fail_arr[-1])

        self._json_data["episodes"].append(episode_info)
        self._dump_json(self._json_path, self._json_data, indent=2)
        self.clear_env_buffer(env_idx)

        save_time = _time_mod.perf_counter() - t0_save
        self.recent_save_times.append(save_time)

    def flush(
        self,
        ignore_empty: bool = True,
        env_idxs=None,
        save: bool = True,
    ):
        """Flush trajectory buffers for the given (or all) environments."""
        if env_idxs is None:
            env_idxs = range(self.num_envs)
        for i in env_idxs:
            if ignore_empty and len(self._traj_actions[i]) == 0:
                continue
            self.flush_env(i, save=save)

    def clean(self):
        """Remove empty trajectories and renumber them to be consecutive."""
        episodes = self._json_data["episodes"]
        prefix_len = len("traj_")
        ep_ids = sorted(int(k[prefix_len:]) for k in self._h5_file)

        new_episodes: list[dict] = []
        new_id = 0
        for i, ep_id in enumerate(ep_ids):
            traj_id = f"traj_{ep_id}"
            ep = episodes[i]
            if ep["elapsed_steps"] == 0:
                del self._h5_file[traj_id]
                continue
            new_traj_id = f"traj_{new_id}"
            if new_traj_id != traj_id:
                self._h5_file[new_traj_id] = self._h5_file[traj_id]
                del self._h5_file[traj_id]
                ep["episode_id"] = new_id
            new_episodes.append(ep)
            new_id += 1

        self._json_data["episodes"] = new_episodes
        self._dump_json(self._json_path, self._json_data, indent=2)

    def close(self, clean: bool = True):
        """Close the HDF5 file, optionally cleaning empty trajectories first."""
        if clean:
            self.clean()
        self._h5_file.close()

    @staticmethod
    def _write_dict_to_hdf5(group, key: str, data) -> None:
        """Recursively write a nested-dict-or-array into an HDF5 group."""
        if isinstance(data, dict):
            subgrp = group.create_group(key, track_order=True)
            for k, v in data.items():
                IsaacDataRecorderHDF5._write_dict_to_hdf5(subgrp, k, v)
        else:
            kwargs = {}
            if key in ("rgb", "depth", "seg"):
                kwargs = dict(compression="gzip", compression_opts=5)
            group.create_dataset(key, data=data, dtype=data.dtype, **kwargs)

    @staticmethod
    def _dump_json(path: str, data: dict, **kwargs) -> None:
        with open(path, "w") as f:
            json.dump(data, f, cls=_NumpyEncoder, **kwargs)


# The one LeRobot recorder (datagen and eval), see toyblocks.vecgen.recording.
from toyblocks.vecgen.recording.lerobot_recorder import IsaacDataRecorderLeRobot  # noqa: E402

# ---------------------------------------------------------------------------
# Evaluation script helpers (used by ``scripts/evaluate.py``)
# ---------------------------------------------------------------------------


def _build_run_name(args: Any) -> str:
    if args.exp_name is not None:
        return args.exp_name
    return f"{args.task}__eval__{args.seed if args.seed is not None else 'none'}__{int(_time_mod.time())}"


def _env_control_hz(base_env: BaseEnv) -> int:
    """Control (policy) steps per second: 1 / (decimation × sim.dt)."""
    cfg = base_env.cfg
    period_s = float(cfg.decimation) * float(cfg.sim.dt)
    return max(1, int(round(1.0 / period_s)))


def _validated_video_env_ids(num_envs: int, ids: tuple[int, ...]) -> tuple[int, ...]:
    """Return ``ids`` with duplicates removed (order preserved); validate range."""
    seen: set[int] = set()
    out: list[int] = []
    for e in ids:
        if e < 0 or e >= num_envs:
            raise ValueError(
                f"video_env_ids contains {e}, which is out of range for num_envs={num_envs} "
                "(valid indices are 0 .. num_envs-1)."
            )
        if e not in seen:
            seen.add(e)
            out.append(e)
    if not out:
        raise ValueError("video_env_ids must contain at least one environment index.")
    return tuple(out)


def _is_json_serializable(v: object) -> bool:
    try:
        json.dumps(v)
        return True
    except (TypeError, ValueError):
        return False


def _frame_all_envs_in_viewport(base_env: BaseEnv) -> None:
    """Point the sim viewport camera at the whole grid of parallel envs.

    Uses the actual ``scene.env_origins`` to compute a world-anchored,
    zoomed-out diagonal view so every env is visible in the ``rgb_array``
    render (the default viewer is anchored to a single env).
    """
    vcc = getattr(base_env, "viewport_camera_controller", None)
    if vcc is None:
        return
    try:
        origins = base_env.scene.env_origins.detach().cpu()
        mins = origins.min(dim=0).values
        maxs = origins.max(dim=0).values
        center = (mins + maxs) / 2.0
        span = float((maxs - mins).max().item())
        span = max(span, float(getattr(base_env.scene.cfg, "env_spacing", 2.0)))
        # Distance chosen empirically to keep the full grid in frame.
        dist = 0.1 * span + 1.0
        eye = (
            float(center[0].item()) + dist,
            float(center[1].item()) + dist,
            float(center[2].item()) + 0.8 * dist,
        )
        lookat = (
            float(center[0].item()) - 2,
            float(center[1].item()) - 2,
            float(center[2].item()),
        )
        vcc.update_view_to_world()
        vcc.update_view_location(eye=eye, lookat=lookat)
    except Exception as exc:  # pragma: no cover - viewer is best-effort
        print(f"[WARN]: could not reframe viewport to show all envs: {exc}")


def _make_env(
    args: Any,
    output_dir: str,
    on_video_saved: Callable[[int, int, Path], None] | None = None,
) -> tuple[gym.Env, BaseEnv, int]:
    """Create and wrap the Isaac Lab environment for evaluation.

    ``on_video_saved`` is forwarded to :class:`RecordEpisode`.
    """
    # Lazy import: ``record_episode`` imports this module; importing it at
    # module load would create a circular dependency.
    from toyblocks.wrappers.record_episode import RecordEpisode

    env_cfg: BaseEnvCfg = args.env_cfg
    env_cfg.sim.device = args.device
    env_cfg.sim.use_fabric = not args.disable_fabric
    env_cfg.scene.num_envs = args.num_envs
    if getattr(args, "env_spacing", None) is not None:
        env_cfg.scene.env_spacing = args.env_spacing
    env_cfg.obs_mode = args.obs_mode
    if args.seed is not None:
        env_cfg.seed = args.seed
    if args.deterministic_cuda:
        env_cfg.sim.physx.enable_enhanced_determinism = True
        env_cfg.sim.physx.gpu_max_num_partitions = 1
    if args.control_mode is not None:
        env_cfg.control_mode = args.control_mode

    print(
        f"[INFO]: control_mode={env_cfg.control_mode!r} "
        f"({'--control_mode' if args.control_mode is not None else 'env cfg default'})",
        flush=True,
    )

    # A policy eval only renders the cameras its obs transform reads (checked
    # against the transform in _make_policy_client); replay and transform-less
    # runs keep every camera.
    if args.obs_transform != "none" and not getattr(args, "replay", False):
        drop = tuple(getattr(args, "drop_cameras", ()))
        for flag, camera in (
            ("--video_camera", args.video_camera),
            ("--hl_overlay_camera", getattr(args, "hl_overlay_camera", None)),
        ):
            if camera is not None and camera in drop:
                raise ValueError(
                    f"{flag} {camera!r} is in --drop_cameras {drop}; drop a different "
                    "camera set or pick another camera."
                )
        env_cfg.drop_cameras = drop
        if drop:
            print(f"[INFO]: cameras not spawned for this eval: {list(drop)}", flush=True)

    env_cfg.track_joint_metrics = args.track_joint_metrics or args.track_metrics
    env_cfg.track_ee_metrics = args.track_ee_metrics or args.track_metrics
    render_mode = None

    # --use_render_params: the render's build-time scene config has to land before
    # the sim exists; the per-episode records are applied by apply_setup at each reset.
    render_replay = None
    if getattr(args, "use_render_params", False) and getattr(args, "dataset_path", None):
        from toyblocks.eval.render_replay import RecordedRenderReplay

        render_replay = RecordedRenderReplay.from_dataset(
            args.dataset_path, assets_paths=args.render_replay_assets_paths
        )
        if render_replay is None:
            print(
                f"[INFO]: render replay off: {args.dataset_path} has no render_params.json",
                flush=True,
            )
        else:
            render_replay.configure_build(env_cfg)
            env_cfg.num_renders_on_reset = max(
                int(env_cfg.num_renders_on_reset), int(args.render_replay_warmup_frames)
            )

    env = gym.make(id=args.task, cfg=env_cfg, render_mode=render_mode)
    base_env: BaseEnv = env.unwrapped
    # Read by episode_queue.apply_setup: paint each setup's recorded glyphs/colours.
    base_env.replay_block_identity = bool(getattr(args, "replay_block_identity", False))

    if render_replay is not None:
        render_replay.bind(base_env)
        base_env.render_replay = render_replay
        base_env.render_replay_wave = bool(getattr(args, "wave_resets", False))
        print(
            "[INFO]: render replay on: recorded camera pose + appearance per episode "
            f"(warm-up {env_cfg.num_renders_on_reset} renders, "
            f"wave_resets={base_env.render_replay_wave})",
            flush=True,
        )
        if args.num_envs > 1 and not base_env.render_replay_wave:
            print(
                "[WARN]: render replay with --num_envs > 1 and no --wave_resets: the "
                "recorded sky, dome, sun and exposure are set once at the first reset "
                "(from the lowest env with a record) and never changed; only per-env parts follow "
                "each episode.",
                flush=True,
            )

    # change camera to see all envs in parallel
    if render_mode == "rgb_array" and args.num_envs > 1:
        _frame_all_envs_in_viewport(base_env)

    video_fps = args.video_fps if args.video_fps is not None else _env_control_hz(base_env)

    if args.obs_mode == "state":
        # The policy-server protocol sends a dict of named observations; obs_mode "state"
        # flattens them into a single tensor, so the two cannot be combined.
        if getattr(args, "policy_host", None):
            raise ValueError(
                "--obs_mode state cannot be used with a policy server (--policy_host): the "
                "server protocol needs a dict of named observations, and --obs_mode state "
                "flattens them into one tensor. Pass --obs_mode rgb+state (add "
                "--obs_transform none if your policy wants the raw keys)."
            )
        env = FlattenObservationWrapper(env)

    any_tracking = (
        args.track_metrics
        or args.track_joint_metrics
        or args.track_ee_metrics
    )
    # Per-step CSVs are opt-in (--save_step_metrics); the episode summaries the
    # tracking flags add come from RecordMetrics' accumulators either way.
    save_step_csvs = any_tracking and getattr(args, "save_step_metrics", False)
    env = RecordMetrics(env, output_dir=output_dir if save_step_csvs else None)

    if args.save_sensor_obs_video and "rgb" not in args.obs_mode:
        raise ValueError(
            "--save_sensor_obs_video requires --obs_mode to include 'rgb' "
            f"(got obs_mode={args.obs_mode!r})."
        )

    any_video = args.save_sensor_obs_video
    need_recorder = any_video or args.save_lerobot
    if need_recorder:
        video_env_ids_for_recorder: tuple[int, ...] | None = None
        if any_video:
            video_env_ids_for_recorder = _validated_video_env_ids(args.num_envs, args.video_env_ids)

        max_steps_per_video = args.max_steps_per_video
        if max_steps_per_video is None and args.num_envs > 1:
            max_steps_per_video = base_env.max_episode_length

        data_recorder = None
        if args.save_lerobot:
            data_recorder = _make_lerobot_recorder(args, base_env, output_dir, video_fps)

        env = RecordEpisode(
            env,
            output_dir=output_dir,
            save_trajectory=args.save_lerobot,
            save_rendering_video=False,
            save_sensor_obs_video=args.save_sensor_obs_video,
            video_camera=args.video_camera if args.save_sensor_obs_video else None,
            max_steps_per_video=max_steps_per_video,
            video_fps=video_fps,
            data_recorder=data_recorder,
            video_env_ids=video_env_ids_for_recorder,
            on_video_saved=on_video_saved if args.save_sensor_obs_video else None,
            hl_overlay_camera=args.hl_overlay_camera,
            save_hl_latency_video=hl_latency_video_enabled(args),
        )

    return env, base_env, video_fps


def _make_lerobot_recorder(
    args: Any, base_env: Any, output_dir: str | Path, video_fps: int
) -> IsaacDataRecorderLeRobot:
    """The eval's LeRobot recorder: the datagen recorder with the eval extras switched on.

    Rows carry the datagen schema (FK'd ``action.ee_target`` included) plus the raw policy
    action, the learned HL's output (``--hl_from_model``) and the live PhysX masses/materials;
    each env's scene-profile columns come from the profile it was spawned with.
    """
    from toyblocks.robots.base_robot import BaseRobotCfg
    from toyblocks.vecgen.recording.articulation_fk import ArticulationEEFK

    scene = base_env.scene
    robot_cfg = getattr(base_env, "_robot_cfg", None)
    # The recorded EE pose has to come from the same link the env's ``ee_pos`` observation
    # does, per BaseRobotCfg.ee_body_name.
    ee_body_name = getattr(robot_cfg, "ee_body_name", None) or "robotiq_arg2f_base_link"
    # FK chain for action.ee_target, from the robot config (class defaults without one).
    arm_joint_names = getattr(robot_cfg, "arm_joint_names", BaseRobotCfg.arm_joint_names)
    arm_link_names = getattr(robot_cfg, "arm_link_names", BaseRobotCfg.arm_link_names)
    robot_id = str(getattr(base_env.cfg, "robot_id", "") or "")
    articulation = next(iter(scene.articulations.values()))
    action_dim = base_env.action_space.shape[-1] if hasattr(base_env, "action_space") else None
    recorder = IsaacDataRecorderLeRobot(
        scene,
        data_dir=str(Path(output_dir) / "lerobot_dataset"),
        dataset_name=args.lerobot_dataset_name or _build_run_name(args),
        num_envs=args.num_envs,
        record_every_n_steps=1,
        fps=video_fps,
        ee_body_name=ee_body_name,
        robotiq="robotiq" in robot_id,
        ee_target_fk=ArticulationEEFK(articulation, ee_body_name, arm_joint_names, arm_link_names),
        enable_sharding=False,
        num_writers=int(args.num_shards),
        action_dim=action_dim,
        record_hl_model=bool(getattr(args, "hl_from_model", False)),
        query_physx_properties=True,
        rgb_codec=str(args.video_codec),
        hl_overlay_camera=args.hl_overlay_camera if hl_overlay_enabled(args) else None,
        hl_overlay_side_by_side=bool(getattr(args, "oracle_comparison", False)),
    )
    scene_id = str(getattr(base_env.cfg, "scene_preset_id", "") or "")
    for env_idx in range(args.num_envs):
        profile = base_env.env_scene_profile(env_idx) if hasattr(base_env, "env_scene_profile") else None
        if profile is None:
            recorder.set_scene_params(env_idx=env_idx, scene_id=scene_id, robot_id=robot_id)
            continue
        recorder.set_scale_properties(
            {name: tuple(size) for name, size in profile["block_sizes"].items()}, env_idx=env_idx
        )
        recorder.set_scene_params(
            cube_scale=float(next(iter(profile["block_sizes"].values()))[0]),
            table_height_delta=float(profile["table_height"]),
            table_friction=float(profile["static_friction"]),
            profile_id=int(profile["profile_id"]),
            scene_id=scene_id,
            robot_id=robot_id,
            env_idx=env_idx,
        )
    return recorder


def hl_overlay_enabled(args: Any) -> bool:
    """Whether the LeRobot recorder writes the HL overlay video for this run.

    Recorded as ``observation.images.<hl_overlay_camera>_hl_overlay`` in the
    ``--save_lerobot`` dataset, so it requires that, and an HL-conditioning
    source available next to the recorded frames: either
    the client-side oracle HL controller (``--use_oracle``), which injects
    ``target_heatmap`` + ``oracle_task_name`` into the obs, or the learned HL
    model on the server (``--hl_from_model``), whose HL is pulled back from the
    server ``hl_model_*`` aux into the obs each step; its Molmo2 previous action
    and prompt are drawn on the same frame. ``pi`` (neither flag) stays
    disabled, so no redundant videos.
    """
    return bool(
        getattr(args, "save_hl_overlay_video", False)
        and getattr(args, "save_lerobot", False)
        and (getattr(args, "use_oracle", False) or getattr(args, "hl_from_model", False))
    )


def hl_overlay_video_key(args: Any) -> str | None:
    """LeRobot video key of the HL overlay stream, or ``None`` when it is not recorded."""
    if not hl_overlay_enabled(args):
        return None
    return f"observation.images.{args.hl_overlay_camera}_hl_overlay"


def validate_hl_latency_video_args(args: Any) -> None:
    """Reject ``--save_hl_latency_video`` without ``--save_sensor_obs_video``.

    The latency video reuses the sensor-obs per-episode frame buffers, so asking
    for it alone would silently record nothing. Called once from the eval
    entrypoint's up-front argument validation, which keeps
    :func:`hl_latency_video_enabled` a pure query -- a predicate that raises is a
    trap for any caller that reads it inside an expression.
    """
    if getattr(args, "save_hl_latency_video", False) and not args.save_sensor_obs_video:
        raise ValueError(
            "--save_hl_latency_video requires --save_sensor_obs_video "
            "(shares its per-episode frame buffers)."
        )


def hl_latency_video_enabled(args: Any) -> bool:
    """Whether to record the side-by-side HL-latency video for this run.

    Pure query; :func:`validate_hl_latency_video_args` enforces the flag
    combination. The fresh/latent conditioning is injected into the recorded obs
    by the eval loop (client oracle) or pulled from the server aux (``--hl
    model``), so there is no extra gating on ``use_oracle`` here.
    """
    return bool(getattr(args, "save_hl_latency_video", False))


def server_reads_prev_front_frame(policy: Any) -> bool:
    """Whether the served checkpoint declares the HL-aged ``prev_front_rgb`` input.

    Read off the connected client's metadata, which is also what configures the
    obs transform to emit the key (see :func:`_make_policy_client`). This is what
    keeps the client-side HL-latency buffer alive for such a checkpoint even at
    zero latency, since that buffer is the only thing that produces the frame.
    """
    if policy is None:
        return False
    meta = policy.metadata if hasattr(policy, "metadata") else {}
    return "prev_front_rgb" in {str(k) for k in meta.get("obs_keys") or ()}


def _check_prev_front_frame_source(args: Any, obs_keys: list[str], front_camera: str) -> None:
    """Check the run can produce the LL's ``prev_front_rgb``, and that it is aged.

    The frame is the HL's own (stale) input observation, so it only exists where
    an HL is being ticked: the client-side buffer under ``--use_oracle``, or the
    policy server's buffer under ``--hl model``. With neither, the run would reach
    the first inference and be rejected there for a missing obs key, with no hint
    that the HL source is the cause -- so that is fatal.

    ``M`` (``--estimated_hl_inference_time_ms``) is the only knob that puts a
    floor under the conditioning age, so ``M <= 0`` is *warned about* rather than
    rejected: it is a deliberate configuration (a synchronous HL, where every
    commit lands on the same step as an inference) as well as a likely mistake.
    Without M the age is bounded below by nothing, and it is systematically
    *zero* whenever the commit cadence lines up with the action chunk: the HL
    tick precedes ``get_action`` in the same iteration, so every inference
    selects the entry committed microseconds earlier and
    ``prev_front_rgb == front_rgb`` bit-for-bit. Training always paired the frame
    with one ``n_action_steps * U{1,2,3,4}`` steps back, so that is
    off-distribution.

    The client-side buffer caches one frame per commit and that single frame
    serves both the overlay panel and ``prev_front_rgb``, so
    ``--hl_overlay_camera`` has to name the transform's front camera as well.
    """
    if "prev_front_rgb" not in obs_keys or args.hl_from_model:
        return
    if not args.use_oracle:
        raise ValueError(
            "the served checkpoint consumes 'prev_front_rgb' (it was trained with "
            "use_previous_front_frame), which is the HL's own input frame. Nothing "
            "in this run ticks an HL to produce one: pass --hl_from_model to let "
            "the policy server age it, or --use_oracle to age it client-side."
        )
    if args.hl_overlay_camera != front_camera:
        raise ValueError(
            f"the served checkpoint consumes 'prev_front_rgb', which the HL-latency "
            f"buffer caches from --hl_overlay_camera={args.hl_overlay_camera!r}, but "
            f"obs_transform {args.obs_transform!r} feeds front_rgb from "
            f"{front_camera!r}. The aged frame would be a different view than the "
            f"current one; pass --hl_overlay_camera {front_camera}."
        )
    if args.estimated_hl_inference_time_ms <= 0:
        print(
            "[WARNING]: the served checkpoint consumes 'prev_front_rgb' but this run "
            f"has --estimated_hl_inference_time_ms={args.estimated_hl_inference_time_ms}, "
            "which is the only floor on the conditioning age. Without it the HL "
            "commit that lands just before an inference is the one selected, so "
            "age at every chunk boundary = 0 ms (prev_front_rgb == front_rgb, "
            "bit-for-bit) and the policy runs off-distribution (training always "
            "offset it by n_action_steps * U{1,2,3,4} steps). If that zero-latency "
            "HL is not what you meant, set --estimated_hl_inference_time_ms ~= "
            "n_action_steps * step_dt * 1000 (n_action_steps is in the checkpoint's "
            "config.yaml) and size --hl_update_interval_ms so the age reaches ~4x "
            "that.",
            flush=True,
        )


def _connect_policy_server(args: Any):
    """Open the eval's single connection to the remote policy server.

    Called before the env is built so the checkpoint's advertised inputs can
    decide the env's ``obs_mode`` (see :func:`_reconcile_obs_mode_with_server`);
    the same connected client is then handed to :func:`_make_policy_client`.
    Blocks until the server accepts, and reads its metadata on connect.
    """
    from toyblocks.serving import WebsocketClientPolicy

    return WebsocketClientPolicy(host=args.policy_host, port=args.policy_port)


def _reconcile_obs_mode_with_server(args: Any, server_meta: dict[str, Any]) -> None:
    """Add ``depth`` to ``args.obs_mode`` when the served checkpoint reads depth.

    Which modalities the LL consumes is a property of the checkpoint, not of the
    eval invocation, so the env follows the model here rather than a hand-typed
    ``--obs_mode`` component that can disagree with it (which surfaces as a
    ``KeyError`` at the first inference step).

    ``rgb`` cannot be added the same way: ``--enable_cameras`` is derived from
    raw argv before ``launch_app()``, long before any server is reachable.
    """
    obs_keys = [str(k) for k in server_meta.get("obs_keys") or ()]
    if not any(k.endswith("_depth") for k in obs_keys):
        return
    parts = args.obs_mode.split("+")
    if "rgb" not in parts:
        raise ValueError(
            f"the served checkpoint reads depth (obs_keys={sorted(obs_keys)}) but "
            f"--obs_mode={args.obs_mode!r} has no 'rgb', and depth is emitted "
            "alongside rgb under observations['images'][cam]. Cameras are enabled "
            "from argv before the sim starts, so this cannot be negotiated here: "
            "pass --obs_mode rgb+state."
        )
    if "depth" not in parts:
        args.obs_mode = "+".join([*parts, "depth"])
        print(
            f"[INFO]: served checkpoint reads depth; obs_mode -> {args.obs_mode!r}",
            flush=True,
        )


def _make_policy_client(args: Any, prompt_resolver: Any, policy: Any):
    """Wrap the connected policy server client in the eval adapter.

    ``policy`` is the client from :func:`_connect_policy_server`, already
    connected (and already used to negotiate ``obs_mode``), so the eval holds
    exactly one connection. Maniflow obs transforms read
    ``prompt_resolver.prompt_for_env`` at every infer call so each env gets its
    own queue-derived goal; other prompt transforms use ``current_prompt``.
    """
    from toyblocks.serving import PolicyClientAdapter
    from toyblocks.serving.action_chunk_broker import ActionChunkBroker, PerEnvActionChunkBroker
    from toyblocks.serving.obs_transforms import ObsTransformPolicy, get_obs_transform_registry
    from toyblocks.serving.maniflow_serving_utils import crop_eval_active
    from toyblocks.serving.obs_transforms.maniflow import (
        ManiflowObsTransform,
        transform_kwargs_from_server_metadata,
    )

    if args.obs_transform != "none":
        registry = get_obs_transform_registry()
        if args.obs_transform not in registry:
            available = ", ".join(sorted(registry.keys()))
            raise ValueError(
                f"Unknown obs_transform {args.obs_transform!r}. Available: {available}"
            )
        prompt_transforms = (
            "maniflow",
            "maniflow_blocks_table",
        )
        if args.obs_transform in prompt_transforms and not getattr(args, "queue_url", None):
            raise ValueError(
                f"obs_transform {args.obs_transform!r} requires a prompt source. "
                f"Pass --queue_url to provide per-episode prompts via the queue."
            )
        prompt_source = prompt_resolver.current_prompt
        server_meta = policy.metadata if hasattr(policy, "metadata") else {}
        state_dim = server_meta.get("state_dim")
        maniflow_transforms = ("maniflow", "maniflow_blocks_table")
        if args.obs_transform in prompt_transforms:
            if args.obs_transform in maniflow_transforms:
                kwargs: dict[str, Any] = {"per_env_prompt": prompt_resolver.prompt_for_env}
            else:
                kwargs = {"prompt": prompt_source}
            advertised = server_meta.get("obs_keys")
            obs_keys = None if advertised is None else [str(k) for k in advertised]
            if args.obs_transform in maniflow_transforms:
                kwargs.update(transform_kwargs_from_server_metadata(server_meta))
                if getattr(args, "center_crop_ratio", None) is not None:
                    kwargs["center_crop_ratio"] = args.center_crop_ratio
                if getattr(args, "depth_aug", None) is not None:
                    from toyblocks.serving.maniflow_serving_utils import require_depth_aug_supported

                    aug_name = str(args.depth_aug).strip().lower()
                    require_depth_aug_supported(aug_name)
                    kwargs["augment_depth"] = aug_name
                    if not kwargs.get("depth_keys"):
                        raise ValueError(
                            f"--depth_aug {args.depth_aug!r} requires a depth-conditioned "
                            "checkpoint (server metadata advertises no *_depth obs keys)."
                        )
                if crop_eval_active(args, server_meta):
                    kwargs["enable_target_front_crop"] = True
                if kwargs.get("enable_target_front_crop"):
                    kwargs["orig_cam_w"] = 640.0  # fallback front-camera size (HL oracle)
                    kwargs["orig_cam_h"] = 360.0
                if getattr(args, "hl_from_model", False):
                    # No client-side target: the HL server crops native_front_rgb.
                    kwargs["pass_native_front_for_server_crop"] = True
            # The server-advertised state_dim decides the transform's agent_pos
            # width, and its obs keys decide which optional modalities are sent,
            # so record what was negotiated.
            print(
                f"[INFO]: obs_transform={args.obs_transform!r} "
                f"server_state_dim={state_dim!r} "
                f"depth_keys={kwargs.get('depth_keys', ())!r} "
                f"include_ptmap={kwargs.get('include_ptmap', True)!r} "
                f"include_prev_front_frame={kwargs.get('include_prev_front_frame', False)!r} "
                f"center_crop_ratio={kwargs.get('center_crop_ratio')!r} "
                f"augment_depth={kwargs.get('augment_depth')!r} "
                f"per_camera_ptmap_bounds={kwargs.get('per_camera_ptmap_bounds')!r} "
                f"enable_target_front_crop={kwargs.get('enable_target_front_crop')!r} "
                f"pass_native_front_for_server_crop="
                f"{kwargs.get('pass_native_front_for_server_crop', False)!r}",
                flush=True,
            )
            transform = registry[args.obs_transform](**kwargs)
            # After construction so the front camera name comes from the transform
            # that will actually source the frame (it differs per subclass).
            if isinstance(transform, ManiflowObsTransform) and obs_keys is not None:
                _check_prev_front_frame_source(args, obs_keys, transform.external_cam)
        else:
            transform = registry[args.obs_transform]()
        # _make_env did not spawn --drop_cameras; the transform must not need one.
        if not getattr(args, "replay", False):
            drop = set(getattr(args, "drop_cameras", ()))
            needed = {
                getattr(transform, attr, None) for attr in ("external_cam", "wrist_cam")
            } - {None}
            if needed & drop:
                raise ValueError(
                    f"obs_transform {args.obs_transform!r} reads cameras "
                    f"{sorted(needed & drop)}, which --drop_cameras {sorted(drop)} "
                    "removes from the scene; drop a different camera set."
                )
        policy = ObsTransformPolicy(policy, transform)

    if args.action_horizon > 0:
        if args.unbatched:
            policy = PerEnvActionChunkBroker(policy, action_horizon=args.action_horizon)
        else:
            policy = ActionChunkBroker(
                policy, action_horizon=args.action_horizon, sync=args.sync_action_chunks
            )

    return PolicyClientAdapter(
        policy=policy,
        action_key=args.action_key,
        device=args.device,
        batched=not args.unbatched,
    )


def _collect_per_episode_metrics(
    raw_metrics: dict[str, list[torch.Tensor]],
) -> dict[str, list[float]]:
    """Flatten per-episode metric tensors into JSON-serializable lists."""
    per_episode: dict[str, list[float]] = {}
    for key, tensors in raw_metrics.items():
        per_episode[key] = torch.cat([v.float().flatten() for v in tensors]).tolist()
    return per_episode


def _aggregate_metrics(
    raw_metrics: dict[str, list[torch.Tensor]],
) -> dict[str, dict[str, float]]:
    """Compute mean / std / min / max for each metric key.

    Uses NaN-safe reductions so metrics like ``first_success_step`` (NaN for
    failed episodes) produce meaningful aggregates over successful episodes.
    """
    aggregated: dict[str, dict[str, float]] = {}
    for key, tensors in raw_metrics.items():
        values = torch.cat([v.float().flatten() for v in tensors])
        finite = values[~torch.isnan(values)]
        aggregated[key] = {
            "mean": torch.nanmean(values).item(),
            "std": finite.std().item() if len(finite) > 1 else 0.0,
            "min": finite.min().item() if len(finite) > 0 else float("nan"),
            "max": finite.max().item() if len(finite) > 0 else float("nan"),
            "count": len(values),
            "valid_count": len(finite),
        }
    return aggregated


# Diagnostic kinematics/collision metric families (from ``--track_metrics``).
# They dominate the console table but are rarely what you scan for at a glance,
# so the summary hides them (they're still written in full to eval_results.json).
_VERBOSE_METRIC_PREFIXES = (
    "ee_pos", "ee_quat", "ee_vel", "ee_acc",
    "joint_pos", "joint_vel", "joint_acc",
    "joint_stiffness", "joint_damping", "joint_effort",
    "collision", "contact",
)


def _print_results(
    aggregated: dict[str, dict[str, float]], num_episodes: int, elapsed: float
) -> None:
    print("\n" + "=" * 60)
    print(f"  Evaluation complete  |  episodes: {num_episodes}  |  time: {elapsed:.1f}s")
    print("=" * 60)
    header = f"{'metric':<20} {'mean':>10} {'std':>10} {'min':>10} {'max':>10} {'count':>8}"
    print(header)
    print("-" * len(header))
    hidden = 0
    for key, stats in sorted(aggregated.items()):
        if key.startswith(_VERBOSE_METRIC_PREFIXES):
            hidden += 1
            continue
        print(
            f"{key:<20} {stats['mean']:>10.4f} {stats['std']:>10.4f} "
            f"{stats['min']:>10.4f} {stats['max']:>10.4f} {stats['count']:>8}"
        )
    if hidden:
        print(
            f"  (+{hidden} diagnostic kinematics/collision metrics hidden; "
            "see eval_results.json)"
        )
    print("=" * 60 + "\n")


def _toyblocks_git_info() -> dict[str, str | bool | dict[str, str | None] | None]:
    """Commits of the ToyBlocks checkout this process runs from and its submodules.

    Submodules are ignored for the dirty check: the eval pipeline checks out the
    pinned hvla commit at runtime, which would otherwise flag every maniflow run.
    ``submodule_commits`` holds each submodule's checked-out HEAD (not the commit
    pinned by ToyBlocks), or ``None`` for an uninitialized submodule.
    """
    repo_dir = str(Path(__file__).resolve().parent)

    def _git(*cmd: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", repo_dir, *cmd],
                capture_output=True, text=True, check=True, timeout=60,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        # rstrip only: `git submodule status` lines start with a meaningful flag column.
        return out.stdout.rstrip()

    # Run from the top level so submodule paths are reported relative to the repo root.
    repo_dir = _git("rev-parse", "--show-toplevel") or repo_dir
    status = _git("status", "--porcelain", "--untracked-files=no", "--ignore-submodules")
    submodule_commits: dict[str, str | None] = {}
    for line in (_git("submodule", "status") or "").splitlines():
        # "<flag><sha> <path> (<describe>)"; flag "-" means not checked out.
        fields = line[1:].split()
        if len(fields) >= 2:
            submodule_commits[fields[1]] = None if line[0] == "-" else fields[0]
    return {
        "toyblocks_commit": _git("rev-parse", "HEAD"),
        "toyblocks_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "toyblocks_dirty": None if status is None else bool(status),
        "submodule_commits": submodule_commits,
    }


def _save_results(
    output_dir: str,
    aggregated: dict[str, Any],
    args: Any,
    num_episodes: int,
    elapsed: float,
    per_episode: dict[str, list[float]] | None = None,
    per_episode_goal_conjuncts: list[list[str] | None] | None = None,
    success_efficiency: float | None = None,
) -> None:
    """Write ``eval_results.json``.

    ``per_episode_goal_conjuncts[i]`` is the PDDL text of each ``goal_conj_<c>`` of
    episode ``i`` (``None`` without a compiled PDDL goal). It lines up with the
    episode-level ``per_episode`` arrays (``success_once``, ``goal_conj_*``, ...),
    not with keys only some episodes carry (e.g. ``replay_object_<name>_*``).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result = {
        **_toyblocks_git_info(),
        "args": {k: v for k, v in vars(args).items() if _is_json_serializable(v)},
        "num_episodes": num_episodes,
        "elapsed_seconds": elapsed,
        "per_episode": per_episode or {},
        "per_episode_goal_conjuncts": per_episode_goal_conjuncts or [],
        "metrics": aggregated,
    }
    # Molmo2 HL runs only: HL commands taken over the demonstration's.
    if success_efficiency is not None:
        result["success_efficiency"] = success_efficiency
    path = os.path.join(output_dir, "eval_results.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {path}")


def _extract_frame(obs: dict, camera: str | None) -> np.ndarray | None:
    """Pull an RGB frame (HWC uint8) from the observation dict.

    If *camera* is given, use that single camera.  Otherwise tile all cameras.
    Returns ``None`` when no images are present.
    """
    images = obs.get("images")
    if images is None:
        return None

    from toyblocks.utils.visualization_utils import tile_images

    if camera is not None:
        img = images[camera]["rgb"]
    else:
        imgs = [cam_data["rgb"] for cam_data in images.values()]
        img = tile_images(imgs)

    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if img.shape[-1] == 4:
        img = img[..., :3]
    if np.issubdtype(img.dtype, np.floating):
        img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    return img
