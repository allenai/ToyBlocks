"""LeRobot recording for the vecgen engine, via IsaacDataRecorderLeRobot (``vecgen.recording``).

The recorder keeps the stage-3 dataset schema (robot state, actions with target feed-
forward in ``action.joint_target``, object poses, symbolic action strings, success /
failure split, shards). Frames are recorded on every tick in which an env executes a
segment; reset settles are not recorded.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from blocksuite.vecgen.engine import Recorder


class _HLState:
    """``current_hl_state`` for the recorder: the ``getTrueProps`` of TAMP's PDDLState."""

    def __init__(self, props: tuple[str, ...]) -> None:
        self._props = set(props)

    def getTrueProps(self) -> set[str]:  # noqa: N802 - TAMP's name
        return self._props


def _ll_args(ll: str | None, hl_state: tuple[str, ...]) -> dict[str, Any]:
    args: dict[str, Any] = {} if ll is None else {"ll_action_name": ll}
    if hl_state:
        args["current_hl_state"] = _HLState(hl_state)
    return args


class LeRobotRecorder(Recorder):
    def __init__(
        self,
        sim: Any,
        geo: Any,
        data_dir: str | Path,
        problem_name: str,
        pddl_goal: str,
        provenance: dict[str, Any] | None = None,
        goal_of_reset_state: Callable[[int], str] | None = None,
        episode_of_reset_state: Callable[[int], tuple[str, int, list[str]]] | None = None,
        record_videos: bool = False,
    ) -> None:
        """``episode_of_reset_state(rid)`` -> ``(source problem, reset-state id within it,
        scene objects outside it)`` for multi-problem packages; without it every episode is
        the data dir's problem and its ``instance_id`` is the reset-state id.
        ``record_videos`` also encodes the scene cameras (the sim must have spawned them)."""
        from blocksuite.vecgen.recording import IsaacDataRecorderLeRobot

        n = sim.num_envs
        self.sim = sim
        self.n = n
        self.problem_name = problem_name
        prov = dict(provenance or {})
        prov.setdefault("instance_json_ids", list(range(n)))
        prov.setdefault("generator", "vecgen")

        def ee_fk(q: Any) -> Any:
            # the recorder takes a host array (np.asarray on a CUDA tensor raises, and the
            # recorder then writes zeros for ee_target / relative_ee_target)
            qt = torch.as_tensor(q, device=sim.device, dtype=torch.float32)
            return geo.fk(qt).detach().cpu().numpy().astype(np.float32)

        self.rec = IsaacDataRecorderLeRobot(
            sim.scene,
            data_dir=str(data_dir),
            num_envs=n,
            record_every_n_steps=1,
            fps=25,
            recording_provenance=prov,
            instance_ids=list(range(n)),
            robotiq=True,
            record_videos=record_videos,
            ee_target_fk=ee_fk,
        )
        scene_id = getattr(sim.base_env.cfg, "scene_preset_id", "") or ""
        robot_id = getattr(getattr(sim.base_env, "robot", None), "id", "") or ""
        slots = sim.env_slots()
        for e in range(n):
            # the scene profile env e was spawned with (fixed for the run); the columns
            # match stage 3's: cube_scale = first block's edge, table friction = the
            # profile's static friction (TemplateEnv writes it to the table per slot)
            profile = sim.env_scene_profile(e)
            self.rec.set_scene_params(
                cube_scale=float(next(iter(profile["block_sizes"].values()))[0]),
                table_height_delta=float(profile["table_height"]),
                table_friction=float(profile["static_friction"]),
                env_idx=e,
                scene_id=scene_id,
                robot_id=robot_id,
                profile_id=slots[e],
            )
            self.rec.set_friction_properties(
                {
                    "static_friction": float(profile["static_friction"]),
                    "dynamic_friction": float(profile["dynamic_friction"]),
                    "restitution": float(profile["restitution"]),
                },
                env_idx=e,
            )
            self.rec.set_pddl_goal(pddl_goal, env_idx=e)
        self.instance_ids: list[int] = []
        self._open: set[int] = set()
        self.goal_of_reset_state = goal_of_reset_state  # problem streaming: goal per episode
        self.episode_of_reset_state = episode_of_reset_state

    def start_episode(
        self, env_id: int, episode_id: int, reset_state_id: int, hl_state: tuple[str, ...] = ()
    ) -> None:
        if self.goal_of_reset_state is not None:
            self.rec.set_pddl_goal(self.goal_of_reset_state(reset_state_id), env_idx=env_id)
        if self.episode_of_reset_state is not None:
            problem, local_id, inactive = self.episode_of_reset_state(reset_state_id)
            self.rec.set_episode_provenance(
                env_id, source_problem=problem, instance_id=local_id, inactive_objects=inactive
            )
        else:
            self.rec.set_episode_provenance(env_id, instance_id=reset_state_id)
        thd = self._placed_table_height_delta(env_id)
        if thd is not None:
            self.rec.set_table_height_delta(thd, env_idx=env_id)
        self.rec.update_symbolic_actions("init", [], "init", _ll_args(None, hl_state), env_idx=env_id)
        self.instance_ids.append(reset_state_id)
        self._open.add(env_id)

    def _placed_table_height_delta(self, env_id: int) -> float | None:
        """Height of the table the reset state actually placed, relative to its spawn pose.

        A reset state runs in any slot whose block sizes and friction match, regardless of the
        slot's table height, and restores its own table pose -- so the env slot's profile height
        (set at construction) can be centimetres off. None when the table cannot be read.
        """
        try:
            scene = self.sim.base_env.scene
            table = scene["workspace_table"]
            z = float(table.data.root_pose_w[env_id, 2] - scene.env_origins[env_id, 2])
            return z - float(table.data.default_root_state[env_id, 2])
        except Exception:  # noqa: BLE001 - keep the slot's value rather than fail the episode
            return None

    def set_goal(self, env_id: int, goal_text: str) -> None:
        self.rec.set_pddl_goal(goal_text, env_idx=env_id)

    def symbolic(
        self, env_id: int, hl: str, hl_args: tuple[str, ...], ll: str, hl_state: tuple[str, ...] = ()
    ) -> None:
        self.rec.update_symbolic_actions(
            hl, list(hl_args), ll, _ll_args(ll, hl_state), env_idx=env_id
        )

    def step(self, env_ids: Tensor) -> None:
        if env_ids.numel():
            self.rec.record_step_batched([e for e in env_ids.tolist() if e in self._open])

    def frame_mark(self, env_id: int) -> int:
        return self.rec.frames_recorded(env_id) if env_id in self._open else 0

    def drop_since(self, env_id: int, mark: int) -> None:
        if env_id in self._open:
            self.rec.drop_frames_since(env_id, mark)

    def end_episode(self, env_id: int, success: bool, error: str) -> None:
        if env_id not in self._open:
            return
        if error:
            self.rec.set_error(error, env_idx=env_id)
        self.rec.write_success(bool(success), env_idx=env_id)
        self.rec.save_episode(env_idx=env_id)
        self._open.discard(env_id)

    def finalize(self) -> None:
        self.rec.finalize(self.instance_ids)


def write_run_manifest(path: str | Path, info: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
