from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from toyblocks.tasks.base_env import BaseEnv
from toyblocks.tasks.base_env_cfg import BaseEnvCfg
from toyblocks.utils import common_utils, tree_utils
from toyblocks.utils.randomization_utils import set_rigid_body_friction
from toyblocks.utils.reset_states_io import (
    describe_spawn_profile,
    load_reset_states_with_profiles,
    reset_state_slot_compat,
    sample_compatible_reset_states,
)

from .template_env_cfg import TemplateEnvCfg

if TYPE_CHECKING:
    from toyblocks.utils.episode_queue import EpisodeQueueClient

logger = logging.getLogger(__name__)


class TemplateEnv(BaseEnv):
    cfg: TemplateEnvCfg
    reset_states: dict[str, dict[str, dict[str, torch.Tensor]]] | None = None
    reset_state_profile_ids: torch.Tensor | None = None
    reset_state_profiles: list[dict[str, Any]] | None = None
    num_scene_profiles: int | None = None
    queue_client: EpisodeQueueClient | None = None
    # (num_reset_states, num_slots) bool: reset state k may run in an env spawned
    # at slot s. Built on first use; None when the reset states record no profile.
    _reset_state_slot_ok: torch.Tensor | None = None
    _reset_state_slot_ok_built: bool = False

    def __init__(self, cfg: TemplateEnvCfg, **kwargs):
        if os.path.exists(cfg.reset_states_path):
            reset_states, profile_ids, num_profiles, profiles = load_reset_states_with_profiles(
                cfg.reset_states_path
            )
            self.reset_state_profiles = profiles
            self.reset_states = common_utils.to_torch(reset_states, device=cfg.sim.device)
            self.num_reset_states = tree_utils.shape(reset_states, first_only=True)[0]
            if profile_ids is not None:
                self.reset_state_profile_ids = torch.as_tensor(
                    profile_ids, device=cfg.sim.device
                )
            self.num_scene_profiles = num_profiles
        super().__init__(cfg, **kwargs)

    # --- autogen: instance layout json ---
    def _merge_instance_layout_json_into_cfg(self, cfg: BaseEnvCfg) -> None:
        pass

    def _write_instance_layout_json_block_poses(self, env_ids: Sequence[int] | None) -> None:
        pass

    # --- end autogen: instance layout json ---

    def _setup_scene_objects(self):
        pass

    def _after_scene_setup(self):
        # before the first environment is reset, we call this function once
        # useful for initializing anything that may need access to instantiated objects e.g. kinematics
        return

    def _compute_intermediate_values(self):
        # Codegen injects PDDL goal checks here that set self._goal_success.
        # Default to all-False so _get_rewards works without codegen.
        if not hasattr(self, "_goal_success"):
            self._goal_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _get_obs_robot(self) -> dict:
        return super()._get_obs_robot()

    def _get_obs_extra(self) -> dict:
        # By default observations only include the robot's information returned by _get_obs_robot
        # to add more information like ground truth object poses return them here as a dictionary
        # Whatever is returned here will also be used to automatically create the observation space
        observations = {"random_pose": torch.rand(self.num_envs, 7, device=self.device)}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        return self._goal_success.float()

    def evaluate(self) -> dict:
        # return optionally additional metrics here
        # include the keys "success" and/or "fail" to indicate if the environment state is in a success or failed state
        # and will update some recorded metrics
        # (success from PDDL goal is injected by scripts/generate_task_from_pddl.py)
        result = super().evaluate()
        result["success"] = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return result

    def _env_slot_ids(self, env_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
        """Spawn slot (cube scale / randomization profile) of each env in ``env_ids``."""
        ids = torch.as_tensor(env_ids, device=self.env_slots_t.device, dtype=torch.long)
        return self.env_slots_t[ids]

    def scene_profile(self, slot: int) -> dict[str, Any]:
        """Values of spawn slot ``slot``, in the schema of a ``profiles.json`` entry."""
        randomizations = self.randomizations
        friction = randomizations["friction"]
        return {
            "profile_id": int(slot),
            "block_sizes": {
                name: [float(v) for v in sizes[slot]]
                for name, sizes in randomizations["block_sizes"].items()
                if len(sizes) > 0
            },
            "table_height": float(randomizations["table_height"][slot]),
            "static_friction": float(friction["static_friction"][slot]),
            "dynamic_friction": float(friction["dynamic_friction"][slot]),
            "restitution": float(friction["restitution"][slot]),
        }

    def env_scene_profile(self, env_idx: int) -> dict[str, Any]:
        """Profile env ``env_idx`` was actually spawned with (its slot, plus overrides)."""
        profile = self.scene_profile(int(self.env_slots[env_idx]))
        # Single-env replay spawns every block at the replayed side length instead
        # of the slot's size (see _setup_scene_objects).
        replay_side = getattr(self.cfg, "replay_block_side_length", None)
        if replay_side is not None and self.num_envs == 1:
            side = float(replay_side)
            profile["block_sizes"] = {name: [side, side, side] for name in profile["block_sizes"]}
        return profile

    def _reset_state_slot_compat(self) -> torch.Tensor | None:
        """Which spawn slots can run each reset state, matched by profile value.

        A reset state's block poses and plan are only valid at the block sizes and
        friction it was sampled under, so it may only run in an env whose spawned
        profile has those values (see :func:`reset_state_slot_compat`). Packages
        that record no profile per reset state cannot be checked and return None.
        """
        if self._reset_state_slot_ok_built:
            return self._reset_state_slot_ok
        self._reset_state_slot_ok_built = True
        if self.reset_state_profile_ids is None:
            logger.warning(
                "%s records no scene profile per reset state, so reset states cannot be "
                "matched to env profiles; regenerate the task to enforce the match.",
                self.cfg.reset_states_path,
            )
            return None
        slot_profiles = [
            self.scene_profile(s) for s in range(int(self.MAX_NON_SIM_STATE_RANDOMIZATIONS))
        ]
        compat = reset_state_slot_compat(
            self.reset_state_profile_ids.cpu().numpy(), self.reset_state_profiles, slot_profiles
        )
        self._reset_state_slot_ok = torch.as_tensor(compat, device=self.device)
        return self._reset_state_slot_ok

    def _check_reset_state_profiles(
        self, env_ids: Sequence[int] | torch.Tensor, instance_ids: Sequence[int] | torch.Tensor
    ) -> None:
        """Raise when an env is asked to run a reset state from a different profile."""
        ok = self._reset_state_slot_compat()
        if ok is None:
            return
        env_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        inst_t = torch.as_tensor(instance_ids, device=self.device, dtype=torch.long).reshape(-1)
        bad = (~ok[inst_t, self._env_slot_ids(env_t)]).nonzero().flatten().tolist()
        if not bad:
            return
        recorded = {int(p["profile_id"]): p for p in self.reset_state_profiles or []}
        lines = []
        for j in bad[:5]:
            env_idx, inst = int(env_t[j]), int(inst_t[j])
            pid = int(self.reset_state_profile_ids[inst])
            lines.append(
                f"  env {env_idx} ({describe_spawn_profile(self.env_scene_profile(env_idx))}) "
                f"<- reset state {inst} "
                f"({describe_spawn_profile(recorded.get(pid, {'profile_id': pid}))})"
            )
        raise ValueError(
            f"{len(bad)} env(s) were asked to run a reset state sampled under a different "
            "scene profile; its block poses and plan are only valid at the recorded block "
            "sizes / friction:\n" + "\n".join(lines)
        )

    def _sample_profile_matched_reset_states(
        self, env_ids: Sequence[int] | torch.Tensor
    ) -> torch.Tensor:
        """Draw one reset state per env from those recorded under the env's profile."""
        env_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        ok = self._reset_state_slot_compat()
        if ok is None:
            return torch.randint(0, self.num_reset_states, (len(env_t),), device=self.device)
        return sample_compatible_reset_states(ok, self._env_slot_ids(env_t))

    def _reset_idx(self, env_ids: Sequence[int] | None, instance_ids: Sequence[int] | None = None):
        super()._reset_idx(env_ids)
        if self.queue_client is not None:
            from toyblocks.utils.episode_queue import apply_setup

            self._write_instance_layout_json_block_poses(env_ids)
            setups = self.queue_client.fetch_reset_states(env_ids)
            for eid_t, setup in zip(env_ids, setups):
                eid = int(eid_t) if isinstance(eid_t, torch.Tensor) else eid_t
                if setup is not None:
                    apply_setup(
                        self,
                        eid,
                        setup,
                        apply_recorded_joint_velocity=(
                            self.queue_client.apply_recorded_joint_velocity
                        ),
                    )
            return
        if self.reset_states is not None:
            if instance_ids is not None:
                self._check_reset_state_profiles(env_ids, instance_ids)
                self.set_state_dict(
                    tree_utils.slice(self.reset_states, instance_ids),
                    env_ids=env_ids,
                    is_relative=True,
                )
            else:
                self.set_state_dict(
                    tree_utils.slice(
                        self.reset_states, self._sample_profile_matched_reset_states(env_ids)
                    ),
                    env_ids=env_ids,
                    is_relative=True,
                )
        else:
            pose = torch.tensor(
                [0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, 0.0], device=self.device
            )
            ids = [0, 1, 2, 3, 4, 5, 6]
            self.robot.articulation.write_joint_position_to_sim(
                pose, joint_ids=ids, env_ids=env_ids
            )
            self.robot.articulation.set_joint_position_target(pose, joint_ids=ids, env_ids=env_ids)
        # randomize table friction
        table = self.scene.rigid_objects["workspace_table"]
        set_rigid_body_friction(
            asset=table,
            static_friction=self.randomizations["friction"]["static_friction"][
                self._env_slot_ids(env_ids)
            ],
            dynamic_friction=self.randomizations["friction"]["dynamic_friction"][
                self._env_slot_ids(env_ids)
            ],
            restitution=self.randomizations["friction"]["restitution"][
                self._env_slot_ids(env_ids)
            ],
            env_ids=env_ids,
        )
        # randomize table height (delta relative to scene spawn pose)
        # Only apply when reset_states are NOT used: during generation, the table
        # shift is applied here, and the resulting state (table + blocks) is saved
        # into reset_states.pkl. When reset_states ARE loaded, set_state_dict
        # already restores the table and blocks in correct relative positions —
        # overriding the table z again would double-apply the offset.
        if self.reset_states is None:
            pose = table.data.root_pose_w[env_ids].clone()
            if not hasattr(self, "_workspace_table_spawn_root_z"):
                self._workspace_table_spawn_root_z = table.data.root_pose_w[:, 2].clone()
            env_idx_t = (
                env_ids
                if isinstance(env_ids, torch.Tensor)
                else torch.tensor(env_ids, device=self.device, dtype=torch.long)
            )
            slot = self._env_slot_ids(env_idx_t)
            dz = self.randomizations["table_height"][slot].to(device=pose.device, dtype=pose.dtype)
            pose[:, 2] = (
                self._workspace_table_spawn_root_z[env_idx_t].to(
                    device=pose.device,
                    dtype=pose.dtype,
                )
                + dz
            )
            table.write_root_pose_to_sim(pose, env_ids=env_ids)
        self._write_instance_layout_json_block_poses(env_ids)

    def _before_sim_step(self, substep: int) -> None:
        super()._before_sim_step(substep)
        if self.queue_client is None:
            return
        from toyblocks.utils.episode_queue import park_placed_blocks

        # A frame rendered after this substep must not show the blocks parked;
        # step() renders once the incremented counter hits the render interval.
        renders = (self._sim_step_counter + 1) % self.cfg.sim.render_interval == 0
        if not (renders and (self.sim.has_gui() or self.sim.has_rtx_sensors())):
            park_placed_blocks(self)

    def _after_sim_step(self, substep: int) -> None:
        super()._after_sim_step(substep)
        if self.queue_client is not None:
            from toyblocks.utils.episode_queue import restore_placed_blocks

            restore_placed_blocks(self)
