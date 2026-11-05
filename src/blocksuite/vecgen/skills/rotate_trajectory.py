"""Scripted-rotate geometry for the vecgen rotate skill.

The constants and methods the rotate skill calls (anchor search, segment/hold timing, the
sub-goal split), ported from the former TAMP rotate executor.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from curobo.types import JointState, Pose

from blocksuite.vecgen.skills._quat import quat_apply, quat_inv, quat_mul

# Dictionary mapping face names to quaternions
ORI_QUATERNIONS = {
    "o1": [1.0,0.0,0.0,0.0],
    "o2": [0.7071,0,0,-0.7071],
    "o3": [0.0,0.0,0.0,-1.0],
    "o4": [0.7071,0,0,0.7071],
    "o5": [0.7071,0,-0.7071,0],
    "o6": [0.5,-0.5,-0.5,-0.5],
    "o7": [0,0.7071,0,0.7071],
    "o8": [0.5,0.5,-0.5,0.5],
    "o9": [0.0,0.0,1.0,0.0],
    "o10":[0,0.7071,0.7071,0],
    "o11": [0.0,-1.0,0.0,0.0],
    "o12": [0.0,-0.7071,0.7071,0.0],
    "o13": [0.7071,0.0,0.7071,0.0],
    "o14": [0.5,0.5,0.5,-0.5],
    "o15": [0,-0.7071,0,0.7071],
    "o16": [0.5,-0.5,0.5,0.5],
    "o17": [0.5,0.5,-0.5,-0.5],
    "o18": [0,0,0.7071,0.7071],
    "o19": [0.5,0.5,0.5,0.5],
    "o20": [0.7071,0.7071,0,0],
    "o21": [0.5,-0.5,0.5,-0.5],
    "o22": [0,0,0.7071,-0.7071],
    "o23": [0.5,-0.5,-0.5,0.5],
    "o24": [0.7071,-0.7071,0,0],
}


# ---- tmp3.Utils.ik_seed
def _ik_joint_names(motion_gen: Any) -> list | None:
    """Joint order the IK solver expects, if it can be read off *motion_gen*."""
    try:
        return list(motion_gen.ik_solver.joint_names)
    except Exception:
        return None


def _seed_position(motion_gen: Any, joint_state: JointState | torch.Tensor | None) -> torch.Tensor | None:
    """Flatten *joint_state* into a ``(1, dof)`` seed ordered for the IK solver."""
    if joint_state is None:
        return None

    if isinstance(joint_state, JointState):
        ik_joint_names = _ik_joint_names(motion_gen)
        if ik_joint_names is not None and joint_state.joint_names is not None:
            if list(joint_state.joint_names) != ik_joint_names:
                if set(ik_joint_names) - set(joint_state.joint_names):
                    # The solver optimizes joints we don't have a value for; a
                    # partial seed would be silently misaligned, so skip seeding.
                    return None
                joint_state = joint_state.reorder(ik_joint_names)
        position = joint_state.position
    else:
        position = joint_state

    if position is None:
        return None
    position = position.detach().reshape(1, -1).to(dtype=torch.float32)

    dof = getattr(getattr(motion_gen, "ik_solver", None), "dof", None)
    if dof is not None and position.shape[-1] != dof:
        return None
    return position


def current_ik_seed(
    motion_gen: Any,
    joint_state: JointState | torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``(retract_config, seed_config)`` for *joint_state*, or None if unusable."""
    position = _seed_position(motion_gen, joint_state)
    if position is None:
        return None
    return position, position.unsqueeze(1)


def solve_ik_seeded(
    motion_gen: Any,
    pose: Any,
    joint_state: JointState | torch.Tensor | None,
    **kwargs: Any,
) -> Any:
    """``motion_gen.solve_ik(pose)`` seeded with the current joint configuration.

    Falls back to an unseeded solve when the seed cannot be built or the
    installed CuRobo rejects it, so IK never fails purely because of seeding.
    """
    seed = current_ik_seed(motion_gen, joint_state)
    if seed is not None:
        retract_config, seed_config = seed
        try:
            return motion_gen.solve_ik(
                pose, retract_config=retract_config, seed_config=seed_config, **kwargs
            )
        except (TypeError, RuntimeError) as exc:
            print(f"[ik_seed] seeded IK rejected ({exc}); solving unseeded")
    return motion_gen.solve_ik(pose, **kwargs)


class RotateTrajectory:
    def __init__(self, argument_name: str) -> None:
        self.argument_name = argument_name

    # ------------------------------------------------------------------
    # Scripted grasp-and-twist rotate (motion-planning heuristic).
    #
    # The rotate is a heuristic waypoint sequence -- descend onto the block
    # from the top-down hover the motion plan left us at, grip it, lift it a
    # few centimetres clear of the table, then spin ONLY the wrist joint
    # (Franka panda_joint7, the last arm joint) so the gripped block rotates
    # in place about the vertical axis to the goal yaw, lower it back down,
    # release, and retreat.  Every non-twist waypoint is turned into a joint
    # target via IK; the twist phase adds the yaw delta to joint 7 alone.  The
    # whole thing runs under the soft stiffness profile so grip/contact forces
    # stay bounded.
    # ------------------------------------------------------------------
    ROTATE_ACTION_DT: float = 0.02
    ROTATE_DECIMATION: int = 4
    # IK link -> fingertip (pad bottom) once closed on the block, along the
    # approach axis. The Robotiq tips travel down as the fingers close (pad
    # bottom is 0.1589 m below robotiq_arg2f_base_link open, 0.173 m fully
    # closed); closed on a 4 cm block under the soft profile the recorded tip
    # sits 0.169 m below it (rotate_fix run, 2026-09-22). 0.1348 is the Panda
    # pad centre (same constant as ReOrientTrajectory).
    ROTATE_EE_TO_TIP_OFFSET: float = 0.1348
    ROTATE_ROBOTIQ_EE_TO_TIP_OFFSET: float = 0.169
    # Gripped fingertip height above the block bottom, as a fraction of the
    # block's vertical extent.
    ROTATE_GRASP_TIP_HEIGHT_FRAC: float = 1.0 / 3.0
    # Lift the gripped block this far off the table before twisting -- just
    # enough clearance to avoid scraping the table, not a visible pick-up.
    ROTATE_LIFT_HEIGHT: float = 0.015
    # Clearance the gripper retreats to between the two sub-steps of a split
    # rotate, before re-solving the grasp and descending again. ROTATE_LIFT_HEIGHT
    # alone leaves the fingers ~1.5 cm off the block, close enough that the
    # re-approach can clip it. Matches LiftTrajectory.LIFT_HEIGHT (and the retract
    # in ori_of.apply) so the scripted path clears by the same margin the rest of
    # the pipeline does.
    ROTATE_INTERSTAGE_LIFT_HEIGHT: float = 0.05
    ROTATE_INTERSTAGE_LIFT_DURATION: float = 0.35
    # Hold at the top of the inter-stage lift. Without it the cycle-2 descent
    # starts pulling the arm down before it has tracked to the clear pose under
    # the soft profile, and it reaches only ~3.5 cm of the commanded 5 cm.
    ROTATE_INTERSTAGE_HOLD_DURATION: float = 0.16
    ROTATE_DESCENT_DURATION: float = 0.30
    ROTATE_GRASP_SETTLE_DURATION: float = 0.10
    ROTATE_LIFT_DURATION: float = 0.30
    ROTATE_TWIST_DURATION: float = 0.60
    ROTATE_TWIST_HOLD_DURATION: float = 0.10
    ROTATE_LOWER_DURATION: float = 0.30
    ROTATE_RETREAT_DURATION: float = 0.40
    # Commanded gripper (EE) rotation speed cap for every scripted segment;
    # segments that would turn the gripper faster (the twist, at 90 deg) are
    # stretched. Each joint target is held for one recorded frame
    # (ROTATE_DECIMATION physics steps at 0.01 s = one 25 Hz render interval),
    # so ROTATE_TARGET_PERIOD, not ROTATE_ACTION_DT, is the sim time a target lasts.
    ROTATE_MAX_GRIPPER_ANGULAR_SPEED: float = 1.0
    ROTATE_TARGET_PERIOD: float = 0.04
    # FK samples along a joint-space segment when measuring its gripper rotation.
    ROTATE_ROTATION_SAMPLES: int = 64
    # Nominal per-action yaw (the domain's yaw_next is a 90 deg step); used only
    # as a fallback when the goal quaternion does not yield a clean z-rotation.
    ROTATE_STEP_YAW: float = float(np.pi / 2)
    ROTATE_GRIPPER_CLOSE_WAIT_STEPS: int = 20
    ROTATE_GRIPPER_OPEN_WAIT_STEPS: int = 20
    ROTATE_STIFFNESS_SETTLE_STEPS: int = 0
    # Overall sign relating a +joint7 rotation to a +world-z block rotation.
    # Derived per-run from the grasp approach axis; this multiplier flips it.
    ROTATE_WRIST_SIGN: float = 1.0
    # A rotate whose total world-z yaw exceeds 90 deg by this margin is split
    # into an align-to-current-ori correction followed by the nominal 90 deg
    # step.  The residual comes from the reset-state spawn perturbation
    # (ori_of.sample uses yaw_range_deg=30.0), so the total lands in 60..120 deg
    # and the test is unambiguous either way.
    ROTATE_SPLIT_TOLERANCE_DEG: float = 5.0
    # Corrections below this are not worth a full release-and-regrasp cycle.
    ROTATE_MIN_CORRECTION_DEG: float = 5.0

    # ------------------------------------------------------------------
    # IK / joint-target helpers (shared shape with the scripted reorient port)
    # ------------------------------------------------------------------

    @staticmethod
    def _ik_solution_to_joint_target(ik_result: Any, robot: Any) -> torch.Tensor | None:
        try:
            success = bool(ik_result.success.flatten()[0].item())
        except Exception:
            success = False
        if not success:
            return None

        try:
            js = ik_result.js_solution[0].reorder(robot.joint_names)
            pos = js.position
        except Exception:
            try:
                js = ik_result.js_solution[0]
                pos = js.position
            except Exception:
                pos = getattr(ik_result, "solution", None)
        if pos is None:
            return None
        if isinstance(pos, torch.Tensor):
            return pos.reshape(-1)[:7].detach().clone()
        device = getattr(robot, "device", getattr(getattr(robot, "_scene", None), "device", "cpu"))
        return torch.as_tensor(pos, device=device, dtype=torch.float32).reshape(-1)[:7]

    @staticmethod
    def _clamp_franka_joints(q: torch.Tensor) -> torch.Tensor:
        lower = torch.tensor(
            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
            device=q.device,
            dtype=q.dtype,
        )
        upper = torch.tensor(
            [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
            device=q.device,
            dtype=q.dtype,
        )
        return torch.max(torch.min(q, upper), lower)

    def _current_joint_target(self, robot: Any, env_index: int) -> torch.Tensor:
        if hasattr(robot, "get_joint_states_batched"):
            env_ids = torch.tensor([env_index], device=robot.device, dtype=torch.long)
            return robot.get_joint_states_batched(env_ids)[0].detach().clone()
        return robot.get_joint_state().position.detach().clone()

    def _solve_ik(self, robot: Any, target_pose: Pose, env_index: int) -> Any:
        if hasattr(robot, "_motion_gen"):
            return solve_ik_seeded(
                robot._motion_gen, target_pose, self._current_joint_target(robot, env_index)
            )
        return robot.solve_ik(target_pose, env_index) if hasattr(robot, "get_joint_states_batched") else robot.solve_ik(target_pose)

    def _prepare_contact_ik_world(self, robot: Any, env_index: int, obj_name: str) -> bool:
        if hasattr(robot, "update_scene_obstacles_for_env"):
            robot.update_scene_obstacles_for_env(
                env_index,
                event="scripted_rotate_grasp_ik",
                exclude_objects=[obj_name],
            )
            self._load_planning_world(robot, env_index, exclude_objects=[obj_name])
            return True
        if hasattr(robot, "update_scene_obstacles"):
            try:
                robot.update_scene_obstacles(exclude_objects={obj_name})
                return True
            except TypeError:
                return False
        return False

    def _restore_ik_world(self, robot: Any, env_index: int, did_prepare: bool) -> None:
        if not did_prepare:
            return
        if hasattr(robot, "update_scene_obstacles_for_env"):
            robot.update_scene_obstacles_for_env(env_index, event="scripted_rotate_restore_ik")
            self._load_planning_world(robot, env_index)
        elif hasattr(robot, "update_scene_obstacles"):
            robot.update_scene_obstacles()

    @staticmethod
    def _load_planning_world(robot: Any, env_index: int, exclude_objects: list[str] | None = None) -> bool:
        coordinator = getattr(robot, "planning_coordinator", None)
        if coordinator is None or not hasattr(coordinator, "_load_scene_for_env"):
            return False
        lock = getattr(coordinator, "_mg_lock", None)
        if lock is not None:
            lock.acquire()
        try:
            coordinator._load_scene_for_env(env_index, exclude_objects=exclude_objects)
            return True
        finally:
            if lock is not None:
                lock.release()

    def _duration_to_steps(self, duration_s: float) -> int:
        return max(2, int(round(float(duration_s) / self.ROTATE_ACTION_DT)))

    def _gripper_rotation_per_alpha(self, robot: Any, q0: torch.Tensor, q1: torch.Tensor) -> float:
        """Peak gripper rotation (rad per unit interpolation fraction) along the
        joint-space line ``q0 -> q1``, from FK on dense samples of it. Without FK
        falls back to the revolute-chain bound ``|omega| <= sum_i |dq_i|``."""
        dq_bound = float((q1[:7] - q0[:7]).abs().sum())
        fk = getattr(robot, "compute_ee_targets_batched", None)
        if fk is None:
            return dq_bound
        try:
            alphas = torch.linspace(
                0.0, 1.0, self.ROTATE_ROTATION_SAMPLES, device=q0.device, dtype=torch.float32
            ).unsqueeze(1)
            path = (1.0 - alphas) * q0[:7].reshape(1, 7).float() + alphas * q1[:7].reshape(1, 7).float()
            quats = fk(path)[:, 3:7]
            dots = (quats[1:] * quats[:-1]).sum(dim=-1).abs().clamp(max=1.0)
            return float((2.0 * torch.acos(dots)).max()) * (self.ROTATE_ROTATION_SAMPLES - 1)
        except Exception as exc:
            print(f"[RotateTrajectory] gripper rotation FK unavailable ({exc}); using joint bound")
            return dq_bound

    def _append_segment(
        self,
        targets: list[torch.Tensor],
        q0: torch.Tensor,
        q1: torch.Tensor,
        duration_s: float,
        robot: Any,
    ) -> None:
        # Stretched (never shortened) so the gripper rotation stays under the cap.
        max_turn_per_target = self.ROTATE_MAX_GRIPPER_ANGULAR_SPEED * self.ROTATE_TARGET_PERIOD
        min_intervals = math.ceil(
            self._gripper_rotation_per_alpha(robot, q0, q1) / max_turn_per_target - 1e-6
        )
        steps = max(self._duration_to_steps(duration_s), min_intervals + 1)
        for step in range(steps):
            alpha = float(step) / float(max(1, steps - 1))
            if targets and step == 0:
                continue
            targets.append(((1.0 - alpha) * q0 + alpha * q1).detach().clone())

    def _append_hold(self, targets: list[torch.Tensor], q: torch.Tensor, duration_s: float) -> None:
        for _ in range(self._duration_to_steps(duration_s)):
            targets.append(q.detach().clone())

    @staticmethod
    def _block_world_half_height(
        simulator: Any, obj_name: str, obj_quat: torch.Tensor, env_index: int
    ) -> float:
        """World-z half extent of the block (its collider when the simulator measured it
        (vecgen), else its spawn bbox, rotated by ``obj_quat``)."""
        from blocksuite.utils import obs_utils

        measured = getattr(simulator, "block_half_extents", None)
        half = measured(obj_name, env_index) if measured is not None else None
        if half is None:
            bbox = obs_utils.get_bounding_box(simulator.scene[obj_name])  # (num_envs, 2, 3)
            half = 0.5 * (bbox[env_index, 1] - bbox[env_index, 0])
        half = half.to(device=obj_quat.device, dtype=torch.float32)
        axes = quat_apply(obj_quat.reshape(1, 4).expand(3, 4), torch.eye(3, device=obj_quat.device))
        return float((axes[:, 2].abs() * half).sum())

    def _fallback_yaw(self, direction: str) -> float:
        # Right-hand rule about world +z: ccw is positive, cw is negative.
        if str(direction).strip().lower() == "cw":
            return -self.ROTATE_STEP_YAW
        return self.ROTATE_STEP_YAW

    @staticmethod
    def _yaw_about_z(q_from: torch.Tensor, q_to: torch.Tensor) -> float | None:
        """Signed rotation about world +z that carries ``q_from`` to ``q_to``.

        Returns ``None`` when the relative rotation is not (close to) a pure
        z-rotation, so the caller can fall back to the symbolic direction.
        """
        q_rel = quat_mul(q_to.reshape(1, 4), quat_inv(q_from.reshape(1, 4))).reshape(-1)
        w = float(q_rel[0])
        vec = q_rel[1:4]
        n = float(torch.linalg.norm(vec))
        if n < 1e-6:
            return 0.0
        if abs(float(vec[2]) / n) < 0.7:
            return None
        yaw = 2.0 * float(np.arctan2(float(vec[2]), w))
        return (yaw + np.pi) % (2.0 * np.pi) - np.pi

    def _wrist_delta(
        self,
        obj_quat: torch.Tensor,
        target_quat: torch.Tensor,
        ee_quat: torch.Tensor,
        direction: str | None,
    ) -> float:
        yaw = self._yaw_about_z(obj_quat, target_quat)
        if yaw is None or abs(yaw) < 1e-3:
            # ``direction`` is the symbolic 90 deg step, so it is only a sane
            # fallback for the sub-step that actually carries that step. For an
            # align-only correction (direction=None) an unreadable yaw means
            # there is nothing to correct, not that we should turn 90 deg.
            yaw = 0.0 if direction is None else self._fallback_yaw(direction)

        # A +joint7 turn rotates the gripped block about the gripper's approach
        # axis; its world-z component maps that wrist turn onto a world-z block
        # rotation.  delta = yaw / a_z inverts that map (a_z ~ -1 for a top-down
        # grasp -> delta ~ -yaw).
        flange_z = quat_apply(
            ee_quat.reshape(1, 4),
            torch.tensor([[0.0, 0.0, 1.0]], device=ee_quat.device, dtype=ee_quat.dtype),
        ).reshape(-1)
        a_z = float(flange_z[2])
        if abs(a_z) < 0.3:
            a_z = -1.0 if a_z <= 0 else 1.0

        delta = yaw / a_z

        return delta * self.ROTATE_WRIST_SIGN

    def _rotate_sub_targets(
        self,
        simulator: Any,
        generated_values: dict,
        env_index: int,
        target_quat: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Goal quaternions this rotate should be executed against, in order.

        A rotate's symbolic effect is one ``yaw_next`` step (90 deg), but its
        goal quaternion is the *canonical* ``ORI_QUATERNIONS[ori2]``, so the
        action also has to absorb whatever yaw the block picked up from the
        reset-state spawn perturbation.  Those add: the true rotation is
        ``90 deg -/+ psi`` with ``|psi| <= 30 deg``.

        When the total exceeds 90 deg, return two goals -- first the canonical
        pose of the block's *current* ori (an align-only correction that leaves
        the symbolic ori untouched), then the real target.  Otherwise return the
        single target, which is the pre-existing behavior.
        """
        single = [target_quat]
        try:
            obj_pose = simulator.get_pose(generated_values["obj"], env_idx=env_index, local=True)
            obj_quat = torch.as_tensor(
                obj_pose, device=target_quat.device, dtype=torch.float32
            ).reshape(-1)[3:7]

            total_yaw = self._yaw_about_z(obj_quat, target_quat)
            if total_yaw is None:
                return single
            limit = float(np.pi / 2) + float(np.deg2rad(self.ROTATE_SPLIT_TOLERANCE_DEG))
            if abs(total_yaw) <= limit:
                return single

            ori_name = generated_values.get("ori")
            if ori_name not in ORI_QUATERNIONS:
                print(
                    f"[RotateTrajectory] rotate needs {np.rad2deg(total_yaw):.1f} deg but "
                    f"source ori {ori_name!r} has no canonical quaternion; not splitting"
                )
                return single

            aligned = torch.tensor(
                ORI_QUATERNIONS[ori_name], device=target_quat.device, dtype=torch.float32
            ).reshape(-1)[:4]
            # obj_quat = q_psi (x) canonical(ori) with q_psi a world-frame yaw, so
            # this relative rotation is pure-z by construction and _yaw_about_z
            # returns -psi rather than None.
            correction = self._yaw_about_z(obj_quat, aligned)
            if correction is None or abs(correction) < float(np.deg2rad(self.ROTATE_MIN_CORRECTION_DEG)):
                return single

            print(
                f"[RotateTrajectory] splitting {np.rad2deg(total_yaw):.1f} deg rotate into "
                f"align {np.rad2deg(correction):.1f} deg + step "
                f"{np.rad2deg(total_yaw - correction):.1f} deg"
            )
            return [aligned, target_quat]
        except Exception as exc:
            print(f"[RotateTrajectory] rotate sub-target split unavailable ({exc}); using single goal")
            return single

    def _solve_anchor(
        self,
        robot: Any,
        pos: torch.Tensor,
        quat: torch.Tensor,
        env_index: int,
        name: str,
    ) -> torch.Tensor | None:
        target_pose = Pose(position=pos.reshape(1, 3), quaternion=quat.reshape(1, 4))
        ik = self._solve_ik(robot, target_pose, env_index)
        target_q = self._ik_solution_to_joint_target(ik, robot)
        if target_q is None:
            print(f"[RotateTrajectory] IK failed for scripted rotate {name}: pos={pos.detach().cpu().tolist()}")
            return None
        return self._clamp_franka_joints(target_q.to(pos.device))

    def _build_scripted_rotate_anchors(
        self,
        robot: Any,
        simulator: Any,
        generated_values: dict,
        env_index: int,
        target_quat: torch.Tensor,
        current_q: torch.Tensor,
        fallback_direction: str | None = None,
        need_clearance: bool = False,
    ) -> dict | None:
        try:
            obj_name: str = generated_values["obj"]

            ee_pose = robot.get_curr_pose(env_index)
            ee_quat = ee_pose.quaternion.reshape(-1)[:4].to(current_q.device, torch.float32)

            obj_pose = simulator.get_pose(obj_name, env_idx=env_index, local=True)
            obj_pose = torch.as_tensor(obj_pose, device=current_q.device, dtype=torch.float32).reshape(-1)
            obj_pos = obj_pose[:3]
            obj_quat = obj_pose[3:7]

            world_z = torch.tensor([0.0, 0.0, 1.0], device=current_q.device, dtype=torch.float32)
            half_z = self._block_world_half_height(simulator, obj_name, obj_quat, env_index)
            tip_offset = (
                self.ROTATE_ROBOTIQ_EE_TO_TIP_OFFSET
                if getattr(robot, "ROBOTIQ", False)
                else self.ROTATE_EE_TO_TIP_OFFSET
            )
            grasp_pos = obj_pos.clone()
            grasp_pos[2] = (
                obj_pos[2] - half_z + 2.0 * half_z * self.ROTATE_GRASP_TIP_HEIGHT_FRAC + tip_offset
            )
            lift_pos = grasp_pos + world_z * self.ROTATE_LIFT_HEIGHT
            clear_pos = grasp_pos + world_z * self.ROTATE_INTERSTAGE_LIFT_HEIGHT

            prepared = self._prepare_contact_ik_world(robot, env_index, obj_name)
            try:
                if not hasattr(robot, "get_curr_pose"):
                    return None
                q_grasp = self._solve_anchor(robot, grasp_pos, ee_quat, env_index, "grasp")
                q_lift = self._solve_anchor(robot, lift_pos, ee_quat, env_index, "lift")
                # Only a sub-step that hands over to another one needs the taller
                # clearance; the final sub-step is followed by rotate_retract_traj.
                q_clear = (
                    self._solve_anchor(robot, clear_pos, ee_quat, env_index, "interstage_clear")
                    if need_clearance else None
                )
            finally:
                self._restore_ik_world(robot, env_index, prepared)

            if q_grasp is None or q_lift is None:
                return None

            wrist_delta = self._wrist_delta(obj_quat, target_quat, ee_quat, fallback_direction)
            e6 = torch.zeros_like(q_grasp)
            e6[6] = 1.0
            # Twist rotates ONLY the last arm joint (panda_joint7).
            q_lift_rot = self._clamp_franka_joints(q_lift + wrist_delta * e6)
            q_grasp_rot = self._clamp_franka_joints(q_grasp + wrist_delta * e6)
            # Keep the twisted wrist on the way up: unwinding joint 7 here would
            # just be stretched out by the gripper speed cap for no benefit, and
            # the block has already been released.
            q_clear_rot = (
                self._clamp_franka_joints(q_clear + wrist_delta * e6)
                if q_clear is not None else None
            )

            return {
                "q_grasp": q_grasp,
                "q_lift": q_lift,
                "q_lift_rot": q_lift_rot,
                "q_grasp_rot": q_grasp_rot,
                "q_clear_rot": q_clear_rot,
                "wrist_delta": wrist_delta,
            }
        except Exception as exc:
            print(f"[RotateTrajectory] scripted rotate anchors unavailable: {exc}")
            return None
