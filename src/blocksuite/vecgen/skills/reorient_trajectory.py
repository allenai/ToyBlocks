"""Scripted-flip geometry for the vecgen flip skill.

The constants and methods the flip skills call (tip offset, waypoint geometry, press depth,
anchor cost), ported from the former TAMP reorient executor.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from blocksuite.vecgen.skills._quat import (
    quat_apply,
    quat_apply_inverse,
    quat_from_angle_axis,
    quat_mul,
)


class ReOrientTrajectory:
    def __init__(self, argument_name: str) -> None:
        self.argument_name = argument_name

    # Scripted contact-flip defaults (ported from TAMP blocks-classic).
    # The flip is a heuristic waypoint sequence — hover over the block's top
    # edge, press into it, pull along the flip direction, retreat — with each
    # waypoint turned into a joint target via IK, executed under the soft
    # "flip" stiffness profile so contact forces stay bounded.
    FLIP_ACTION_DT: float = 0.02
    FLIP_DECIMATION: int = 4
    # Tip placement along the edge axis, positive = inside the block. The
    # gripper is aligned to the pull direction for every flip, so one inset
    # serves all directions (the classic -5 mm outward inset for side flips
    # belonged to its fixed gripper orientation). The compliant arm's tip
    # drifts ~1 cm toward the robot while descending onto the edge, so with no
    # inset forward flips (edge nearest the robot) pressed past the edge and
    # dragged air (4/50 in play); 8 mm keeps the tip on the top face, and the
    # drag slides it onto the edge.
    FLIP_EDGE_INSET: float = 0.008
    # Press depth below the top face. The PD arm cannot enter the block, so
    # the commanded depth sets the normal force (~k_cart * depth), and it is
    # that force's friction that drags the edge over. Recorded flips that
    # succeeded pressed ~40 mm (by accident, through the tip-offset error) for
    # 40-55 N at kp=300; a 3 mm press gave 2-6 N and the tip slid over the
    # edge with the block tilting <10 deg. The depth is scaled inversely with
    # the active profile's first-joint kp so the force, not the depth, stays
    # constant across profiles (kp=600 -> ~10 mm, which is why "stiff" used to
    # scoot blocks with the fixed 40 mm), and capped so the tip stays
    # FLIP_TABLE_CLEARANCE above the table for small blocks (the legacy 40 mm
    # on a 30 mm block put the IK target under the table -> IK failure ->
    # blind joint fallback). 20 mm (~25 N) is the middle ground: at 35-40 mm
    # the arm deflected 15-25 mm sideways under load before the pull even
    # started, which yawed the block by 15-35 deg (the dominant baseline
    # failure); at 3-6 mm the tip slid over the edge instead of dragging it.
    FLIP_PRESS_DEPTH_REF: float = 0.020
    FLIP_PRESS_REF_KP: float = 300.0
    # Friction-adaptive press: the drag works through tip friction mu*N, and
    # the block's static friction is randomized 0.1-0.4 per env, so the press
    # is scaled by FLIP_PRESS_REF_FRICTION / mu (gain >= 1, capped). At mu=0.11
    # a 20-30 N press never tipped the block (12-50 deg tilts that fell back)
    # while mu>=0.24 blocks flipped at the same press.
    FLIP_PRESS_REF_FRICTION: float = 0.3
    FLIP_PRESS_MAX_FRICTION_GAIN: float = 2.5
    FLIP_CONTACT_DEPTH_MIN: float = 0.003
    FLIP_CONTACT_DEPTH_MAX: float = 0.05
    FLIP_TABLE_CLEARANCE: float = 0.005
    FLIP_FREE_MOTION_DURATION: float = 0.20
    # Commanded gripper (EE) rotation speed cap for every scripted segment;
    # segments that would turn the gripper faster are stretched. Each joint
    # target is held for one recorded frame (FLIP_DECIMATION physics steps at
    # 0.01 s = one 25 Hz render interval), so FLIP_TARGET_PERIOD, not
    # FLIP_ACTION_DT, is the sim time a target lasts.
    FLIP_MAX_GRIPPER_ANGULAR_SPEED: float = 1.0
    FLIP_TARGET_PERIOD: float = 0.04
    # FK samples along a joint-space segment when measuring its gripper rotation.
    FLIP_ROTATION_SAMPLES: int = 64
    # Minimum sim time of the free motion into the edge-align pose on the first
    # side (left/right) flip attempt, where the gripper yaws onto the pull
    # direction. Retries start from the already-yawed retreat pose.
    FLIP_SIDE_ROTATE_MIN_TIME: float = 1.0
    FLIP_CONTACT_DESCENT_DURATION: float = 0.20
    FLIP_CONTACT_SETTLE_DURATION: float = 0.10
    # Hold over the edge before descending so the overshoot of the approach
    # (velocity feedforward) settles above the block instead of carrying the
    # tip past the edge.
    FLIP_HOVER_SETTLE_DURATION: float = 0.15
    # Pull length scales with block height: the top edge only has to travel
    # ~H*sin(45deg) to pass the tipping point, and a fixed 9 cm drag under a
    # firm press rolled small blocks a second time (180 deg results).
    FLIP_PULL_DISTANCE: float = 0.11
    FLIP_PULL_DISTANCE_MIN: float = 0.045
    FLIP_PULL_HEIGHT_FACTOR: float = 1.6
    # Pull-down bias (fraction of block height) applied from the first attempt:
    # the contact corner descends by H*(1-cos(theta)) as the block pivots, so a
    # level pull loses it just before the tipping point (observed as 20-40 deg
    # tilts that fall back). Retries use the larger retry fraction below.
    FLIP_PULL_DOWN_BIAS: float = 0.0
    FLIP_PULL_DOWN_HEIGHT_FRACTION: float = 0.2
    FLIP_PULL_DURATION: float = 0.275
    FLIP_PULL_HOLD_DURATION: float = 0.15
    FLIP_RETREAT_HEIGHT: float = 0.12
    FLIP_RETREAT_DURATION: float = 1.00
    FLIP_EDGE_HOVER_HEIGHT: float = 0.012
    # IK link -> closed fingertip contact surface, along the approach axis.
    # 0.1348 is the Robotiq *pad center* with the gripper open (the grasp
    # TCP), not the closed fingertip: the pad bottom is 0.1589 m below
    # robotiq_arg2f_base_link when open and 0.173 m when fully closed (gripper
    # USD collision extents + URDF FK at finger_joint=0.9). With 0.1348 the tip
    # landed ~3.8 cm deeper than every waypoint intended and pinned the block
    # instead of hooking its top edge (the pin force scales with kp, which is
    # what made the flip look stiffness-sensitive). Kept for the Panda hand
    # only; the Robotiq path measures the closed fingertip live (below) and
    # falls back to FLIP_ROBOTIQ_EE_TO_TIP_OFFSET.
    FLIP_EE_TO_TIP_OFFSET: float = 0.1348
    FLIP_ROBOTIQ_EE_TO_TIP_OFFSET: float = 0.173
    # Live fingertip: lowest point of the inner-finger pad in the
    # ``*_inner_finger`` body frame (pad bottom, from the gripper USD), read
    # from the articulation once the gripper has closed. Same contact-point
    # model as flip_env, at the pad bottom rather than the pad middle.
    FLIP_ROBOTIQ_FINGER_TIP_LOCAL: tuple[float, float, float] = (0.0433, 0.0, 0.1481)
    FLIP_ROBOTIQ_FINGER_BODY_NAMES: tuple[str, str] = ("left_inner_finger", "right_inner_finger")
    FLIP_PANDA_FULL_CLOSE_TARGET: float = 0.0
    FLIP_ROBOTIQ_FULL_CLOSE_TARGET: float = 0.9
    # Wait for the fingers to actually close before moving: the tip offset is
    # measured on the closed gripper, and an open Robotiq straddles a small
    # block instead of pressing its top edge (the legacy 0 let the fingers ramp
    # closed during the 0.2 s free-motion segment, which is not enough; 20
    # steps measured only finger_joint=0.5 with a 4.1 cm tip gap).
    FLIP_GRIPPER_FULL_CLOSE_WAIT_STEPS: int = 40
    # Seeded IK: each waypoint is solved with the previous anchor as seed and
    # retract config so consecutive anchors stay on one arm configuration.
    # Unseeded solves jumped up to ~0.3 rad between the hover and contact
    # anchors, and joint-space interpolation across such a jump sweeps the tip
    # sideways during a 1.5 cm descent. Larger jumps are logged.
    FLIP_IK_MAX_JOINT_JUMP_RAD: float = 0.6
    # The fingertip footprint is symmetric under a 180 deg gripper yaw, so every
    # flip has two equivalent gripper orientations. Solves IK for both
    # and keeps the better-conditioned one instead of always taking
    # the (-90, 90] wrap, which near the robot base folds the arm.
    FLIP_YAW_SELECT_DEFAULT: bool = True
    # When no IK solution exists, retry with a
    # shorter drag (the drag end is the waypoint that leaves the workspace near
    # the base) and, failing that, skip the attempt instead of running the
    # fixed joint-offset fallback, which cannot flip anything.
    FLIP_NO_BLIND_FALLBACK_DEFAULT: bool = True
    FLIP_PULL_SCALES: tuple[float, ...] = (1.0, 0.75, 0.55)
    # Solve the drag (contact -> pull) at evenly spaced
    # points on the straight fingertip line, each IK seeded from the previous one,
    # and execute those dense targets. Linear joint interpolation between the two
    # anchors swings the tip off the edge whenever they sit in different arm
    # configurations (a >1 rad joint jump at 'pull' near the robot), so the block
    # never tips. A drag needing a step above FLIP_DRAG_MAX_STEP_RAD is rejected.
    FLIP_CARTESIAN_DRAG_DEFAULT: bool = True
    FLIP_DRAG_STEPS: int = 12
    FLIP_DRAG_MAX_STEP_RAD: float = 0.35
    # Extra IK seeds for the edge-align pose (with the Cartesian drag): the arm may
    # change configuration during the free move to the edge (in the air) instead of
    # mid-drag. First: mean start configuration of successful forward flips in play
    # (sep30_play_100_small_rr, more extended elbow than failures); second: Franka ready.
    # Each retry drags this fraction further. Near the robot, forward flips used to tip
    # 20-45 deg and fall back: the drag ended just short of the tipping point, and
    # retries only pressed deeper.
    FLIP_RETRY_PULL_GROWTH: float = 0.25
    FLIP_EDGE_SEEDS: tuple[tuple[float, ...], ...] = (
        (0.03, 0.30, 0.03, -2.16, 0.01, 2.52, 0.14),
        (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785),
    )
    FLIP_STIFFNESS_SETTLE_STEPS: int = 0
    # Extra scripted-flip attempts when the settled block misses the target
    # orientation (the dominant failure: the fingertip grazes the top edge and
    # the block scoots away instead of tipping). Waypoints are rebuilt from the
    # block's current pose each attempt.
    FLIP_SCRIPTED_RETRIES: int = 2
    # Retry escalation: repeating identical waypoints fails identically (the
    # miss is geometric, not stochastic), so each retry presses deeper into the
    # edge and holds contact longer before pulling. Depth stays capped by the
    # table clearance (see _retry_contact_depth).
    FLIP_RETRY_CONTACT_DEPTH_STEP: float = 0.004
    FLIP_RETRY_CONTACT_SETTLE_DURATION: float = 0.25
    # Retries bias the pull further downward than the first attempt
    # (FLIP_PULL_DOWN_HEIGHT_FRACTION). The contact corner descends by
    # H*(1-cos(theta)) as the block pivots — ~0.3*H at the 45-degree tipping
    # point — so a too-level pull loses the corner just before it commits
    # (observed as 20-46 degree near-miss tilts). Following the corner down by
    # ~0.35*H keeps the tip pressing through the tipping point; the bias is
    # capped so the pull target stays FLIP_TABLE_CLEARANCE above the table.
    FLIP_RETRY_PULL_DOWN_HEIGHT_FRACTION: float = 0.35
    # Low-level action recorded for each attempt's retreat (and the settle after it). The reorient
    # action's own reorient_retract_traj step is skipped for the scripted flip (LiftTrajectory), so
    # the flip labels its built-in retreat with that name, as the other skills' retracts are labeled.
    RETRACT_LL_ACTION: str = "reorient_retract_traj"

    @staticmethod
    def _direction_vector(direction: str, device: torch.device) -> torch.Tensor:
        mapping = {
            "forward": (-1.0, 0.0, 0.0),
            "backward": (1.0, 0.0, 0.0),
            "left_flip": (0.0, -1.0, 0.0),
            "left": (0.0, -1.0, 0.0),
            "right_flip": (0.0, 1.0, 0.0),
            "right": (0.0, 1.0, 0.0),
        }
        return torch.tensor(
            mapping.get(str(direction).strip().lower(), (-1.0, 0.0, 0.0)),
            device=device,
            dtype=torch.float32,
        )

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

    @staticmethod
    def _forward_flip_gripper_quat(device: torch.device) -> torch.Tensor:
        return torch.tensor(
            [0.0, -0.70710678, 0.70710678, 0.0],
            device=device,
            dtype=torch.float32,
        )

    @classmethod
    def _yawed_flip_gripper_quat(cls, device: torch.device, yaw_rad: float) -> torch.Tensor:
        """Flip gripper orientation rotated about world z by ``yaw_rad``.

        The legacy orientation is fixed in the world frame; rotating it by the
        block's yaw (relative to the requested world flip direction) keeps the
        fingertip square to the edge it presses, for any block yaw.
        """
        base = cls._forward_flip_gripper_quat(device)
        if abs(float(yaw_rad)) < 1e-6:
            return base
        yaw_q = quat_from_angle_axis(
            torch.tensor([float(yaw_rad)], device=device, dtype=torch.float32),
            torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float32),
        )
        return quat_mul(yaw_q, base.reshape(1, 4)).reshape(4)

    @classmethod
    def _constant_ee_to_tip_offset(cls, robot: Any, device: torch.device) -> torch.Tensor:
        """Fallback IK-link -> fingertip offset (IK-link frame)."""
        z = cls.FLIP_ROBOTIQ_EE_TO_TIP_OFFSET if getattr(robot, "ROBOTIQ", False) else cls.FLIP_EE_TO_TIP_OFFSET
        return torch.tensor([0.0, 0.0, z], device=device, dtype=torch.float32)

    def _measure_ee_to_tip_offset(self, robot: Any, env_index: int) -> torch.Tensor | None:
        """Closed-fingertip midpoint relative to the IK link, in the IK-link frame.

        Reads the live inner-finger body poses (Robotiq only), so the offset is
        exact for the gripper as it actually closed in this sim. Returns
        ``None`` (caller uses the constant) when unavailable or when the reading
        is implausible (e.g. the fingers did not close).
        """
        if not getattr(robot, "ROBOTIQ", False):
            return None
        art = getattr(robot, "_robot", None)
        ee_ids = getattr(robot, "_ee_link_id", None)
        if art is None or not ee_ids:
            return None
        try:
            finger_ids, _ = art.find_bodies(list(self.FLIP_ROBOTIQ_FINGER_BODY_NAMES))
            body_pos = art.data.body_pos_w[env_index]
            body_quat = art.data.body_quat_w[env_index]
            tip_local = torch.tensor(
                self.FLIP_ROBOTIQ_FINGER_TIP_LOCAL, device=body_pos.device, dtype=body_pos.dtype
            ).expand(len(finger_ids), 3)
            tips = body_pos[finger_ids] + quat_apply(body_quat[finger_ids], tip_local)
            tip_mid = tips.mean(dim=0)
            ee_pos = body_pos[ee_ids[0]]
            ee_quat = body_quat[ee_ids[0]]
            offset = quat_apply_inverse(ee_quat, tip_mid - ee_pos).to(dtype=torch.float32)
            gap = float(torch.linalg.norm(tips[0] - tips[1]))
            finger_q = None
            gripper_ids = getattr(robot, "_gripper_joint_ids", None)
            if gripper_ids is not None:
                finger_q = float(art.data.joint_pos[env_index, gripper_ids].reshape(-1)[0])
            print(
                f"[ReOrientTrajectory] env {env_index}: measured ee->tip offset "
                f"{[round(v, 4) for v in offset.tolist()]} m (tip gap {gap:.4f} m, "
                f"finger_joint={finger_q if finger_q is None else round(finger_q, 3)}; "
                f"legacy constant 0.1348)"
            )
            if not (0.12 <= float(offset[2]) <= 0.20) or float(torch.linalg.norm(offset[:2])) > 0.02:
                print(
                    f"[ReOrientTrajectory] env {env_index}: implausible tip offset, "
                    "falling back to the constant"
                )
                return None
            return offset
        except Exception as exc:
            print(f"[ReOrientTrajectory] tip offset measurement unavailable: {exc}")
            return None

    @staticmethod
    def _block_half_extents(simulator: Any, obj_name: str, env_index: int, device: torch.device) -> torch.Tensor:
        """Per-env half extents of the block in its local frame: the collider's, when the
        simulator measured them (vecgen), else the spawn bbox."""
        from blocksuite.utils import obs_utils

        measured = getattr(simulator, "block_half_extents", None)
        half = measured(obj_name, env_index) if measured is not None else None
        if half is not None:
            return half.to(device=device, dtype=torch.float32)

        bbox = obs_utils.get_bounding_box(simulator.scene[obj_name])  # (num_envs, 2, 3)
        return (0.5 * (bbox[env_index, 1] - bbox[env_index, 0])).to(device=device, dtype=torch.float32)

    @classmethod
    def _press_depth_for_profile(cls, profile: str) -> float:
        """Commanded press depth for ``profile``, the stiffness profile the arm runs the whole
        flip at (vecgen runs every phase at one profile)."""
        kp = cls.FLIP_PRESS_REF_KP
        try:
            from blocksuite.utils.stiffness_profiles import get_profile

            kp = float(get_profile(profile).stiffness[0])
        except Exception as exc:
            print(f"[ReOrientTrajectory] press depth: profile lookup failed ({exc}); assuming kp={kp}")
        depth = cls.FLIP_PRESS_DEPTH_REF * cls.FLIP_PRESS_REF_KP / max(kp, 1.0)
        return max(cls.FLIP_CONTACT_DEPTH_MIN, min(depth, cls.FLIP_CONTACT_DEPTH_MAX))

    @classmethod
    def _press_friction_gain(cls, simulator: Any, obj_name: str, env_index: int) -> float:
        """Press multiplier from the block's static friction (1.0 when unknown)."""
        try:
            mat = simulator.scene[obj_name].root_physx_view.get_material_properties()
            mu = float(mat[env_index, 0, 0])
        except Exception:
            return 1.0
        if not (mu > 1e-3):
            return cls.FLIP_PRESS_MAX_FRICTION_GAIN
        return max(1.0, min(cls.FLIP_PRESS_REF_FRICTION / mu, cls.FLIP_PRESS_MAX_FRICTION_GAIN))

    def _retry_contact_depth(self, attempt: int, top_half_extent: float, friction_gain: float = 1.0) -> float:
        """Contact press depth for a given attempt, kept clear of the table."""
        base = getattr(self, "_flip_press_depth", None)
        if base is None:
            base = self._press_depth_for_profile("medium")
        depth = base * friction_gain + attempt * self.FLIP_RETRY_CONTACT_DEPTH_STEP
        table_cap = 2.0 * top_half_extent - self.FLIP_TABLE_CLEARANCE
        return max(self.FLIP_CONTACT_DEPTH_MIN, min(depth, table_cap))

    def _build_scripted_flip_waypoint_positions(
        self,
        simulator: Any,
        env_index: int,
        obj_name: str,
        obj_pose: torch.Tensor,
        direction: str,
        attempt: int = 0,
        pull_scale: float = 1.0,
    ) -> tuple[list[torch.Tensor], float]:
        """Fingertip waypoints (world) and the gripper yaw offset for this flip.

        The edge point has always been built from the block's own axes; the
        pull now runs along the block's edge axis too (projected to the table
        plane) instead of the raw world direction, and the returned yaw is the
        gripper rotation about world z that aligns the forward-flip fingertip
        footprint with that pull direction (wrapped to (-90, 90] deg). A pull
        along the world axis on a yawed block has a component along the edge,
        which yaws the block, slides the tip toward a corner, and tumbles the
        block onto a neighbouring face.
        """
        obj_pos = obj_pose[:3]
        obj_quat = obj_pose[3:7]
        device = obj_pose.device

        half_extents = self._block_half_extents(simulator, obj_name, env_index, device)

        x_axis = quat_apply(obj_quat, torch.tensor([1.0, 0.0, 0.0], device=device))
        y_axis = quat_apply(obj_quat, torch.tensor([0.0, 1.0, 0.0], device=device))
        z_axis_obj = quat_apply(obj_quat, torch.tensor([0.0, 0.0, 1.0], device=device))
        world_z = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)
        push_axis = self._direction_vector(direction, device)

        axes = [x_axis, y_axis, z_axis_obj]
        top_axis_idx = int(np.argmax([torch.abs(torch.dot(axis, world_z)).item() for axis in axes]))
        top_axis = axes[top_axis_idx].clone()
        if torch.dot(top_axis, world_z).item() < 0.0:
            top_axis = -top_axis

        push_axis_xy = push_axis[:2] / torch.clamp(torch.linalg.norm(push_axis[:2]), min=1e-8)
        candidate_indices = [idx for idx in range(3) if idx != top_axis_idx]
        candidates = [axes[idx] for idx in candidate_indices]
        lateral_dots = [
            torch.abs(torch.dot(axis[:2], push_axis_xy)).item()
            for axis in candidates
        ]
        edge_candidate_idx = int(np.argmax(lateral_dots))
        edge_axis = candidates[edge_candidate_idx].clone()
        if torch.dot(edge_axis[:2], push_axis_xy).item() < 0.0:
            edge_axis = -edge_axis
        # All directions use the same drag flip: hook the top edge on the pull
        # side and drag it outward. The classic port special-cased left flips
        # to the opposite (trailing) edge and pushed across the top; with the
        # tip at its true height that press lands on the top face and shoves
        # the block onto arbitrary faces (0/32 on left-flip problems).
        edge_axis_idx = candidate_indices[edge_candidate_idx]

        # Block-frame pull direction: the edge axis flattened onto the table,
        # signed to match the requested world direction.
        pull_dir = edge_axis.clone()
        pull_dir[2] = 0.0
        pull_dir = pull_dir / torch.clamp(torch.linalg.norm(pull_dir), min=1e-8)
        if torch.dot(pull_dir[:2], push_axis_xy).item() < 0.0:
            pull_dir = -pull_dir
        # Gripper yaw: rotate the forward-flip gripper orientation so the
        # fingertip footprint relative to the pull is the same for every
        # direction. The closed Robotiq tip is ~22 mm wide across the closing
        # axis but only ~8 mm along it; the legacy fixed orientation put the
        # thin dimension along the pull for side flips, so a tip placed on the
        # edge overlapped the block by about a millimetre and dragged air. The
        # footprint is symmetric under 180 deg, so wrap to (-90, 90] deg.
        forward_xy = torch.tensor([-1.0, 0.0], device=device, dtype=torch.float32)
        flip_yaw = math.atan2(
            float(forward_xy[0] * pull_dir[1] - forward_xy[1] * pull_dir[0]),
            float(torch.dot(forward_xy, pull_dir[:2])),
        )
        if flip_yaw > math.pi / 2.0:
            flip_yaw -= math.pi
        elif flip_yaw <= -math.pi / 2.0:
            flip_yaw += math.pi

        top_center = obj_pos + top_axis * half_extents[top_axis_idx]
        edge_inset = min(
            self.FLIP_EDGE_INSET,
            0.5 * float(half_extents[edge_axis_idx]),
        )
        edge_extent = max(0.0, float(half_extents[edge_axis_idx]) - edge_inset)
        edge_point = top_center + edge_axis * edge_extent
        contact_depth = self._retry_contact_depth(
            attempt,
            float(half_extents[top_axis_idx]),
            friction_gain=self._press_friction_gain(simulator, obj_name, env_index),
        )
        contact_pos = edge_point - top_axis * contact_depth
        block_height = 2.0 * float(half_extents[top_axis_idx])
        down_fraction = (
            self.FLIP_PULL_DOWN_HEIGHT_FRACTION if attempt == 0 else self.FLIP_RETRY_PULL_DOWN_HEIGHT_FRACTION
        )
        # Never let the pull target drop below FLIP_TABLE_CLEARANCE above the table.
        pull_down_bias = max(
            self.FLIP_PULL_DOWN_BIAS,
            min(
                down_fraction * block_height,
                (block_height - contact_depth) - self.FLIP_TABLE_CLEARANCE,
            ),
        )
        pull_factor = self.FLIP_PULL_HEIGHT_FACTOR
        pull_max = self.FLIP_PULL_DISTANCE
        retry_growth = self.FLIP_RETRY_PULL_GROWTH
        pull_distance = min(
            pull_max,
            max(self.FLIP_PULL_DISTANCE_MIN, pull_factor * block_height),
        ) * float(pull_scale) * (1.0 + retry_growth * attempt)
        pull_pos = contact_pos + pull_dir * pull_distance - top_axis * pull_down_bias
        return [
            edge_point + world_z * self.FLIP_EDGE_HOVER_HEIGHT,
            contact_pos,
            pull_pos,
            pull_pos + world_z * self.FLIP_RETREAT_HEIGHT,
        ], flip_yaw

    def _flip_anchor_cost(self, anchors: list[torch.Tensor], drag_path: list[torch.Tensor] | None = None) -> float:
        """Lower is better: no reconfiguration during contact, margin from joint limits, little travel."""
        lower = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], device=anchors[0].device)
        upper = torch.tensor([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], device=anchors[0].device)
        span = upper - lower
        if drag_path:
            # contact -> pull is the dense path; score its largest step, plus the other contact-phase jumps
            pts = [anchors[2], *drag_path]
            drag_step = max(float((b[:7] - a[:7]).abs().max()) for a, b in zip(pts[:-1], pts[1:]))
            other = [float((anchors[2][:7] - anchors[1][:7]).abs().max()), float((anchors[4][:7] - anchors[3][:7]).abs().max())]
            contact_jump = max([drag_step * (self.FLIP_DRAG_STEPS / 2.0)] + other)
        else:
            contact_jump = max(float((b[:7] - a[:7]).abs().max()) for a, b in zip(anchors[1:-1], anchors[2:]))
        margin = min(float(torch.min(torch.minimum(q[:7] - lower, upper - q[:7]) / span)) for q in anchors[1:])
        travel = float((anchors[1][:7] - anchors[0][:7]).abs().sum())
        jump_pen = 10.0 * max(0.0, contact_jump - self.FLIP_IK_MAX_JOINT_JUMP_RAD) + contact_jump
        return jump_pen + 4.0 * max(0.0, 0.08 - margin) / 0.08 + 0.05 * travel
