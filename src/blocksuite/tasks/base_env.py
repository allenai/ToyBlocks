"""
BaseEnv class that modifies the DirectRLEnv class with some useful utilities for block suite.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import isaaclab.sim as sim_utils
import omni.kit.app  # noqa
import omni.physx  # noqa
import torch
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.rigid_object.rigid_object import RigidObject
from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.envs.common import VecEnvObs, VecEnvStepReturn
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.markers.visualization_markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors.camera.camera import Camera
from isaaclab.sensors.contact_sensor import ContactSensor, ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer import FrameTransformer
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_apply, quat_mul
from isaacsim.core.prims import XFormPrim
from isaacsim.core.simulation_manager import SimulationManager

from blocksuite.robots import REGISTERED_ROBOTS
from blocksuite.robots.kinematics import Kinematics, KinematicsCfg
from blocksuite.scenes import SCENE_PRESETS
from blocksuite.utils import common_utils, gym_utils, isaac_utils, randomization_utils
from blocksuite.utils.metrics import MetricsTracker

from .base_env_cfg import BaseEnvCfg


class BaseEnv(DirectRLEnv):
    cfg: BaseEnvCfg

    AGENT_OBS_KEYS: tuple[str, ...] = ("joint_pos", "joint_targets", "joint_vel", "ee_pos")
    """Canonical emission order of the ``agent`` observation group (see :meth:`_get_obs_robot`).
    This order defines the flat RL observation layout and must never change: RL checkpoints
    bake it into their input widths."""

    _logged_ee_pos_convention: bool = False
    """Class-level default; :meth:`_get_obs_robot` sets it per instance after logging
    which ``ee_pos`` convention (see :attr:`BaseEnvCfg.legacy_ee_pos`) is active."""

    cameras: dict[str, Camera] = dict()
    """dictionary of cameras available in the environment."""

    frame_transformers: dict[str, FrameTransformer] = dict()
    """dictionary of frame transformations available in the environment."""

    _DEBUG_GOAL_KEY: str = "goal"
    """key used for the goal entry in the debug marker dicts (see ``_setup_debug_block_markers``)."""

    ### Visualization / debug marker utilities specific to blocks ###

    @property
    def visualization_markers(self) -> dict:
        if hasattr(self, "cube_corner_visuals"):
            return {
                "block_arrow": self.block_arrow,
                "goal_arrow": self.goal_arrow,
                "cube_corners": self.cube_corner_visuals,
                "goal_cube_corners": self.goal_corner_visuals,
            }
        return {}

    def _setup_visualization_markers(self):
        if not hasattr(self, "cube_corner_visuals"):
            colors = [
                (1.0, 0.0, 0.0),
                (1.0, 0.5, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 1.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (1.0, 0.0, 1.0),
                (0.0, 1.0, 0.5),
            ]
            cube_corner_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/CubeCornerVisuals",
                markers={
                    f"corner_pt_visual_{i}": sim_utils.SphereCfg(
                        radius=0.01,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(colors[i])),
                    )
                    for i in range(8)
                },
            )
            self.cube_corner_visuals = VisualizationMarkers(cfg=cube_corner_cfg)
            self.cube_corner_visuals.set_visibility(self.cfg.enable_visual_markers)
            self.goal_corner_visuals = VisualizationMarkers(
                cfg=cube_corner_cfg.replace(prim_path="/Visuals/GoalCornerVisuals")
            )
            self.goal_corner_visuals.set_visibility(self.cfg.enable_visual_markers)
            self.block_arrow = VisualizationMarkers(
                cfg=GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/BlockArrow")
            )
            self.goal_arrow = VisualizationMarkers(
                cfg=BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/GoalArrow")
            )
            self.block_arrow.set_visibility(False)
            self.goal_arrow.set_visibility(False)
        # 3-axis frame arrows: red=X, green=Y, blue=Z; always created so they appear in camera recordings
        _frame_arrow_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/BlockFrameArrows",
            markers={
                "x_arrow": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.1, 0.01, 0.01),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
                "y_arrow": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.1, 0.01, 0.01),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
                "z_arrow": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.1, 0.01, 0.01),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                ),
            },
        )
        self.block_frame_arrows = VisualizationMarkers(cfg=_frame_arrow_cfg)
        self.block_frame_arrows.set_visibility(False)
        self.goal_frame_arrows = VisualizationMarkers(
            cfg=_frame_arrow_cfg.replace(prim_path="/Visuals/GoalFrameArrows")
        )
        self.goal_frame_arrows.set_visibility(False)

    def _update_visualization_markers(self):
        # Generic debug visualization
        if getattr(self.cfg, "enable_visual_markers", False):
            self._update_debug_block_markers()
        self._render_legacy_visualization_markers()
        # Both goal passes above read the same freshness flags, so they are consumed once here at the end of
        # the step. Any env that does not re-publish a goal before the next render stops being drawn.
        if hasattr(self, "_goal_vis_fresh"):
            self._goal_vis_fresh.zero_()

    def _render_legacy_visualization_markers(self) -> None:
        """Draw the pre-``--debug_vis`` block/goal marker sets (used by the viser viewer)."""
        if not hasattr(self, "cube_corner_visuals"):
            return
        if hasattr(self, "block") and hasattr(self, "block_corner_pts"):
            self.block_arrow.visualize(self.block.data.root_pos_w, self.block.data.root_quat_w)
            corner_translations = (
                self.block_corner_pts + self.scene.env_origins.unsqueeze(1)
            ).reshape(self.num_envs * 8, 3)
            self.cube_corner_visuals.visualize(
                translations=corner_translations, marker_indices=list(range(8)) * self.num_envs
            )
            if hasattr(self, "block_frame_arrows"):
                self._visualize_frame_arrows(
                    self.block_frame_arrows, self.block.data.root_pos_w, self.block.data.root_quat_w
                )
        # Legacy goal pass. Driven off the same per-env buffers as the debug markers so it no longer depends
        # on ``goal_pos``, whose frame differs per RL env. Both prims here are hidden whenever the debug
        # markers own the goal, so this leaves visibility alone and lets the debug pass consume the flags.
        if hasattr(self, "goal_frame_arrows"):
            self._visualize_goal_markers(
                self.goal_corner_visuals,
                self.goal_frame_arrows,
                consume_fresh=False,
                manage_visibility=False,
            )

    def _ensure_goal_visualization_state(self) -> None:
        """Lazily allocate the per-env goal visualization buffers."""
        if hasattr(self, "_goal_vis_corner_pts"):
            return
        self._goal_vis_corner_pts: torch.Tensor = torch.zeros(
            (self.num_envs, 8, 3), device=self.device
        )
        self._goal_vis_quat: torch.Tensor = torch.zeros((self.num_envs, 4), device=self.device)
        self._goal_vis_quat[:, 0] = 1.0
        self._goal_vis_fresh: torch.Tensor = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def set_goal_visualization(
        self, env_idx: int, goal_corner_pts: torch.Tensor, goal_quat: torch.Tensor
    ) -> None:
        """Publish a single env slot's RL skill goal for the debug markers.

        Called once per RL step by the skill executors (``_sync_visualization_data``). Writing one row at a
        time is what keeps one env's goal from being drawn in every other env: the executors run per env,
        so rebinding the whole batch tensor here would show env ``i``'s goal in the slots running unrelated
        actions.

        Args:
            env_idx: env slot the goal belongs to.
            goal_corner_pts: ``(8, 3)`` goal corner points in the **env-local** frame.
            goal_quat: ``(4,)`` goal orientation as ``(w, x, y, z)``.
        """
        self._ensure_goal_visualization_state()
        self._goal_vis_corner_pts[env_idx] = goal_corner_pts
        self._goal_vis_quat[env_idx] = goal_quat
        self._goal_vis_fresh[env_idx] = True

    def _visualize_goal_markers(
        self,
        corner_marker: VisualizationMarkers,
        arrow_marker: VisualizationMarkers,
        consume_fresh: bool = True,
        manage_visibility: bool = True,
    ) -> None:
        """Draw goal markers only for the env slots that published a goal since the last render.

        The freshness flags are consumed here, so a slot has to re-publish every step to stay drawn. That is
        what makes the goal disappear as soon as an env moves off an RL skill onto an analytic action
        (pick/place/pnp), which has no goal pose of its own.

        Args:
            corner_marker: marker set for the 8 goal corner points.
            arrow_marker: marker set for the goal frame arrows.
            consume_fresh: clear the freshness flags after drawing. The last caller in a step must pass
                True; earlier callers pass False so they see the same active set.
            manage_visibility: toggle the marker prims' visibility to match the active set. The legacy
                ``goal_corner_visuals`` pass leaves this False so it keeps whatever visibility
                ``_setup_debug_block_markers`` gave it (it is deliberately hidden when debug markers own
                the goal, to avoid drawing a second, larger set on top).
        """
        self._ensure_goal_visualization_state()
        active = self._goal_vis_fresh.nonzero(as_tuple=False).flatten()
        if consume_fresh:
            self._goal_vis_fresh.zero_()
        if active.numel() == 0:
            if manage_visibility:
                corner_marker.set_visibility(False)
                arrow_marker.set_visibility(False)
            return
        if manage_visibility:
            corner_marker.set_visibility(True)
            arrow_marker.set_visibility(True)
        n = int(active.numel())
        corners_w = self._goal_vis_corner_pts[active] + self.scene.env_origins[active].unsqueeze(1)
        corner_marker.visualize(
            translations=corners_w.reshape(n * 8, 3), marker_indices=list(range(8)) * n
        )
        # Anchored at the centroid of the goal corners so the arrows sit at the center of the corner
        # markers drawn above, independent of whatever frame each RL env keeps ``goal_pos`` in.
        self._visualize_frame_arrows(arrow_marker, corners_w.mean(dim=1), self._goal_vis_quat[active])

    def _visualize_frame_arrows(
        self, marker: VisualizationMarkers, pos: torch.Tensor, quat: torch.Tensor
    ):
        """Visualize 3 arrows aligned to the X (red), Y (green), Z (blue) axes of the given frame.

        ``pos``/``quat`` may cover a subset of the envs (see ``_visualize_goal_markers``), so the marker
        count is taken from the inputs rather than from ``num_envs``.
        """
        n = pos.shape[0]
        sqrt2_2 = 0.7071067811865476
        # R_z(+90°) rotates the X-axis arrow to point along +Y
        rz90 = torch.tensor(
            [[sqrt2_2, 0.0, 0.0, sqrt2_2]], device=self.device, dtype=quat.dtype
        ).expand(n, -1)
        # R_y(-90°) rotates the X-axis arrow to point along +Z
        ry_neg90 = torch.tensor(
            [[sqrt2_2, 0.0, -sqrt2_2, 0.0]], device=self.device, dtype=quat.dtype
        ).expand(n, -1)
        quat_y = quat_mul(quat, rz90)
        quat_z = quat_mul(quat, ry_neg90)
        translations = pos.repeat_interleave(3, dim=0)
        orientations = torch.stack([quat, quat_y, quat_z], dim=1).reshape(3 * n, 4)
        marker.visualize(
            translations=translations, orientations=orientations, marker_indices=[0, 1, 2] * n
        )

    def _setup_debug_block_markers(self):
        """Lazily create per-block debug markers (corner-point spheres + RGB frame arrows) for every block-like
        rigid object in the scene, plus one identically configured set for the goal pose.

        Used by ``execute_plans.py --debug_vis``. Generated TAMP scenes have several named blocks (e.g.
        ``block_r``, ``block_blue``) and no single ``block``/``goal``, so this is generic over all rigid objects
        whose name contains ``block``. Created lazily on the first update call so it works even for generated
        envs whose ``_after_scene_setup`` does not call ``super()``.

        The goal set goes through the same creation code as the blocks (same sphere radius, same per-corner
        colors, same arrow scales/colors), so corner ``i`` of the goal matches corner ``i`` of the block and the
        two sets coincide exactly when the block pose matches the goal pose.
        """
        self._debug_block_names = [n for n in self.scene.rigid_objects if "block" in n.lower()]
        corner_colors = [
            (1.0, 0.0, 0.0),
            (1.0, 0.5, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 1.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 0.5),
        ]
        frame_arrow_markers = {
            axis: sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.1, 0.01, 0.01),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
            for axis, color in (
                ("x_arrow", (1.0, 0.0, 0.0)),
                ("y_arrow", (0.0, 1.0, 0.0)),
                ("z_arrow", (0.0, 0.0, 1.0)),
            )
        }
        self._debug_corner_markers = {}
        self._debug_frame_arrows = {}
        # The goal is a virtual pose (no rigid object), but it is created here alongside the blocks so both
        # marker sets are built from the exact same configs.
        for name in [*self._debug_block_names, self._DEBUG_GOAL_KEY]:
            corner_cfg = VisualizationMarkersCfg(
                prim_path=f"/Visuals/Debug/{name}/Corners",
                markers={
                    f"corner_pt_visual_{i}": sim_utils.SphereCfg(
                        radius=0.01,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=corner_colors[i]),
                    )
                    for i in range(8)
                },
            )
            corner_marker = VisualizationMarkers(cfg=corner_cfg)
            corner_marker.set_visibility(True)
            self._debug_corner_markers[name] = corner_marker

            frame_arrows = VisualizationMarkers(
                cfg=VisualizationMarkersCfg(
                    prim_path=f"/Visuals/Debug/{name}/FrameArrows", markers=frame_arrow_markers
                )
            )
            frame_arrows.set_visibility(True)
            self._debug_frame_arrows[name] = frame_arrows
        # The generic corner visuals use a larger sphere radius and only track poses while an RL skill executor
        # is running, so leaving them on would draw a second, differently sized (and stale) set on top of these.
        if hasattr(self, "cube_corner_visuals"):
            self.cube_corner_visuals.set_visibility(False)
            self.goal_corner_visuals.set_visibility(False)
        self._debug_block_markers_ready = True

    def _update_debug_block_markers(self):
        """Update the per-block and goal debug corner-point and frame-arrow markers to track the current poses."""
        from blocksuite.utils import obs_utils

        if not getattr(self, "_debug_block_markers_ready", False):
            self._setup_debug_block_markers()
        for name in self._debug_block_names:
            obj = self.scene.rigid_objects[name]
            # cube_corner_points returns world-frame corners (num_envs, 8, 3), so no env-origin offset needed.
            corners_w = obs_utils.cube_corner_points(obj).reshape(self.num_envs * 8, 3)
            self._debug_corner_markers[name].visualize(
                translations=corners_w, marker_indices=list(range(8)) * self.num_envs
            )
            self._visualize_frame_arrows(
                self._debug_frame_arrows[name], obj.data.root_pos_w, obj.data.root_quat_w
            )
        # The goal is published per env by the RL skill executors (``set_goal_visualization``) from the same
        # cube_corner_points() call on the same block asset, so corner i of the goal is corner i of the block.
        # Only slots that published a goal this step are drawn, so the marker is absent for envs running an
        # analytic action rather than showing another env's goal or a leftover one from an earlier skill.
        self._visualize_goal_markers(
            self._debug_corner_markers[self._DEBUG_GOAL_KEY],
            self._debug_frame_arrows[self._DEBUG_GOAL_KEY],
            consume_fresh=False,
        )

    def _merge_instance_layout_json_into_cfg(self, cfg: BaseEnvCfg) -> None:
        """Optional hook before ``camera_overrides_json``; override in generated tasks."""
        pass

    def set_episode_goal(self, env_idx: int, pddl_goal: str) -> None:
        """Set the active PDDL goal for a single sub-env.

        Called by :func:`~blocksuite.utils.episode_queue.apply_setup` when the
        queue setup carries a ``pddl_goal`` field. Generated multi-goal envs
        override this to update ``_per_env_goal_idx``; the base implementation
        is a no-op (single-goal envs ignore it).
        """
        pass

    def request_episode_end(self, env_ids: Sequence[int]) -> None:
        """Externally request that the given sub-envs end their episode on the
        next :meth:`step` call.

        The request is treated as a *truncation* (like a timeout), so it
        survives ``cfg.ignore_terminations`` and the episode finalizes through
        the normal path: :meth:`evaluate` runs for the final step, success /
        fail metrics land in ``final_info``, and the env auto-resets.  Used by
        the eval harness when a client-side oracle reports its plan complete.
        """
        ids = torch.as_tensor(list(env_ids), device=self.device, dtype=torch.long)
        self._episode_end_requests[ids] = True

    def __init__(self, cfg: BaseEnvCfg, config_setup_only=False, **kwargs):
        self.cfg = cfg
        # Every entry point (evaluate.py --env-cfg.legacy-ee-pos, replay,
        # training, an RL checkpoint's recorded config.json) funnels through
        # here, so this is the one place that can keep the retired convention
        # from being reintroduced silently.
        if cfg.legacy_ee_pos:
            raise ValueError(
                "BaseEnvCfg.legacy_ee_pos is phased out and can no longer be enabled. "
                "ee_pos now comes from the robot config's ee_body_name in the env "
                "frame, matching what blocksuite.vecgen.recording.lerobot_recorder writes."
            )
        if cfg.scene_preset_id is not None:
            # SCENE_PRESETS holds module-level instances, so aliasing the dict
            # here would make every write below (the robot's cameras, the
            # instance-layout and override merges, the depth data_types) a
            # permanent edit to the preset that the next env in this process
            # inherits.
            cfg.camera_cfgs = copy.deepcopy(SCENE_PRESETS[cfg.scene_preset_id].camera_cfgs)
            self._scene_preset_cfg = SCENE_PRESETS[cfg.scene_preset_id]
        if cfg.robot_id is not None:
            self._robot_class = REGISTERED_ROBOTS[cfg.robot_id].robot_class
            self._robot_cfg = REGISTERED_ROBOTS[cfg.robot_id].cfg_dataclass()
            self._robot_cfg.articulation = self._robot_cfg.articulation.replace(
                prim_path="{ENV_REGEX_NS}/Robot"
            )
            for controller in self._robot_cfg.controllers.values():
                for _k, v in controller.items():
                    v.asset_name = "robot"
            if cfg.replay_gripper_close_target is not None:
                from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg

                close_target = float(cfg.replay_gripper_close_target)
                for controller in self._robot_cfg.controllers.values():
                    gripper_cfg = controller.get("gripper")
                    if isinstance(gripper_cfg, BinaryJointPositionActionCfg):
                        gripper_cfg.close_command_expr = {"finger_joint": close_target}
        else:
            raise NotImplementedError(
                "Currently not supported to make an environment without a robot."
            )
        if not isinstance(cfg.frame_transformers, dict):
            cfg.frame_transformers = self._robot_cfg.frame_transformers
        cfg.camera_cfgs.update(self._robot_cfg.cameras)
        for name in cfg.drop_cameras:
            cfg.camera_cfgs.pop(name, None)
        self._merge_instance_layout_json_into_cfg(cfg)
        common_utils.apply_camera_overrides_json(cfg)
        # TODO (stao): debatable whether we should stuff robot config into the main env config.
        isaac_utils.replace_env_regex_ns(cfg)
        isaac_utils.replace_env_regex_ns(self._robot_cfg)
        isaac_utils.replace_env_regex_ns(self._scene_preset_cfg)

        ### process observation mode ###
        obs_parts = set(cfg.obs_mode.split("+"))
        self._include_state_in_obs = "state" in obs_parts
        self._include_rgb_in_obs = "rgb" in obs_parts
        # "depth" adds metric depth + intrinsics to the per-camera entries the
        # "rgb" component already builds, so it is an extension of that group
        # rather than a standalone one.
        self._include_depth_in_obs = "depth" in obs_parts
        if self._include_depth_in_obs and not self._include_rgb_in_obs:
            raise ValueError(
                f"obs_mode={cfg.obs_mode!r} requests 'depth' without 'rgb', but depth "
                "is emitted alongside rgb under observations['images'][cam]; use e.g. "
                "'rgb+state+depth'."
            )
        if self._include_depth_in_obs:
            # Cameras are built from ``cfg.camera_cfgs`` inside
            # ``super().__init__()`` below, so the depth request declares the
            # sensor output it needs here instead of failing in
            # ``_get_observations`` for a preset that renders rgb only (e.g.
            # droid_franka's wrist camera).
            for cam_cfg in cfg.camera_cfgs.values():
                if "distance_to_image_plane" not in cam_cfg.data_types:
                    # New list, not append: the scene cfgs are private copies by
                    # now, but the robot's camera cfgs entered this dict from the
                    # robot preset above.
                    cam_cfg.data_types = [*cam_cfg.data_types, "distance_to_image_plane"]
                # Under "max" a background pixel comes back as the far clipping
                # plane, which for the front camera's (0.5, 2.4) range is a
                # plausible-looking 2.4 m surface that survives the policy's
                # MAX_DEPTH_M cutoff. "none" leaves it +inf so it is zeroed as
                # invalid, matching the recorded training configs.
                cam_cfg.depth_clipping_behavior = "none"

        if config_setup_only:
            # Pseudo-env for TAMP RL inference: skip Isaac sim setup. Mark closed so
            # DirectRLEnv.__del__ -> close() does not touch uninitialized attributes.
            self._is_closed = True
            return

        super().__init__(cfg, **kwargs)
        # Initialized here (not only in _after_scene_setup) so autogen tasks that
        # override _after_scene_setup without super() still work with _reset_idx.
        self.table_height_offset = torch.zeros(self.num_envs, device=self.device)
        # Per-env sampled friction (static, dynamic), stored for privileged/asymmetric-critic observations.
        # Populated in _reset_idx when randomize_friction is enabled; otherwise stays 0.
        self.table_friction = torch.zeros(self.num_envs, 2, device=self.device)
        self.block_friction = torch.zeros(self.num_envs, 2, device=self.device)
        ### Setup controllers based on manager style action configs ###
        self.robot.control_mode = self.cfg.control_mode  # set the initial default control mode

        ### setup various buffers ###
        self.actions = torch.zeros(size=self.action_space.shape, device=self.device)
        """buffer for raw actions sent to env.step calls"""
        self.prev_actions = torch.zeros(size=self.action_space.shape, device=self.device)
        """buffer for the previous step's raw actions, useful for action smoothness penalties"""
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        self.joint_targets = self.robot.data.joint_pos_target
        self._episode_end_requests = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool,
        )
        """per-env external episode-end requests; see :meth:`request_episode_end`."""
        # Allocated regardless of ``record_metrics``: :meth:`step` updates these
        # unconditionally, and the success buffers are also control inputs for
        # eval termination (see blocksuite.eval.run_eval), not just reported
        # metrics. ``record_metrics`` gates per-episode reporting, not the buffers.
        self._returns = torch.zeros(
            size=(self.num_envs,), device=self.device, dtype=torch.float32
        )
        self._success_once = torch.zeros(
            size=(self.num_envs,), device=self.device, dtype=torch.bool
        )
        self._success_streak = torch.zeros(
            size=(self.num_envs,), device=self.device, dtype=torch.long
        )
        self._fail_once = torch.zeros(
            size=(self.num_envs,), device=self.device, dtype=torch.bool
        )
        self._first_success_step = torch.full(
            (self.num_envs,),
            -1,
            device=self.device,
            dtype=torch.long,
        )

        self._after_scene_setup()
        self._setup_visualization_markers()

        ### Setup metrics tracker ###
        self._metrics_tracker: MetricsTracker | None = None
        self._tracker_stats: dict[str, torch.Tensor] = {}
        any_tracking = (
            self.cfg.track_joint_metrics or self.cfg.track_ee_metrics or self.cfg.track_collisions
        )
        if any_tracking:
            ee_frame = self.scene.sensors.get("frame_transformer:ee")
            contact_sensor = (
                self.scene.sensors.get("contact_sensor") if self.cfg.track_collisions else None
            )
            contact_filter_sensors: dict[str, ContactSensor] = {}
            if hasattr(self, "_collision_filter_body_sensors"):
                for body_name, sensor_key in self._collision_filter_body_sensors.items():
                    contact_filter_sensors[body_name] = self.scene.sensors[sensor_key]

            self._metrics_tracker = MetricsTracker(
                robot=self.robot,
                ee_frame=ee_frame,
                contact_sensor=contact_sensor,
                contact_filter_sensors=contact_filter_sensors or None,
                track_joint_kinematics=self.cfg.track_joint_metrics,
                track_ee_kinematics=self.cfg.track_ee_metrics,
                track_collisions=self.cfg.track_collisions,
                dt=self.step_dt,
                num_envs=self.num_envs,
                device=self.device,
            )

        ### Automatically determine observation and action spaces ###
        obs, _ = self.reset()
        self.update_observation_space(obs)
        self.update_action_space()

    def set_robot_control_mode(self, control_mode: str):
        self.robot.control_mode = control_mode
        self.update_action_space()
        # NOTE (stao): If someone modifies action space with wrappers, we can't easily track it here, someone needs to manually update action space
        # for each action wrapper

    def update_action_space(self):
        self.single_action_space = gym.spaces.Box(
            low=-1, high=1, shape=(len(self.robot.actuated_indices),)
        )
        self.cfg.action_space = gym.vector.utils.batch_space(
            self.single_action_space, n=self.num_envs
        )
        self.action_space = self.cfg.action_space

    def update_observation_space(self, obs):
        self._init_raw_obs = common_utils.to_numpy(obs)
        self.single_observation_space = gym_utils.convert_observation_to_space(
            self._init_raw_obs, unbatched=True
        )
        self.cfg.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, n=self.num_envs
        )
        self.observation_space = self.cfg.observation_space

    def _setup_scene_objects(self):
        raise NotImplementedError("""_setup_scene_objects needs to be implemented.
        This should add any objects to the scene that aren't robots.
        All of these will be cloned to all parallel environments.
        """)

    def _after_scene_setup(self):
        """
        This hook is called after the scene is setup. This means data for objects can be accessed.
        """
        self.kinematics = Kinematics(
            self,
            KinematicsCfg(
                articulation_cfg=SceneEntityCfg(
                    name="robot", joint_names=["panda_joint.*"], body_names="panda_link7"
                ),
                tcp_offset=(0.0, 0.0, 0.2418),
            ),
        )

    def _resolve_ee_body_idx(self):
        """Resolve the EE body index used for ``ee_pos`` observations.

        Uses ``ee_body_name`` from the robot config so the body matches what
        was recorded as ``observation.state.ee_position`` during datagen.
        """
        names = list(self.robot.articulation.body_names)
        ee_name = self._robot_cfg.ee_body_name
        if ee_name not in names:
            raise ValueError(
                f"ee_body_name '{ee_name}' not found in articulation body names: {names}"
            )
        return names.index(ee_name)

    def _setup_scene_lights(self):
        """
        This should add lights to the scene.
        """
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(1.0, 1.0, 1.0))
        light_cfg.func("/World/Light", light_cfg)

    def _apply_actuator_gain_overrides(self):
        """Apply the ``BaseEnvCfg`` PD gain overrides to the robot's actuator configs.

        Called from :meth:`_setup_scene` after ``actuator_profile`` selection, in
        increasing order of precedence: gains recorded in an RL checkpoint's
        ``config.json`` → ``actuator_gains_json`` → ``actuator_stiffness`` /
        ``actuator_damping``.
        """
        import json
        from copy import deepcopy

        from blocksuite.robots.actuator_profiles import (
            apply_gains_to_actuators,
            apply_scalar_gain_overrides,
            load_gains_from_checkpoint_config,
        )

        if not (
            self.cfg.actuator_gains_ckpt_config
            or self.cfg.actuator_gains_json
            or self.cfg.actuator_stiffness
            or self.cfg.actuator_damping
        ):
            return
        # The actuator cfgs may be shared module-level objects (a named
        # ``ACTUATOR_PROFILES`` entry or a robot config's class default), so copy
        # before mutating to avoid leaking gains into other envs in this process.
        actuators = deepcopy(self._robot_cfg.articulation.actuators)
        self._robot_cfg.articulation.actuators = actuators

        if self.cfg.actuator_gains_ckpt_config:
            gains = load_gains_from_checkpoint_config(self.cfg.actuator_gains_ckpt_config)
            if gains:
                apply_gains_to_actuators(actuators, gains)
                print(
                    "[INFO]: Overrode robot PD gains with those recorded in "
                    f"{self.cfg.actuator_gains_ckpt_config}"
                )
        if self.cfg.actuator_gains_json:
            apply_gains_to_actuators(actuators, json.loads(self.cfg.actuator_gains_json))
        if self.cfg.actuator_stiffness:
            apply_scalar_gain_overrides(actuators, "stiffness", self.cfg.actuator_stiffness)
        if self.cfg.actuator_damping:
            apply_scalar_gain_overrides(actuators, "damping", self.cfg.actuator_damping)

    def _setup_scene(self):
        if self.cfg.actuator_profile is not None:
            from blocksuite.robots.actuator_profiles import ACTUATOR_PROFILES

            profile = ACTUATOR_PROFILES[self.cfg.actuator_profile]
            self._robot_cfg.articulation.actuators = profile
        self._apply_actuator_gain_overrides()

        # add robot
        self.robot = self._robot_class(cfg=self._robot_cfg, env=self)
        self.scene.articulations["robot"] = self.robot.articulation

        if self.cfg.scene_preset_id is not None:
            # static scene assets don't need to recorded to sim state so we only spawn them here
            for k, asset_cfg in self._scene_preset_cfg.asset_cfgs.items():
                if isinstance(asset_cfg, RigidObjectCfg):
                    self.scene.rigid_objects[k] = asset_cfg.class_type(asset_cfg)
                elif isinstance(asset_cfg, AssetBaseCfg):
                    if asset_cfg.spawn is not None:
                        asset_cfg.spawn.func(
                            asset_cfg.prim_path,
                            asset_cfg.spawn,
                            translation=asset_cfg.init_state.pos,
                            orientation=asset_cfg.init_state.rot,
                        )
                        self.scene._extras[k] = XFormPrim(
                            asset_cfg.prim_path, reset_xform_properties=False
                        )
                else:
                    asset_cfg.class_type(asset_cfg)
            # Safety net: some scene-preset RigidObject USDs ship only a collider (e.g. the
            # toyblocks_real table in some asset versions), and UsdFileCfg + rigid_props only
            # MODIFIES an existing RigidBodyAPI -- it never applies one -- so RigidObject init would
            # fail with "Failed to find a rigid body". Apply a kinematic body on the env_0 template
            # (before cloning) for any such asset that lacks one. Skips assets whose USD already
            # authors a rigid body (avoids nested-rigid-body errors -> composes with a store-side
            # fix), and is durable across asset re-download (unlike editing the USD).
            from pxr import UsdPhysics, PhysxSchema
            for _k, _acfg in self._scene_preset_cfg.asset_cfgs.items():
                if not isinstance(_acfg, RigidObjectCfg):
                    continue
                _tmpl = sim_utils.find_first_matching_prim(_acfg.prim_path)
                if _tmpl is None:
                    continue
                _existing = sim_utils.get_all_matching_child_prims(
                    _tmpl.GetPath().pathString,
                    predicate=lambda p: p.HasAPI(UsdPhysics.RigidBodyAPI),
                    traverse_instance_prims=False,
                )
                if not _existing:
                    UsdPhysics.RigidBodyAPI.Apply(_tmpl).CreateKinematicEnabledAttr(True)
                    PhysxSchema.PhysxRigidBodyAPI.Apply(_tmpl)
                    print(f"[base_env] applied kinematic RigidBodyAPI to scene asset '{_k}' "
                          f"({_tmpl.GetPath()}) -- USD shipped none")
        spawn_ground_plane(
            prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05)
        )

        # add user defined objects, e.g. blocks
        self._setup_scene_objects()

        # add all cameras
        if self._include_rgb_in_obs or self.cfg.always_spawn_cameras:
            only = self.cfg.spawn_only_cameras
            for k, v in self.cfg.camera_cfgs.items():
                if only is not None and k not in only:
                    continue
                self.scene.sensors[f"camera:{k}"] = v.class_type(v)
                self.cameras[k] = self.scene.sensors[f"camera:{k}"]
        # add all frame transformations
        for k, v in self.cfg.frame_transformers.items():
            self.scene.sensors[f"frame_transformer:{k}"] = FrameTransformer(v)

        for name, sensor_cfg in self._robot_cfg.contact_sensors.items():
            self.scene.sensors[f"contact:{name}"] = ContactSensor(sensor_cfg)

        # add contact sensor for collision tracking
        if self.cfg.track_collisions:
            env_regex_ns = self.scene.env_regex_ns
            resolved_filters = [
                p.format(ENV_REGEX_NS=env_regex_ns) for p in self.cfg.collision_filter_prim_paths
            ]

            contact_cfg = ContactSensorCfg(
                prim_path=f"{env_regex_ns}/Robot/.*",
                track_air_time=False,
            )
            self.scene.sensors["contact_sensor"] = ContactSensor(contact_cfg)

            # Per-body filter sensors for contact_pair metrics.
            # Isaac Lab's filter_prim_paths_expr requires the sensor's
            # prim_path to match exactly one prim per env, so the multi-body
            # sensor above cannot use filters.  We create lightweight
            # single-body sensors for each robot body instead.
            if resolved_filters:
                self._collision_filter_body_sensors: dict[str, str] = {}
                for body_name in self.cfg.collision_filter_body_names:
                    sensor_key = f"contact_filter:{body_name}"
                    filter_cfg = ContactSensorCfg(
                        prim_path=f"{env_regex_ns}/Robot/{body_name}",
                        filter_prim_paths_expr=resolved_filters,
                        track_air_time=False,
                    )
                    self.scene.sensors[sensor_key] = ContactSensor(filter_cfg)
                    self._collision_filter_body_sensors[body_name] = sensor_key
        if self.scene.cfg.replicate_physics:
            self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self._setup_scene_lights()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone()

    def _before_sim_step(self, substep: int) -> None:
        """
        This hook is called before actions are applied and a sim sub step is taken
        """
        pass

    def _after_sim_step(self, substep: int) -> None:
        """
        This hook is called after a sim sub step is taken
        """
        pass

    def _after_control_step(self) -> None:
        """
        This hook is called after actions are applied, all sim sub steps are taken, and _compute_intermediate_values is called
        """
        pass

    def _before_auto_reset(self, reset_env_ids: Sequence[int]) -> None:
        """
        This hook is called before an environment is auto reset internally by BaseEnv. Useful to use
        if you need to record any sim data before we auto-reset
        """
        pass

    def _after_reset_idx(self, reset_env_ids: Sequence[int]) -> None:
        """
        This hook is called after any call to self._reset_idx and intermediate values are recomputed in BaseEnv.
        """
        for v in self.robot.controller.values():
            v.reset(reset_env_ids)

    def _update_sensor_buffers(self, env_ids: Sequence[int]) -> None:
        """Copy the latest RTX annotator output into ``sensor.data``.

        ``sim.render()`` updates annotators only. ``_get_observations`` reads
        ``camera.data.output``, which stays leftover from the previous episode
        until this runs. ``set_state_dict`` already did this; ``reset`` and the
        auto-reset path in ``step`` did not, so the first post-reset RGB showed
        the prior episode (robot not at the just-written home pose).
        """
        for sensor in self.scene.sensors.values():
            sensor._update_buffers_impl(env_ids)

    def _compute_goals(self):
        """
        This function can be overriden to compute any intermediate goal related data to save to self for use
        by _compute_intermediate_values and other functions like self.evaluate and self._get_rewards

        Differrent to standard Isaac Lab, instead of a single _compute_intermediate_values function, we have a separate compute_goals
        function that is called before _compute_intermediate_values. This way other code can re-use an environment's functions without having to actually
        create the environment (e.g. _get_observations). Now a user just needs to remake the _compute_goals function themselves, and make sure
        any scene assets referenced by _compute_intermediate_values are referenced in some way.

        NOTE (stao): In reality this is also just a work around the fact we cannot create two isaac environment instances in the same process. If we could,
        then we would simply set the other instance's sim state and re-use functionalities.
        """

    def _compute_intermediate_values(self):
        """
        This function can be overriden to compute any intermediate values needed for other functions such as observations or rewards.

        It is called before self.evaluate, self._get_observations, and self._get_rewards are called. Called after all physics steps are taken.

        This function is also called whenever the environment state is set via self.set_state or self.set_state_dict.
        """
        pass

    def _compute_goals_and_intermediate_values(self):
        self._compute_goals()
        self._compute_intermediate_values()

    def _apply_action(self) -> None:
        current_dim = 0
        for v in self.robot.controller.values():
            v.process_actions(self.actions[:, current_dim : current_dim + v.action_dim])
            current_dim += v.action_dim
            v.apply_actions()

    def _run_gripper_settle(self, action: torch.Tensor, is_rendering: bool) -> None:
        """Run settle steps with the arm frozen and only the gripper actuated.

        Constructs a temporary action that holds the arm joints at their current
        positions and applies only the gripper command from *action*.
        """
        robot = self.robot.articulation
        arm_joint_pos = robot.data.joint_pos[:, :7].clone()
        gripper_dim = action.shape[-1] - 1
        gripper_cmd = action[:, gripper_dim : gripper_dim + 1]

        settle_action = torch.cat([arm_joint_pos, gripper_cmd], dim=-1)
        saved_actions = self.actions.clone()
        self.actions = settle_action

        settle_substeps = self.cfg.gripper_settle_steps * self.cfg.decimation
        for substep in range(settle_substeps):
            self._before_sim_step(substep)
            self._sim_step_counter += 1
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)
            self._after_sim_step(substep)

        self.actions = saved_actions

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[VecEnvObs, dict]:
        """Resets all the environments and returns observations.

        This function calls the :meth:`_reset_idx` function to reset all the environments.
        However, certain operations, such as procedural terrain generation, that happened during initialization
        are not repeated.

        Args:
            seed: The seed to use for randomization. Defaults to None, in which case the seed is not set.
            options: Additional information to specify how the environment is reset. Defaults to None.

                Note:
                    This argument is used for compatibility with Gymnasium environment definition.

        Returns:
            A tuple containing the observations and extras.
        """
        # NOTE (stao): changes to original reset are to include computing intermediate values and evaluate/update extras before returning
        # set the seed
        if seed is not None:
            self.seed(seed)

        # reset state of scene
        indices = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        self._reset_idx(indices)

        # update articulation kinematics
        self.scene.write_data_to_sim()
        self.sim.forward()

        # if sensors are added to the scene, make sure we render to reflect changes in reset
        if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
            for _ in range(self.cfg.num_rerenders_on_reset):
                self.sim.render()

        if self.cfg.wait_for_textures and self.sim.has_rtx_sensors():
            while SimulationManager.assets_loading():
                self.sim.render()
        self._update_sensor_buffers(indices)

        # return observations
        self._compute_goals_and_intermediate_values()
        self.extras.update(self.evaluate())
        self._after_reset_idx(indices)
        return self._get_observations(), self.extras

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics.

        The environment steps forward at a fixed time-step, while the physics simulation is decimated at a
        lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
        independently using the :attr:`DirectRLEnvCfg.decimation` (number of simulation steps per environment step)
        and the :attr:`DirectRLEnvCfg.sim.physics_dt` (physics time-step). Based on these parameters, the environment
        time-step is computed as the product of the two.

        This function performs the following steps:

        1. Pre-process the actions before stepping through the physics.
        2. Apply the actions to the simulator and step through the physics in a decimated manner.
        3. Compute the reward and done signals.
        4. Reset environments that have terminated or reached the maximum episode length.
        5. Apply interval events if they are enabled.
        6. Compute observations.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim). Can also be None which means to not perform any additional control and to simply step forward physics.

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        if action is not None:
            action = action.to(self.device)
            # add action noise
            if self.cfg.action_noise_model:
                action = self._action_noise_model(action)

            # process actions
            self._pre_physics_step(action)
        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # Gripper settle: freeze arm and only actuate gripper until it fully
        # opens/closes before allowing normal arm motion.
        if action is not None and self.cfg.gripper_settle_steps > 0:
            gripper_dim = action.shape[-1] - 1
            gripper_val = action[:, gripper_dim]
            prev_gripper = (
                self.prev_actions[:, gripper_dim]
                if self.prev_actions.shape[-1] > gripper_dim
                else gripper_val
            )
            gripper_changed = ((gripper_val < 0) != (prev_gripper < 0)).any()
            if gripper_changed:
                self._run_gripper_settle(action, is_rendering)

        # Detect gripper close transition for extra decimation
        extra_gripper_steps = 0
        if action is not None and self.cfg.gripper_close_decimation > 0:
            gripper_dim = action.shape[-1] - 1
            gripper_now_close = (action[:, gripper_dim] < 0).any()
            prev_has_gripper = self.prev_actions.shape[-1] > gripper_dim
            gripper_prev_open = prev_has_gripper and (self.prev_actions[:, gripper_dim] >= 0).any()
            if gripper_now_close and gripper_prev_open:
                extra_gripper_steps = self.cfg.gripper_close_decimation

        # perform physics stepping
        total_substeps = self.cfg.decimation + extra_gripper_steps
        for substep in range(total_substeps):
            self._before_sim_step(substep)
            self._sim_step_counter += 1
            if action is not None:
                # set actions into buffers
                self._apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)
            self._after_sim_step(substep)
        self._compute_goals_and_intermediate_values()
        self._after_control_step()

        if self._metrics_tracker is not None:
            self._tracker_stats = self._metrics_tracker.step()

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        if self.cfg.ignore_terminations:
            self.reset_terminated[:] = False
        # Externally requested episode ends (see request_episode_end) count as
        # truncations so they bypass ignore_terminations and still finalize with
        # full evaluation/metrics below.
        if self._episode_end_requests.any():
            self.reset_time_outs |= self._episode_end_requests
            self._episode_end_requests[:] = False
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()
        self._returns += self.reward_buf

        # update observations
        self.obs_buf = self._get_observations()
        self.extras = self.evaluate()
        if "success" in self.extras:
            success = self.extras["success"].bool()
            newly_succeeded = success & ~self._success_once
            self._first_success_step[newly_succeeded] = self.episode_length_buf[newly_succeeded]
            self._success_once |= success
            self._success_streak[success] += 1
            self._success_streak[~success] = 0
        if "fail" in self.extras:
            self._fail_once = self._fail_once | self.extras["fail"]

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0 and self.cfg.auto_reset:
            # modification: support proper vector env api
            self.final_obs_buf = common_utils.torch_clone_dict(self.obs_buf)
            self.extras = dict(
                _final_info=self.reset_buf,
                final_observation=self.final_obs_buf,
                final_info=self.extras,
            )

            if self.cfg.record_metrics:
                episode_info = dict()
                if "success" in self.extras["final_info"]:
                    episode_info["success_once"] = self._success_once.clone()
                    episode_info["success_at_end"] = self.extras["final_info"]["success"].clone()
                    episode_info["first_success_step"] = self._first_success_step.clone().float()
                    episode_info["first_success_step"][~self._success_once] = float("nan")
                if "fail" in self.extras["final_info"]:
                    episode_info["fail_once"] = self._fail_once.clone()
                    episode_info["fail_at_end"] = self.extras["final_info"]["fail"].clone()
                episode_info["return"] = self._returns.clone()
                episode_info["episode_len"] = self.episode_length_buf.clone()
                episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
                self.extras["final_info"]["episode"] = episode_info
            self._before_auto_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            # Commit buffered writes + rerender RTX sensors *before* sampling
            # _get_observations. The previous order rendered after, so the
            # policy / recorded video saw stale RGB from the previous episode.
            # sim.render() only updates RTX annotators; camera.data.output is
            # filled by _update_sensor_buffers (see set_state_dict).
            self.scene.write_data_to_sim()
            self.sim.forward()
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()  # pyright: ignore[reportCallIssue]
            self._update_sensor_buffers(reset_env_ids)
            self._compute_goals_and_intermediate_values()
            self.extras.update(self.evaluate())
            self._after_reset_idx(reset_env_ids)
            self.obs_buf = self._get_observations()

        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # add observation noise
        # note: we apply no noise to the state space (since it is used for critic networks)
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model(self.obs_buf["policy"])

        # return observations, rewards, resets and extras
        return (
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )

    def get_state_dict(
        self, env_ids: Sequence[int] | None = None, is_relative: bool = False
    ) -> dict:
        if env_ids is None:
            env_ids = self.scene._ALL_INDICES

        state = dict()
        # articulations
        state["articulation"] = dict()
        for asset_name, articulation in self.scene.articulations.items():
            asset_state = dict()
            asset_state["root_pose"] = articulation.data.root_pose_w[env_ids].clone()
            if is_relative:
                asset_state["root_pose"][:, :3] -= self.scene.env_origins[env_ids]
            asset_state["root_velocity"] = articulation.data.root_vel_w[env_ids].clone()
            asset_state["joint_position"] = articulation.data.joint_pos[env_ids].clone()
            asset_state["joint_velocity"] = articulation.data.joint_vel[env_ids].clone()
            asset_state["joint_position_target"] = articulation.data.joint_pos_target[
                env_ids
            ].clone()
            state["articulation"][asset_name] = asset_state

        # rigid objects
        state["rigid_object"] = dict()
        for asset_name, rigid_object in self.scene.rigid_objects.items():
            asset_state = dict()
            asset_state["root_pose"] = rigid_object.data.root_pose_w[env_ids].clone()
            if is_relative:
                asset_state["root_pose"][:, :3] -= self.scene.env_origins[env_ids]
            asset_state["root_velocity"] = rigid_object.data.root_vel_w[env_ids].clone()
            state["rigid_object"][asset_name] = asset_state

        return state

    def set_state_dict(
        self,
        state_dict: dict[str, dict[str, dict[str, torch.Tensor]]],
        env_ids: Sequence[int] | None = None,
        is_relative: bool = False,
    ) -> None:
        if env_ids is None:
            env_ids = self.scene._ALL_INDICES

        # articulations
        for asset_name, articulation in self.scene.articulations.items():
            asset_state = state_dict["articulation"][asset_name]
            # root state
            root_pose = asset_state["root_pose"].clone()
            if is_relative:
                root_pose[:, :3] += self.scene.env_origins[env_ids]
            root_velocity = asset_state["root_velocity"].clone()
            articulation.write_root_pose_to_sim(root_pose, env_ids=env_ids)
            articulation.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
            # joint state
            joint_position = asset_state["joint_position"].clone()
            joint_velocity = asset_state["joint_velocity"].clone()
            articulation.write_joint_state_to_sim(joint_position, joint_velocity, env_ids=env_ids)
            articulation.set_joint_position_target(joint_position, env_ids=env_ids)
            articulation.set_joint_velocity_target(joint_velocity, env_ids=env_ids)

        # rigid objects
        for asset_name, rigid_object in self.scene.rigid_objects.items():
            asset_state = state_dict["rigid_object"][asset_name]
            root_pose = asset_state["root_pose"].clone()
            if is_relative:
                root_pose[:, :3] += self.scene.env_origins[env_ids]
            root_velocity = asset_state["root_velocity"].clone()
            rigid_object.write_root_pose_to_sim(root_pose, env_ids=env_ids)
            rigid_object.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
        self.scene.write_data_to_sim()
        self.sim.forward()
        # if sensors are added to the scene, make sure we render to reflect changes in reset
        if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
            for _ in range(self.cfg.num_rerenders_on_reset):
                self.sim.render()

        if self.cfg.wait_for_textures and self.sim.has_rtx_sensors():
            while SimulationManager.assets_loading():
                self.sim.render()
        self._update_sensor_buffers(env_ids)

        self._compute_goals_and_intermediate_values()
        self.extras.update(self.evaluate())
        self._after_reset_idx(env_ids)

    def _get_obs_robot(self) -> dict:
        """Retrieve observations from the robot.

        Emits every key in :attr:`AGENT_OBS_KEYS` unconditionally so dict-observation
        consumers (the ManiFlow / DROID obs transforms) can read any of them without
        an env-side flag. The flat RL vector selects a subset via :meth:`rl_agent_obs`.
        """
        robot = self.robot.articulation
        if self.cfg.legacy_ee_pos:
            # Pre-``ee_body_name`` convention: last body in the chain, robot base frame.
            # See BaseEnvCfg.legacy_ee_pos -- this is what the recorded
            # observation.state.ee_position stats were fit against.
            ee_pose_w = robot.data.body_link_pose_w[:, len(robot.body_names) - 1, :]
            base_quat_inv = robot.data.root_quat_w.clone()
            base_quat_inv[:, 1:] = -base_quat_inv[:, 1:]
            ee_pos = quat_apply(base_quat_inv, ee_pose_w[:, :3] - robot.data.root_pos_w)
        else:
            ee_body_idx = self._resolve_ee_body_idx()
            ee_pos = robot.data.body_link_pose_w[:, ee_body_idx, :3] - self.scene.env_origins
        if not self._logged_ee_pos_convention:
            body_name = (
                robot.body_names[-1] if self.cfg.legacy_ee_pos else self._robot_cfg.ee_body_name
            )
            frame = "robot base" if self.cfg.legacy_ee_pos else "env origin"
            print(
                f"[INFO]: ee_pos observation from body '{body_name}' in the {frame} frame "
                f"(legacy_ee_pos={self.cfg.legacy_ee_pos})"
            )
            self._logged_ee_pos_convention = True
        return {
            "joint_pos": self.joint_pos,
            "joint_targets": self.joint_targets,
            "joint_vel": self.joint_vel,
            "ee_pos": ee_pos,
        }

    def rl_agent_obs(self, agent_obs: dict) -> dict:
        """Subset of the ``agent`` obs group that the flat RL vector consumes.

        Always ordered by :attr:`AGENT_OBS_KEYS`, never by ``cfg.rl_agent_obs_keys``, so
        the flat width and layout an RL checkpoint was trained against cannot shift
        under a reordered config.
        """
        allowed = set(self.cfg.rl_agent_obs_keys)
        unknown = allowed - set(self.AGENT_OBS_KEYS)
        if unknown:
            raise ValueError(
                f"rl_agent_obs_keys contains unknown key(s) {sorted(unknown)}; "
                f"valid keys are {list(self.AGENT_OBS_KEYS)}."
            )
        return {k: agent_obs[k] for k in self.AGENT_OBS_KEYS if k in allowed}

    def _get_obs_extra(self) -> dict:
        """Retrieve extra observations from the environment. Usually this is ground truth data like object poses."""
        return dict()

    def _get_obs_goal(self) -> dict:
        """Retrieve goal related observations from the environment."""
        return dict()

    def _get_obs_privileged(self) -> dict:
        """Retrieve privileged observations available only to the critic under asymmetric actor-critic."""
        return dict()

    def _get_observations(self) -> dict:
        observations = {"agent": self._get_obs_robot()}
        if self._include_state_in_obs:
            observations.update({"state": self._get_obs_extra()})
            observations.update({"goal": self._get_obs_goal()})
            # Privileged group MUST be appended last so it lands at the tail of the flat obs vector;
            # the agent slices off this tail for the actor while the critic consumes the full vector.
            if self.cfg.asymmetric_critic:
                observations.update({"privileged": self._get_obs_privileged()})
        if self._include_rgb_in_obs:
            observations["images"] = dict()
            for k, v in self.cameras.items():
                entry = {"rgb": v.data.output["rgb"]}
                if self._include_depth_in_obs:
                    # Raw metric depth (meters) + native-resolution pinhole
                    # intrinsics, i.e. what a depth policy back-projects with.
                    # A camera preset that does not render
                    # ``distance_to_image_plane`` raises here rather than
                    # handing the policy a plausible-looking zero depth map.
                    entry["depth"] = v.data.output["distance_to_image_plane"]
                    entry["intrinsics"] = v.data.intrinsic_matrices
                observations["images"][k] = entry
        return observations

    def _get_rewards(self) -> torch.Tensor:
        return self.reset_terminated.float()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length
        return time_out, time_out

    def evaluate(self) -> dict:
        """
        Evaluate the current state of the environment. Should return a dictionary of various metrics. For compatability with data generation / planning code
        this should not use any objects in the scene and should rely on values generated by ``self._compute_goals`` and ``self._compute_intermediate_values`` functions.

        There are two reserved keys, "success" and "fail". Success indicates the state is a successful one, and fail indicates the
        state is a failure and generally irrecoverable.

        Subclasses should call ``super().evaluate()`` and merge their own
        stats into the returned dict so that :class:`~blocksuite.utils.metrics.MetricsTracker`
        stats are included automatically.
        """
        result: dict = {}
        if self._tracker_stats:
            result["stats"] = dict(self._tracker_stats)
        return result

    ### Reset functions ###
    def _reset_robots(self, env_ids: Sequence[int]):
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.articulation.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.articulation.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.articulation.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self.robot.articulation.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.robot.articulation.set_joint_velocity_target(joint_vel, env_ids=env_ids)

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        b = len(env_ids)
        self.prev_actions[env_ids] = 0.0
        # Cleared regardless of ``record_metrics``: step() accumulates into these
        # every step, and stale success state could end the next episode of this
        # env immediately.
        self._returns[env_ids] = 0.0
        self._success_once[env_ids] = False
        self._success_streak[env_ids] = 0
        self._fail_once[env_ids] = False
        self._first_success_step[env_ids] = -1
        self._reset_robots(env_ids)
        if self._metrics_tracker is not None:
            env_ids_t = (
                env_ids
                if isinstance(env_ids, torch.Tensor)
                else torch.tensor(env_ids, device=self.device)
            )
            self._metrics_tracker.reset(env_ids_t)

        # table height randomization, if enabled
        table_dz = torch.zeros(b, device=self.device)
        if self.cfg.table_height_range > 0:
            table_dz = torch.empty(b, device=self.device).uniform_(
                -self.cfg.table_height_range, self.cfg.table_height_range
            )
            table_default_local = (
                torch.tensor(
                    self.scene.rigid_objects["workspace_table"].cfg.init_state.pos,
                    device=self.device,
                    dtype=torch.float32,
                )
                .unsqueeze(0)
                .expand(b, -1)
                .clone()
            )
            table_pos_w = self.scene.env_origins[env_ids] + table_default_local
            table_pos_w[:, 2] += table_dz
            table_quat = (
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(b, -1)
            )
            table_pose_w = torch.cat([table_pos_w, table_quat], dim=-1)
            self.scene.rigid_objects["workspace_table"].write_root_pose_to_sim(
                table_pose_w, env_ids=env_ids
            )
            # TODO (sumeetb) I think we can get rid of this, but commenting it out just for now.
            # self.scene.rigid_objects["table_collider"].write_root_pose_to_sim(table_pose_w, env_ids=env_ids)
        self.table_height_offset[env_ids] = table_dz

        # randomize table friction if enabled
        if self.cfg.randomize_friction:
            if "workspace_table" not in self.scene.rigid_objects:
                print(
                    "workspace_table not found in scene.rigid_objects. continuing without randomizing friction"
                )
            else:
                table_friction_samples = randomization_utils.randomize_rigid_body_friction(
                    asset=self.scene.rigid_objects["workspace_table"],
                    static_friction_range=self.cfg.table_static_friction_range,
                    dynamic_friction_range=self.cfg.table_dynamic_friction_range,
                    restitution_range=self.cfg.table_restitution_range,
                    env_ids=env_ids,
                    make_consistent=self.cfg.make_consistent,
                )
                # store (static, dynamic) for privileged observations
                self.table_friction[env_ids] = table_friction_samples[:, :2].to(self.device)

                if "block" in self.scene.rigid_objects:
                    block_friction_samples = randomization_utils.randomize_rigid_body_friction(
                        asset=self.scene.rigid_objects["block"],
                        static_friction_range=self.cfg.block_static_friction_range,
                        dynamic_friction_range=self.cfg.block_dynamic_friction_range,
                        restitution_range=self.cfg.block_restitution_range,
                        env_ids=env_ids,
                        make_consistent=self.cfg.make_consistent,
                    )
                    self.block_friction[env_ids] = block_friction_samples[:, :2].to(self.device)

        if self.cfg.randomize_init_joint_vel:
            arm_joint_vel_noise = 0.2  # rad/s, per-joint uniform half-range
            arm_joint_ids = self.kinematics.joint_ids
            n_arm_joints = self.kinematics.n_joints
            vel_noise = torch.empty(b, n_arm_joints, device=self.device).uniform_(
                -arm_joint_vel_noise, arm_joint_vel_noise
            )
            self.kinematics.articulation.write_joint_velocity_to_sim(
                vel_noise, joint_ids=arm_joint_ids, env_ids=env_ids,
            )

    ### Actuator parameter control ###

    def set_actuator_gains(
        self,
        stiffness: torch.Tensor | None = None,
        damping: torch.Tensor | None = None,
        effort_limits: torch.Tensor | None = None,
        velocity_limits: torch.Tensor | None = None,
        settle_steps: int = 0,
        settle_env_mask: torch.Tensor | None = None,
        settle_joint_ids: Sequence[int] | torch.Tensor | None = None,
        settle_zero_velocity: bool = True,
    ):
        """Change actuator physics parameters at runtime.

        Writes new gains to the live simulation.  If *settle_steps* > 0, runs
        that many raw sim substeps while holding the current joint position so
        the physics can stabilize under the new gains before the next
        ``env.step()`` call.

        Args:
            stiffness: Joint stiffness tensor, shape ``(num_envs, num_joints)``.
            damping: Joint damping tensor, same shape.
            effort_limits: Joint effort-limit tensor, same shape.
            velocity_limits: Joint velocity-limit tensor, same shape.
            settle_steps: Number of raw sim substeps to run (holding position)
                after writing the new gains.  0 means no settle period.
            settle_env_mask: Optional boolean tensor of shape ``(num_envs,)``.
                When provided, only the masked environments have their
                position/velocity targets frozen during settle steps;
                other environments keep their existing targets undisturbed.
                When ``None``, all environments are frozen (legacy behavior).
            settle_joint_ids: Optional joint indices to freeze while settling.
                Other joints retain their active targets. When ``None``, all
                joints are frozen (legacy behavior).
            settle_zero_velocity: When ``True`` (legacy behavior), the frozen
                joints' velocities are also zeroed in the sim before settling.
                ``False`` only retargets them to their current position with a
                zero velocity target, leaving any residual motion for the new
                gains to damp out (as TAMP's stiffness-profile switch does).
        """
        robot = self.robot.articulation
        if stiffness is not None:
            robot.write_joint_stiffness_to_sim(stiffness)
        if damping is not None:
            robot.write_joint_damping_to_sim(damping)
        if effort_limits is not None:
            robot.write_joint_effort_limit_to_sim(effort_limits)
        if velocity_limits is not None:
            robot.write_joint_velocity_limit_to_sim(velocity_limits)

        if settle_steps > 0:
            if settle_env_mask is not None:
                env_ids = settle_env_mask.nonzero(as_tuple=False).flatten()
                if env_ids.numel() == 0:
                    return
                if settle_joint_ids is None:
                    cur_pos = robot.data.joint_pos[env_ids].clone()
                    robot.set_joint_position_target(cur_pos, env_ids=env_ids)
                    robot.set_joint_velocity_target(
                        torch.zeros_like(cur_pos), env_ids=env_ids
                    )
                    if settle_zero_velocity:
                        robot.write_joint_velocity_to_sim(
                            torch.zeros_like(robot.data.joint_vel[env_ids]),
                            env_ids=env_ids,
                        )
                else:
                    cur_pos = robot.data.joint_pos[env_ids][
                        :, settle_joint_ids
                    ].clone()
                    cur_vel = robot.data.joint_vel[env_ids][:, settle_joint_ids]
                    robot.set_joint_position_target(
                        cur_pos, joint_ids=settle_joint_ids, env_ids=env_ids
                    )
                    robot.set_joint_velocity_target(
                        torch.zeros_like(cur_pos),
                        joint_ids=settle_joint_ids,
                        env_ids=env_ids,
                    )
                    if settle_zero_velocity:
                        robot.write_joint_velocity_to_sim(
                            torch.zeros_like(cur_vel),
                            joint_ids=settle_joint_ids,
                            env_ids=env_ids,
                        )
            else:
                if settle_joint_ids is None:
                    cur_pos = robot.data.joint_pos.clone()
                    robot.set_joint_position_target(cur_pos)
                    robot.set_joint_velocity_target(torch.zeros_like(cur_pos))
                    if settle_zero_velocity:
                        robot.write_joint_velocity_to_sim(
                            torch.zeros_like(robot.data.joint_vel)
                        )
                else:
                    cur_pos = robot.data.joint_pos[:, settle_joint_ids].clone()
                    cur_vel = robot.data.joint_vel[:, settle_joint_ids]
                    robot.set_joint_position_target(
                        cur_pos, joint_ids=settle_joint_ids
                    )
                    robot.set_joint_velocity_target(
                        torch.zeros_like(cur_pos), joint_ids=settle_joint_ids
                    )
                    if settle_zero_velocity:
                        robot.write_joint_velocity_to_sim(
                            torch.zeros_like(cur_vel), joint_ids=settle_joint_ids
                        )
            robot.write_data_to_sim()
            for _ in range(settle_steps):
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.scene.update(self.physics_dt)

    ### some additional functions pulled from IsaacLabSimulator ###
    # TODO (stao): these functions are not batched. They should be, but predicates assume not batched at the moment
    def get_pose(self, obj_name: str, is_relative: bool = False) -> torch.Tensor:
        try:
            obj = self.scene[obj_name]
        except KeyError as exc:
            raise ValueError(f"Object '{obj_name}' not found in the simulator.") from exc

        if not isinstance(obj, XFormPrim):
            data = obj.data
            pose = data.root_state_w[:, :7].clone()
            if is_relative:
                pose[:, :3] -= self.scene.env_origins
            return pose
        else:
            # NOTE (stao): Is there a better way to access pose data without having to add a new get_pose function...?
            # why do some assets only have pose data stored like this? this makes it not possible for an object oriented way to access some data
            pos = obj.get_local_poses()[0]
            if not is_relative:
                pos += self.scene.env_origins
            quat = obj.get_local_poses()[1]
            pose = torch.cat([pos, quat], axis=-1)
            return pose

    def set_pose(self, obj_name: str, pose: torch.Tensor) -> None:
        if obj_name in self.scene:
            if isinstance(self.scene[obj_name], RigidObject):
                obj = self.scene[obj_name]
                pos = pose[:3]
                quat = pose[3:]  # [w,x,y,z]
                self.scene[obj_name].write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
            elif isinstance(self.scene[obj_name], XFormPrim):
                obj = self.scene[obj_name]
                pos = pose[:3]
                quat = pose[3:]
                obj.set_world_poses(position=pos, quaternion=quat)
        else:
            raise ValueError(f"Rigid object '{obj_name}' not found in the simulator.")

    def export_scene_config(self):
        pass

    def import_scene_config(self, scene_config: dict):
        pass
