"""Scene preset aligned with ``apr15_pnp/envs/problem_color-only-*/init.py`` ``BlocksSceneCfg``.

Static Seattle lab table at ``(0.5, 0, 0)`` and the front / left pinhole cameras
(ZED-style third-person and a world-fixed front view) used in those Isaac Lab
environments. Wrist-mounted cameras belong on the robot config
(:class:`~blocksuite.robots.configs.franka_robotiq_wrist_cam_contact_ft_sensors.FrankaRobotiqWristCamContactFtSensorsCfg`).
"""

from blocksuite.scenes.base import BaseSceneConfig
from isaaclab.assets.asset_base_cfg import AssetBaseCfg
from isaaclab.sensors import TiledCameraCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


@configclass
class ColorOnlyBlocksTableSceneConfig(BaseSceneConfig):
    asset_cfgs = dict(
        table1=AssetBaseCfg(
            # Must be a global path: ``BaseEnv._setup_scene`` spawns ``SCENE_PRESETS`` assets
            # without ``replace_env_regex_ns`` (unlike ``cfg.camera_cfgs``).
            prim_path="/World/envs/env_.*/table1",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            ),
        ),
    )
    camera_cfgs = dict(
        front_camera=TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/FrontCamera",
            update_period=0.04,
            height=360,
            width=640,
            data_types=["rgb", "distance_to_image_plane", "instance_segmentation_fast", "semantic_segmentation"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=30.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.5, 2.4),
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=(2.0, 0.0, 1.0),
                rot=(-0.3535534, 0.6123724, 0.6123724, -0.3535534),
                convention="ros",
            ),
        ),
        left_camera=TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/LeftCamera",
            update_period=0.04,
            height=360,
            width=640,
            data_types=["rgb", "distance_to_image_plane", "instance_segmentation_fast", "semantic_segmentation"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.12,
                focus_distance=0.6,
                horizontal_aperture=5.139,
                clipping_range=(0.1, 3.0),
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.15, 0.40, 0.45),
                rot=(0.151496, -0.317428, 0.844820, -0.403198),
                convention="ros",
            ),
        ),
    )
