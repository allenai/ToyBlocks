"""Scene preset aligned with ``may1_pnp/envs/problem_*/init.py`` ``BlocksSceneCfg``.

Dark cuboid table with invisible collision box, front / left pinhole cameras
(ZED-style third-person and a world-fixed front view).  Wrist-mounted cameras
belong on the robot config
(:class:`~blocksuite.robots.configs.franka_robotiq_wrist_cam_contact_ft_sensors.FrankaRobotiqWristCamContactFtSensorsCfg`).
"""

import isaaclab.sim as sim_utils
from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from blocksuite.scenes.base import BaseSceneConfig


@configclass
class TableSceneConfig(BaseSceneConfig):
    asset_cfgs = dict(
        table_collider=RigidObjectCfg(
            prim_path="/World/envs/env_.*/table_collider",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.29, 0, -0.385], rot=[1, 0, 0, 0]),
            spawn=sim_utils.CuboidCfg(
                size=(1.5, 1.13, 0.77),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    kinematic_enabled=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.1,
                    dynamic_friction=0.1,
                    restitution=0.0,
                ),
                visible=False,
                activate_contact_sensors=True,
            ),
        ),
        table=RigidObjectCfg(
            prim_path="/World/envs/env_.*/table",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.518, 0.0, -0.035]),
            spawn=sim_utils.CuboidCfg(
                size=(0.762, 1.735, 0.05),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    kinematic_enabled=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.22, 0.26, 0.28),
                ),
            ),
        ),
        table2=RigidObjectCfg(
            prim_path="/World/envs/env_.*/table2",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.285, 0, -0.39]),
            spawn=sim_utils.CuboidCfg(
                size=(0.82, 1.735, 0.76),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    kinematic_enabled=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.22, 0.26, 0.28),
                ),
            ),
        ),
    )
    camera_cfgs = dict(
        front_camera=TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/FrontCamera",
            update_period=0.04,
            height=360,
            width=640,
            data_types=[
                "rgb",
                "distance_to_image_plane",
                "instance_segmentation_fast",
                "semantic_segmentation",
            ],
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
            data_types=[
                "rgb",
                "distance_to_image_plane",
                "instance_segmentation_fast",
                "semantic_segmentation",
            ],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.12,
                focus_distance=0.6,
                horizontal_aperture=5.139,
                clipping_range=(0.1, 2.5),
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.15, 0.40, 0.45),
                rot=(0.151496, -0.317428, 0.844820, -0.403198),
                convention="ros",
            ),
        ),
    )
