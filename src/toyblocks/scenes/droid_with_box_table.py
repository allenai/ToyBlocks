
from toyblocks.scenes.base import BaseSceneConfig
from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from isaaclab.sensors.camera.tiled_camera_cfg import TiledCameraCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

@configclass
class DroidWithBoxTableSceneConfig(BaseSceneConfig):
    asset_cfgs = dict(
        workspace_table = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/workspace_table",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.637, 0.0, -0.025], rot=[1, 0, 0, 0]),
            spawn=sim_utils.CuboidCfg(
                size=(1, 1.735, 0.05),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    kinematic_enabled=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.3,
                    dynamic_friction=0.3,
                    restitution=0.0,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.22, 0.26, 0.28)
                ),
                activate_contact_sensors=True,
            ),
        ),
        robot_table = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/robot_table",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.285, 0, -0.38]),
            spawn=sim_utils.CuboidCfg(
                size=(0.82, 1.735, 0.76),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    kinematic_enabled=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=False,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.22, 0.26, 0.28)
                ),
            ),
        )
    )
    camera_cfgs = dict(
        front_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/FrontCamera",
            update_period=0.04,
            height=360,
            width=640,
            data_types=["rgb", "distance_to_image_plane", "instance_segmentation_fast", "semantic_segmentation"],  # ADD semantic_segmentation
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=30.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.5, 2.4)
            ),
            offset=TiledCameraCfg.OffsetCfg(pos=(2.0, 0.0, 1.0), rot=(-0.3535534, 0.6123724, 0.6123724, -0.3535534 ), convention="ros"),
        ),
        left_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/LeftCamera",
            update_period=0.04,
            height=360,
            width=640,
            data_types=["rgb", "distance_to_image_plane", "instance_segmentation_fast", "semantic_segmentation"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.12, focus_distance=0.6, horizontal_aperture=5.139, clipping_range=(0.1, 3.0)
            ),
            offset=TiledCameraCfg.OffsetCfg(pos=(0.15, 0.40, 0.45), rot=(0.151496, -0.317428, 0.844820, -0.403198), convention="ros"),
        )
    )