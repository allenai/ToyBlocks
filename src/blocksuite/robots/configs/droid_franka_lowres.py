"""``droid_franka`` with a 640×360 wrist camera (same 16∶9 as 1280×720).

Horizontal field of view matches the standard ``droid_franka`` wrist camera;
``horizontal_aperture``, ``vertical_aperture``, and ``focal_length`` are unchanged
because the aspect ratio is still 16∶9 (``vertical_aperture / horizontal_aperture == 9/16``).
"""

from blocksuite.robots.configs.droid_franka import DroidFranka, DroidFrankaCfg
from blocksuite.robots.registration import register_robot
from isaaclab.sensors import TiledCameraCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

_WRIST_WIDTH = 640
_WRIST_HEIGHT = 360


@configclass
class DroidFrankaLowresCfg(DroidFrankaCfg):
    cameras = {
        "wrist_camera": TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/base_link/wrist_cam",
            height=_WRIST_HEIGHT,
            width=_WRIST_WIDTH,
            data_types=["rgb", "semantic_segmentation"],
            colorize_semantic_segmentation=False,
            update_latest_camera_pose=True,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.8,
                focus_distance=28.0,
                horizontal_aperture=5.376,
                vertical_aperture=3.024,
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.011, -0.031, -0.074),
                rot=(-0.420, 0.570, 0.576, -0.409),
                convention="opengl",
            ),
        )
    }


@register_robot()
class DroidFrankaLowres(DroidFranka):
    cfg: DroidFrankaLowresCfg
    id = "droid_franka_lowres"

    def open_gripper(self):
        raise NotImplementedError("Open gripper not implemented for DroidFrankaLowres")

    def close_gripper(self):
        raise NotImplementedError("Close gripper not implemented for DroidFrankaLowres")
