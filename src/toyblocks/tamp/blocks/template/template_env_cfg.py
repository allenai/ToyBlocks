import os

from isaaclab.utils import configclass

from toyblocks.tasks.base_env_cfg import BaseEnvCfg


@configclass
class TemplateEnvCfg(BaseEnvCfg):
    episode_length_s = 99999
    scene_preset_id = "sim2real"
    # scene_preset_id = "droid_with_box_table"

    reset_states_path: str = os.path.join(os.path.dirname(__file__), "reset_states.pkl")
    robot_id: str = "fr3_robotiq_gripper_sysid"
