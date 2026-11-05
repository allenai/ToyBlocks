from blocksuite.tasks.base_env_cfg import BaseEnvCfg
from isaaclab.utils import configclass

@configclass
class BlocksuiteEnvCfg(BaseEnvCfg):
    episode_length_s = 1000.0

    # customizable parameters
    block_spawn_mode = "debug"
    num_blocks = 1
    block_size = (0.03, 0.03, 0.03)