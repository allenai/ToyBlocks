from blocksuite.tasks.base_env_cfg import BaseEnvCfg
from isaaclab.utils import configclass
@configclass
class LiftEnvCfg(BaseEnvCfg):
    episode_length_s = 2.56 # episode horizon = 64

    block_size = (0.04, 0.04, 0.04)