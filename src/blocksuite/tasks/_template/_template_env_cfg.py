"""
Minimal template environment. If making a new environment, copy the _template folder this file is in
"""

from blocksuite.tasks.base_env_cfg import BaseEnvCfg
from isaaclab.utils import configclass
@configclass
class TemplateEnvCfg(BaseEnvCfg):
    episode_length_s = 2.56 # episode horizon = 64 by default