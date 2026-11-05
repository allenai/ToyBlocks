import gymnasium as gym

from toyblocks.tasks.base_env import BaseEnv
from toyblocks.utils import common_utils
class FlattenObservationWrapper(gym.ObservationWrapper):
    """
    Flattens the observations into a single vector
    """

    def __init__(self, env) -> None:
        super().__init__(env)
        self.base_env.update_observation_space(
            self.observation(self.base_env._init_raw_obs)
        )

    @property
    def base_env(self) -> BaseEnv:
        return self.env.unwrapped

    def observation(self, observation):
        # The env emits every agent key; the flat vector carries only the subset the
        # RL policy was trained on. Rebinding "agent" keeps it first in the dict, which
        # is what fixes the flat layout as [agent | state | goal | privileged].
        observation = dict(observation)
        observation["agent"] = self.base_env.rl_agent_obs(observation["agent"])
        return common_utils.flatten_state_dict(observation)
    
    def get_observations(self):
        return self.observation(self.base_env._get_observations())

    def step(self, *args, **kwargs):
        ret = super().step(*args, **kwargs)
        infos = ret[-1]
        if "final_observation" in infos:
            infos["final_observation"] = self.observation(infos["final_observation"])
        return ret