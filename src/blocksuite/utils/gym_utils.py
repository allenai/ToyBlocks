"""Gymnasium utilities"""

import numpy as np
from gymnasium import spaces

def get_dtype_bounds(dtype: np.dtype):
    """Gets the min and max values of a given numpy type"""
    if np.issubdtype(dtype, np.floating):
        info = np.finfo(dtype)
        return info.min, info.max
    elif np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return info.min, info.max
    elif np.issubdtype(dtype, np.bool_):
        return 0, 1
    else:
        raise TypeError(dtype)


def convert_observation_to_space(observation, prefix="", unbatched=False):
    """Convert observation to OpenAI gym observation space (recursively).
    Adapted from https://github.com/haosulab/ManiSkill/blob/main/mani_skill/utils/gym_utils.py
    """
    if isinstance(observation, (dict)):
        # CATUION: Explicitly create a list of key-value tuples
        # Otherwise, spaces.Dict will sort keys if a dict is provided
        space = spaces.Dict(
            [
                (
                    k,
                    convert_observation_to_space(
                        v, prefix + "/" + k, unbatched=unbatched
                    ),
                )
                for k, v in observation.items()
            ]
        )
    elif isinstance(observation, np.ndarray):
        if unbatched:
            shape = observation.shape[1:]
        else:
            shape = observation.shape
        dtype = observation.dtype
        low, high = get_dtype_bounds(dtype)
        if np.issubdtype(dtype, np.floating):
            low, high = -np.inf, np.inf
        space = spaces.Box(low, high, shape=shape, dtype=dtype)
    elif isinstance(observation, (float, np.float32, np.float64)):
        space = spaces.Box(-np.inf, np.inf, shape=[1], dtype=np.float32)
    elif isinstance(observation, (int, np.int32, np.int64)):
        space = spaces.Box(-np.inf, np.inf, shape=[1], dtype=int)
    elif isinstance(observation, (bool, np.bool_)):
        space = spaces.Box(0, 1, shape=[1], dtype=np.bool_)
    else:
        raise NotImplementedError(type(observation), observation)

    return space