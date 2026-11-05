import gymnasium as gym

gym.register(
    id="ToyBlocks-Rotate-v0",
    entry_point=f"{__name__}.rotate_env:RotateEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rotate_env_cfg:RotateEnvCfg",
    },
)