import gymnasium as gym

gym.register(
    id="ToyBlocks-Align-v0",
    entry_point=f"{__name__}.align_env:AlignEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.align_env_cfg:AlignEnvCfg",
    },
)