import gymnasium as gym

gym.register(
    id="ToyBlocks-Push-Stacked-v0",
    entry_point=f"{__name__}.push_stacked_env:PushStackedEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.push_stacked_env_cfg:PushStackedEnvCfg",
    },
)