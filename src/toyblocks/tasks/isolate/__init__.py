import gymnasium as gym

gym.register(
    id="ToyBlocks-Isolate-v0",
    entry_point=f"{__name__}.isolate_env:IsolateEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isolate_env_cfg:IsolateEnvCfg",
    },
)