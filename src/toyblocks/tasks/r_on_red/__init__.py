import gymnasium as gym

gym.register(
    id="ToyBlocks-ROnRed-v0",
    entry_point=f"{__name__}.r_on_red_env:ROnRedEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.r_on_red_env_cfg:ROnRedEnvCfg",
    },
)
