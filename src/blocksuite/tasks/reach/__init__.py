import gymnasium as gym

gym.register(
    id="Blocksuite-Reach-v0",
    entry_point=f"{__name__}.reach_env:ReachEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reach_env_cfg:ReachEnvCfg",
    },
)
