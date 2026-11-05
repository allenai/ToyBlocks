import gymnasium as gym

gym.register(
    id="ToyBlocks-Pile-v0",
    entry_point=f"{__name__}.toyblocks_env:ToyBlocksEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.toyblocks_env_cfg:ToyBlocksEnvCfg",
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)