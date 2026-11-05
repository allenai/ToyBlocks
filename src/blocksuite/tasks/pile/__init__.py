import gymnasium as gym

gym.register(
    id="Blocksuite-Pile-v0",
    entry_point=f"{__name__}.blocksuite_env:BlocksuiteEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.blocksuite_env_cfg:BlocksuiteEnvCfg",
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)