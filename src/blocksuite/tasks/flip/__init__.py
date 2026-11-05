import gymnasium as gym

gym.register(
    id="Blocksuite-Flip-v0",
    entry_point=f"{__name__}.flip_env:FlipEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flip_env_cfg:FlipEnvCfg",
    },
)