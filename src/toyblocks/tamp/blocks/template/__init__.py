import gymnasium as gym

gym.register(
    id="ToyBlocks-Template-v0",
    entry_point=f"{__name__}._template_env_cfg:TemplateEnvCfg",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}._template_env_cfg:TemplateEnvCfg",
    },
)