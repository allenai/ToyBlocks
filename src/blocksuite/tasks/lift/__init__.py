import gymnasium as gym

gym.register(
    id="Blocksuite-Lift-v0",
    entry_point=f"{__name__}.lift_env:LiftEnv",
    disable_env_checker=True,
    # TODO (stao): most of the time a gym registration is repetitive. maniskill uses a registration decorator to make it one line
    # alternatively we keep it more verbose like this if we want to do environment variations via different env registrations?
    kwargs={
        # TODO (stao): what is env_cfg_entry_point for? where is it used in isaac lab?
        "env_cfg_entry_point": f"{__name__}.lift_env_cfg:LiftEnvCfg",
    },
)