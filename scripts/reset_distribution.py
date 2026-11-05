import tyro
from blocksuite.cli.isaac_app import launch_app
simulation_app = launch_app()
from dataclasses import dataclass
from typing import Annotated
from blocksuite.cli.base_args import BaseArgs, add_env_cfg_args
import gymnasium as gym
import blocksuite.tasks  # noqa: F401
@add_env_cfg_args
@dataclass
class Args(BaseArgs):
    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    """Number of parallel environments."""
    task: str = "Blocksuite-Lift-v0"
    """The task to run."""
    seed: Annotated[int, tyro.conf.arg(aliases=["-s"])] = 0
    """The seed to use"""
    num_resets: int = 256
    """Number of resets to run."""
    obs_mode: str = "state"
    """the obs mode to use. Can be state, rgb, or rgb+state"""
    record_video: bool = False
    """whether to record video. Will create a grid of videos if there are parallel environments. If there are multiple cameras, they will be
    combined."""

def main(args: Args):
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = args.env_cfg
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.sim.use_fabric = not args.disable_fabric
    env_cfg.obs_mode = args.obs_mode
    # create environment
    env = gym.make(args.task, cfg=env_cfg)
    if args.headless:
        from blocksuite.viewer.viewer import Viewer, ViewerCfg
        viewer = Viewer(cfg=ViewerCfg(
            include_cameras=True,
            env_ids=list(range(min(args.num_envs, 16))),
            ignore_empty_trajectories=False,
            max_trajectories_per_env=args.num_resets
        ))
        viewer.hook_to_env(env)
    seed = args.seed
    while simulation_app.is_running():
        env.reset(seed=seed)
        env.render()
        seed += 1
        if seed == args.num_resets + args.seed:
            break
    env.close()
    if args.headless:
        print(f"Scripted Finished. Viewer is still running at {viewer.get_server_url()}")
        while True:
            import time
            time.sleep(0.1)


if __name__ == "__main__":
    main(tyro.cli(Args))
    simulation_app.close()
