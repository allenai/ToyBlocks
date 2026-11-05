from toyblocks.cli.isaac_app import launch_app
simulation_app = launch_app()
from dataclasses import dataclass
from typing import Annotated
from toyblocks.cli.base_args import BaseArgs, add_env_cfg_args
import gymnasium as gym
import torch
import tyro
import toyblocks.tasks  # noqa: F401

@add_env_cfg_args
@dataclass
class Args(BaseArgs):
    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    """Number of parallel environments."""
    task: str = "ToyBlocks-Lift-v0"
    """The task to run."""
    seed: Annotated[int, tyro.conf.arg(aliases=["-s"])] = 0
    """The seed to use"""
    obs_mode: str = "state"
    """the obs mode to use. Can be state, rgb, or rgb+state"""
    control_mode: str = "arm_relative_joint_gripper_binary"
    """The control mode to use"""
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
    env_cfg.control_mode = args.control_mode
    # create environment
    env = gym.make(args.task, cfg=env_cfg)
    if args.headless:
        from toyblocks.viewer.viewer import Viewer, ViewerCfg
        viewer = Viewer(cfg=ViewerCfg(
            env_ids=list(range(min(args.num_envs, 16))),
        ))
        viewer.hook_to_env(env)
    env.reset(seed=args.seed)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    t = 0
    while simulation_app.is_running():
        print(f"Timestep {t}")
        actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
        t += 1
        obs, reward, terminated, truncated, info = env.step(actions)
        dones = terminated | truncated
        if dones.any():
            break
    # close the simulator
    env.close()
    if args.headless:
        print(f"Scripted Finished. Viewer is still running at {viewer.get_server_url()}")
        while True:
            import time
            time.sleep(0.1)


if __name__ == "__main__":
    main(tyro.cli(Args))
    simulation_app.close()
