from blocksuite.cli.isaac_app import launch_app
simulation_app = launch_app()
from dataclasses import dataclass
from typing import Annotated
import gymnasium as gym
import torch
import tyro
import blocksuite.tasks  # noqa: F401
from blocksuite.tasks.lift.lift_env import LiftEnv
from blocksuite.cli.base_args import BaseArgs, add_env_cfg_args

@add_env_cfg_args
@dataclass
class Args(BaseArgs):
    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    """Number of parallel environments."""
    task: str = "Blocksuite-Lift-v0"
    """The task to run."""
    seed: Annotated[int, tyro.conf.arg(aliases=["-s"])] = 0
    """The seed to use"""
    test: str = "sweep"
    """test to run. Can be sweep, nudge, grasp, flip_backward, flip_forward"""
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
    env_cfg.episode_length_s = 4
    env_cfg.obs_mode = args.obs_mode
    # create environment
    env = gym.make(args.task, cfg=env_cfg)
    base_env: LiftEnv = env.unwrapped
    if args.headless:
        from blocksuite.viewer.viewer import Viewer, ViewerCfg
        viewer = Viewer(cfg=ViewerCfg(
            include_cameras=True,
            include_substeps=False,
            env_ids=list(range(min(args.num_envs, 16))),
        ))
        viewer.hook_to_env(env)
    env.reset(seed=args.seed)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    
    if args.test == "sweep":
        block_pose = base_env.block.data.root_pose_w.clone()
        block_pose[:, :3] = base_env.scene.env_origins
        block_pose[:, 0] += 0.57
        block_pose[:, 1] += 0.09
        block_pose[:, 2] = base_env.cfg.block_size[2] / 2 + 5e-4
        block_pose[:, 3:] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        base_env.block.write_root_pose_to_sim(block_pose)
    elif args.test == "nudge":
        block_pose = base_env.block.data.root_pose_w.clone()
        block_pose[:, :3] = base_env.scene.env_origins
        block_pose[:, 0] += 0.6
        block_pose[:, 1] += 0.09
        block_pose[:, 2] = base_env.cfg.block_size[2] / 2 + 5e-4
        block_pose[:, 3:] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        base_env.block.write_root_pose_to_sim(block_pose)
    elif args.test == "grasp":
        block_pose = base_env.block.data.root_pose_w.clone()
        block_pose[:, :3] = base_env.scene.env_origins
        block_pose[:, 0] += 0.58
        block_pose[:, 2] = base_env.cfg.block_size[2] / 2 + 5e-4
        block_pose[:, 3:] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        base_env.block.write_root_pose_to_sim(block_pose)
    elif args.test == "flip_backward":
        block_pose = base_env.block.data.root_pose_w.clone()
        block_pose[:, :3] = base_env.scene.env_origins
        block_pose[:, 0] += 0.574
        block_pose[:, 2] = base_env.cfg.block_size[2] / 2 + 5e-4
        block_pose[:, 3:] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        base_env.block.write_root_pose_to_sim(block_pose)
    elif args.test == "flip_forward":
        block_pose = base_env.block.data.root_pose_w.clone()
        block_pose[:, :3] = base_env.scene.env_origins
        block_pose[:, 0] += 0.61
        block_pose[:, 2] = base_env.cfg.block_size[2] / 2 + 5e-4
        block_pose[:, 3:] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        base_env.block.write_root_pose_to_sim(block_pose)
    
    t = 0
    while simulation_app.is_running():
        print(f"Timestep {t}")
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        t += 1
        if args.test == "sweep" or args.test == "nudge":
            if t <= 15:
                actions[:, -1] = 1
            elif t <= 30:
                actions[:, -1] = 1
                actions[:, 1] = 0.2
            elif t < 45:
                actions[:, -1] = -1
            elif t < 75:
                actions[:, 0] = 0.3
                actions[:, -1] = -1
            else:
                actions[:, -1] = -1
        elif args.test == "grasp":
            if t <= 15:
                actions[:, -1] = 1
            elif t <= 35:
                actions[:, -1] = 1
                actions[:, 1] = 0.2
            elif t < 45:
                actions[:, -1] = -1
            elif t < 75:
                actions[:, 1] = -0.2
                actions[:, -1] = -1
            else:
                actions[:, -1] = -1
        elif args.test == "flip_backward":
            if t <= 15:
                actions[:, -1] = -1
            elif t <= 28:
                actions[:, -1] = -1
                actions[:, 1] = 0.05
                actions[:, 3] = -0.5
            elif t < 45:
                actions[:, -1] = -1
            elif t < 75:
                actions[:, 1] = -0.25
                actions[:, -1] = -1
            else:
                actions[:, 0] = 0.1
                actions[:, -1] = -1
        elif args.test == "flip_forward":
            if t <= 15:
                actions[:, -1] = -1
            elif t <= 28:
                actions[:, -1] = -1
                actions[:, 1] = 0.2
                actions[:, 3] = -0.2
            elif t < 45:
                actions[:, -1] = -1
                actions[:, 3] = 0.45
            elif t < 75:
                actions[:, -1] = -1
            else:
                actions[:, -1] = -1
        obs, reward, terminated, truncated, info = env.step(actions)
        dones = terminated | truncated
        if dones.any():
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
