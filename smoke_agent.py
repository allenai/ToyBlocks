"""CPU smoke test for the ManiFlowPolicy vlagents agent against a dummy checkpoint.

Exercises the exact code path franka.py -> vlagents server uses (initialize /
reset / act with obs shaped like obs_rcs2agents produces), without needing the
rpyc server or the rcs sim. Runs on CPU (hides the GPU).

    python vlagents/utils/smoke_agent.py \
        --checkpoint ckpts/dummy_maniflow/checkpoints/dummy_model.safetensors \
        --stats ckpts/dummy_maniflow/stats.json
"""

import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # force CPU (before torch import)

import numpy as np

from vlagents.policies import ManiFlowPolicy, Obs


def make_obs() -> Obs:
    return Obs(
        cameras={
            "front_rgb": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "gripper_rgb": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        },
        state=np.zeros(8, dtype=np.float32),  # 7 joints + 1 gripper
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="ckpts/dummy_maniflow/checkpoints/dummy_model.safetensors")
    p.add_argument("--stats", default="ckpts/dummy_maniflow/stats.json")
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--execution-horizon", type=int, default=4)
    args = p.parse_args()

    agent = ManiFlowPolicy(
        checkpoint_path=args.checkpoint,
        stats_path=args.stats,
        device="cpu",
        execution_horizon=args.execution_horizon,
    )
    print(">> initialize()")
    agent.initialize()

    print(">> reset()")
    agent.reset(make_obs(), instruction="pick up the green cube")

    print(f">> act() x{args.steps} (exercises chunk buffering)")
    for i in range(args.steps):
        a = np.asarray(agent.act(make_obs()).action)
        gripper = a[7] if a.shape[0] > 7 else "NA"
        print(f"  step {i}: shape={a.shape} dtype={a.dtype} first3={np.round(a[:3], 4).tolist()} gripper={gripper}")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
