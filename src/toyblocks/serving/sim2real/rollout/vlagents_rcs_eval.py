"""Evaluate a maniflow vlagents policy in simulation over multiple episodes.

Uses the same vlagents protocol as ``vlagent_rcs_client`` (obs conversion,
remote ``act`` calls, action unpacking) but runs headless-style automatic
episodes instead of the interactive command loop. Every step's ``info`` dict is
recorded and dumped to a JSON file together with aggregate averages over all
steps and episodes -- most importantly the success rate.

An episode ends as soon as ``info["success"]`` is true or ``max_steps`` is
reached. Runtime-tunable policy properties (instruction, n_action_steps, host,
port, model, ...) are read from ``vlagent_client.json`` next to this file, the
same config the interactive client uses.

Example:

    python -m toyblocks.serving.sim2real.rollout.vlagents_rcs_eval \
        --num-episodes 20 --max-steps 300 --output-dir vlagent_eval_runs --seed 0

Each run creates a timestamped subfolder under ``--output-dir`` containing
``results.json`` and, per episode, ``episode_<i>_front_rgb.mp4`` and
``episode_<i>_gripper_rgb.mp4`` (encoded via ffmpeg, as in ``vlagents.eval``).
"""

import argparse
import copy
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from toyblocks.serving.sim2real.rollout.goal_utils import InferenceConfig
from duobench.tasks.flipping import FlippingEnvConfig
import numpy as np
import rcs
from duobench.tasks.pick_place import PickPlaceEnvConfig
from rcs._core.sim import SimConfig
from rcs.envs.base import ControlMode, RelativeTo
from rcs.envs.storage_wrapper import StorageWrapper

from toyblocks.serving.sim2real.vlagent_rcs_client import (
    ModelInference,
    load_inference_config,
)

logger = logging.getLogger(__name__)

HOME_POSE = rcs.HOME_POSITIONS["FR3_DROID"]
CONTROL_MODE = ControlMode.JOINTS
RELATIVETO = RelativeTo.NONE

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def get_env2(cfg: InferenceConfig):
    """Build the simulation pick-place env (headless, non-realtime for eval)."""


    # scene = PickPlaceEnvConfig()
    scene = FlippingEnvConfig()
    sim_cfg_data = scene.config()
    # sim_cfg_data.robot_cfgs["right"].kp = [100., 100., 100., 100., 75., 150., 50.]
    # sim_cfg_data.robot_cfgs["right"].kv = [20., 20., 20., 20., 7.5, 15.0, 5.0]
    # sim_cfg_data.robot_cfgs["right"].kp = np.array([600, 600, 600, 600, 250, 150, 50])
    # sim_cfg_data.robot_cfgs["right"].kv = np.array([50, 50, 50, 50, 30, 25, 15])
    
    sim_cfg_data.robot_cfgs["right"].kp = np.array([200.,200.,200.,200.,50.,50.,30.])
    sim_cfg_data.robot_cfgs["right"].kv = np.array([40.,40.,40.,40.,10.,10.,8.])

    sim_cfg_data.robot_cfgs["right"].q_home = HOME_POSE
    sim_cfg_data.sim_cfg = SimConfig(
        async_control=True,
        realtime=False,
        frequency=cfg.fps,
        max_convergence_steps=500,
    )
    sim_cfg_data.wrapper_cfg.include_depth = False
    sim_cfg_data.control_mode = ControlMode.JOINTS
    sim_cfg_data.relative_to = RELATIVETO
    sim_cfg_data.wrapper_cfg.binary_gripper = True
    sim_cfg_data.headless = True
    sim_cfg_data.max_relative_movement = (
        cfg.max_rel_mov_joints
        if CONTROL_MODE == ControlMode.JOINTS
        else cfg.max_rel_mov_cart
    )

    env_rel = scene.create_env(sim_cfg_data)

    return StorageWrapper(
        env_rel,
        cfg.record_path,
        cfg.instruction,
        batch_size=32,
        max_rows_per_group=2,
        max_rows_per_file=10,
        allow_wrapper_instruction=False,
    )


EVAL_CAMERAS = ("front_rgb", "gripper_rgb")


def _write_camera_mp4(
    frames: list[np.ndarray], output_path: Path, fps: int = 30
) -> None:
    """Encode a list of RGB frames into an mp4 via ffmpeg (as in vlagents.eval)."""
    if not frames:
        return

    height, width = frames[0].shape[:2]
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert process.stdin is not None
    for frame in frames:
        process.stdin.write(np.ascontiguousarray(frame).astype(np.uint8).tobytes())
    process.stdin.close()
    process.wait()


def _to_jsonable(value: Any) -> Any:
    """Recursively convert numpy / non-native types into JSON-serializable ones."""
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _strip_frames(obs: dict) -> dict:
    """Return the low-dim observation (per-robot state) without camera frames."""
    return {k: v for k, v in obs.items() if k != "frames"}


def _collect_frames(obs: dict, buffers: dict[str, list[np.ndarray]]) -> None:
    """Append this step's RGB frame for each eval camera to its buffer."""
    frames = obs.get("frames", {})
    for camera in EVAL_CAMERAS:
        rgb = frames.get(camera, {}).get("rgb", {}).get("data")
        if rgb is not None:
            buffers[camera].append(np.asarray(rgb))


def run_episode(
    controller: ModelInference,
    max_steps: int,
    episode_idx: int,
    output_dir: Path,
    fps: int,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run one episode until success or ``max_steps``, returning its record."""
    env = controller.env
    obs, _ = env.reset(seed=seed)
    controller._action_source.reset()
    obs_dict = controller.obs_rcs2agents(obs)

    step_infos: list[dict[str, Any]] = []
    frame_buffers: dict[str, list[np.ndarray]] = {cam: [] for cam in EVAL_CAMERAS}
    _collect_frames(obs, frame_buffers)
    success = False

    for step in range(max_steps):
        action = controller.act(copy.deepcopy(obs_dict))
        if any(robot_action.done for robot_action in action.values()):
            logger.info("episode %d: agent issued done at step %d", episode_idx, step)
            break

        rcs_action, _pd_mode = controller.action_agents2rcs(action)
        obs, reward, _, _, info = env.step(rcs_action)
        obs_dict = controller.obs_rcs2agents(obs)
        _collect_frames(obs, frame_buffers)

        step_infos.append(
            {
                "step": step,
                "reward": float(reward),
                "info": _to_jsonable(info),
                "action": _to_jsonable(rcs_action),
                "observation": _to_jsonable(_strip_frames(obs)),
            }
        )

        if bool(info.get("success", False)):
            success = True
            logger.info("episode %d: success at step %d", episode_idx, step)
            break

    for camera, frames in frame_buffers.items():
        _write_camera_mp4(
            frames, output_dir / f"episode_{episode_idx:03d}_{camera}.mp4", fps=fps
        )

    if not success:
        logger.info(
            "episode %d: finished without success after %d steps",
            episode_idx,
            len(step_infos),
        )

    return {
        "episode": episode_idx,
        "success": success,
        "num_steps": len(step_infos),
        "steps": step_infos,
    }


def aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute averages across all episodes and steps."""
    num_episodes = len(episodes)
    episode_success = [1.0 if ep["success"] else 0.0 for ep in episodes]
    step_counts = [ep["num_steps"] for ep in episodes]

    # Average over every recorded step of every numeric info field.
    numeric_fields: dict[str, list[float]] = {}
    for ep in episodes:
        for step in ep["steps"]:
            for key, val in step["info"].items():
                if isinstance(val, bool):
                    numeric_fields.setdefault(key, []).append(1.0 if val else 0.0)
                elif isinstance(val, (int, float)):
                    numeric_fields.setdefault(key, []).append(float(val))

    per_step_averages = {key: _mean(vals) for key, vals in numeric_fields.items()}

    return {
        "num_episodes": num_episodes,
        "success_rate": _mean(episode_success),
        "num_successes": int(sum(episode_success)),
        "avg_steps_per_episode": _mean([float(c) for c in step_counts]),
        "per_step_averages": per_step_averages,
    }


def build_lean(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduced view: per-step stage + success per episode, plus aggregates."""
    lean_episodes = []
    episode_max_stage = []
    for ep in episodes:
        stages = [s["info"].get("stage") for s in ep["steps"]]
        episode_max_stage.append(max(stages) if stages else None)
        lean_episodes.append(
            {
                "episode": ep["episode"],
                "success": ep["success"],
                "num_steps": ep["num_steps"],
                "stages": stages,
            }
        )

    episode_success = [1.0 if ep["success"] else 0.0 for ep in episodes]
    return {
        "summary": {
            "num_episodes": len(episodes),
            "success_rate": _mean(episode_success),
            "num_successes": int(sum(episode_success)),
            "avg_max_stage": _mean([s for s in episode_max_stage if s is not None]),
        },
        "episodes": lean_episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-episodes", type=int, default=10, help="number of episodes to evaluate"
    )
    parser.add_argument(
        "--max-steps", type=int, default=300, help="max steps per episode"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="base seed; episode i is reset with seed+i for deterministic scenes",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("vlagent_eval_runs"),
        help="parent directory; a timestamped subfolder is created per run",
    )
    args = parser.parse_args()

    logger.info("Building env")
    np.random.seed(args.seed)
    cfg = load_inference_config()

    run_dir = args.output_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("writing run outputs to %s", run_dir)

    env = get_env2(cfg)
    controller = ModelInference(env, cfg)

    episodes: list[dict[str, Any]] = []
    with env:
        controller.remote_agent.ensure_connected()
        for episode_idx in range(args.num_episodes):
            episodes.append(
                run_episode(
                    controller,
                    args.max_steps,
                    episode_idx,
                    run_dir,
                    cfg.fps,
                    seed=args.seed + episode_idx,
                )
            )

    summary = aggregate(episodes)
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_episodes": args.num_episodes,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "instruction": cfg.instruction,
            "vlagents_model": cfg.vlagents_model,
            "n_action_steps": cfg.n_action_steps,
        },
        "summary": summary,
        "episodes": episodes,
    }

    output_json = run_dir / "results.json"
    output_json.write_text(json.dumps(_to_jsonable(results), indent=2) + "\n")

    lean = {
        "timestamp": results["timestamp"],
        "config": results["config"],
        **build_lean(episodes),
    }
    lean_json = run_dir / "results_lean.json"
    lean_json.write_text(json.dumps(_to_jsonable(lean), indent=2) + "\n")

    logger.info(
        "wrote results to %s | success_rate=%.3f (%d/%d)",
        output_json,
        summary["success_rate"] if summary["success_rate"] is not None else float("nan"),
        summary["num_successes"],
        summary["num_episodes"],
    )


if __name__ == "__main__":
    main()