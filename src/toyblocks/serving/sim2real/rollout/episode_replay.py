"""Replay a recorded LeRobot dataset episode on the real robot.

Purpose: isolate policy vs. execution. We run the *exact* recorded joint
targets (no policy in the loop) and compare what the robot measures against what
the dataset recorded. If the replay reproduces the dataset motion (and grasps
the cube), the sim2real execution is faithful and any task failure lives in the
policy; if the replay itself is off (e.g. arm too high), the gap is in
execution.

It reuses the vlagent client's env/gain/TAM setup verbatim (``env_setup.get_env``
driven by ``vlagent_rcs_client.json`` and the ``tam_utils`` helpers) so the
execution path matches the policy runs, and it writes:
  - a per-frame replay log (replay_frames.csv) pairing each sent target with the
    measured joints and the dataset's own recorded observation.

Dataset: ``~/dataset/compliant_test_flip_pnp_softer/blocks_success`` (soft PD,
pd_mode=0), 25 fps, absolute targets in ``action.joint_target`` (radians).

Usage:
    python -m toyblocks.serving.sim2real.rollout.episode_replay                 # episode 0
    EPISODE=3 python -m toyblocks.serving.sim2real.rollout.episode_replay       # pick an episode
    REPLAY_DATASET=/path/to/blocks_success python -m toyblocks.serving.sim2real.rollout.episode_replay

``fps``, ``max_rel_mov_joints`` and the ``tam_*`` settings come from
``vlagent_rcs_client.json``; ``ReplayConfig.tam`` overrides the ``tam`` switch.
"""

from __future__ import annotations

import os

# JAX (TAM history encoder) must see this before it initializes its backend.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import json  # noqa: E402
import logging  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from rcs.utils import SimpleFrameRate  # noqa: E402

# Reuse the exact env/gain/TAM setup and TAM helpers from the vlagent client so
# the execution path is identical (no drift between replay and live inference).
from toyblocks.serving.sim2real.rollout.env_setup import get_env  # noqa: E402
from toyblocks.serving.sim2real.rollout.inference_config import load_inference_config  # noqa: E402
from toyblocks.serving.sim2real.rollout.tam_utils import (  # noqa: E402
    check_ideal_model_alignment,
    hardware_robot,
    init_tam_runtime,
    launch_history_encoder_thread,
    tam_supported,
    warmup_history_encoder,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

# Override with REPLAY_DATASET=/path/to/<dataset>/blocks_success
DATASET_ROOT = Path(
    os.environ.get(
        "REPLAY_DATASET",
        # "/home/tobi/dataset/aug13_sim2real_kp40_pnp/blocks_success",
        #"/home/tobi/dataset/problem_letter_color_10_20260903_174710_662062_5419/datasets/blocks_success/",
        #"/home/yejin/code/robot-control-stack/toyblocks/datasets/sep14_flip_left_right_play/"
        # "/home/tobi/dataset/naman_mp_flip_aug_26/blocks_success",
        #"/home/yejin/code/robot-control-stack/toyblocks/datasets/sep28_cmp2/softer/blocks_success",
        "/home/yejin/code/robot-control-stack/toyblocks/datasets/sep28_cmp2/sysid/blocks_success",
    )
)
ROBOT_KEY = "right"


@dataclass
class ReplayConfig:
    dataset_root: Path = DATASET_ROOT
    episode: int = int(os.environ.get("EPISODE", "0"))
    # Run the replay with TAM enabled? Start with False to see the raw sim2real
    # execution gap of the bare soft controller, then re-run with True to see
    # whether TAM closes it. (Run both for the aug13 dataset, same episode.)
    # Overrides ``tam`` from vlagent_rcs_client.json.
    tam: bool = False
    # Some datasets store gripper as 0=open/1=closed; the live obs convention in
    # the vlagent client is 0=closed/1=open. Flip here if the gripper acts inverted.
    invert_gripper: bool = True
    # Safety: skip a frame's target if any joint would jump more than this in one
    # step (rad). None disables the check (the env still clamps to
    # max_relative_movement). Purely a warning aid.
    warn_step_jump_rad: float | None = np.deg2rad(6)
    # Where the per-frame replay CSV + meta are written (None disables).
    log_dir: str | None = None
    # Override the env's per-step joint clamp (deg). 
    # The clamp doubles as a speed limit: ``LimitedAbsoluteAction`` clips the
    # target to ``measured +- clamp``, so a lagging joint is held a constant
    # ``clamp`` ahead of itself and the PD loop settles at ``kp/kd * clamp``
    # rad/s. The client's 0.2 rad gives the wrist (kp 50, kd 10) 1.0 rad/s.
    max_rel_mov_deg: float | None = None


# --- Dataset loading (mirrors the reference notebook) ---------------------------

def _load_episodes_meta(root: Path) -> pd.DataFrame:
    cols = [
        "episode_index",
        "length",
        "tasks",
        "data/chunk_index",
        "data/file_index",
    ]
    frames = [
        pd.read_parquet(f, columns=cols)
        for f in sorted((root / "meta/episodes/chunk-000").glob("*.parquet"))
    ]
    return pd.concat(frames, ignore_index=True).set_index("episode_index")


def _load_episode(root: Path, meta: pd.DataFrame, episode_index: int) -> pd.DataFrame:
    row = meta.loc[episode_index]
    chunk = int(row["data/chunk_index"])
    file = int(row["data/file_index"])
    df = pd.read_parquet(root / f"data/chunk-{chunk:03d}/file-{file:03d}.parquet")
    return df[df["episode_index"] == episode_index].reset_index(drop=True)


def _stack(df: pd.DataFrame, col: str) -> np.ndarray:
    return np.stack(df[col].to_numpy())


def load_trajectory(root: Path, episode: int):
    """Return (q_target[T,7], gripper[T], pd_mode[T], q_obs[T,7], t[T], task).

    Frames with ``action.is_valid == False`` are dropped — these are terminal /
    "done" frames whose ``action.joint_target`` is garbage (e.g. a 130 deg jump)
    and must never be sent to the robot.
    """
    meta = _load_episodes_meta(root)
    ep = _load_episode(root, meta, episode)
    if "action.is_valid" in ep:
        valid = _stack(ep, "action.is_valid").astype(bool).reshape(-1)
        if not valid.all():
            logger.warning("dropping %d invalid frame(s) of %d", (~valid).sum(), len(ep))
            ep = ep[valid].reset_index(drop=True)
    q_target = _stack(ep, "action.joint_target").astype(np.float64)  # absolute rad
    q_obs = _stack(ep, "observation.state.joint_position").astype(np.float64)
    gripper = _stack(ep, "action.gripper_position").astype(np.float64).reshape(-1)
    action = _stack(ep, "action").astype(np.float64)
    pd_mode = action[:, 8] if action.shape[1] > 8 else np.zeros(len(ep))
    t = _stack(ep, "timestamp").astype(np.float64).reshape(-1)
    task = meta.loc[episode, "tasks"]
    return q_target, gripper, pd_mode, q_obs, t, task


# --- Replay ---------------------------------------------------------------------

def main() -> None:
    rcfg = ReplayConfig()
    cfg = load_inference_config()  # gains, TAM flags, warmup, fps from the client JSON
    cfg.tam = rcfg.tam  # replay decides TAM on/off independently of the client default
    if cfg.tam and not tam_supported():
        logger.warning("TAM only exists in the Franka hardware controller; disabling")
        cfg.tam = False

    if rcfg.log_dir is None:
        # Replay logs are run artifacts: they live under the repo's gitignored
        # outputs/, not wherever the process happened to be started.
        _outputs = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), *[os.pardir] * 5, "outputs"
        )
        rcfg.log_dir = os.path.join(
            os.path.normpath(_outputs), "replay_logs", time.strftime("%Y%m%d_%H%M%S")
        )
    os.makedirs(rcfg.log_dir, exist_ok=True)
    logger.info("replay logs -> %s", rcfg.log_dir)

    q_target, gripper, pd_mode, q_obs_ds, t_ds, task = load_trajectory(
        rcfg.dataset_root, rcfg.episode
    )
    n = len(q_target)
    logger.info(
        "episode %d: %d frames (%.1f s @ %d fps), task=%s",
        rcfg.episode, n, n / cfg.fps, cfg.fps, task,
    )
    if not np.allclose(pd_mode, 0.0):
        logger.warning(
            "dataset pd_mode is not all-0 (soft); this dataset should be soft. "
            "min=%.2f max=%.2f", pd_mode.min(), pd_mode.max()
        )
    # Report how often the env's per-step clamp will bite (targets vs clamp).
    clamp_deg = rcfg.max_rel_mov_deg if rcfg.max_rel_mov_deg is not None else np.rad2deg(cfg.max_rel_mov_joints)
    step_jump_deg = np.rad2deg(np.abs(np.diff(q_target, axis=0)).max(axis=1))
    n_clamped = int(np.sum(step_jump_deg > clamp_deg))
    logger.info(
        "per-step target jump: max %.1f deg, p99 %.1f deg; clamp %.1f deg would "
        "engage on %d/%d frames", step_jump_deg.max(), np.percentile(step_jump_deg, 99),
        clamp_deg, n_clamped, len(q_target),
    )
    if rcfg.max_rel_mov_deg is not None:
        cfg.max_rel_mov_joints = float(np.deg2rad(rcfg.max_rel_mov_deg))

    # Same env (incl. StorageWrapper; nothing is recorded unless start_record
    # is called) and the same gain/TAM controller config as the policy runs.
    env = get_env(cfg)

    # Warn about targets it still cannot reach.
    limits = np.array(hardware_robot(env).get_config().joint_limits, dtype=np.float64)
    out = (q_target < limits[0]) | (q_target > limits[1])
    for j in np.flatnonzero(out.any(axis=0)):
        logger.warning(
            "joint %d: %d/%d dataset targets outside the joint limits "
            "(target range %.4f..%.4f, limits %.4f..%.4f)",
            j + 1, int(out[:, j].sum()), len(q_target),
            q_target[:, j].min(), q_target[:, j].max(),
            limits[0][j], limits[1][j],
        )

    # Stand up the TAM runtime the same way the vlagent client does, without
    # pulling in the policy server (ModelInference opens a RemoteAgent).
    tam_runtime = None
    if cfg.tam:
        tam_runtime = init_tam_runtime()
        check_ideal_model_alignment(hardware_robot(env), tam_runtime)
        warmup_history_encoder(tam_runtime, cfg)

    replay_log = None
    if rcfg.log_dir:
        replay_log = open(os.path.join(rcfg.log_dir, "replay_frames.csv"), "w")
        header = ["frame", "t_ds", "gripper_sent"]
        header += [f"q_sent{j}" for j in range(7)]
        header += [f"q_meas{j}" for j in range(7)]
        header += [f"q_ds{j}" for j in range(7)]  # dataset-recorded measured q
        replay_log.write(",".join(header) + "\n")
        replay_log.flush()
        # Save the episode/dataset provenance next to the logs.
        with open(os.path.join(rcfg.log_dir, "replay_meta.json"), "w") as f:
            json.dump(
                {
                    "dataset_root": str(rcfg.dataset_root),
                    "episode": rcfg.episode,
                    "n_frames": int(n),
                    "fps": cfg.fps,
                    "tam": bool(cfg.tam),
                    "tam_residual_clip": list(cfg.tam_residual_clip)
                    if cfg.tam_residual_clip is not None else None,
                    "task": list(task) if hasattr(task, "__len__") else str(task),
                },
                f, indent=2,
            )

    input("Press Enter to start replay (Ctrl+C to abort)...")
    tam_stop = threading.Event()
    if tam_runtime is not None:
        launch_history_encoder_thread(hardware_robot(env), tam_runtime, tam_stop)

    # frame_rate = SimpleFrameRate(cfg.fps)
    with env:
        obs, _ = env.reset()
        try:
            for i in range(n):
                g = 1.0 - gripper[i] if rcfg.invert_gripper else gripper[i]
                act = {
                    ROBOT_KEY: {
                        "joints": q_target[i].astype(np.float32),
                        "gripper": np.array([g], dtype=np.float32),
                    }
                }
                obs, _, _, _, _ = env.step(act)
                if replay_log is not None:
                    q_meas = np.asarray(obs[ROBOT_KEY]["joints"], dtype=np.float64)
                    vals = [i, t_ds[i], g]
                    vals += q_target[i].tolist() + q_meas.tolist() + q_obs_ds[i].tolist()
                    replay_log.write(",".join(f"{v:.6f}" for v in vals) + "\n")
                # frame_rate()
            logger.info("replay finished (%d frames)", n)
        except KeyboardInterrupt:
            logger.info("replay aborted by user")
        finally:
            if replay_log is not None:
                replay_log.close()
            env.reset()
            tam_stop.set()
            logger.info("replay logs in %s", rcfg.log_dir)


if __name__ == "__main__":
    main()
