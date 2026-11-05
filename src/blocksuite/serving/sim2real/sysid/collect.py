"""Step 1 of sysid: run the excitation on the real robot and record it.

Builds the env with ``env_setup.get_env(load_inference_config())`` -- the same
call ``vlagent_rcs_client.py`` and ``episode_replay.py`` make -- so the PD gains,
torque limits, joint-limit overrides, TAM switch, camera set, control mode, rate
and per-step clamp are whatever the policy runs use, verbatim. Change a gain in
``env_setup.py`` and the sysid run follows automatically; nothing is redeclared
here, so a recording can never describe a plant we do not deploy.

Writes, under ``--record-path``:
  real/ (or sim/)        the StorageWrapper parquet: env_action, obs, dq, tau_J
  sysid_run_meta.json    the gains actually in force (read back from
                         ``env.stiffness_profile``), fps, per-step clamp,
                         excitation settings, seed, commanded and measured centre

The fits read that metadata so the replay runs at the same gains -- otherwise
the optimiser absorbs a gain mismatch into friction/armature. ISAAC_SYSID_PLAN.md
S1: stiffness/damping are FIXED, not fit.

Usage:
    python -m blocksuite.serving.sim2real.sysid.collect --record-path outputs/sysid_data_kp300 --dry-run
    python -m blocksuite.serving.sim2real.sysid.collect --record-path outputs/sysid_data_kp300
    python -m blocksuite.serving.sim2real.sysid.collect --center 0.0,-0.6,0.0,-2.2,0.0,1.6,0.8

Record the paired sim run with the same --seed and --center (flip
ROBOT_INSTANCE in hardware_config.py) and the two line up sample for sample.
Then: python -m blocksuite.serving.sim2real.sysid.fit_mujoco <record-path>
"""

from __future__ import annotations

import os

# JAX (TAM history encoder) must see this before it initializes its backend.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from rcs._core.common import RobotPlatform  # noqa: E402
from rcs.utils import SimpleFrameRate  # noqa: E402

from blocksuite.serving.sim2real.rollout import env_setup  # noqa: E402
from blocksuite.serving.sim2real.rollout.inference_config import load_inference_config  # noqa: E402
from blocksuite.serving.sim2real.sysid import common  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


def _write_run_meta(path: Path, env, cfg, ex: common.Excitation, q_home, q_des) -> None:
    """Dump the controller + excitation settings the fit has to mirror in sim."""
    meta = {
        "source": "sysid/collect.py (env via env_setup.get_env)",
        "stiffness_profile": getattr(env, "stiffness_profile", None),
        "controller": {
            "fps": cfg.fps,
            "control_mode": str(env_setup.CONTROL_MODE),
            "max_rel_mov_joints": cfg.max_rel_mov_joints,
            "tam": cfg.tam,
        },
        "excitation": ex.as_dict(),
        "q_center_commanded": ex.center_pose().tolist(),
        "q_center_measured": np.asarray(q_home).tolist(),
        "num_samples": int(len(q_des)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, default=str) + "\n")
    logger.info("Wrote run metadata to %s", path)


def _check_step_clamp(q_des: np.ndarray, max_rel_mov_joints: float) -> None:
    """Warn if the excitation asks for steps the env's clamp would truncate.

    ``LimitedAbsoluteAction`` clips each target to ``measured +- clamp``. A
    trajectory that exceeds it is silently flattened, so the recorded
    ``env_action`` would disagree with what the controller really saw.
    """
    step = float(np.abs(np.diff(q_des, axis=0)).max())
    logger.info(
        "Max commanded step %.4f rad/tick vs env clamp %.4f rad", step, max_rel_mov_joints
    )
    if step >= max_rel_mov_joints:
        logger.warning(
            "Excitation exceeds max_rel_mov_joints -- targets will be clamped. "
            "Lower --amp-scale, or raise max_rel_mov_joints in vlagent_rcs_client.json, "
            "or the fit will see a trajectory the robot was never commanded."
        )


def main(args: argparse.Namespace) -> None:
    cfg = load_inference_config()
    if args.tam is not None:
        cfg.tam = args.tam
    if args.fps is not None:
        cfg.fps = args.fps

    # The fits read ``<folder>/real`` (or ``/sim``), but env_setup's StorageWrapper
    # writes straight into ``record_path``. Append the split here so the recording
    # lands in the layout the fit expects.
    base = Path(args.record_path if args.record_path is not None else cfg.record_path)
    split = "sim" if env_setup.ROBOT_INSTANCE != RobotPlatform.HARDWARE else "real"
    cfg.record_path = str(base / split)

    center = None
    if args.center is not None:
        center = np.array([float(x) for x in args.center.split(",")], dtype=float)
        if center.shape != (7,):
            raise SystemExit("--center must be 7 comma-separated joint angles [rad]")
        # env_setup.get_env() homes to its module-level HOME_POSE; point it at the
        # excitation centre so the arm starts where the trajectory is centred.
        env_setup.HOME_POSE = center

    ex = common.Excitation(
        fps=cfg.fps,
        seed=args.seed,
        duration=args.duration,
        ramp_time=args.ramp_time,
        amp_scale=args.amp_scale,
        center=center,
    )

    logger.info(
        "Excitation centre: %s (seed=%d, %.0f s @ %d Hz)",
        np.round(ex.center_pose(), 3).tolist(), ex.seed, ex.duration, ex.fps,
    )
    logger.info("Recording to %s", cfg.record_path)
    logger.info("Building env via env_setup.get_env (same path as the policy runs)")
    env_rel = env_setup.get_env(cfg)
    logger.info("Active PD profile: %s", getattr(env_rel, "stiffness_profile", None))

    with env_rel:
        env_rel.reset()
        robot = env_rel.get_wrapper_attr("robot")[common.ROBOT_KEY]

        if not args.yes:
            input(
                f"The robot will move to {np.round(ex.center_pose(), 3).tolist()} and run a "
                f"{ex.duration:.0f} s excitation (amp_scale={ex.amp_scale}) with the gains "
                "above. Clear the workspace, fix the gripper state, keep the e-stop ready, "
                "then press enter."
            )
        robot.move_home()

        # Centre on the pose actually reached, not the commanded one.
        q_home = np.asarray(robot.get_joint_position(), dtype=float)
        q_des = ex.build(q_home)
        ex.safety_report(q_des)
        _check_step_clamp(q_des, cfg.max_rel_mov_joints)

        _write_run_meta(base / "sysid_run_meta.json", env_rel, cfg, ex, q_home, q_des)
        if args.dry_run:
            logger.info("--dry-run: trajectory planned and metadata written, not moving.")
            return

        gripper = np.array([common.GRIPPER_ACTION], dtype=np.float32)
        env_rel.start_record()
        rate = SimpleFrameRate(cfg.fps, "sysid excitation loop")
        try:
            for target in q_des:
                env_rel.step(
                    {common.ROBOT_KEY: {"joints": target.copy(), "gripper": gripper.copy()}}
                )
                rate()
        finally:
            env_rel.stop_record()
            logger.info("Excitation finished, returning home.")
            robot.move_home()

    logger.info("Done. Fit with: python -m blocksuite.serving.sim2real.sysid.fit_mujoco %s", base)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--record-path", default=None, help="Output folder (default: record_path from vlagent_rcs_client.json).")
    p.add_argument("--seed", type=int, default=common.SEED, help="Excitation seed. Use the same seed for the paired sim and real runs.")
    p.add_argument("--center", default=None, help="7 comma-separated joint angles [rad] to excite around (default: env_setup's HOME_POSE).")
    p.add_argument("--duration", type=float, default=common.DURATION, help="Excitation duration [s].")
    p.add_argument("--ramp-time", type=float, default=common.RAMP_TIME, help="Amplitude fade in/out [s].")
    p.add_argument("--amp-scale", type=float, default=common.AMP_SCALE, help="Global multiplier on the per-joint amplitudes.")
    p.add_argument("--fps", type=int, default=None, help="Override the config's control rate [Hz].")
    p.add_argument("--tam", type=lambda s: s.lower() == "true", default=None, help="Override the config's TAM switch (true/false). Leave off for a bare-PD plant.")
    p.add_argument("--dry-run", action="store_true", help="Plan the trajectory and write metadata, but do not move the robot.")
    p.add_argument("-y", "--yes", action="store_true", help="Skip the confirmation prompt.")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
