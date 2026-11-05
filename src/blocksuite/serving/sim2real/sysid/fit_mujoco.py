"""Step 2a of sysid: fit MuJoCo joint dynamics to a recording. See SYSID.md.

Replays the recording open-loop in a headless free-space sim at the *recorded*
gains and fits, per joint, the three parameters that are not observable from the
FR3's link-side ``tau_J`` sensor -- Coulomb friction, viscous damping and rotor
inertia -- by matching sim ``q(t)`` to real ``q(t)``.

    python -m blocksuite.serving.sim2real.sysid.fit_mujoco outputs/sysid_data_kp300

Keep the budget modest: more passes keep lowering the training RMS but push
damping/armature to their bounds -- the three are collinear at these frequencies,
so a lower RMS can mean less physical parameters. One pass stays physical.

Originally the ``calibrate()`` half of Tobias Juelg's ``sysid.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
from rcs._core.sim import SimConfig
from rcs.envs.base import ControlMode, RelativeTo

from blocksuite.serving.sim2real.sysid import common

logger = logging.getLogger(__name__)

PASSES = 1  # coordinate-descent sweeps over the 7 joints
MAX_EVALS = 40  # sim rollouts per joint per sweep
LO = np.array([0.0, 0.0, 0.005])  # bounds on [frictionloss, damping, armature]
HI = np.array([6.0, 6.0, 1.0])


def build_sim(q_home, kp, kd, fps, max_rel_mov):
    """Free-space single-arm sim: no task scene, no rendering, realtime off.

    ``q_home`` is the recording's first target, so an extended-configuration
    dataset replays without a large move-in transient.
    """
    from duobench.utils.single_arm import SingleArm

    scene = SingleArm()
    cfg = scene.config()
    cfg.robot_cfgs[common.ROBOT_KEY].kp = np.asarray(kp, dtype=float).tolist()
    cfg.robot_cfgs[common.ROBOT_KEY].kv = np.asarray(kd, dtype=float).tolist()
    cfg.robot_cfgs[common.ROBOT_KEY].q_home = np.asarray(q_home, dtype=float)
    cfg.sim_cfg = SimConfig(async_control=True, realtime=False, frequency=fps,
                            max_convergence_steps=500)
    cfg.control_mode = ControlMode.JOINTS
    cfg.relative_to = RelativeTo.NONE
    cfg.wrapper_cfg.binary_gripper = True
    cfg.wrapper_cfg.include_depth = False
    cfg.headless = True
    cfg.camera_cfgs = None
    if max_rel_mov is not None:
        cfg.max_relative_movement = max_rel_mov
    return scene.create_env(cfg)


def fit(folder, out_path, *, passes=PASSES, max_evals=MAX_EVALS, plots=True) -> dict:
    q_des, real_q = common.load_recording(folder)
    logger.info("Loaded %d real samples from %s/real", len(real_q), folder)
    kp, kd, max_rel_mov, fps = common.resolve_controller(folder)

    env = build_sim(q_des[0], kp, kd, fps, max_rel_mov)
    model = env.get_wrapper_attr("sim").model
    names = sorted(model.joint(i).name for i in range(model.njnt)
                   if "fr3_joint" in model.joint(i).name)
    dof = [int(model.jnt_dofadr[model.joint(n).id]) for n in names]
    gripper = np.array([common.GRIPPER_ACTION], dtype=np.float32)
    T = min(len(q_des), len(real_q))
    real_q = real_q[:T]

    def rollout(fric, damp, arm) -> np.ndarray:
        """Set params, replay q_des, return simulated q of shape (T, 7)."""
        for a, f, d, r in zip(dof, fric, damp, arm):
            model.dof_frictionloss[a], model.dof_damping[a], model.dof_armature[a] = f, d, r
        env.reset()
        sim_q = np.empty((T, 7))
        for i in range(T):
            obs, *_ = env.step(
                {common.ROBOT_KEY: {"joints": q_des[i].copy(), "gripper": gripper.copy()}}
            )
            sim_q[i] = obs[common.ROBOT_KEY]["joints"]
        return sim_q

    def rms(sim_q) -> np.ndarray:
        return np.rad2deg(np.sqrt(((sim_q - real_q) ** 2).mean(0)))

    fric, damp, arm = (model.dof_frictionloss[dof].copy(), model.dof_damping[dof].copy(),
                       model.dof_armature[dof].copy())
    fric0, damp0, arm0 = fric.copy(), damp.copy(), arm.copy()
    sim_before = rollout(fric, damp, arm)
    before = rms(sim_before)
    logger.info("initial per-joint RMS [deg]: %s (mean %.3f)",
                np.round(before, 3).tolist(), before.mean())

    # Coordinate descent over joints; per joint a pattern search over the three.
    history = []
    for sweep in range(passes):
        for j in range(7):
            x = np.array([fric[j], damp[j], arm[j]])
            step = np.maximum(0.3 * np.abs(x), np.array([0.2, 0.1, 0.03]))

            def obj(xx) -> float:
                fric[j], damp[j], arm[j] = xx
                return rms(rollout(fric, damp, arm))[j]

            best, evals = obj(x), 1
            while evals < max_evals and np.any(step > np.array([0.02, 0.01, 0.005])):
                improved = False
                for i in range(3):
                    for s in (step[i], -step[i]):
                        if evals >= max_evals:
                            break
                        xt = x.copy()
                        xt[i] = float(np.clip(x[i] + s, LO[i], HI[i]))
                        if xt[i] == x[i]:
                            continue
                        v, evals = obj(xt), evals + 1
                        if v < best - 1e-9:
                            best, x, improved = v, xt, True
                if not improved:
                    step *= 0.5
            fric[j], damp[j], arm[j] = x
            history.append((f"p{sweep + 1}j{j + 1}", float(best)))
            logger.info("sweep %d joint %d: fric %.3f->%.3f  damp %.3f->%.3f  "
                        "arm %.3f->%.3f  (RMS %.3f->%.3f deg)", sweep + 1, j + 1,
                        fric0[j], fric[j], damp0[j], damp[j], arm0[j], arm[j], before[j], best)

    sim_after = rollout(fric, damp, arm)
    after = rms(sim_after)
    result = {
        "data_folder": os.path.abspath(folder),
        "engine": "mujoco",
        "note": "Closed-loop replay fit (sim q(t) matched to real q(t)). Training error only -- no held-out split.",
        "joints": names,
        "gains": {"kp": np.asarray(kp).tolist(), "kd": np.asarray(kd).tolist(), "fixed": True},
        "fps": fps, "passes": passes, "max_evals_per_joint": max_evals,
        "initial": {"frictionloss": fric0.tolist(), "damping": damp0.tolist(), "armature": arm0.tolist()},
        "calibrated": {"frictionloss": fric.tolist(), "damping": damp.tolist(), "armature": arm.tolist()},
        "rms_deg_before": before.tolist(), "rms_deg_after": after.tolist(),
        "rms_deg_before_mean": float(before.mean()), "rms_deg_after_mean": float(after.mean()),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("=== calibration result (paste into assets/robots/fr3/fr3.xml) ===")
    logger.info("joint | frictionloss | damping | armature | RMS[deg] before->after")
    for j in range(7):
        logger.info("  %s | %6.3f | %6.3f | %6.3f | %.3f -> %.3f",
                    names[j], fric[j], damp[j], arm[j], before[j], after[j])
    logger.info("mean RMS %.3f -> %.3f deg. Wrote %s", before.mean(), after.mean(), out_path)

    if plots:
        common.plot_fit(Path(folder) / "sysid_fit", real_q, sim_before, sim_after,
                        before, after, fps, convergence=history, title="MuJoCo: ")
    env.close()
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", help="Recording folder from collect.py (expects <folder>/real).")
    p.add_argument("--out", default=None, help="Output JSON (default: <folder>/calibrated_params.json).")
    p.add_argument("--passes", type=int, default=PASSES, help="Coordinate-descent sweeps. More lowers training RMS but pushes parameters to their bounds.")
    p.add_argument("--max-evals", type=int, default=MAX_EVALS, help="Sim rollouts per joint per sweep.")
    p.add_argument("--no-plots", action="store_true")
    a = p.parse_args()
    fit(a.folder, a.out or os.path.join(a.folder, "calibrated_params.json"),
        passes=a.passes, max_evals=a.max_evals, plots=not a.no_plots)
