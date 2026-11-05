"""Step 2b of sysid: fit Isaac/PhysX joint dynamics to a recording. See SYSID.md.

Same input as ``fit_mujoco.py``, same controller (toyblocks's **joint-space** PD
at the recorded gains), different engine and parameters: joint ``armature`` and the three
PhysX joint-friction terms (static, dynamic, viscous).

Method: batched single-transition prediction error. Seed env *i* with the real
state at *i*, command the recorded target, step one control step, compare the
predicted ``(q, dq)`` to the real next state -- so ``num_envs`` transitions
evaluate in one batched step, and re-seeding every step kills the drift that
makes a free rollout ill-conditioned. PhysX has no gradients, so CMA-ES over the
28 bounded parameters.

    python -m toyblocks.serving.sim2real.sysid.fit_isaac outputs/sysid_data_kp300 --headless

frankatwin's Isaac sysid was the reference for the optimizer and parameter
handling, but it fits under a Cartesian impedance controller; that control law is
deliberately not reused. Needs ``uv sync --extra isaac`` and ``cmaes``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

import numpy as np

from toyblocks.serving.sim2real.sysid import common

logger = logging.getLogger(__name__)

# Same parameterisation as frankatwin's sysid, so the two results are directly
# comparable: 7 armature (reflected rotor inertia) + PhysX 5's three-term joint
# friction model -- static (mu_s), dynamic (mu_d) and viscous (c_v) -- giving
# 4 * 7 = 28 free parameters. Mass/COM/inertia are not fit: a single-centre
# excitation identifies them only weakly.
#
# mu_d is encoded as a ratio in [0, 1] of mu_s rather than fitted directly, which
# enforces mu_d <= mu_s. Fitting the two independently would let the optimiser
# pick a dynamic friction above the static one, which is not physical.
#
# Motor delay is the 29th parameter in frankatwin and is held at 0 here (see
# DELAY_STEPS below); it is still written to the output so the two files line up.
NJ = 7
ARMATURE_MAX, FRICTION_MAX, VISCOUS_MAX = 0.5, 5.0, 5.0
BOUNDS = np.array([[0.0, ARMATURE_MAX]] * NJ      # armature
                  + [[0.0, FRICTION_MAX]] * NJ    # mu_static
                  + [[0.0, 1.0]] * NJ             # dynamic_ratio -> mu_dynamic = ratio * mu_static
                  + [[0.0, VISCOUS_MAX]] * NJ)    # mu_viscous
NP = len(BOUNDS)  # 28

# Held fixed. toyblocks's actuators have no delay buffer to set it on, so this
# is not a free parameter here; it is recorded for parity with frankatwin.
DELAY_STEPS = 0


def decode(params: np.ndarray) -> dict:
    """Flat (28,) vector -> the named parameter groups, matching frankatwin's JSON."""
    armature, mu_static, ratio, mu_viscous = (params[i * NJ:(i + 1) * NJ] for i in range(4))
    return {
        "armature": armature.tolist(),
        "mu_static": mu_static.tolist(),
        "dynamic_ratio": ratio.tolist(),
        "mu_dynamic": (ratio * mu_static).tolist(),
        "mu_viscous": mu_viscous.tolist(),
        "motor_delay_steps": DELAY_STEPS,
    }


def fit(args) -> dict:
    from toyblocks.cli.isaac_app import launch_app  # must precede any isaaclab import

    app = launch_app()

    import gymnasium as gym
    import torch
    from cmaes import CMA

    import toyblocks.tasks  # noqa: F401  registers the gym ids

    q_des, real_q, real_dq = common.load_recording(args.folder, with_vel=True)
    kp, kd, _, fps = common.resolve_controller(args.folder)
    dt, T = 1.0 / fps, len(q_des) - 1  # T transitions
    logger.info("Loaded %d transitions at %d Hz from %s", T, fps, args.folder)

    from isaaclab_tasks.utils import parse_env_cfg

    # The gym id takes a config object, not kwargs; --headless is consumed by launch_app.
    env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)
    # Isaac Lab swaps the root logging handlers during env creation and pins the
    # console one to WARNING, which would silence every progress line below.
    for h in logging.getLogger().handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            h.setLevel(logging.INFO)
    uw = env.unwrapped
    robot = uw.scene["robot"]
    dev = uw.device

    names = list(robot.data.joint_names)
    # Match the 7 numbered arm joints only: a plain "joint" substring also catches
    # the gripper's finger/knuckle joints, which sort ahead of "panda_joint*".
    numbered = {int(m.group(1)): m.group(0) for n in names if (m := re.fullmatch(r".*joint([1-7])", n))}
    if len(numbered) != NJ:
        raise RuntimeError(f"expected {NJ} numbered arm joints, found {sorted(numbered.values())} in {names}")
    arm = [names.index(numbered[i]) for i in range(1, NJ + 1)]
    logger.info("Arm joints (Isaac order): %s", [names[i] for i in arm])

    # Joint-space PD at the recorded gains, held FIXED for the whole fit: a gain
    # mismatch would otherwise be explained away as dynamics.
    stiff, damp = robot.data.joint_stiffness.clone(), robot.data.joint_damping.clone()
    stiff[:, arm] = torch.tensor(kp, device=dev, dtype=torch.float32)
    damp[:, arm] = torch.tensor(kd, device=dev, dtype=torch.float32)
    uw.set_actuator_gains(stiffness=stiff, damping=damp, settle_steps=8)
    logger.info("Joint PD gains set: kp=%s kd=%s", kp.tolist(), kd.tolist())

    T32 = lambda a: torch.tensor(a, device=dev, dtype=torch.float32)  # noqa: E731
    q_des_t, real_q_t, real_dq_t = T32(q_des), T32(real_q), T32(real_dq)

    def apply(params, sl: slice) -> None:
        """Write one candidate's armature and three friction terms into a slice of envs."""
        n = sl.stop - sl.start
        ids = torch.arange(sl.start, sl.stop, device=dev)
        armature, mu_s, ratio, mu_v = (params[i * NJ:(i + 1) * NJ] for i in range(4))
        a = robot.data.joint_armature[sl].clone()
        fs = robot.data.joint_friction_coeff[sl].clone()
        fd = robot.data.joint_dynamic_friction_coeff[sl].clone()
        fv = robot.data.joint_viscous_friction_coeff[sl].clone()
        a[:, arm] = T32(armature).repeat(n, 1)
        fs[:, arm] = T32(mu_s).repeat(n, 1)
        fd[:, arm] = T32(ratio * mu_s).repeat(n, 1)
        fv[:, arm] = T32(mu_v).repeat(n, 1)
        robot.write_joint_armature_to_sim(a, env_ids=ids)
        robot.write_joint_friction_coefficient_to_sim(
            fs, joint_dynamic_friction_coeff=fd, joint_viscous_friction_coeff=fv, env_ids=ids)

    def predict(idx):
        """Seed envs with the real state at ``idx``, command q_des, step once."""
        # Reset first. Seeding only the arm leaves the gripper joints, the scene
        # objects and the solver state from the previous evaluation in place, and
        # that carry-over compounds: re-running the *same* transitions with the
        # *same* parameters drove the single-step error from 0.11 deg to >1000 deg
        # within eight calls, which is what made the whole fit diverge.
        env.reset()
        pos, vel = robot.data.joint_pos.clone(), robot.data.joint_vel.clone()
        pos[:, arm], vel[:, arm] = real_q_t[idx], real_dq_t[idx]
        robot.write_joint_state_to_sim(pos, vel)
        target = robot.data.joint_pos_target.clone()
        target[:, arm] = q_des_t[idx]
        robot.set_joint_position_target(target)
        uw.scene.write_data_to_sim()
        for _ in range(uw.cfg.decimation):
            uw.sim.step(render=False)
        robot.update(uw.physics_dt)
        return robot.data.joint_pos[:, arm], robot.data.joint_vel[:, arm]

    def chunks(width):
        """Transition indices in batches of ``width``, tail padded to stay rectangular."""
        for start in range(0, T, width):
            n = min(width, T - start)
            idx = torch.arange(start, start + n, device=dev)
            if n < width:
                idx = torch.cat([idx, torch.full((width - n,), start, device=dev)])
            yield idx, n

    def score(population) -> np.ndarray:
        """Velocity-weighted single-step error per candidate; lower is better.

        Residuals are scaled by 1/dt so the position and velocity terms are
        commensurate. Every candidate sees the same transitions in the same order.
        """
        pop = len(population)
        per_env = args.num_envs // pop
        if per_env == 0:
            raise SystemExit(f"--num-envs ({args.num_envs}) must be >= --population ({pop})")
        for c in range(pop):
            apply(population[c], slice(c * per_env, (c + 1) * per_env))

        total, count = torch.zeros(pop, device=dev), 0
        for idx_chunk, n in chunks(per_env):
            idx = torch.cat([idx_chunk] * pop)
            q_hat, dq_hat = predict(idx)
            r_q = args.w_q * (q_hat - real_q_t[idx + 1]) / dt
            r_dq = args.w_dq * (dq_hat - real_dq_t[idx + 1]) / dt
            total += (r_q**2 + r_dq**2).sum(dim=1).view(pop, per_env)[:, :n].sum(dim=1)
            count += n
        return (total / max(count, 1)).cpu().numpy()

    def rms_deg(params):
        """Per-joint single-step position RMS [deg], and the predicted q (T, 7)."""
        apply(params, slice(0, args.num_envs))
        pred = np.empty((T, NJ))
        for idx, n in chunks(args.num_envs):
            q_hat, _ = predict(idx)
            pred[idx[0].item() : idx[0].item() + n] = q_hat[:n].cpu().numpy()
        return np.rad2deg(np.sqrt(((pred - real_q[1:]) ** 2).mean(0))), pred

    # The articulation's own values, re-encoded into the fit's parameterisation.
    # mu_s = 0 makes the ratio undefined; 0 is the right choice there because
    # mu_d = ratio * mu_s is 0 either way.
    mu_s0 = robot.data.joint_friction_coeff[0, arm].cpu().numpy()
    mu_d0 = robot.data.joint_dynamic_friction_coeff[0, arm].cpu().numpy()
    baseline = np.concatenate([
        robot.data.joint_armature[0, arm].cpu().numpy(),
        mu_s0,
        np.divide(mu_d0, mu_s0, out=np.zeros_like(mu_d0), where=mu_s0 > 0),
        robot.data.joint_viscous_friction_coeff[0, arm].cpu().numpy(),
    ])
    before, pred_before = rms_deg(baseline)
    logger.info("baseline per-joint single-step RMS [deg]: %s (mean %.4f)",
                np.round(before, 4).tolist(), before.mean())

    cma = CMA(mean=BOUNDS.mean(axis=1), sigma=args.sigma, bounds=BOUNDS,
              population_size=args.population)
    best, best_params, history, t0 = np.inf, BOUNDS.mean(axis=1), [], time.time()
    for it in range(1, args.max_iter + 1):
        asked = [cma.ask() for _ in range(cma.population_size)]
        scores = score(np.stack(asked))
        cma.tell([(x, float(s)) for x, s in zip(asked, scores)])
        i = int(np.argmin(scores))
        if scores[i] < best:
            best, best_params = float(scores[i]), asked[i].copy()
        el = time.time() - t0
        history.append({"iter": it, "min": float(scores.min()),
                        "mean": float(scores.mean()), "best": best})
        logger.info("iter %3d/%d  min %.6f  mean %.6f  best %.6f  (%.0f s, ETA %.0f s)",
                    it, args.max_iter, scores.min(), scores.mean(), best,
                    el, el / it * (args.max_iter - it))

    after, pred_after = rms_deg(best_params)
    result = {
        "data_folder": os.path.abspath(args.folder),
        "engine": "isaac/physx",
        "note": "Batched single-transition prediction-error fit under joint-space PD. Training error only -- no held-out split.",
        "task": args.task, "fps": fps, "joints": [names[i] for i in arm],
        "gains": {"kp": np.asarray(kp).tolist(), "kd": np.asarray(kd).tolist(), "fixed": True},
        "initial": decode(baseline),
        "calibrated": decode(best_params),
        # Flat vector in frankatwin's order, delay last, so the two files can be
        # diffed directly.
        "best_params_raw": best_params.tolist() + [float(DELAY_STEPS)],
        "best_score": best,
        "rms_deg_before": before.tolist(), "rms_deg_after": after.tolist(),
        "rms_deg_before_mean": float(before.mean()), "rms_deg_after_mean": float(after.mean()),
        "bounds": BOUNDS.tolist(), "history": history, "args": vars(args),
    }
    out_path = args.out or os.path.join(args.folder, "isaac_calibrated_params.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    cal = decode(best_params)
    logger.info("=== isaac calibration result ===")
    logger.info("joint | armature |    mu_s |    mu_d |    mu_v | RMS[deg] before->after")
    for j in range(NJ):
        logger.info("  %s | %8.4f | %7.4f | %7.4f | %7.4f | %.4f -> %.4f", names[arm[j]],
                    cal["armature"][j], cal["mu_static"][j], cal["mu_dynamic"][j],
                    cal["mu_viscous"][j], before[j], after[j])
    logger.info("mean RMS %.4f -> %.4f deg. Wrote %s", before.mean(), after.mean(), out_path)

    groups = ("armature", "mu_static", "dynamic_ratio", "mu_viscous")
    at_bound = [f"{names[arm[j]]}.{groups[g]}" for g in range(4) for j in range(NJ)
                if best_params[g * NJ + j] > 0.95 * BOUNDS[g * NJ + j, 1]]
    if at_bound:
        logger.warning("Parameters at their upper bound on %s -- the fit may be compensating "
                       "for something other than dynamics. Check the gains, or widen the "
                       "bounds.", at_bound)

    if not args.no_plots:
        common.plot_fit(Path(args.folder) / "sysid_fit_isaac", real_q[1:], pred_before,
                        pred_after, before, after, fps, convergence=history, title="Isaac: ")
    env.close()
    app.close()
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", help="Recording folder from collect.py (expects <folder>/real).")
    p.add_argument("--out", default=None, help="Output JSON (default: <folder>/isaac_calibrated_params.json).")
    p.add_argument("--task", default="ToyBlocks-Lift-v0", help="ToyBlocks Isaac env to borrow the FR3 articulation from; the scene objects are never touched.")
    p.add_argument("--num-envs", type=int, default=512, help="Parallel envs; must be >= --population.")
    p.add_argument("--population", type=int, default=16, help="CMA-ES population size.")
    p.add_argument("--max-iter", type=int, default=40, help="CMA-ES iterations.")
    p.add_argument("--sigma", type=float, default=0.3, help="CMA-ES initial step size.")
    p.add_argument("--w-q", type=float, default=0.1, help="Position residual weight.")
    p.add_argument("--w-dq", type=float, default=1.0, help="Velocity residual weight; velocity-weighted by default because a position-only loss hides over-damping.")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    fit(p.parse_args())
