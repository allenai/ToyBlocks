"""Shared pieces of the sysid pipeline: excitation, recordings, plots.

Everything both fits and the collector need. See SYSID.md.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import rcs
from rcs import common as rcs_common
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

ROBOT_KEY = "right"
GRIPPER_ACTION = 1.0  # held fixed all run: the gripper is a payload, not an actuator
HOME_POSE = rcs.HOME_POSITIONS["FR3_DROID"]

# --- excitation defaults -----------------------------------------------------
SEED = 0
DURATION = 40.0  # [s]
RAMP_TIME = 3.0  # amplitude fade in/out [s]
AMP_SCALE = 0.6  # global multiplier on JOINT_AMPLITUDE
BASE_FREQ = 0.2  # fundamental f0 [Hz]
N_HARMONICS = 6  # sines per joint: f0, 2*f0, ... N*f0
# Per-joint frequency shift in units of BASE_FREQ, so no two joints share a
# frequency and the regressor stays well conditioned. Max freq ~1.3 Hz.
FREQ_OFFSET = 0.12
JOINT_AMPLITUDE = np.array([0.5, 0.4, 0.5, 0.4, 0.6, 0.6, 0.8])  # peak [rad]
JOINT_LIMITS = np.asarray(rcs.ROBOTS[rcs.common.RobotType.FR3].joint_limits).T  # (7,2)
LIMIT_MARGIN = 0.1  # stay this far [rad] from every limit


class Excitation:
    """A smooth multi-sine free-space excitation centred on one configuration.

    Fully determined by ``seed`` and the settings below, so a hardware run and a
    sim run with the same ones line up sample-for-sample.
    """

    def __init__(self, fps, *, seed=SEED, duration=DURATION, ramp_time=RAMP_TIME,
                 amp_scale=AMP_SCALE, center=None):
        self.fps, self.seed = int(fps), int(seed)
        self.duration, self.ramp_time = float(duration), float(ramp_time)
        self.amp_scale = float(amp_scale)
        self.center = None if center is None else np.asarray(center, dtype=float)

    def center_pose(self) -> np.ndarray:
        return np.asarray(HOME_POSE if self.center is None else self.center, dtype=float)

    def build(self, q_home: np.ndarray) -> np.ndarray:
        """Joint targets ``(T, 7)``.

        Harmonic amplitudes decay as ``1/k``, normalised so their sum is the
        per-joint amplitude (bounded excursion and velocity). A raised-cosine
        envelope fades 0->1 over ``ramp_time`` at both ends, so the arm starts
        and ends at ``q_home`` with near-zero velocity. Clamped to the limits.
        """
        rng = np.random.default_rng(self.seed)
        k = np.arange(1, N_HARMONICS + 1)
        weights = (1.0 / k) / np.sum(1.0 / k)
        amps = self.amp_scale * JOINT_AMPLITUDE[:, None] * weights[None, :]  # (7,K)
        phases = rng.uniform(-np.pi, np.pi, size=(7, N_HARMONICS))
        freqs = BASE_FREQ * (k[None, :] + np.arange(7)[:, None] * FREQ_OFFSET)

        t = np.arange(int(round(self.duration * self.fps))) / self.fps
        env = np.ones_like(t)
        up, down = t < self.ramp_time, t > (self.duration - self.ramp_time)
        env[up] = 0.5 * (1.0 - np.cos(np.pi * t[up] / self.ramp_time))
        env[down] = 0.5 * (1.0 - np.cos(np.pi * (self.duration - t[down]) / self.ramp_time))

        ang = 2 * np.pi * freqs[:, :, None] * t[None, None, :] + phases[:, :, None]
        osc = np.einsum("jk,jkt->tj", amps, np.sin(ang))  # (T,7)
        q_des = np.asarray(q_home, dtype=float)[None, :] + env[:, None] * osc
        lo, hi = JOINT_LIMITS[:, 0] + LIMIT_MARGIN, JOINT_LIMITS[:, 1] - LIMIT_MARGIN
        return np.clip(q_des, lo[None, :], hi[None, :])

    def safety_report(self, q_des: np.ndarray) -> None:
        """Log per-joint excursion and peak velocity of the planned trajectory."""
        dq = np.gradient(q_des, 1.0 / self.fps, axis=0)
        lo, hi = JOINT_LIMITS[:, 0] + LIMIT_MARGIN, JOINT_LIMITS[:, 1] - LIMIT_MARGIN
        logger.info("Planned excitation (%d samples, %.1f s @ %d Hz):",
                    len(q_des), len(q_des) / self.fps, self.fps)
        logger.info("  joint |    min q |    max q | limit lo | limit hi | peak |dq|")
        clipped = False
        for j in range(7):
            qmin, qmax = q_des[:, j].min(), q_des[:, j].max()
            hit = qmin <= lo[j] + 1e-6 or qmax >= hi[j] - 1e-6
            clipped |= hit
            logger.info("   %d    | %7.3f | %7.3f | %7.3f | %7.3f | %7.3f%s", j + 1,
                        qmin, qmax, lo[j], hi[j], np.abs(dq[:, j]).max(),
                        "  <-- clamped" if hit else "")
        if clipped:
            logger.warning("Trajectory hits the clamp on some joints. Lower --amp-scale.")

    def as_dict(self) -> dict:
        """Settings, for the run metadata the fits read back."""
        return {
            "seed": self.seed, "duration_s": self.duration, "ramp_time_s": self.ramp_time,
            "amp_scale": self.amp_scale, "fps": self.fps, "base_freq_hz": BASE_FREQ,
            "n_harmonics": N_HARMONICS, "freq_offset": FREQ_OFFSET,
            "joint_amplitude": JOINT_AMPLITUDE.tolist(), "limit_margin": LIMIT_MARGIN,
            "gripper_action": GRIPPER_ACTION,
        }


# --- reading a recording -----------------------------------------------------

def load_recording(folder: str, split: str = "real", with_vel: bool = False):
    """``(q_des, q)`` from ``<folder>/<split>``, plus ``dq`` if ``with_vel``.

    ``q_des`` is the commanded targets (``env_action``), ``q`` the measured
    positions (``obs``). ``dq`` falls back to a finite difference when the
    recording has no ``robot_state.dq``.
    """
    import pyarrow.dataset as pds

    rows = pds.dataset(os.path.join(folder, split), format="parquet").to_table().to_pylist()
    rows.sort(key=lambda r: r["step"])
    q_des = np.array([r["env_action"][ROBOT_KEY]["joints"] for r in rows], dtype=float)
    q = np.array([r["obs"][ROBOT_KEY]["joints"] for r in rows], dtype=float)
    if not with_vel:
        return q_des, q
    try:
        dq = np.array([r["obs"][ROBOT_KEY]["robot_state"]["dq"] for r in rows], dtype=float)
    except (KeyError, TypeError):
        logger.warning("No robot_state.dq in the recording -- finite-differencing q.")
        dq = np.gradient(q, axis=0)
    return q_des, q, dq


def resolve_controller(folder: str):
    """``(kp, kd, max_rel_mov, fps)`` for the replay, from the recording's metadata.

    Replaying at gains other than the recording's makes the optimiser absorb the
    gain error into the dynamics parameters, so this is not optional -- the
    fallback to env_setup's current profile is a loud warning, not a default.
    """
    from toyblocks.serving.sim2real.rollout import env_setup

    path = os.path.join(folder, "sysid_run_meta.json")
    if not os.path.exists(path):
        logger.warning("No sysid_run_meta.json in %s -- using env_setup's CURRENT gains. "
                       "If the recording used another PD profile, the fit will be wrong.", folder)
        return env_setup.PD_KP, env_setup.PD_KD, None, 25
    with open(path) as f:
        meta = json.load(f)
    prof, ctrl = meta.get("stiffness_profile") or {}, meta.get("controller") or {}
    if prof.get("kp") is None:
        return env_setup.PD_KP, env_setup.PD_KD, ctrl.get("max_rel_mov_joints"), int(ctrl.get("fps", 25))
    kp = np.asarray(prof["kp"], dtype=float)
    kd = np.asarray(prof.get("kd", prof.get("kv")), dtype=float)
    logger.info("Replaying with recorded gains (profile %s): kp=%s kd=%s",
                prof.get("profile"), kp.tolist(), kd.tolist())
    return kp, kd, ctrl.get("max_rel_mov_joints"), int(ctrl.get("fps", 25))



# --- forward kinematics ------------------------------------------------------

def wrap(a: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi] so orientation errors don't show fake 2pi jumps."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def fk(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(T,7) joint angles [rad] -> (T,3) TCP position [m] and (T,3,3) rotations.

    Same model as ``replay_report``: FR3 with the Robotiq 2F-85 TCP offset, so
    the Cartesian numbers here line up with the replay reports.
    """
    robot = rcs.ROBOTS[rcs_common.RobotType.FR3]
    pin = rcs_common.Pin(robot.mjcf_model_path, robot.attachment_site, False)
    tcp = rcs.GRIPPER_TCP_OFFSETS[rcs_common.GripperType("Robotiq2F85")]
    poses = [pin.forward(qi, tcp) for qi in q]
    return (np.array([p.xyzrpy()[:3] for p in poses]),
            np.array([p.rotation_m() for p in poses]))


def cartesian_error(sim_q: np.ndarray, real_q: np.ndarray) -> np.ndarray:
    """(T,6) TCP error: position [mm] and roll/pitch/yaw [deg].

    The angular part is the roll/pitch/yaw of the *relative* rotation
    ``R_real^T @ R_sim``, not the difference of two absolute RPY triples.
    Differencing absolute RPY is unusable here: this excitation sweeps pitch
    through the full +-180 deg range, so the Euler convention flips mid-run and
    a 0.17 deg pose error reads as 37 deg. The relative rotation stays near
    identity, where RPY is well conditioned and each component is the error
    about that axis.
    """
    pos_s, rot_s = fk(sim_q)
    pos_r, rot_r = fk(real_q)
    rel = np.einsum("tji,tjk->tik", rot_r, rot_s)  # rot_real^T @ rot_sim
    rpy = Rotation.from_matrix(rel).as_euler("xyz", degrees=True)
    return np.column_stack([(pos_s - pos_r) * 1e3, rpy])


# --- plots -------------------------------------------------------------------

def plot_fit(out_dir, real_q, sim_before, sim_after, before, after, fps,
             convergence=None, title="", n_overlays=3):
    """Per-joint RMS bars, q(t) overlays for the most-improved joints, convergence.

    ``convergence`` is either ``[(label, value), ...]`` (coordinate descent) or
    ``[{"iter","min","mean","best"}, ...]`` (CMA-ES); both are plotted.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    BASE, FIT = "#1f77b4", "#ff7f0e"

    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(7)
    ax.bar(x - 0.19, before, 0.38, color=BASE, label=f"baseline (mean {before.mean():.4f})")
    ax.bar(x + 0.19, after, 0.38, color=FIT, label=f"fitted (mean {after.mean():.4f})")
    ax.set_xticks(x, [f"j{j + 1}" for j in range(7)])
    ax.set_ylabel("RMS(sim q - real q) [deg]")
    ax.set_title(f"{title}Per-joint tracking error, before and after the fit")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "rms_per_joint.png", dpi=150)
    plt.close(fig)

    _plot_cartesian(out_dir, real_q, sim_before, sim_after, fps, title, BASE, FIT)

    t = np.arange(len(real_q)) / fps
    for j in np.argsort(-(before - after))[:n_overlays]:
        fig, (ax, axe) = plt.subplots(2, 1, figsize=(9, 5), sharex=True,
                                      gridspec_kw={"height_ratios": [2, 1]})
        ax.plot(t, np.rad2deg(real_q[:, j]), color="k", lw=1.2, label="real")
        ax.plot(t, np.rad2deg(sim_before[:, j]), color=BASE, lw=1.0, alpha=0.8,
                label=f"sim baseline (RMS {before[j]:.4f} deg)")
        ax.plot(t, np.rad2deg(sim_after[:, j]), color=FIT, lw=1.0,
                label=f"sim fitted (RMS {after[j]:.4f} deg)")
        ax.set_ylabel(f"j{j + 1} position [deg]")
        ax.set_title(f"{title}Joint {j + 1} against the real recording")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        for sim, c in ((sim_before, BASE), (sim_after, FIT)):
            axe.plot(t, np.rad2deg(sim[:, j] - real_q[:, j]), color=c, lw=0.9)
        axe.axhline(0, color="k", lw=0.6)
        axe.set_ylabel("error [deg]")
        axe.set_xlabel("time [s]")
        axe.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"overlay_j{j + 1}.png", dpi=150)
        plt.close(fig)

    if convergence:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        if isinstance(convergence[0], dict):
            it = [h["iter"] for h in convergence]
            ax.plot(it, [h["mean"] for h in convergence], color=BASE, lw=1, label="population mean")
            ax.plot(it, [h["min"] for h in convergence], color="#999", lw=0.8, label="iteration min")
            ax.plot(it, [h["best"] for h in convergence], color=FIT, lw=1.6, label="best so far")
            ax.set_xlabel("CMA-ES iteration")
            ax.set_ylabel("objective")
            ax.set_yscale("log")
            ax.legend(fontsize=8)
        else:
            ax.plot(range(1, len(convergence) + 1), [v for _, v in convergence],
                    "o-", ms=4, color=FIT)
            ax.set_xticks(range(1, len(convergence) + 1),
                          [lbl for lbl, _ in convergence], rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("that joint's RMS [deg]")
        ax.set_title(f"{title}Optimiser convergence")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "convergence.png", dpi=150)
        plt.close(fig)

    logger.info("Wrote plots to %s", out_dir)


def _plot_cartesian(out_dir, real_q, sim_before, sim_after, fps, title, BASE, FIT):
    """TCP error over time and its RMS, in mm and deg, from FK on the joint traces.

    Joint-space RMS hides where the error actually lands: a small error on j1
    moves the TCP much further than the same error on j7. These plots are the
    same fit seen at the tool.
    """
    import matplotlib.pyplot as plt

    try:
        err_before = cartesian_error(sim_before, real_q)
        err_after = cartesian_error(sim_after, real_q)
    except Exception as exc:  # FK model unavailable: the joint-space plots still stand
        logger.warning("Skipping the Cartesian plots, forward kinematics failed: %s", exc)
        return

    t = np.arange(len(real_q)) / fps
    labels = ["x", "y", "z", "roll", "pitch", "yaw"]
    units = ["mm"] * 3 + ["deg"] * 3

    for name, cols, unit in (("position", range(3), "mm"), ("orientation", range(3, 6), "deg")):
        fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
        for ax, c in zip(axes, cols):
            for err, colour, tag in ((err_before, BASE, "baseline"), (err_after, FIT, "fitted")):
                rms = np.sqrt((err[:, c] ** 2).mean())
                ax.plot(t, err[:, c], color=colour, lw=0.9,
                        label=f"{tag} (RMS {rms:.3f} {unit})")
            ax.axhline(0, color="k", lw=0.6)
            ax.set_ylabel(f"{labels[c]} error [{unit}]")
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(alpha=0.3)
        axes[0].set_title(f"{title}TCP {name} error against the real recording")
        axes[-1].set_xlabel("time [s]")
        fig.tight_layout()
        fig.savefig(out_dir / f"cartesian_{name}_error.png", dpi=150)
        plt.close(fig)

    rms_before = np.sqrt((err_before ** 2).mean(axis=0))
    rms_after = np.sqrt((err_after ** 2).mean(axis=0))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, cols, unit, name in ((axes[0], slice(0, 3), "mm", "position"),
                                 (axes[1], slice(3, 6), "deg", "orientation")):
        x = np.arange(3)
        ax.bar(x - 0.19, rms_before[cols], 0.38, color=BASE, label="baseline")
        ax.bar(x + 0.19, rms_after[cols], 0.38, color=FIT, label="fitted")
        ax.set_xticks(x, labels[cols])
        ax.set_ylabel(f"RMS TCP {name} error [{unit}]")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"{title}TCP error, before and after the fit", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "rms_cartesian.png", dpi=150)
    plt.close(fig)
    logger.info("TCP RMS before: %s | after: %s (%s)",
                np.round(rms_before, 4).tolist(), np.round(rms_after, 4).tolist(),
                ", ".join(f"{lab} [{u}]" for lab, u in zip(labels, units)))
