"""One real replay vs. two sim trajectories (no-sysid / sysid), as a multi-page PDF.

The companion script ``replay_report.py`` compares *two real replays*. This one
answers the other half of the question: given a single run on the real robot,
which simulator reproduces it better — the one tuned by system identification,
or the one without it?

The setup that makes this a fair comparison: both sim episodes were rolled out
from the *same* recorded joint targets (verified here, they are bit-identical),
and that same target stream is what was sent to the real arm. So three bodies
executed one command sequence and the only difference is the plant:

  real          q_meas   the real FR3, from replay_frames.csv
  sim no-sysid  q_ds     the source dataset episode (softer)
  sim sysid     q_sysid  the same episode re-rolled in the sysid-tuned sim

and the quantity of interest is the *sim-to-real gap*, real vs. each sim. The
controller lag (real vs. command) is plotted once as a reference floor: it is a
property of the arm, shared by both comparisons, and no amount of sysid can
shrink it.

End-effector signals are forward kinematics of those joint series (FR3 +
Robotiq 2F-85 TCP, ``rcs.common.Pin``), so every EE curve lives in one frame.
Velocities are Savitzky-Golay derivatives of the logged positions (identical
filter on every series); each sim's own recorded ``joint_velocity`` is overlaid.

Usage:
    python scripts/sim2real/replay_sim_cmp_report.py
    python scripts/sim2real/replay_sim_cmp_report.py LOG_DIR \
        --no-sysid DS_A --sysid DS_B [--episode 0] [-o out.pdf]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

import rcs
from rcs import common as rcs_common

from blocksuite.serving.sim2real.rollout.episode_replay import (
    _load_episode,
    _load_episodes_meta,
    _stack,
)

# scripts/sim2real/<this file> -> parents[2] is the blocksuite repo root.
REPO = Path(__file__).resolve().parents[2]
LOG_ROOT = REPO / "outputs" / "replay_logs"
DATASETS = REPO / "datasets"

DEFAULT_LOG = LOG_ROOT / "20260928_180800"
DEFAULT_NO_SYSID = DATASETS / "sep28_cmp2/softer/blocks_success"
DEFAULT_SYSID = (
    DATASETS / "sep28_cmp2/softer_sysid/lerobot_dataset/sep28_cmp2_softer_ep0_sysid_replay"
)

JOINTS = [f"joint {j + 1}" for j in range(7)]
EE_DIMS = ["x", "y", "z", "roll", "pitch", "yaw"]
EE_UNITS = ["m", "m", "m", "rad", "rad", "rad"]
# Orientation error is the rotation vector of the relative rotation, not an RPY
# difference: RPY differences are meaningless across wraps and near gimbal lock,
# which this trajectory passes through constantly.
EE_ERR_DIMS = ["x", "y", "z", "ori x", "ori y", "ori z"]
EE_ERR_UNITS = ["mm", "mm", "mm", "deg", "deg", "deg"]

C_REAL = "#d62728"   # the real robot
C_NO = "#1f77b4"     # sim without sysid
C_SYS = "#ff7f0e"    # sim with sysid
C_LAG = "0.45"       # controller lag reference
C_TGT = "#7f7f7f"    # the joint target, shared by all three bodies

# episode_replay writes replay_frames.csv with "%.6f", so anything read back
# through that file carries up to half a unit in the sixth decimal. It is the
# resolution limit of every real-vs-sim comparison here, not a real difference.
QUANT = 5e-7

GAP = "sim-to-real gap"
LAG = "controller lag (real vs. command)"
def signals(sims) -> str:
    """Legend line naming each body, using whatever the two sims were labelled."""
    return "   ·   ".join(
        ["real = joints the FR3 reached (q_meas)"]
        + [f"{s.label} = joints that sim reached" for s in sims]
    )

# Savitzky-Golay window for differentiating the 25 Hz position logs. 9 frames
# (~0.36 s) is the smallest window that keeps the measured velocity readable
# without flattening the fast wrist moves.
VEL_WINDOW = 9
VEL_POLY = 2


def wrap(a: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi] so orientation errors don't show fake 2pi jumps."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def fk(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(T,7) joint angles -> (T,6) TCP [x,y,z,roll,pitch,yaw] and (T,3,3) rotations."""
    robot = rcs.ROBOTS[rcs_common.RobotType.FR3]
    pin = rcs_common.Pin(robot.mjcf_model_path, robot.attachment_site, False)
    tcp = rcs.GRIPPER_TCP_OFFSETS[rcs_common.GripperType("Robotiq2F85")]
    poses = [pin.forward(qi, tcp) for qi in q]
    xyzrpy = np.array([pose.xyzrpy() for pose in poses])
    xyzrpy[:, 3:] = wrap(xyzrpy[:, 3:])
    rot = np.array([pose.rotation_m() for pose in poses])
    return xyzrpy, rot


def ori_err(rot_a: np.ndarray, rot_b: np.ndarray) -> np.ndarray:
    """(T,3) rotation vector [rad] taking orientation b to orientation a."""
    rel = np.einsum("tji,tjk->tik", rot_b, rot_a)  # rot_b^T @ rot_a
    return Rotation.from_matrix(rel).as_rotvec()


def deriv(q: np.ndarray, dt: float) -> np.ndarray:
    """Smoothed time derivative of (T,N) joint positions [rad] -> [rad/s]."""
    window = min(VEL_WINDOW, len(q) - (len(q) + 1) % 2)
    if window <= VEL_POLY:
        return np.gradient(q, dt, axis=0)
    return savgol_filter(q, window, VEL_POLY, deriv=1, delta=dt, axis=0, mode="interp")


def rmse(e: np.ndarray, axis=0) -> np.ndarray:
    return np.sqrt(np.mean(e**2, axis=axis))


def load_sim(root: Path, episode: int) -> dict[str, np.ndarray]:
    """Joint position / velocity / target of one episode, valid frames only.

    ``episode_replay`` drops ``action.is_valid == False`` frames before sending
    them, so the same frames are dropped here to stay row-aligned with the CSV.
    """
    ep = _load_episode(root, _load_episodes_meta(root), episode)
    if "action.is_valid" in ep:
        ep = ep[_stack(ep, "action.is_valid").astype(bool).reshape(-1)]
    return {
        "q": _stack(ep, "observation.state.joint_position").astype(np.float64),
        "dq": _stack(ep, "observation.state.joint_velocity").astype(np.float64),
        "target": _stack(ep, "action.joint_target").astype(np.float64),
    }


class Sim:
    """One simulated trajectory held against the one real run."""

    def __init__(self, root: Path, episode: int, label: str, color: str,
                 real: "Real"):
        self.root = root
        self.episode = episode
        self.label = label
        self.color = color
        raw = load_sim(root, episode)
        if len(raw["q"]) != len(real.t):
            raise SystemExit(
                f"{root.name}: {len(raw['q'])} valid frames but the replay log has "
                f"{len(real.t)}; these are not the same episode."
            )
        self.q = raw["q"]
        self.dq_rec = raw["dq"]
        self.target = raw["target"]
        # Bit-identical targets are what make this a controlled comparison; say
        # so on the title page rather than silently assuming it.
        self.target_dev = float(np.abs(self.target - real.q_sent).max())
        self.dq = deriv(self.q, real.dt)
        self.ee, self.rot = fk(self.q)
        # sim-to-real gap, always real minus sim
        self.q_err = real.q_meas - self.q
        self.ee_err = np.hstack(
            [real.ee[:, :3] - self.ee[:, :3], ori_err(real.rot, self.rot)]
        )
        self.pos_err = np.linalg.norm(self.ee_err[:, :3], axis=1) * 1e3
        self.ori_err = np.rad2deg(np.linalg.norm(self.ee_err[:, 3:], axis=1))


class Real:
    """The one real replay log: measured joints, the command, and their FK."""

    def __init__(self, path: Path):
        self.path = path
        self.meta = json.loads((path / "replay_meta.json").read_text())
        df = pd.read_csv(path / "replay_frames.csv")
        self.t = df["t_ds"].to_numpy()
        self.gripper = df["gripper_sent"].to_numpy()
        self.q_meas = df[[f"q_meas{j}" for j in range(7)]].to_numpy()
        self.q_sent = df[[f"q_sent{j}" for j in range(7)]].to_numpy()
        self.dt = float(np.median(np.diff(self.t)))
        self.dq = deriv(self.q_meas, self.dt)
        self.ee, self.rot = fk(self.q_meas)
        # The joint target is the shared input to all three bodies, so it is
        # plotted alongside every trajectory as the common reference.
        self.ee_sent, rot_sent = fk(self.q_sent)
        # Controller lag: how far the arm ran behind the target it was given.
        # Shared by both sim comparisons, so it is the floor neither can beat.
        self.q_lag = self.q_sent - self.q_meas
        self.ee_lag = np.hstack(
            [self.ee_sent[:, :3] - self.ee[:, :3], ori_err(rot_sent, self.rot)]
        )
        self.pos_lag = np.linalg.norm(self.ee_lag[:, :3], axis=1) * 1e3
        self.ori_lag = np.rad2deg(np.linalg.norm(self.ee_lag[:, 3:], axis=1))


def _finish(fig, title: str, pdf: PdfPages) -> None:
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    pdf.savefig(fig)
    plt.close(fig)


def _grid(n_rows: int, sims: list[Sim], height: float = 1.85):
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(11.69, min(8.27, 1.0 + height * n_rows)),
        sharex="col", sharey="row", squeeze=False,
    )
    for c, sim in enumerate(sims):
        axes[0][c].set_title(f"real vs. {sim.label}", fontsize=11, fontweight="bold")
    for c in range(2):
        axes[-1][c].set_xlabel("dataset time [s]")
    return fig, axes


def _shade_gripper(ax, real: Real) -> None:
    """Light shading while the gripper is commanded closed."""
    closed = real.gripper < 0.5
    edges = np.flatnonzero(np.diff(closed.astype(int)))
    bounds = np.concatenate(
        ([0] if closed[0] else [], edges + 1, [len(closed) - 1] if closed[-1] else [])
    ).astype(int)
    for a, b in zip(bounds[::2], bounds[1::2]):
        ax.axvspan(real.t[a], real.t[min(b, len(real.t) - 1)],
                   color="0.85", lw=0, zorder=0)


def page_signals(pdf: PdfPages, real: Real, sims: list[Sim], kind: str) -> None:
    """Real overlaid on each sim, one row per joint or EE dimension."""
    if kind == "joint":
        names, units = JOINTS, ["rad"] * 7
        real_sig, sim_sigs = real.q_meas, [s.q for s in sims]
        tgt = real.q_sent
        title = "Joint trajectories — target vs. real robot vs. sim"
    else:
        names, units = EE_DIMS, EE_UNITS
        real_sig, sim_sigs = real.ee, [s.ee for s in sims]
        tgt = real.ee_sent
        title = "End-effector (FK, 2F-85 TCP) — target vs. real robot vs. sim"

    for lo in range(0, len(names), 4):
        hi = min(lo + 4, len(names))
        fig, axes = _grid(hi - lo, sims)
        for c, (sim, sig) in enumerate(zip(sims, sim_sigs)):
            for r, d in enumerate(range(lo, hi)):
                ax = axes[r][c]
                _shade_gripper(ax, real)
                # The target is the same series in both columns: it is the one
                # input all three bodies were given.
                ax.plot(real.t, tgt[:, d], color=C_TGT, lw=2.0, alpha=0.55,
                        solid_capstyle="round", zorder=1, label="joint target (shared)")
                ax.plot(real.t, sig[:, d], color=sim.color, lw=1.0, zorder=2,
                        label=sim.label)
                ax.plot(real.t, real_sig[:, d], color=C_REAL, lw=1.0, ls="--",
                        alpha=0.85, zorder=3, label="real")
                ax.set_ylabel(f"{names[d]}\n[{units[d]}]", fontsize=9)
                ax.grid(alpha=0.25, lw=0.4)
                ax.tick_params(labelsize=8)
                ax.legend(fontsize=6.5, ncol=3, loc="upper right", framealpha=0.9)
        _finish(fig, f"{title} ({names[lo]}–{names[hi - 1]})", pdf)


def page_velocity(pdf: PdfPages, real: Real, sims: list[Sim]) -> None:
    """Joint velocity in each sim vs. on the real robot."""
    title = "Joint velocity — real vs. sim (derivative of the logged positions)"
    for lo in range(0, 7, 4):
        hi = min(lo + 4, 7)
        fig, axes = _grid(hi - lo, sims)
        for c, sim in enumerate(sims):
            for r, j in enumerate(range(lo, hi)):
                ax = axes[r][c]
                _shade_gripper(ax, real)
                ax.axhline(0, color="0.4", lw=0.6)
                ax.plot(real.t, sim.dq_rec[:, j], color="#c6dbef", lw=1.4,
                        label="sim (recorded)")
                ax.plot(real.t, sim.dq[:, j], color=sim.color, lw=0.9,
                        label=f"{sim.label} (diff)  RMS {rmse(sim.dq[:, j]):.3f}")
                ax.plot(real.t, real.dq[:, j], color=C_REAL, lw=0.9, alpha=0.85,
                        label=f"real (diff)  RMS {rmse(real.dq[:, j]):.3f}")
                ax.set_ylabel(f"{JOINTS[j]}\n[rad/s]", fontsize=9)
                ax.grid(alpha=0.25, lw=0.4)
                ax.tick_params(labelsize=8)
                ax.legend(fontsize=6.5, ncol=3, loc="upper right", framealpha=0.9)
        _finish(fig, f"{title} ({JOINTS[lo]}–{JOINTS[hi - 1]})", pdf)


def page_errors(pdf: PdfPages, real: Real, sims: list[Sim], kind: str) -> None:
    """Per-dimension sim-to-real gap over time, with the controller lag behind it."""
    if kind == "joint":
        names, units = JOINTS, ["deg"] * 7
        gaps = [np.rad2deg(s.q_err) for s in sims]
        lag = np.rad2deg(real.q_lag)
        title = f"Joint {GAP} over time"
    else:
        names, units = EE_ERR_DIMS, EE_ERR_UNITS

        def scaled(e: np.ndarray) -> np.ndarray:
            out = e.copy()
            out[:, :3] *= 1e3
            out[:, 3:] = np.rad2deg(out[:, 3:])
            return out

        gaps = [scaled(s.ee_err) for s in sims]
        lag = scaled(real.ee_lag)
        title = f"End-effector {GAP} over time"

    for lo in range(0, len(names), 4):
        hi = min(lo + 4, len(names))
        fig, axes = _grid(hi - lo, sims)
        for c, (sim, gap) in enumerate(zip(sims, gaps)):
            for r, d in enumerate(range(lo, hi)):
                ax = axes[r][c]
                _shade_gripper(ax, real)
                ax.axhline(0, color="0.4", lw=0.6)
                ax.plot(real.t, lag[:, d], color=C_LAG, lw=0.7, alpha=0.7,
                        label=f"{LAG}  RMSE {rmse(lag[:, d]):.2f}")
                ax.plot(real.t, gap[:, d], color=sim.color, lw=0.9,
                        label=f"{GAP} vs. {sim.label}  RMSE {rmse(gap[:, d]):.2f}")
                ax.set_ylabel(f"{names[d]}\n[{units[d]}]", fontsize=9)
                ax.grid(alpha=0.25, lw=0.4)
                ax.tick_params(labelsize=8)
                ax.legend(fontsize=6.5, loc="upper right", framealpha=0.9)
        _finish(fig, f"{title} ({names[lo]}–{names[hi - 1]})", pdf)


def page_norms(pdf: PdfPages, real: Real, sims: list[Sim]) -> None:
    """Scalar TCP position / orientation gap, both sims on one pair of axes."""
    fig, axes = plt.subplots(2, 1, figsize=(11.69, 7.0), sharex=True)

    ax = axes[0]
    _shade_gripper(ax, real)
    for sim in sims:
        ax.plot(real.t, sim.pos_err, color=sim.color, lw=0.9,
                label=f"{sim.label}   RMSE {rmse(sim.pos_err):.1f} mm   "
                      f"max {sim.pos_err.max():.1f}")
    ax.plot(real.t, real.pos_lag, color=C_LAG, lw=0.7, alpha=0.7,
            label=f"{LAG}   RMSE {rmse(real.pos_lag):.1f} mm   "
                  f"max {real.pos_lag.max():.1f}")
    ax.set_ylabel("‖TCP position gap‖ [mm]", fontsize=9)

    ax = axes[1]
    _shade_gripper(ax, real)
    for sim in sims:
        ax.plot(real.t, sim.ori_err, color=sim.color, lw=0.9,
                label=f"{sim.label}   RMSE {rmse(sim.ori_err):.2f}°   "
                      f"max {sim.ori_err.max():.2f}")
    ax.plot(real.t, real.ori_lag, color=C_LAG, lw=0.7, alpha=0.7,
            label=f"{LAG}   RMSE {rmse(real.ori_lag):.2f}°   "
                  f"max {real.ori_lag.max():.2f}")
    ax.set_ylabel("TCP orientation gap [deg]", fontsize=9)
    ax.set_xlabel("dataset time [s]")

    for ax in axes:
        ax.grid(alpha=0.25, lw=0.4)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    _finish(fig, "TCP sim-to-real gap over the whole trajectory", pdf)


def page_targets(pdf: PdfPages, real: Real, sims: list[Sim]) -> None:
    """Premise check: the three bodies really were given the same joint target.

    The whole report rests on this. If these curves are not flat at zero, the
    comparison is between different commands and every later page is void.
    """
    fig, axes = plt.subplots(3, 1, figsize=(11.69, 7.6),
                             gridspec_kw={"height_ratios": [2.2, 1.4, 1.4]})

    ax = axes[0]
    _shade_gripper(ax, real)
    for j in range(7):
        ax.plot(real.t, real.q_sent[:, j], lw=0.8, label=JOINTS[j])
    ax.set_ylabel("joint target\n[rad]", fontsize=9)
    ax.set_title("The commanded joint target — one stream, sent to all three bodies",
                 fontsize=10)
    ax.legend(fontsize=6.5, ncol=7, loc="upper right", framealpha=0.9)

    # Each sim's own action.joint_target against the target sent to the real arm.
    # Per-joint traces here are a solid band of quantization noise and show
    # nothing, so this is the worst joint at each instant -- one curve per sim,
    # and the only thing that matters is whether it ever leaves the noise floor.
    ax = axes[1]
    _shade_gripper(ax, real)
    for sim in sims:
        dev = np.abs(sim.target - real.q_sent).max(axis=1) * 1e6
        ax.plot(real.t, dev, color=sim.color, lw=0.7, alpha=0.8,
                label=f"{sim.label}   max {dev.max():.3f} urad")
    ax.axhline(QUANT * 1e6, color="0.3", ls="--", lw=0.9,
               label=f"replay-log rounding ({QUANT * 1e6:.1f} urad, CSV writes %.6f)")
    ax.set_ylabel("max over joints\n|sim target - sent| [urad]", fontsize=8.5)
    ax.set_title("Deviation of each sim's action.joint_target from the target sent "
                 "to the real arm", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)

    # The two sims against each other: no CSV in the path, so no rounding floor.
    # This is the one comparison that can be, and is, exactly zero.
    ax = axes[2]
    _shade_gripper(ax, real)
    between = np.abs(sims[0].target - sims[1].target).max(axis=1)
    ax.plot(real.t, between, color="0.2", lw=0.8)
    ax.set_ylabel(f"max over joints\n|{sims[1].label} - {sims[0].label}| [rad]",
                  fontsize=8.5)
    ax.set_title(
        "The two sim targets against each other (both read straight from parquet, "
        f"no CSV rounding in between) — max {between.max():.1e} rad, "
        f"{'bit-identical' if between.max() == 0 else 'NOT identical'}",
        fontsize=10,
    )
    ax.set_ylim(-1e-9, 1e-8)

    axes[-1].set_xlabel("dataset time [s]")
    for ax in axes:
        ax.grid(alpha=0.25, lw=0.4)
        ax.tick_params(labelsize=8)
    _finish(fig, "Command-stream check: all three bodies got the same joint target", pdf)


def page_bars(pdf: PdfPages, real: Real, sims: list[Sim]) -> None:
    """Per-joint / per-EE-dimension gap RMSE, the two sims next to each other."""
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 6.6))
    width = 0.38
    specs = [
        (axes[0][0], f"Joint {GAP}\nhow far the real robot ended up from each sim",
         JOINTS, "deg", [np.rad2deg(rmse(s.q_err)) for s in sims],
         np.rad2deg(rmse(real.q_lag))),
        (axes[0][1], "Joint velocity gap\nreal vs. each sim, same SG derivative",
         JOINTS, "rad/s", [rmse(real.dq - s.dq) for s in sims], None),
        (axes[1][0], f"End-effector position {GAP}\nreal TCP vs. each sim TCP",
         EE_DIMS[:3], "mm", [rmse(s.ee_err[:, :3]) * 1e3 for s in sims],
         rmse(real.ee_lag[:, :3]) * 1e3),
        (axes[1][1], f"End-effector orientation {GAP}\nreal TCP vs. each sim TCP "
         "(rotation vector)",
         EE_ERR_DIMS[3:], "deg", [np.rad2deg(rmse(s.ee_err[:, 3:])) for s in sims],
         np.rad2deg(rmse(real.ee_lag[:, 3:]))),
    ]
    for ax, title, names, unit, vals, lag in specs:
        x = np.arange(len(names))
        for k, (sim, v) in enumerate(zip(sims, vals)):
            bars = ax.bar(x + (k - 0.5) * width, v, width,
                          label=sim.label, color=sim.color, alpha=0.9)
            ax.bar_label(bars, fmt="%.2f", fontsize=6.5, padding=1)
        if lag is not None:
            ax.plot(x, lag, ls="none", marker="_", ms=18, mew=1.6, color=C_LAG,
                    label=LAG)
        ax.set_xticks(x, names, fontsize=8)
        ax.set_ylabel(f"RMSE [{unit}]", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.margins(y=0.18)
        ax.grid(axis="y", alpha=0.25, lw=0.4)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7.5)
    fig.text(0.5, 0.935, signals(sims), ha="center", fontsize=7.5, color="0.35")
    fig.suptitle("Sim-to-real gap summary (RMSE): one real run vs. two sims",
                 fontsize=13, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.915))
    pdf.savefig(fig)
    plt.close(fig)


def page_title(pdf: PdfPages, real: Real, sims: list[Sim]) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.5, 0.945, "Sim-to-real gap: one real replay vs. two simulators",
             ha="center", fontsize=17, fontweight="bold")
    fig.text(0.5, 0.915,
             "The same recorded joint targets executed by the real FR3 and by two "
             "sims — without and with system identification",
             ha="center", fontsize=10, color="0.3")

    dev = max(s.target_dev for s in sims)
    lines = [
        "Three bodies, one command stream:",
        "    real           q_meas    the joint positions the real FR3 reached",
        f"    {sims[0].label:<14s} q_a       the joint positions that sim reached",
        f"    {sims[1].label:<14s} q_b       the joint positions that sim reached",
        "",
        "The two sim datasets' action.joint_target arrays are bit-identical to each other,",
        f"and agree with the target logged for the real arm to {dev:.0e} rad -- exactly the",
        'rounding of replay_frames.csv, which episode_replay.py writes with "%.6f". So the',
        "plant is the only thing that differs and the comparison is controlled. Frames with",
        "action.is_valid == False are dropped from both datasets, matching what was sent.",
        "",
        "One quantity is being compared:",
        "",
        "    sim-to-real gap    real vs. sim, per sim",
        "        Did the real robot end up where that sim did? Whichever sim keeps this",
        "        smaller is the better model of the arm. This is what sysid is meant to",
        "        shrink, and the whole report is a test of whether it did.",
        "",
        "    controller lag     real vs. the command it was given (grey)",
        "        Plotted as a reference floor. It is a property of the controller and the",
        "        arm, identical under both comparisons, so sysid cannot move it. A sim",
        "        that tracks its own targets perfectly inherits this lag as a gap.",
        "",
        "End-effector values are forward kinematics of those joint series (FR3 + Robotiq 2F-85",
        "TCP), so all curves live in one consistent frame. Grey bands mark frames where the",
        "gripper is commanded closed. Roll/pitch/yaw rows are wrapped to (-pi, pi] and so they",
        "jump; orientation ERROR is never an RPY difference but the rotation vector of the",
        "relative rotation (ori x/y/z), well behaved through wraps and gimbal lock.",
        "",
        f"Velocity is not logged on the real arm, so it is a Savitzky-Golay derivative (window",
        f"{VEL_WINDOW}, order {VEL_POLY}) of the logged positions, applied identically to every series; each",
        "sim's recorded joint_velocity is overlaid in pale blue. The replay loop is not",
        "rate-paced, so real velocity is per dataset frame, not per wall-clock second.",
        "",
        "real replay",
        f"    log        {real.path}",
        f"    dataset    {real.meta['dataset_root']}",
        f"    episode    {real.meta['episode']}   frames {len(real.t)}   "
        f"{real.t[-1]:.1f} s @ {real.meta['fps']} fps   TAM {real.meta['tam']}",
        f"    lag        {np.rad2deg(rmse(real.q_lag.ravel())):.3f} deg RMSE joints   "
        f"{rmse(real.pos_lag):.1f} mm RMSE TCP",
        "",
    ]
    for sim in sims:
        lines += [
            f"{sim.label}",
            f"    dataset    {sim.root}",
            f"    episode    {sim.episode}   frames {len(sim.q)}",
            f"    joints     gap {np.rad2deg(rmse(sim.q_err.ravel())):.3f} deg RMSE",
            f"    TCP pos    gap {rmse(sim.pos_err):.1f} mm RMSE "
            f"(max {sim.pos_err.max():.1f})",
            f"    TCP ori    gap {rmse(sim.ori_err):.2f} deg RMSE "
            f"(max {sim.ori_err.max():.2f})",
            "",
        ]

    a, b = sims
    better = b if rmse(b.pos_err) < rmse(a.pos_err) else a
    lines += [
        f"Smaller TCP position gap: {better.label}  "
        f"({rmse(a.pos_err):.1f} mm {a.label} vs. {rmse(b.pos_err):.1f} mm {b.label}).",
    ]
    fig.text(0.055, 0.885, "\n".join(lines), va="top", ha="left",
             fontsize=7.4, family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", nargs="?", type=Path, default=DEFAULT_LOG,
                    help="outputs/replay_logs/<stamp> dir holding the real run")
    ap.add_argument("--no-sysid", type=Path, default=DEFAULT_NO_SYSID,
                    help="LeRobot dataset root of the sim rollout without sysid")
    ap.add_argument("--sysid", type=Path, default=DEFAULT_SYSID,
                    help="LeRobot dataset root of the sim rollout with sysid")
    ap.add_argument("--label-a", default="sim WITHOUT sysid",
                    help="display name for the --no-sysid rollout")
    ap.add_argument("--label-b", default="sim WITH sysid",
                    help="display name for the --sysid rollout")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("-o", "--out", type=Path,
                    default=LOG_ROOT / "real_vs_sim_sysid_report.pdf")
    args = ap.parse_args()

    real = Real(args.log)
    sims = [
        Sim(args.no_sysid, args.episode, args.label_a, C_NO, real),
        Sim(args.sysid, args.episode, args.label_b, C_SYS, real),
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.out) as pdf:
        page_title(pdf, real, sims)
        page_targets(pdf, real, sims)
        page_bars(pdf, real, sims)
        page_norms(pdf, real, sims)
        page_signals(pdf, real, sims, "joint")
        page_errors(pdf, real, sims, "joint")
        page_velocity(pdf, real, sims)
        page_signals(pdf, real, sims, "ee")
        page_errors(pdf, real, sims, "ee")
        pdf.infodict()["Title"] = "Sim-to-real gap: one real replay vs. two sims"
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
