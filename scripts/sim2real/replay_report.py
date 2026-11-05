"""Sim2real replay report: no-sysid vs sysid, as a multi-page PDF.

Reads two ``outputs/replay_logs/<stamp>/`` directories written by ``episode_replay.py``
(each holding ``replay_frames.csv`` + ``replay_meta.json``) and produces a PDF
comparing, side by side, how faithfully the real robot reproduced a sim
trajectory generated WITHOUT system identification versus WITH it.

Three signals per frame, all in the CSV, named in the plots as:
  sim motion  q_ds    the joint positions simulation actually reached
  command     q_sent  the joint target that was sent to the real robot
  real        q_meas  the joint positions the real robot actually reached

and two comparisons, both against ``real``, named rather than written as a
subtraction because the algebra is what made earlier drafts unreadable:
  sim-to-real gap   real vs. sim motion   (what sysid is meant to shrink)
  controller lag    real vs. command      (a property of the arm, not of sim)

Joint velocities are not logged, so they are Savitzky-Golay derivatives of the
logged positions (identical filter on every series, so the curves are
comparable); where the dataset parquet is reachable its recorded
``observation.state.joint_velocity`` is overlaid as a reference.

End-effector signals are forward kinematics of those three joint series
(FR3 + Robotiq 2F-85 TCP, ``rcs.common.Pin``), so the EE comparison uses one
consistent frame instead of mixing the dataset's flange pose with FK.

Usage:
    python scripts/sim2real/replay_report.py                 # 2 newest logs
    python scripts/sim2real/replay_report.py A_dir B_dir [-o out.pdf]
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
LOG_ROOT = Path(__file__).resolve().parents[2] / "outputs" / "replay_logs"

JOINTS = [f"joint {j + 1}" for j in range(7)]
EE_DIMS = ["x", "y", "z", "roll", "pitch", "yaw"]
EE_UNITS = ["m", "m", "m", "rad", "rad", "rad"]
# Orientation error is the rotation vector of the relative rotation, not a
# roll/pitch/yaw difference: RPY differences are meaningless across wraps and
# near gimbal lock, which this trajectory passes through constantly.
EE_ERR_DIMS = ["x", "y", "z", "ori x", "ori y", "ori z"]
EE_ERR_UNITS = ["mm", "mm", "mm", "deg", "deg", "deg"]

C_DS = "#1f77b4"    # dataset / sim
C_MEAS = "#d62728"  # measured on the real robot
C_SENT = "#2ca02c"  # action sent

# The two comparisons this report makes. Both involve the real robot; they
# differ in what it is held against, so they are always named, never written as
# a subtraction.
GAP = "sim-to-real gap"      # real robot vs. the motion sim itself achieved
LAG = "controller lag"       # real robot vs. the target it was commanded
SIGNALS = ("sim motion = joints sim reached (q_ds)   ·   "
           "command = joints sent to the real robot (q_sent)   ·   "
           "real = joints the real robot reached (q_meas)")

# Savitzky-Golay window for differentiating the 25 Hz position logs. 9 frames
# (~0.36 s) is the smallest window that keeps the measured velocity readable
# without flattening the fast wrist moves.
VEL_WINDOW = 9
VEL_POLY = 2


def wrap(a: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi] so orientation errors don't show fake 2pi jumps."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def fk(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(T,7) joint angles -> (T,6) TCP [x,y,z,roll,pitch,yaw] and (T,3,3) rotations.

    RPY is left wrapped to (-pi, pi]; it is only ever plotted, never differenced.
    """
    robot = rcs.ROBOTS[rcs_common.RobotType.FR3]
    pin = rcs_common.Pin(robot.mjcf_model_path, robot.attachment_site, False)
    tcp = rcs.GRIPPER_TCP_OFFSETS[rcs_common.GripperType("Robotiq2F85")]
    poses = [pin.forward(qi, tcp) for qi in q]
    xyzrpy = np.array([pose.xyzrpy() for pose in poses])
    xyzrpy[:, 3:] = wrap(xyzrpy[:, 3:])
    rot = np.array([pose.rotation_m() for pose in poses])
    return xyzrpy, rot


def ori_err(rot_meas: np.ndarray, rot_ref: np.ndarray) -> np.ndarray:
    """(T,3) rotation vector [rad] taking the reference orientation to the measured one."""
    rel = np.einsum("tji,tjk->tik", rot_ref, rot_meas)  # rot_ref^T @ rot_meas
    return Rotation.from_matrix(rel).as_rotvec()


def deriv(q: np.ndarray, dt: float) -> np.ndarray:
    """Smoothed time derivative of (T,N) joint positions [rad] -> [rad/s]."""
    window = min(VEL_WINDOW, len(q) - (len(q) + 1) % 2)
    if window <= VEL_POLY:
        return np.gradient(q, dt, axis=0)
    return savgol_filter(q, window, VEL_POLY, deriv=1, delta=dt, axis=0, mode="interp")


def dataset_velocity(meta: dict, n_frames: int) -> np.ndarray | None:
    """The sim's own recorded joint velocity, or None if the dataset is unreachable.

    ``episode_replay`` drops ``action.is_valid == False`` frames before sending
    them, so the same frames are dropped here to stay row-aligned with the CSV.
    """
    try:
        root = Path(meta["dataset_root"])
        episodes = _load_episodes_meta(root)
        ep = _load_episode(root, episodes, int(meta["episode"]))
        if "action.is_valid" in ep:
            ep = ep[_stack(ep, "action.is_valid").astype(bool).reshape(-1)]
        vel = _stack(ep, "observation.state.joint_velocity").astype(np.float64)
    except Exception as exc:  # dataset moved / not mounted: the plot still works
        print(f"  note: dataset velocity unavailable ({exc.__class__.__name__}: {exc})")
        return None
    if len(vel) != n_frames:
        print(f"  note: dataset has {len(vel)} valid frames, log has {n_frames}; "
              "skipping the recorded-velocity reference")
        return None
    return vel


class Run:
    """One replay log: joint + EE signals and their tracking errors."""

    def __init__(self, path: Path, label: str):
        self.path = path
        self.label = label
        self.meta = json.loads((path / "replay_meta.json").read_text())
        df = pd.read_csv(path / "replay_frames.csv")
        self.df = df
        self.t = df["t_ds"].to_numpy()
        self.gripper = df["gripper_sent"].to_numpy()
        self.q_ds = df[[f"q_ds{j}" for j in range(7)]].to_numpy()
        self.q_sent = df[[f"q_sent{j}" for j in range(7)]].to_numpy()
        self.q_meas = df[[f"q_meas{j}" for j in range(7)]].to_numpy()
        (self.ee_ds, r_ds), (self.ee_sent, r_sent), (self.ee_meas, r_meas) = (
            fk(self.q_ds), fk(self.q_sent), fk(self.q_meas)
        )
        # sim2real gap: real robot vs what sim recorded
        self.q_err_s2r = self.q_meas - self.q_ds
        # tracking error: how far the commanded target sits ahead of the real robot
        self.q_err_track = self.q_sent - self.q_meas
        # EE error = [dx, dy, dz (m), rotation vector (rad)]
        self.ee_err_s2r = np.hstack(
            [self.ee_meas[:, :3] - self.ee_ds[:, :3], ori_err(r_meas, r_ds)]
        )
        self.ee_err_track = np.hstack(
            [self.ee_sent[:, :3] - self.ee_meas[:, :3], ori_err(r_sent, r_meas)]
        )
        # Velocities are not logged; differentiate every series the same way so
        # the sim and real curves stay comparable.
        self.dt = float(np.median(np.diff(self.t)))
        self.dq_ds = deriv(self.q_ds, self.dt)
        self.dq_meas = deriv(self.q_meas, self.dt)
        self.dq_ds_rec = dataset_velocity(self.meta, len(self.t))
        # single scalar orientation error (geodesic angle of the relative rotation)
        self.ori_err_s2r = np.linalg.norm(self.ee_err_s2r[:, 3:], axis=1)
        self.ori_err_track = np.linalg.norm(self.ee_err_track[:, 3:], axis=1)

def rmse(e: np.ndarray, axis=0) -> np.ndarray:
    return np.sqrt(np.mean(e ** 2, axis=axis))


def _finish(fig, title: str, pdf: PdfPages) -> None:
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    pdf.savefig(fig)
    plt.close(fig)


def _grid(n_rows: int, runs: list[Run], height: float = 1.75):
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(11.69, min(8.27, 1.0 + height * n_rows)),
        sharex="col", sharey="row", squeeze=False,
    )
    for c, run in enumerate(runs):
        axes[0][c].set_title(run.label, fontsize=11, fontweight="bold")
    for c in range(2):
        axes[-1][c].set_xlabel("dataset time [s]")
    return fig, axes


def _shade_gripper(ax, run: Run) -> None:
    """Light shading while the gripper is commanded closed."""
    closed = run.gripper < 0.5
    edges = np.flatnonzero(np.diff(closed.astype(int)))
    bounds = np.concatenate(
        ([0] if closed[0] else [], edges + 1, [len(closed) - 1] if closed[-1] else [])
    ).astype(int)
    for a, b in zip(bounds[::2], bounds[1::2]):
        ax.axvspan(run.t[a], run.t[min(b, len(run.t) - 1)], color="0.85", lw=0, zorder=0)


def page_signals(pdf: PdfPages, runs: list[Run], kind: str) -> None:
    """Dataset / sent / measured overlaid, one row per joint or EE dimension."""
    if kind == "joint":
        names, units = JOINTS, ["rad"] * 7
        sigs = [(r.q_ds, r.q_sent, r.q_meas) for r in runs]
        title = "Joint trajectories — sim motion vs. command vs. real robot"
    else:
        names, units = EE_DIMS, EE_UNITS
        sigs = [(r.ee_ds, r.ee_sent, r.ee_meas) for r in runs]
        title = "End-effector (FK, 2F-85 TCP) — sim motion vs. command vs. real robot"

    for lo in range(0, len(names), 4):
        hi = min(lo + 4, len(names))
        fig, axes = _grid(hi - lo, runs, height=1.85)
        for c, (run, (ds, sent, meas)) in enumerate(zip(runs, sigs)):
            for r, d in enumerate(range(lo, hi)):
                ax = axes[r][c]
                _shade_gripper(ax, run)
                ax.plot(run.t, ds[:, d], color=C_DS, lw=1.0, label="sim motion")
                ax.plot(run.t, sent[:, d], color=C_SENT, lw=0.8, ls="--", label="command")
                ax.plot(run.t, meas[:, d], color=C_MEAS, lw=1.0, alpha=0.85, label="real")
                ax.set_ylabel(f"{names[d]}\n[{units[d]}]", fontsize=9)
                ax.grid(alpha=0.25, lw=0.4)
                ax.tick_params(labelsize=8)
        axes[0][0].legend(fontsize=7, ncol=3, loc="upper right", framealpha=0.9)
        suffix = f" ({names[lo]}–{names[hi - 1]})"
        _finish(fig, title + suffix, pdf)


def page_velocity(pdf: PdfPages, runs: list[Run]) -> None:
    """Joint velocity in sim vs. on the real robot, one row per joint."""
    title = "Joint velocity \u2014 sim vs. real (derivative of the logged positions)"
    for lo in range(0, 7, 4):
        hi = min(lo + 4, 7)
        fig, axes = _grid(hi - lo, runs, height=1.85)
        for c, run in enumerate(runs):
            for r, j in enumerate(range(lo, hi)):
                ax = axes[r][c]
                _shade_gripper(ax, run)
                ax.axhline(0, color="0.4", lw=0.6)
                if run.dq_ds_rec is not None:
                    ax.plot(run.t, run.dq_ds_rec[:, j], color="#9ecae1", lw=1.4,
                            label="sim (recorded)")
                ax.plot(run.t, run.dq_ds[:, j], color=C_DS, lw=0.9,
                        label=f"sim motion (diff)  RMS {rmse(run.dq_ds[:, j]):.3f}")
                ax.plot(run.t, run.dq_meas[:, j], color=C_MEAS, lw=0.9, alpha=0.85,
                        label=f"real (diff)  RMS {rmse(run.dq_meas[:, j]):.3f}")
                ax.set_ylabel(f"{JOINTS[j]}\n[rad/s]", fontsize=9)
                ax.grid(alpha=0.25, lw=0.4)
                ax.tick_params(labelsize=8)
                ax.legend(fontsize=6.5, ncol=3, loc="upper right", framealpha=0.9)
        _finish(fig, f"{title} ({JOINTS[lo]}\u2013{JOINTS[hi - 1]})", pdf)


def page_errors(pdf: PdfPages, runs: list[Run], kind: str) -> None:
    """Sim2real gap and tracking error per dimension, with RMSE annotated."""
    if kind == "joint":
        names, units, scale = JOINTS, ["deg"] * 7, np.rad2deg
        errs = [(scale(r.q_err_s2r), scale(r.q_err_track)) for r in runs]
        title = f"Joint error over time — {GAP} and {LAG}"
    else:
        names, units = EE_ERR_DIMS, EE_ERR_UNITS
        errs = []
        for r in runs:
            a, b = r.ee_err_s2r.copy(), r.ee_err_track.copy()
            for e in (a, b):
                e[:, :3] *= 1e3
                e[:, 3:] = np.rad2deg(e[:, 3:])
            errs.append((a, b))
        title = f"End-effector error over time — {GAP} and {LAG}"

    for lo in range(0, len(names), 4):
        hi = min(lo + 4, len(names))
        fig, axes = _grid(hi - lo, runs, height=1.85)
        for c, (run, (e_gap, e_trk)) in enumerate(zip(runs, errs)):
            for r, d in enumerate(range(lo, hi)):
                ax = axes[r][c]
                _shade_gripper(ax, run)
                ax.axhline(0, color="0.4", lw=0.6)
                ax.plot(run.t, e_gap[:, d], color=C_DS, lw=0.9,
                        label=f"{GAP}  RMSE {rmse(e_gap[:, d]):.2f}")
                ax.plot(run.t, e_trk[:, d], color=C_MEAS, lw=0.9, alpha=0.8,
                        label=f"{LAG}  RMSE {rmse(e_trk[:, d]):.2f}")
                ax.set_ylabel(f"{names[d]}\n[{units[d]}]", fontsize=9)
                ax.grid(alpha=0.25, lw=0.4)
                ax.tick_params(labelsize=8)
                ax.legend(fontsize=6.5, loc="upper right", framealpha=0.9)
        suffix = f" ({names[lo]}–{names[hi - 1]})"
        _finish(fig, title + suffix, pdf)


def page_norms(pdf: PdfPages, runs: list[Run]) -> None:
    """Scalar position / orientation error over the whole trajectory."""
    fig, axes = _grid(2, runs, height=2.6)
    for c, run in enumerate(runs):
        ax = axes[0][c]
        _shade_gripper(ax, run)
        pos_gap = np.linalg.norm(run.ee_err_s2r[:, :3], axis=1) * 1e3
        pos_trk = np.linalg.norm(run.ee_err_track[:, :3], axis=1) * 1e3
        ax.plot(run.t, pos_gap, color=C_DS, lw=0.9,
                label=f"{GAP}  RMSE {rmse(pos_gap):.1f} mm  max {pos_gap.max():.1f}")
        ax.plot(run.t, pos_trk, color=C_MEAS, lw=0.9, alpha=0.8,
                label=f"{LAG}  RMSE {rmse(pos_trk):.1f} mm  max {pos_trk.max():.1f}")
        ax.set_ylabel("‖TCP position\nerror‖ [mm]", fontsize=9)

        ax = axes[1][c]
        _shade_gripper(ax, run)
        ori_gap = np.rad2deg(run.ori_err_s2r)
        ori_trk = np.rad2deg(run.ori_err_track)
        ax.plot(run.t, ori_gap, color=C_DS, lw=0.9,
                label=f"{GAP}  RMSE {rmse(ori_gap):.2f}°  max {ori_gap.max():.2f}")
        ax.plot(run.t, ori_trk, color=C_MEAS, lw=0.9, alpha=0.8,
                label=f"{LAG}  RMSE {rmse(ori_trk):.2f}°  max {ori_trk.max():.2f}")
        ax.set_ylabel("TCP orientation\nerror [deg]", fontsize=9)

        for ax in (axes[0][c], axes[1][c]):
            ax.grid(alpha=0.25, lw=0.4)
            ax.tick_params(labelsize=8)
            ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    _finish(fig, "TCP error magnitude over the whole trajectory", pdf)


def page_bars(pdf: PdfPages, runs: list[Run]) -> None:
    """Per-joint / per-EE-dimension RMSE, the two runs next to each other."""
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 6.6))
    width = 0.38
    specs = [
        (axes[0][0], f"Joint {GAP}\nhow far the real robot ended up from the sim motion",
         JOINTS, "deg", [np.rad2deg(rmse(r.q_err_s2r)) for r in runs]),
        (axes[0][1], f"Joint {LAG}\nhow far the real robot ran behind its command",
         JOINTS, "deg", [np.rad2deg(rmse(r.q_err_track)) for r in runs]),
        (axes[1][0], f"End-effector position {GAP}\nreal TCP vs. the sim TCP motion",
         EE_DIMS[:3], "mm", [rmse(r.ee_err_s2r[:, :3]) * 1e3 for r in runs]),
        (axes[1][1], f"End-effector orientation {GAP}\nreal TCP vs. the sim TCP motion "
         "(rotation vector)",
         EE_ERR_DIMS[3:], "deg", [np.rad2deg(rmse(r.ee_err_s2r[:, 3:])) for r in runs]),
    ]
    for ax, title, names, unit, vals in specs:
        x = np.arange(len(names))
        for k, (run, v) in enumerate(zip(runs, vals)):
            bars = ax.bar(x + (k - 0.5) * width, v, width,
                          label=run.label, color=[C_DS, C_MEAS][k], alpha=0.85)
            ax.bar_label(bars, fmt="%.2f", fontsize=6.5, padding=1)
        ax.set_xticks(x, names, fontsize=8)
        ax.set_ylabel(f"RMSE [{unit}]", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.margins(y=0.16)
        ax.grid(axis="y", alpha=0.25, lw=0.4)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8)
    fig.text(0.5, 0.935, SIGNALS, ha="center", fontsize=7.5, color="0.35")
    fig.suptitle("Error summary (RMSE): without sysid vs. with sysid",
                 fontsize=13, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.915))
    pdf.savefig(fig)
    plt.close(fig)


def page_title(pdf: PdfPages, runs: list[Run]) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.5, 0.93, "Sim2real replay: without sysid vs. with sysid",
             ha="center", fontsize=17, fontweight="bold")
    fig.text(0.5, 0.895,
             "Recorded sim trajectories replayed open-loop on the real FR3 "
             "(no policy in the loop)", ha="center", fontsize=10, color="0.3")

    lines = [
        "Three signals per frame, all from replay_frames.csv:",
        "    sim motion   q_ds     — the joint positions simulation actually reached",
        "    command      q_sent   — the joint target that was sent to the real robot",
        "    real         q_meas   — the joint positions the real robot actually reached",
        "",
        "Two comparisons, both against 'real'. They are named, never written as a subtraction:",
        "",
        "    sim-to-real gap    real vs. sim motion",
        "        Did the real robot end up where sim did? This is the sim2real question the",
        "        replay exists to answer, and it is what sysid is meant to shrink.",
        "",
        "    controller lag     real vs. command",
        "        Did the real robot reach the target it was told to reach? This is a property",
        "        of the controller and the arm, not of sim; sysid cannot change it.",
        "",
        "    They differ because 'sim motion' is what sim reached, while 'command' is what was",
        "    asked for — sim lagged its own targets too, so neither equals the other.",
        "",
        "End-effector values are forward kinematics of those joint series (FR3 + Robotiq 2F-85 TCP),",
        "so all three EE curves live in one consistent frame.",
        "Grey bands mark frames where the gripper is commanded closed.",
        "",
        f"Velocity is not logged, so it is a Savitzky-Golay derivative (window {VEL_WINDOW}, order "
        f"{VEL_POLY})",
        "of the logged positions, applied identically to the sim and real series; the sim's own",
        "recorded joint_velocity is overlaid where the dataset parquet is reachable. Note the replay",
        "loop is not rate-paced, so real velocity is per dataset frame, not per wall-clock second.",
        "",
        "The roll/pitch/yaw rows are wrapped to (-pi, pi], so they jump; orientation ERROR is never",
        "an RPY difference but the rotation vector of the relative rotation (ori x/y/z), which is",
        "well behaved through wraps and gimbal lock.",
        "",
    ]
    for run in runs:
        pos = np.linalg.norm(run.ee_err_s2r[:, :3], axis=1) * 1e3
        trk = np.linalg.norm(run.ee_err_track[:, :3], axis=1) * 1e3
        lines += [
            f"{run.label}",
            f"    log       {run.path.name}",
            f"    dataset   {run.meta['dataset_root']}",
            f"    episode   {run.meta['episode']}   frames {len(run.t)}   "
            f"{run.t[-1]:.1f} s @ {run.meta['fps']} fps   TAM {run.meta['tam']}",
            f"    joints   gap {np.rad2deg(rmse(run.q_err_s2r.ravel())):.3f} deg RMSE   "
            f"lag {np.rad2deg(rmse(run.q_err_track.ravel())):.3f} deg RMSE",
            f"    TCP pos  gap {rmse(pos):.1f} mm RMSE (max {pos.max():.1f})   "
            f"lag {rmse(trk):.1f} mm RMSE (max {trk.max():.1f})",
            f"    TCP ori  gap {np.rad2deg(rmse(run.ori_err_s2r)):.2f} deg RMSE "
            f"(max {np.rad2deg(run.ori_err_s2r.max()):.2f})",
            "",
        ]
    fig.text(0.06, 0.85, "\n".join(lines), va="top", ha="left",
             fontsize=8.5, family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def newest_logs(n: int = 2) -> list[Path]:
    dirs = [d for d in LOG_ROOT.iterdir() if (d / "replay_frames.csv").exists()]
    good = [d for d in dirs if sum(1 for _ in (d / "replay_frames.csv").open()) > 2]
    return sorted(good, key=lambda d: d.name)[-n:]


def label_for(path: Path, meta: dict) -> str:
    root = meta["dataset_root"]
    if "sysid" in root:
        return "WITH sysid"
    return "WITHOUT sysid"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", nargs="*", type=Path,
                    help="two outputs/replay_logs/<stamp> dirs (default: the two newest)")
    ap.add_argument("-o", "--out", type=Path,
                    default=LOG_ROOT / "sysid_replay_report.pdf")
    args = ap.parse_args()

    paths = args.logs if args.logs else newest_logs(2)
    if len(paths) != 2:
        ap.error(f"need exactly two replay log dirs, got {len(paths)}")

    runs = []
    for p in paths:
        meta = json.loads((p / "replay_meta.json").read_text())
        runs.append(Run(p, label_for(p, meta)))
    # Always show "without sysid" on the left.
    runs.sort(key=lambda r: "WITHOUT" not in r.label)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.out) as pdf:
        page_title(pdf, runs)
        page_bars(pdf, runs)
        page_norms(pdf, runs)
        page_signals(pdf, runs, "joint")
        page_errors(pdf, runs, "joint")
        page_velocity(pdf, runs)
        page_signals(pdf, runs, "ee")
        page_errors(pdf, runs, "ee")
        pdf.infodict()["Title"] = "Sim2real replay report: without sysid vs with sysid"
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
