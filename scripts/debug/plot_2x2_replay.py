#!/usr/bin/env python3
"""Clean per-cell replay plots (one figure per 2x2 cell), presentation-ready.

For each cell it shows, per arm joint, the tracking ERROR (replayed - recorded)
for all 8 episodes over normalized episode progress on a shared tight scale, plus
an aggregate RMS-error panel. This replaces the earlier tangle of 16 faint
trajectory traces where the error was invisible.

    RG_OUT=scratchpad/red_green_2block/test1 .venv/bin/python scripts/debug/plot_2x2_replay.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("RG_OUT", "scratchpad/red_green_2block/test1"))
if not OUT.is_absolute():
    OUT = REPO / OUT

DS = {"A": np.load(OUT / "dsA.npz"), "B": np.load(OUT / "dsB.npz")}
DSNAME = {"A": "blocksuite", "B": "classic"}
CELLS = [  # (dataset, sim, file, diagonal?)
    ("A", "blocksuite", "replay_A_in_blocksuite.npz", True),
    ("A", "classic", "replay_A_in_classic.npz", False),
    ("B", "blocksuite", "replay_B_in_blocksuite.npz", False),
    ("B", "classic", "replay_B_in_classic.npz", True),
]
JOINTS = [f"panda_joint{i+1}" for i in range(7)]
GRID = np.linspace(0.0, 1.0, 120)
YLIM = 0.10  # rad (~5.7 deg) shared error scale across cells; spikes annotated
# Drop the recorder's trailing zero-action frames (action.joint_target=0 -> the
# arm is commanded toward 0 on the last row); comparing on them is meaningless.
TRIM = int(os.environ.get("RG_TRIM_TAIL", "3"))
CMAP = plt.get_cmap("tab10")
summary = {}


def resample(v, L):  # (L,) -> (120,)
    return np.interp(GRID, np.linspace(0, 1, L), v)


for ds, sim, fname, diag in CELLS:
    fpath = OUT / fname
    if not fpath.exists():
        print(f"SKIP {fname}")
        continue
    rp = np.load(fpath)
    rec = DS[ds]
    lengths = np.maximum(rec["lengths"].astype(int) - TRIM, 2)
    jp_rec = rec["joint_position"]                     # (8,T,7)
    jp_rep = np.transpose(rp["joint_pos"], (1, 0, 2))  # (8,T,7)
    T = min(jp_rec.shape[1], jp_rep.shape[1])
    n = jp_rec.shape[0]

    # per-episode, per-joint error resampled to normalized progress
    jerr = np.zeros((n, 120, 7))
    for e in range(n):
        L = int(min(lengths[e], T))
        for j in range(7):
            jerr[e, :, j] = resample(jp_rep[e, :L, j] - jp_rec[e, :L, j], L)
    rms = np.sqrt((jerr ** 2).mean(axis=2))            # (n,120) rms over joints

    kind = "SELF-CONSISTENCY" if diag else "CROSS-REPO"
    accent = "#2ca02c" if diag else "#d62728"
    fig, axes = plt.subplots(3, 3, figsize=(15, 9))
    fig.suptitle(f"Replay error:  dataset {ds} ({DSNAME[ds]})  →  {sim} sim   [{kind}]",
                 fontsize=14, color=accent)

    for j in range(7):
        ax = axes.flat[j]
        for e in range(n):
            ax.plot(GRID, jerr[e, :, j], color=CMAP(e % 10), lw=0.9, alpha=0.6)
        ax.plot(GRID, jerr[:, :, j].mean(0), color="k", lw=2.0, label="mean")
        ax.axhline(0, color="0.5", lw=0.6)
        mx = np.degrees(np.abs(jerr[:, :, j]).max())
        ax.set_title(f"{JOINTS[j]}   max |err| {mx:.1f}°", fontsize=9)
        ax.set_ylim(-YLIM, YLIM)
        ax.set_xlabel("episode progress", fontsize=7)
        ax.set_ylabel("err (rad)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.25)
        clipped = np.abs(jerr[:, :, j]).max() > YLIM
        if clipped:
            ax.text(0.98, 0.95, f"↑ spikes to {mx:.0f}°", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7, color="#b00")

    # aggregate RMS-over-joints panel
    ax = axes.flat[7]
    ax.fill_between(GRID, rms.min(0), rms.max(0), color=accent, alpha=0.15, lw=0)
    ax.plot(GRID, rms.mean(0), color=accent, lw=2.2)
    ax.axhline(np.radians(1), color="k", lw=0.6, ls=":", alpha=0.5)
    ax.text(0.01, np.radians(1) * 1.1, "1°", fontsize=7, alpha=0.6)
    ax.set_title("RMS error over 7 joints (mean + band)", fontsize=9)
    ax.set_xlabel("episode progress", fontsize=7)
    ax.set_ylabel("RMS err (rad)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.25)

    # text summary
    ax = axes.flat[8]; ax.axis("off")
    mean_deg = np.degrees(rms.mean())
    peak_deg = np.degrees(rms.mean(0).max())
    summary[(ds, sim)] = (mean_deg, peak_deg)
    ax.text(0.02, 0.97,
            f"dataset {ds} ({DSNAME[ds]}) -> {sim} sim\n{kind}\n\n"
            f"each panel: replayed - recorded\n(8 colored lines = episodes,\n black = mean),  y = +/- {np.degrees(YLIM):.0f} deg\n\n"
            f"mean RMS err:  {mean_deg:.2f} deg\npeak RMS err:  {peak_deg:.2f} deg\n\n"
            f"{'diagonal: sim replaying its' if diag else 'off-diagonal: sim replaying'}\n"
            f"{'own dataset' if diag else 'the OTHER dataset'}\n\n"
            f"low & flat = faithful replay.\n"
            f"(block grasp reproduced to ~1-2cm\n in all 4 cells)",
            va="top", ha="left", fontsize=9, family="monospace")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    op = OUT / fname.replace(".npz", ".png")
    fig.savefig(op, dpi=120); plt.close(fig)
    print(f"wrote {op}")

print("\n=== per-cell RMS joint error (mean / peak, deg) ===")
for ds, sim, *_ in CELLS:
    v = summary.get((ds, sim))
    if v:
        print(f"  {ds}->{sim:10s}: mean {v[0]:.2f}  peak {v[1]:.2f}")
