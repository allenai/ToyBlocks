#!/usr/bin/env python3
"""Clean, interpretable summary of the 2x2 replay: replay error vs episode progress.

Instead of 16 faint overlapping traces per joint, this shows the single quantity
that matters — the RMS arm-joint error between replayed and recorded — aggregated
over the 7 joints and the 8 episodes, on a shared scale, with episodes resampled
to normalized progress (0..1) so their different lengths line up.

    RG_OUT=scratchpad/red_green_2block/test1 .venv/bin/python scripts/debug/plot_replay_summary.py
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
CELLS = [  # (dataset, sim, file, color, linestyle, kind)
    ("A", "toyblocks", "replay_A_in_toyblocks.npz", "#1f77b4", "-",  "self"),
    ("A", "classic",    "replay_A_in_classic.npz",    "#1f77b4", "--", "cross"),
    ("B", "toyblocks", "replay_B_in_toyblocks.npz", "#d62728", "--", "cross"),
    ("B", "classic",    "replay_B_in_classic.npz",    "#d62728", "-",  "self"),
]
GRID = np.linspace(0.0, 1.0, 120)
TRIM = int(os.environ.get("RG_TRIM_TAIL", "3"))  # drop trailing zero-action frames


def cell_curves(ds, fname):
    rp = np.load(OUT / fname)
    rec = DS[ds]
    lengths = np.maximum(rec["lengths"].astype(int) - TRIM, 2)
    jp_rec = rec["joint_position"]                     # (8,T,7)
    jp_rep = np.transpose(rp["joint_pos"], (1, 0, 2))  # (8,T,7)
    T = min(jp_rec.shape[1], jp_rep.shape[1])
    curves = []
    for e in range(jp_rec.shape[0]):
        L = int(min(lengths[e], T))
        if L < 2:
            continue
        err = jp_rep[e, :L] - jp_rec[e, :L]            # (L,7)
        rms = np.sqrt((err ** 2).mean(axis=1))         # (L,) rad
        curves.append(np.interp(GRID, np.linspace(0, 1, L), rms))
    return np.asarray(curves)                          # (n_ep,120)


data = {(ds, sim): cell_curves(ds, f) for ds, sim, f, *_ in CELLS}

fig = plt.figure(figsize=(15, 6))
gs = fig.add_gridspec(1, 3, width_ratios=[2.0, 1.0, 0.9], wspace=0.28)
fig.suptitle("Replay arm-joint error (RMS over 7 joints) — matched fr3 USD + gains + frame-0",
             fontsize=13)

# --- left: error vs normalized progress, all 4 cells on one shared axis ---
ax = fig.add_subplot(gs[0, 0])
for ds, sim, f, color, ls, kind in CELLS:
    c = data[(ds, sim)]
    m = c.mean(0)
    ax.fill_between(GRID, c.min(0), c.max(0), color=color, alpha=0.10, linewidth=0)
    ax.plot(GRID, m, color=color, linestyle=ls, linewidth=2.2,
            label=f"{ds}→{sim} ({kind})")
ax.set_xlabel("episode progress (0 = start, 1 = end)")
ax.set_ylabel("RMS joint error (rad)")
ax.set_title("Error over episode  —  lower = replay reproduces recorded")
ax.grid(True, alpha=0.3)
ax.legend(title="dataset → sim", fontsize=9)
ax.axhline(np.radians(1), color="k", lw=0.6, ls=":", alpha=0.5)
ax.text(0.005, np.radians(1) * 1.05, "1°", fontsize=7, alpha=0.6)

# --- middle: bar chart of mean & peak RMS error per cell ---
ax2 = fig.add_subplot(gs[0, 1])
labels = [f"{ds}→{sim[:2]}" for ds, sim, *_ in CELLS]
means = [data[(ds, sim)].mean() for ds, sim, *_ in CELLS]
peaks = [data[(ds, sim)].mean(0).max() for ds, sim, *_ in CELLS]
colors = [c[3] for c in CELLS]
x = np.arange(len(CELLS))
ax2.bar(x - 0.2, means, 0.4, color=colors, alpha=0.9, label="mean")
ax2.bar(x + 0.2, peaks, 0.4, color=colors, alpha=0.45, label="peak")
ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=0, fontsize=8)
ax2.set_ylabel("RMS joint error (rad)")
ax2.set_title("mean vs peak error")
ax2.grid(True, axis="y", alpha=0.3)
ax2.legend(fontsize=8)
for xi, (m, p) in enumerate(zip(means, peaks)):
    ax2.text(xi - 0.2, m, f"{m:.3f}", ha="center", va="bottom", fontsize=7)
    ax2.text(xi + 0.2, p, f"{p:.2f}", ha="center", va="bottom", fontsize=7)

# --- right: text takeaways ---
ax3 = fig.add_subplot(gs[0, 2]); ax3.axis("off")
bmean = {(ds, sim): data[(ds, sim)].mean() for ds, sim, *_ in CELLS}
lines = [
    "READING THIS PLOT",
    "",
    "blue = dataset A (toyblocks)",
    "red  = dataset B (classic)",
    "solid = self  dashed = cross",
    "",
    "Flat & low  = the sim reproduces",
    "the recorded arm motion.",
    "",
    f"B→toyblocks mean {bmean[('B','toyblocks')]*57.3:.2f}°",
    f"B→classic    mean {bmean[('B','classic')]*57.3:.2f}°",
    "  → B identical in both sims",
    "",
    f"A end-spike (place/release):",
    f"  peak ~{max(peaks[:2])*57.3:.0f}°, both sims",
    "  = A trajectory artifact,",
    "    not a pipeline difference",
]
ax3.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace")

fig.tight_layout(rect=(0, 0, 1, 0.95))
op = OUT / "replay_summary.png"
fig.savefig(op, dpi=140)
print(f"wrote {op}")
for ds, sim, *_ in CELLS:
    c = data[(ds, sim)]
    print(f"  {ds}->{sim:10s}: mean {np.degrees(c.mean()):.2f} deg, peak {np.degrees(c.mean(0).max()):.2f} deg")
