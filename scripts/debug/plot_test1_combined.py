#!/usr/bin/env python3
"""Single combined figure: per joint, ToyBlocks value, TAMP value, and their diff.

For each of the 7 arm joints one panel shows (left axis, rad) the env-mean joint
position from the ToyBlocks (blocks_vectorized) run and the TAMP (blocks-classic)
run replaying the same CuRobo plan, and (right axis, rad) their signed difference
classic - vectorized. Within-pipeline env spread is ~0, so the env-mean is
representative.

    RG_OUT=scratchpad/red_green_2block/test1 uv run python scripts/debug/plot_test1_combined.py
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

bv = np.load(OUT / "joint_log.npz", allow_pickle=True)          # ToyBlocks / vectorized
cl = np.load(OUT / "joint_log_classic.npz", allow_pickle=True)  # TAMP / classic

names = [str(n) for n in bv["joint_names"]]
S = min(bv["joint_pos"].shape[0], cl["joint_pos"].shape[0])
bv_mean = bv["joint_pos"][:S].mean(axis=1)   # (S, 7)
cl_mean = cl["joint_pos"][:S].mean(axis=1)   # (S, 7)
tgt = bv["joint_tgt"][:S, 0, :]              # (S, 7) commanded (identical for both)
diff = cl_mean - bv_mean                     # (S, 7)  classic - vectorized
J = bv_mean.shape[1]
steps = np.arange(S)

BV_C, CL_C, D_C = "#1f77b4", "#d62728", "#2ca02c"  # blue / red / green

fig, axes = plt.subplots(3, 3, figsize=(18, 11))
fig.suptitle(
    "Test 1 combined — same CuRobo plan: ToyBlocks (blue) vs TAMP classic (red), "
    "diff = classic - vectorized (green, right axis)",
    fontsize=13,
)

diff_max = float(np.abs(diff).max())
handles_labels = None
for j in range(J):
    ax = axes.flat[j]
    ax.plot(steps, tgt[:, j], color="0.7", linewidth=0.9, linestyle=":", label="target", zorder=1)
    ax.plot(steps, bv_mean[:, j], color=BV_C, linewidth=1.4, label="toyblocks (vectorized)", zorder=3)
    ax.plot(steps, cl_mean[:, j], color=CL_C, linewidth=1.4, linestyle="--",
            label="TAMP (classic)", zorder=4)
    ax.set_title(names[j], fontsize=10)
    ax.set_xlabel("sim step", fontsize=7)
    ax.set_ylabel("joint position (rad)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.25, linewidth=0.4)

    ax2 = ax.twinx()
    ax2.plot(steps, diff[:, j], color=D_C, linewidth=1.0, alpha=0.9,
             label="diff (classic - vectorized)", zorder=2)
    ax2.axhline(0.0, color=D_C, linewidth=0.5, alpha=0.3)
    ax2.set_ylim(-diff_max * 1.1, diff_max * 1.1)  # shared diff scale across joints
    ax2.set_ylabel("diff (rad)", fontsize=7, color=D_C)
    ax2.tick_params(axis="y", labelsize=6, colors=D_C)

    if handles_labels is None:
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        handles_labels = (h1 + h2, l1 + l2)

# panel 8: absolute diff, all joints, shared linear scale
ax = axes.flat[7]
cmap = plt.get_cmap("tab10")
for j in range(J):
    ax.plot(steps, np.abs(diff[:, j]), color=cmap(j), linewidth=1.0,
            label=names[j].replace("panda_", ""))
ax.set_title("|classic - vectorized| per joint", fontsize=10)
ax.set_xlabel("sim step", fontsize=7)
ax.set_ylabel("|diff| (rad)", fontsize=7)
ax.tick_params(labelsize=6)
ax.grid(True, alpha=0.25, linewidth=0.4)
ax.legend(fontsize=6, ncol=2)

# panel 9: legend + numeric summary
ax = axes.flat[8]
ax.axis("off")
peak = np.abs(diff).max(axis=0)
final = np.abs(diff[-1])
lines = ["toyblocks = blocks_vectorized", "TAMP = blocks-classic (matched fr3 USD + stiff gains)",
         "", "peak |classic - vectorized| (rad):"]
lines += [f"  {names[j].replace('panda_', ''):7s} peak {peak[j]:.4f}  final {final[j]:.4f}"
          for j in range(J)]
lines += ["", f"max over all joints/steps: {diff_max:.4f} rad ({np.degrees(diff_max):.2f} deg)"]
ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace")
if handles_labels is not None:
    ax.legend(handles_labels[0], handles_labels[1], loc="lower left", fontsize=8)

fig.tight_layout(rect=(0, 0, 1, 0.96))
out_png = OUT / "test1_combined.png"
fig.savefig(out_png, dpi=130)
print(f"wrote {out_png}")
print(f"max |classic - vectorized| = {diff_max:.5f} rad ({np.degrees(diff_max):.3f} deg)")
