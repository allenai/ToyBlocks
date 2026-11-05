#!/usr/bin/env python3
"""Overlay blocks_vectorized vs blocks-classic joint traces for the same plan.

Loads Test 1's two logs (both (S, 8, 7), step-aligned, same commanded target
sequence) and, per arm joint, plots each pipeline's env-mean with a min..max band
plus the commanded target, then a per-joint pipeline-difference panel.

    RG_OUT=scratchpad/red_green_2block/test1 uv run python scripts/debug/plot_test1_compare.py
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

bv = np.load(OUT / "joint_log.npz", allow_pickle=True)          # blocks_vectorized
cl = np.load(OUT / "joint_log_classic.npz", allow_pickle=True)  # classic

bv_pos, cl_pos = bv["joint_pos"], cl["joint_pos"]  # (S, N, 7) each
tgt = bv["joint_tgt"][:, 0, :]                     # (S, 7) commanded (same for both)
names = [str(n) for n in bv["joint_names"]]
S = min(bv_pos.shape[0], cl_pos.shape[0])
bv_pos, cl_pos, tgt = bv_pos[:S], cl_pos[:S], tgt[:S]
J = bv_pos.shape[2]
steps = np.arange(S)

BV_C, CL_C = "#1f77b4", "#d62728"  # blue = vectorized, red = classic

fig, axes = plt.subplots(3, 3, figsize=(17, 11))
fig.suptitle(
    "Test 1: same CuRobo plan — blocks_vectorized (blue) vs blocks-classic (red), "
    "matched fr3 USD + stiff gains",
    fontsize=13,
)
for j in range(J):
    ax = axes.flat[j]
    for pos, c, lab in ((bv_pos, BV_C, "vectorized"), (cl_pos, CL_C, "classic")):
        mean = pos[:, :, j].mean(axis=1)
        lo, hi = pos[:, :, j].min(axis=1), pos[:, :, j].max(axis=1)
        ax.fill_between(steps, lo, hi, color=c, alpha=0.25, linewidth=0)
        ax.plot(steps, mean, color=c, linewidth=1.3, label=lab)
    ax.plot(steps, tgt[:, j], color="black", linewidth=1.0, linestyle="--", label="target")
    ax.set_title(names[j], fontsize=9)
    ax.set_xlabel("sim step", fontsize=7)
    ax.set_ylabel("rad", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.25, linewidth=0.4)

# pipeline-difference panel: |classic_mean - vectorized_mean| per joint
ax = axes.flat[7]
diff = np.abs(cl_pos.mean(axis=1) - bv_pos.mean(axis=1))  # (S, 7)
cmap = plt.get_cmap("tab10")
for j in range(J):
    ax.plot(steps, diff[:, j], color=cmap(j), linewidth=1.0, label=names[j].replace("panda_", ""))
ax.set_title("|classic - vectorized| env-mean, per joint (rad)", fontsize=9)
ax.set_xlabel("sim step", fontsize=7)
ax.set_ylabel("rad", fontsize=7)
ax.tick_params(labelsize=6)
ax.grid(True, alpha=0.25, linewidth=0.4)
ax.legend(fontsize=6, ncol=2)

# legend / summary
ax = axes.flat[8]
ax.axis("off")
peak = diff.max(axis=0)
final = diff[-1]
lines = ["peak |classic - vectorized| (rad):"]
lines += [f"  {names[j].replace('panda_','')}: peak {peak[j]:.4f}  final {final[j]:.4f}"
          for j in range(J)]
lines += ["", f"max over all joints/steps: {diff.max():.4f} rad",
          f"({np.degrees(diff.max()):.2f} deg)"]
ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace")

fig.tight_layout(rect=(0, 0, 1, 0.96))
out_png = OUT / "test1_compare_pipelines.png"
fig.savefig(out_png, dpi=130)
print(f"wrote {out_png}")
print(f"max |classic - vectorized| = {diff.max():.5f} rad ({np.degrees(diff.max()):.3f} deg)")
for j in range(J):
    print(f"  {names[j]}: peak {peak[j]:.5f}  final {final[j]:.5f} rad")
