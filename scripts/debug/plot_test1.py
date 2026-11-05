#!/usr/bin/env python3
"""Plot the joint-position log from test1_same_plan_8envs.py.

One subplot per arm joint; each shows all 8 env traces plus the commanded
target. Phase boundaries (settle / trajectory / gripper-close) are shaded.

    RG_OUT=scratchpad/red_green_2block/test1 uv run python scripts/debug/plot_test1.py
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

data = np.load(OUT / "joint_log.npz", allow_pickle=True)
pos = data["joint_pos"]            # (S, N, 7)
tgt = data["joint_tgt"]            # (S, N, 7)
phases = data["phases"]            # (S,)
names = [str(n) for n in data["joint_names"]]
S, N, J = pos.shape
steps = np.arange(S)

# 8 distinguishable colors
cmap = plt.get_cmap("tab10")
colors = [cmap(i) for i in range(N)]

# phase shading boundaries
phase_names = {0: "settle", 1: "trajectory", 2: "gripper"}
bounds = [0] + [i for i in range(1, S) if phases[i] != phases[i - 1]] + [S]

fig, axes = plt.subplots(3, 3, figsize=(17, 11))
fig.suptitle(
    f"Test 1: one env-0 red-grasp plan replayed in {N} envs — arm joint positions",
    fontsize=14,
)
for j in range(J):
    ax = axes.flat[j]
    for e in range(N):
        ax.plot(steps, pos[:, e, j], color=colors[e], linewidth=1.0, label=f"env {e}")
    ax.plot(steps, tgt[:, 0, j], color="black", linewidth=1.4, linestyle="--",
            label="target")
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        ph = int(phases[b0])
        if ph == 1:
            ax.axvspan(b0, b1, color="#f0f0f0", alpha=0.5, zorder=0)
    ax.set_title(names[j], fontsize=9)
    ax.set_xlabel("sim step", fontsize=7)
    ax.set_ylabel("rad", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.25, linewidth=0.4)

# divergence panel: max |env_i - env_0| across joints vs time
ax = axes.flat[7]
dev = np.abs(pos - pos[:, :1, :]).max(axis=2)  # (S, N)
for e in range(1, N):
    ax.plot(steps, dev[:, e], color=colors[e], linewidth=1.0, label=f"env {e}")
ax.set_title("max joint deviation from env 0 (rad)", fontsize=9)
ax.set_xlabel("sim step", fontsize=7)
ax.set_ylabel("rad", fontsize=7)
ax.tick_params(labelsize=6)
ax.grid(True, alpha=0.25, linewidth=0.4)

# legend in the last cell
ax = axes.flat[8]
ax.axis("off")
handles, labels = axes.flat[0].get_legend_handles_labels()
ax.legend(handles, labels, loc="center", fontsize=9, ncol=2, title="traces")

fig.tight_layout(rect=(0, 0, 1, 0.97))
out_png = OUT / "test1_joint_positions.png"
fig.savefig(out_png, dpi=130)
print(f"wrote {out_png}")

# also print the peak divergence for a quick numeric read
peak = dev.max(axis=0)  # (N,)
print("peak max-joint deviation from env 0 (rad) per env:")
for e in range(N):
    print(f"  env {e}: {peak[e]:.5f}")
