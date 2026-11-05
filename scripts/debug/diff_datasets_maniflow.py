#!/usr/bin/env python3
"""Diff datasets A (blocksuite) vs B (classic) on the columns ManiFlow consumes.

Only the maniflow-consumed numeric columns are compared (state proprio + action
target + pd_mode); images and unused columns are ignored. Both datasets are keyed
by reset-state instance (A via instance_id, B via frame-0 red-xy) and aligned to
normalized episode progress, since the two pipelines produce different-length
trajectories for the same task.

Outputs:
  maniflow_diff_dist.png   per-dimension value distributions, A vs B
  maniflow_diff_traj.png   per-instance mean trajectory A vs B (joint_target/pos)
and prints a per-column divergence table.

    RG_OUT=scratchpad/red_green_2block/test1 .venv/bin/python scripts/debug/diff_datasets_maniflow.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("RG_OUT", "scratchpad/red_green_2block/test1"))
if not OUT.is_absolute():
    OUT = REPO / OUT

A_DIR = REPO / os.environ.get("RG_A_DIR", "scratchpad/red_green_2block/red_on_green_2block/dataset/blocks_success")
B_DIR = REPO / os.environ.get("RG_B_DIR", "scratchpad/red_green_2block/classic/data_matched/blocks_success")
CLASSIC = REPO / "scratchpad/red_green_2block/classic/red_on_green_2block"
RESET_RED_XY = np.array([
    json.loads((CLASSIC / f"{i+1}.json").read_text())["block_positions"]["block_red1"][:2]
    for i in range(8)
])

# maniflow-consumed numeric columns: (key, dim, short, group)
COLS = [
    ("observation.state.joint_position", 7, "state.jpos", "state(8-dim)"),
    ("observation.state.gripper_position", 1, "state.grip", "state(8-dim)"),
    ("action.joint_target", 7, "act.jtarget", "action(abs)"),
    ("action.gripper_position", 1, "act.grip", "action(abs)"),
    ("action.pd_mode", 1, "act.pd_mode", "action(abs)"),
    ("observation.state.joint_velocity", 7, "state.jvel", "state(19-dim only)"),
    ("observation.state.ee_position", 6, "state.ee", "state(19-dim only)"),
]
GRID = np.linspace(0, 1, 120)
TRIM = int(os.environ.get("RG_TRIM_TAIL", "3"))  # drop trailing zero-action frames


def load(ds_dir):
    pfs = sorted(ds_dir.glob("data/chunk-*/file-*.parquet"))
    fail = ds_dir.parent / "blocks_failure"
    if fail.exists():
        pfs += sorted(fail.glob("data/chunk-*/file-*.parquet"))
    per_inst = {}
    for pf in pfs:
        d = pq.read_table(pf).to_pydict()
        ep = np.asarray(d["episode_index"]).reshape(-1)
        for e in sorted(set(ep.tolist())):
            idx = np.flatnonzero(ep == e)
            red0 = np.asarray(d["observation.state.object_pose.block_red1"][idx[0]])[:2]
            if "instance_id" in d:
                inst = int(np.asarray(d["instance_id"][idx[0]]).reshape(-1)[0])
            else:
                inst = int(np.argmin(np.linalg.norm(RESET_RED_XY - red0, axis=1)))
            if inst in per_inst:
                continue
            keep = max(len(idx) - TRIM, 2)
            idx = idx[:keep]  # drop trailing zero-action frames
            cols = {}
            for key, dim, *_ in COLS:
                if key in d:
                    cols[key] = np.asarray([np.atleast_1d(d[key][i]) for i in idx], dtype=np.float64)
            per_inst[inst] = cols
    return per_inst


A, B = load(A_DIR), load(B_DIR)
insts = sorted(set(A) & set(B))
print(f"common instances: {insts}")

# ---- pooled value distributions per dimension ----
present = [(k, dim, short, grp) for (k, dim, short, grp) in COLS
           if all(k in A[i] and k in B[i] for i in insts)]
panels = sum(dim for _, dim, _, _ in present)
ncol = 7
nrow = int(np.ceil(panels / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.4 * nrow))
axes = np.atleast_1d(axes).flatten()
p = 0
rows = []
for key, dim, short, grp in present:
    Aall = np.concatenate([A[i][key] for i in insts], axis=0)  # (N,dim)
    Ball = np.concatenate([B[i][key] for i in insts], axis=0)
    for c in range(dim):
        ax = axes[p]; p += 1
        a, b = Aall[:, c], Ball[:, c]
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
        rng = (lo - 1e-6, hi + 1e-6)
        ax.hist(a, bins=40, range=rng, color="#1f77b4", alpha=0.55, density=True, label="A")
        ax.hist(b, bins=40, range=rng, color="#d62728", alpha=0.55, density=True, label="B")
        # 1-Wasserstein-ish: mean abs diff of sorted samples (equal-count resample)
        n = min(len(a), len(b))
        w1 = float(np.mean(np.abs(np.sort(np.random.default_rng(0).choice(a, n, replace=len(a) < n))
                                  - np.sort(np.random.default_rng(1).choice(b, n, replace=len(b) < n)))))
        ax.set_title(f"{short}[{c}]  W1={w1:.3f}", fontsize=7)
        ax.tick_params(labelsize=6)
        rows.append((f"{short}[{c}]", a.mean(), a.std(), b.mean(), b.std(), w1))
        if p == 1:
            ax.legend(fontsize=6)
for k in range(p, len(axes)):
    axes[k].axis("off")
fig.suptitle("ManiFlow-consumed columns: value distributions  A=blocksuite (blue)  B=classic (red)",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(OUT / "maniflow_diff_dist.png", dpi=120)
plt.close(fig)
print(f"wrote {OUT / 'maniflow_diff_dist.png'}")

# ---- per-instance mean trajectory diff: joint_target & joint_position ----
def traj_panel(ax, key, dimidx, title):
    Am, Bm = [], []
    for i in insts:
        a, b = A[i][key][:, dimidx], B[i][key][:, dimidx]
        Am.append(np.interp(GRID, np.linspace(0, 1, len(a)), a))
        Bm.append(np.interp(GRID, np.linspace(0, 1, len(b)), b))
    Am, Bm = np.array(Am), np.array(Bm)
    ax.fill_between(GRID, Am.min(0), Am.max(0), color="#1f77b4", alpha=0.15, lw=0)
    ax.fill_between(GRID, Bm.min(0), Bm.max(0), color="#d62728", alpha=0.15, lw=0)
    ax.plot(GRID, Am.mean(0), color="#1f77b4", lw=2, label="A blocksuite")
    ax.plot(GRID, Bm.mean(0), color="#d62728", lw=2, label="B classic")
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6); ax.grid(True, alpha=0.25)


fig, axes = plt.subplots(2, 7, figsize=(19, 6))
for j in range(7):
    traj_panel(axes[0, j], "action.joint_target", j, f"action.joint_target[{j}]")
    traj_panel(axes[1, j], "observation.state.joint_position", j, f"state.joint_position[{j}]")
axes[0, 0].legend(fontsize=7)
fig.suptitle("A vs B mean trajectory over episode progress (band = min..max across 8 instances)",
             fontsize=13)
fig.text(0.5, 0.5, "action.joint_target (top)   /   observation.state.joint_position (bottom)",
         ha="center", fontsize=9, color="0.4")
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT / "maniflow_diff_traj.png", dpi=120)
plt.close(fig)
print(f"wrote {OUT / 'maniflow_diff_traj.png'}")

# ---- printed divergence table ----
print(f"\n{'column':18s}{'A mean':>9}{'A std':>9}{'B mean':>9}{'B std':>9}{'W1 dist':>9}")
for name, am, asd, bm, bsd, w1 in rows:
    print(f"{name:18s}{am:9.3f}{asd:9.3f}{bm:9.3f}{bsd:9.3f}{w1:9.3f}")
