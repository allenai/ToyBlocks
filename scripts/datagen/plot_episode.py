#!/usr/bin/env python3
"""Plot all signals from a single episode in a LeRobot-style dataset.

Usage:
    python scripts/datagen/plot_episode.py \\
        --dataset data/datasets/merged/train/<experiment-id> \\
        --episode 0

    python scripts/datagen/plot_episode.py \\
        --dataset data/datasets/merged/train/<experiment-id> \\
        --episode 0 --output /tmp/my_plot.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from matplotlib.patches import Patch

JOINT_NAMES = [f"joint_{i}" for i in range(7)]

PARQUET_COLUMNS = [
    "episode_index",
    "frame_index",
    "timestamp",
    "observation.state.joint_position",
    "observation.state.joint_velocity",
    "observation.state.ee_position",
    "observation.state.gripper_position",
    "action.joint_target",
    "action.joint_position",
    "action.relative_joint_target",
    "action.ee_position",
    "action.gripper_position",
    "action.pd_mode",
    "action.high_level_action",
    "action.low_level_action",
    "pddl_goal",
    "is_first",
    "is_last",
]

OBJECT_POSE_PREFIX = "observation.state.object_pose."


def _to_array(series) -> np.ndarray:
    return np.stack([np.array(x, dtype=np.float32) for x in series])


def _to_scalar_array(series) -> np.ndarray:
    return np.array([float(x) if not hasattr(x, "__len__") else float(x[0]) for x in series])


def _discover_object_columns(all_columns: list[str]) -> list[str]:
    """Find all observation.state.object_pose.* columns, excluding workspace_table."""
    return [
        c for c in all_columns if c.startswith(OBJECT_POSE_PREFIX) and "workspace_table" not in c
    ]


def _add_event_lines(ax, pd_changes, grip_changes):
    for s in pd_changes:
        ax.axvline(s, color="blue", ls=":", alpha=0.4, lw=0.8)
    for s in grip_changes:
        ax.axvline(s, color="red", ls=":", alpha=0.4, lw=0.8)


def load_episode(dataset_root: Path, episode: int):
    """Load a single episode from parquet files, returning a pandas DataFrame."""
    parquet_files = sorted(dataset_root.glob("data/chunk-*/file-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")

    schema = pq.read_schema(str(parquet_files[0]))
    all_columns = schema.names
    obj_cols = _discover_object_columns(all_columns)
    keep = [c for c in PARQUET_COLUMNS if c in all_columns] + obj_cols

    tables = [pq.read_table(str(p), columns=keep) for p in parquet_files]
    df = pa.concat_tables(tables).to_pandas()
    ep = df[df["episode_index"] == episode].sort_values("frame_index").reset_index(drop=True)
    if ep.empty:
        raise ValueError(
            f"Episode {episode} not found. Available range: "
            f"{int(df['episode_index'].min())}–{int(df['episode_index'].max())}"
        )
    # A merged dataset carries the union of the problems' objects; the columns of objects
    # absent from this episode's scene are null, so plot only the ones it has.
    obj_cols = [c for c in obj_cols if ep[c].notna().all()]
    return ep, obj_cols


def plot_episode(ep, obj_cols: list[str], output_path: Path, episode: int):
    N = len(ep)
    steps = np.arange(N)

    joint_pos = _to_array(ep["observation.state.joint_position"])
    joint_vel = _to_array(ep["observation.state.joint_velocity"])
    ee_pos = _to_array(ep["observation.state.ee_position"])
    gripper_obs = _to_scalar_array(ep["observation.state.gripper_position"])
    action_jt = _to_array(ep["action.joint_target"])
    rel_jt = _to_array(ep["action.relative_joint_target"])
    act_ee = _to_array(ep["action.ee_position"])
    action_grip = _to_scalar_array(ep["action.gripper_position"])
    pd_mode = _to_scalar_array(ep["action.pd_mode"])

    pd_changes = np.where(np.diff(pd_mode) != 0)[0] + 1
    grip_changes = np.where(np.abs(np.diff(action_grip)) > 0.1)[0] + 1

    n_objects = len(obj_cols)
    n_rows = 8 + n_objects  # 8 fixed rows + one per object
    fig = plt.figure(figsize=(22, 4 * n_rows))
    gs = fig.add_gridspec(n_rows, 2, hspace=0.35, wspace=0.25)

    pddl_goal = str(ep["pddl_goal"].iloc[0]).strip()
    hl_actions = ep["action.high_level_action"].unique().tolist()
    fig.suptitle(
        f"Episode {episode} — {pddl_goal}\n({N} steps, HL: {hl_actions})",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    colors = plt.cm.tab10(np.linspace(0, 0.7, 7))

    # --- Row 0: joint positions (obs vs target) ---
    ax = fig.add_subplot(gs[0, :])
    for j in range(7):
        ax.plot(steps, joint_pos[:, j], color=colors[j], lw=1.2, label=f"obs j{j}")
        ax.plot(steps, action_jt[:, j], color=colors[j], lw=0.8, ls="--", alpha=0.6)
    _add_event_lines(ax, pd_changes, grip_changes)
    ax.set_ylabel("Position (rad)")
    ax.set_title("Joint Positions (solid=obs, dashed=action target)")
    ax.legend(fontsize=7, ncol=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    # --- Row 1: joint velocities ---
    ax = fig.add_subplot(gs[1, :])
    for j in range(7):
        ax.plot(steps, joint_vel[:, j], color=colors[j], lw=1.0, label=f"j{j}")
    _add_event_lines(ax, pd_changes, grip_changes)
    ax.set_ylabel("Velocity (rad/s)")
    ax.set_title("Joint Velocities")
    ax.legend(fontsize=7, ncol=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    # --- Row 2: EE position + orientation ---
    ax_xyz = fig.add_subplot(gs[2, 0])
    for i, label in enumerate(["x", "y", "z"]):
        ax_xyz.plot(steps, ee_pos[:, i], lw=1.2, label=label)
    ax_xyz.set_ylabel("Position (m)")
    ax_xyz.set_title("End-Effector Position (XYZ)")
    ax_xyz.legend(fontsize=8)
    ax_xyz.grid(True, alpha=0.3)

    ax_rpy = fig.add_subplot(gs[2, 1])
    for i, label in enumerate(["roll", "pitch", "yaw"]):
        ax_rpy.plot(steps, ee_pos[:, 3 + i], lw=1.2, label=label)
    ax_rpy.set_ylabel("Angle (rad)")
    ax_rpy.set_title("End-Effector Orientation (RPY)")
    ax_rpy.legend(fontsize=8)
    ax_rpy.grid(True, alpha=0.3)

    # --- Row 3: gripper + PD mode ---
    ax_g = fig.add_subplot(gs[3, 0])
    ax_g.plot(steps, gripper_obs, color="green", lw=1.5, label="gripper obs")
    ax_g.plot(steps, action_grip, color="orange", lw=1.2, ls="--", label="gripper action")
    ax_g.set_ylabel("Gripper pos")
    ax_g.set_title("Gripper Position")
    ax_g.legend(fontsize=8)
    ax_g.grid(True, alpha=0.3)

    ax_pd = fig.add_subplot(gs[3, 1])
    ax_pd.plot(steps, pd_mode, color="purple", lw=1.5)
    ax_pd.set_ylabel("PD mode")
    ax_pd.set_title("PD Mode")
    ax_pd.grid(True, alpha=0.3)
    ax_pd.set_yticks(sorted(set(pd_mode.astype(int))))

    # --- Row 4: object poses ---
    row = 4
    for _oi, col in enumerate(obj_cols):
        obj_name = col.removeprefix(OBJECT_POSE_PREFIX)
        obj_pose = _to_array(ep[col])

        ax_op = fig.add_subplot(gs[row, 0])
        for i, label in enumerate(["x", "y", "z"]):
            ax_op.plot(steps, obj_pose[:, i], lw=1.2, label=label)
        ax_op.set_ylabel("Position (m)")
        ax_op.set_title(f"{obj_name} Position (XYZ)")
        ax_op.legend(fontsize=8)
        ax_op.grid(True, alpha=0.3)

        ax_oq = fig.add_subplot(gs[row, 1])
        for i, label in enumerate(["qw", "qx", "qy", "qz"]):
            ax_oq.plot(steps, obj_pose[:, 3 + i], lw=1.2, label=label)
        ax_oq.set_ylabel("Quaternion")
        ax_oq.set_title(f"{obj_name} Orientation (Quaternion)")
        ax_oq.legend(fontsize=8)
        ax_oq.grid(True, alpha=0.3)
        row += 1

    # --- Relative joint target ---
    ax = fig.add_subplot(gs[row, :])
    for j in range(7):
        ax.plot(steps, rel_jt[:, j], color=colors[j], lw=1.0, label=f"j{j}")
    ax.set_ylabel("Relative target (rad)")
    ax.set_title("Relative Joint Target (Action)")
    ax.legend(fontsize=7, ncol=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", lw=0.5, alpha=0.3)
    row += 1

    # --- Action EE ---
    ax_ae_xyz = fig.add_subplot(gs[row, 0])
    for i, label in enumerate(["x", "y", "z"]):
        ax_ae_xyz.plot(steps, act_ee[:, i], lw=1.2, label=label)
    ax_ae_xyz.set_ylabel("Position (m)")
    ax_ae_xyz.set_title("Action EE Position (XYZ)")
    ax_ae_xyz.legend(fontsize=8)
    ax_ae_xyz.grid(True, alpha=0.3)

    ax_ae_rpy = fig.add_subplot(gs[row, 1])
    for i, label in enumerate(["roll", "pitch", "yaw"]):
        ax_ae_rpy.plot(steps, act_ee[:, 3 + i], lw=1.2, label=label)
    ax_ae_rpy.set_ylabel("Angle (rad)")
    ax_ae_rpy.set_title("Action EE Orientation (RPY)")
    ax_ae_rpy.legend(fontsize=8)
    ax_ae_rpy.grid(True, alpha=0.3)
    row += 1

    # --- Action labels timeline ---
    ax = fig.add_subplot(gs[row, :])
    hl = ep["action.high_level_action"].values
    ll = ep["action.low_level_action"].values
    unique_hl = sorted(set(hl))
    unique_ll = sorted(set(ll))
    hl_cmap = plt.cm.Set2(np.linspace(0, 1, max(len(unique_hl), 2)))
    ll_cmap = plt.cm.Set3(np.linspace(0, 1, max(len(unique_ll), 2)))
    hl_map = {a: i for i, a in enumerate(unique_hl)}
    ll_map = {a: i for i, a in enumerate(unique_ll)}

    for i in range(N):
        ax.barh(1.0, 1, left=i, height=0.3, color=hl_cmap[hl_map[hl[i]]], edgecolor="none")
        ax.barh(0.5, 1, left=i, height=0.3, color=ll_cmap[ll_map[ll[i]]], edgecolor="none")

    hl_patches = [Patch(facecolor=hl_cmap[hl_map[a]], label=a) for a in unique_hl]
    ll_patches = [Patch(facecolor=ll_cmap[ll_map[a]], label=a) for a in unique_ll]
    hl_leg = ax.legend(
        handles=hl_patches,
        title="High-level",
        title_fontsize=14,
        fontsize=12,
        ncol=len(unique_hl),
        loc="lower left",
        bbox_to_anchor=(0.0, -0.55),
    )
    ax.add_artist(hl_leg)
    ax.legend(
        handles=ll_patches,
        title="Low-level",
        title_fontsize=14,
        fontsize=12,
        ncol=min(len(unique_ll), 5),
        loc="lower right",
        bbox_to_anchor=(1.0, -0.55),
    )
    ax.set_yticks([0.5, 1.0])
    ax.set_yticklabels(["Low-level", "High-level"])
    ax.set_xlim(0, N)
    ax.set_xlabel("Step")
    ax.set_title("Action Labels")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot all signals from a single episode in a LeRobot-style dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the dataset root directory.",
    )
    parser.add_argument("--episode", type=int, default=0, help="Episode index to plot.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG path (default: <dataset>/episode_<N>.png).",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    if not dataset_root.is_dir():
        parser.error(f"Dataset directory not found: {dataset_root}")

    if args.output is None:
        output_path = dataset_root / f"episode_{args.episode}.png"
    else:
        output_path = Path(args.output)

    ep, obj_cols = load_episode(dataset_root, args.episode)
    print(
        f"Episode {args.episode}: {len(ep)} steps | "
        f"PDDL: {ep['pddl_goal'].iloc[0].strip()} | "
        f"Objects: {[c.removeprefix(OBJECT_POSE_PREFIX) for c in obj_cols]}"
    )
    plot_episode(ep, obj_cols, output_path, args.episode)


if __name__ == "__main__":
    main()
