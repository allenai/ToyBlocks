"""Matplotlib plotting and visual diagnostics for evaluation runs."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


_OBJECT_POSE_PREFIX = "observation.state.object_pose."
_CAMERA_POSE_PREFIX = "observation.camera_pose."
_CAMERA_INTRINSICS_PREFIX = "observation.camera_intrinsics."
_DEFAULT_CUBE_SIDE_LENGTH = 0.04
_N_ARM_JOINTS = 7
# Max distance (m) from the block's surface, per `_distance_to_box_surface`, for a
# fingertip to count as having made contact in `_first_contact_point`. Must stay
# well above the sim's frame-to-frame approach distance (observed true contact
# lands within ~0.03-0.6mm) but well below the block's own half-size, or "first
# contact" fires on an approach frame nowhere near the surface instead of an
# actual touch.
_CONTACT_DISTANCE_THRESHOLD = 0.008

# Cube corner signs in the object's local frame, matching
# blocksuite.utils.obs_utils.ENDPOINT_SIGNS so corner indices line up with the
# online (Isaac Lab) computation.
_CORNER_SIGNS = np.array(
    [
        [-1, -1, +1],
        [-1, +1, +1],
        [+1, -1, +1],
        [+1, +1, +1],
        [+1, +1, -1],
        [+1, -1, -1],
        [-1, +1, -1],
        [-1, -1, -1],
    ],
    dtype=np.float32,
)
_CORNER_LABELS = [
    "(-x,-y,+z)", "(-x,+y,+z)", "(+x,-y,+z)", "(+x,+y,+z)",
    "(+x,+y,-z)", "(+x,-y,-z)", "(-x,+y,-z)", "(-x,-y,-z)",
]

_ROBOT_POSE_PREFIX = "robot.poses."
_LEFT_FINGER_LINK = "left_inner_finger"
_RIGHT_FINGER_LINK = "right_inner_finger"


_LEFT_FINGERTIP_POSITION_IN_LEFT_FINGER_LINK_FRAME = np.array([0.0, 0.0, 0.1388], dtype=np.float32)
_RIGHT_FINGERTIP_POSITION_IN_RIGHT_FINGER_LINK_FRAME = np.array([0.0, 0.0, 0.1388], dtype=np.float32)

# Node markers for time-series line plots, so each plotted line shows the exact
# steps at which the underlying dataset actually has a measurement rather than
# just the interpolated line connecting them. Sim and GT get distinct shapes
# since they otherwise share solid-vs-dashed linestyles per plot.
_SIM_MARKER = "o"
_GT_MARKER = "x"
_MARKER_SIZE = 3.5


def _fingertip_positions(
    finger_pose_in_robot_base_frame: np.ndarray,
    fingertip_position_in_finger_link_frame: np.ndarray,
) -> np.ndarray:
    """Fingertip position ``(n, 3)`` in the robot base frame.

    ``finger_pose_in_robot_base_frame`` is the finger link's own pose ``(n, 7)``
    (position + wxyz quaternion), in the robot base frame.
    ``fingertip_position_in_finger_link_frame`` is the fixed offset ``(3,)`` from
    that link's origin to the fingertip, expressed in the finger link's own local
    frame (see the derivation comment above ``_LEFT_FINGERTIP_POSITION_IN_LEFT_FINGER_LINK_FRAME`` /
    ``_RIGHT_FINGERTIP_POSITION_IN_RIGHT_FINGER_LINK_FRAME``). It is rotated into the robot base frame by
    the finger link's own orientation and added to the finger link's position.
    """
    from blocksuite.utils.tf import quat_rotate_vec_wxyz

    finger_position_in_robot_base_frame = finger_pose_in_robot_base_frame[:, :3]
    finger_orientation_in_robot_base_frame = finger_pose_in_robot_base_frame[:, 3:7]
    return finger_position_in_robot_base_frame + quat_rotate_vec_wxyz(
        finger_orientation_in_robot_base_frame, fingertip_position_in_finger_link_frame
    )


def _distance_to_box_surface(fingertip_position_in_cube_frame: np.ndarray, half_size: float) -> np.ndarray:
    """Euclidean distance ``(n,)`` from each point in ``fingertip_position_in_cube_frame``
    ``(n, 3)`` to the surface of a cube of half-extent ``half_size`` centered at the
    origin of that same frame. Always ``>= 0``: for a point outside the cube this is
    the ordinary nearest-point-on-box distance, for a point that has penetrated the
    cube it's the distance back out to the nearest face.

    The input must already be expressed in the cube's own local/object frame (i.e.
    the caller has already rotated it by the cube's actual quaternion, as
    ``_first_contact_point`` does via ``world_pose_to_local``) — this function itself
    only ever sees an axis-aligned cube, so the cube it was derived from need not be
    axis-aligned in the robot base frame.
    """
    q = np.abs(fingertip_position_in_cube_frame) - half_size
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    penetration = -np.minimum(np.max(q, axis=-1), 0.0)
    return outside + penetration


def _first_contact_point(
    finger_pose_in_robot_base_frame: Optional[np.ndarray],
    block_pose_in_robot_base_frame: Optional[np.ndarray],
    half_size: float,
    fingertip_position_in_finger_link_frame: np.ndarray,
) -> Optional[tuple[int, np.ndarray, np.ndarray]]:
    """First ``(frame_index, fingertip_xyz, fingertip_xyz_in_block_frame)`` at which a
    fingertip's distance to the block's surface (see ``_distance_to_box_surface``)
    drops to ``_CONTACT_DISTANCE_THRESHOLD`` or below, or ``None`` if it never does.
    ``fingertip_xyz`` is in the robot base frame; ``fingertip_xyz_in_block_frame`` is
    the same point re-expressed in the block's own local frame (i.e. with the block's
    pose at that frame divided out), which is what lets a contact point be compared
    against the block's fixed footprint regardless of how the block itself is posed.

    ``finger_pose_in_robot_base_frame`` and ``block_pose_in_robot_base_frame`` must
    both be in the robot base frame (per the rest of this module).
    ``fingertip_position_in_finger_link_frame`` is the fixed local offset passed
    through to ``_fingertip_positions`` to locate the fingertip on that finger link.
    The fingertip is then transformed from the robot base frame into the block's own
    local frame (via ``world_pose_to_local``, using the block's recorded orientation)
    before the distance is computed, so the block's actual orientation — it need not
    be axis-aligned in the robot base frame — is accounted for exactly.
    """
    from blocksuite.utils.tf import world_pose_to_local

    if finger_pose_in_robot_base_frame is None or block_pose_in_robot_base_frame is None:
        return None
    n = min(len(finger_pose_in_robot_base_frame), len(block_pose_in_robot_base_frame))
    if n == 0:
        return None
    fingertip_positions_in_robot_base_frame = _fingertip_positions(
        finger_pose_in_robot_base_frame[:n], fingertip_position_in_finger_link_frame
    )
    # The fingertip is a fixed translation off the finger link, not an independent
    # body, so it shares the finger link's orientation exactly.
    fingertip_orientation_in_robot_base_frame = finger_pose_in_robot_base_frame[:n, 3:7]
    fingertip_pose_in_robot_base_frame = np.concatenate(
        [fingertip_positions_in_robot_base_frame, fingertip_orientation_in_robot_base_frame], axis=-1
    )
    block_pose_in_robot_base_frame = block_pose_in_robot_base_frame[:n]
    fingertip_position_in_cube_frame = world_pose_to_local(
        fingertip_pose_in_robot_base_frame, block_pose_in_robot_base_frame[:, :3], block_pose_in_robot_base_frame[:, 3:7]
    )[:, :3]
    contact = _distance_to_box_surface(fingertip_position_in_cube_frame, half_size) <= _CONTACT_DISTANCE_THRESHOLD
    if not contact.any():
        return None
    idx = int(np.argmax(contact))
    return idx, fingertip_positions_in_robot_base_frame[idx], fingertip_position_in_cube_frame[idx]


def _side_first_contact_step(
    contacts: list[tuple[str, int, np.ndarray, np.ndarray, str, str, Optional[np.ndarray]]],
    side_label: str,
) -> Optional[int]:
    """Earliest ``frame_idx`` among ``contacts`` whose label starts with ``side_label``
    (``"Sim"`` or ``"GT"``), i.e. across whichever finger(s) made contact first on that
    side, or ``None`` if that side made no contact at all.
    """
    steps = [frame_idx for label, frame_idx, *_ in contacts if label.startswith(side_label)]
    return min(steps) if steps else None


def _block_pose_columns(columns) -> dict[str, str]:
    """Column name -> block name, for every non-table ``object_pose`` feature."""
    out: dict[str, str] = {}
    for col in columns:
        if not col.startswith(_OBJECT_POSE_PREFIX):
            continue
        name = col[len(_OBJECT_POSE_PREFIX):]
        if "table" in name.lower():
            continue
        out[col] = name
    return out


def _cube_corners(poses: np.ndarray, half_size: float) -> np.ndarray:
    """8 cube corners per step in the pose's own frame: ``(n, 7)`` -> ``(n, 8, 3)``."""
    from blocksuite.utils.tf import quat_rotate_vec_wxyz

    pos = poses[:, :3]
    quat = poses[:, 3:7]
    corners = np.empty((poses.shape[0], 8, 3), dtype=np.float32)
    for i, sign in enumerate(_CORNER_SIGNS):
        corners[:, i, :] = pos + quat_rotate_vec_wxyz(quat, sign * half_size)
    return corners


def _save_block_pose_comparison(
    *,
    output_dir: str,
    block_name: str,
    eval_block_pose_in_robot_base_frame: Optional[np.ndarray],
    gt_block_pose_in_robot_base_frame: Optional[np.ndarray],
    eval_side_length: float,
    gt_side_length: float,
    steps_arr: np.ndarray,
    gripper_close_steps: Sequence[int] = (),
    gripper_open_steps: Sequence[int] = (),
    eval_friction_properties: Optional[np.ndarray] = None,
    gt_friction_properties: Optional[np.ndarray] = None,
    eval_table_friction: Optional[float] = None,
    gt_table_friction: Optional[float] = None,
    eval_table_height_delta: Optional[float] = None,
    gt_table_height_delta: Optional[float] = None,
    eval_object_mass: Optional[float] = None,
    gt_object_mass: Optional[float] = None,
    eval_left_finger_pose_in_robot_base_frame: Optional[np.ndarray] = None,
    eval_right_finger_pose_in_robot_base_frame: Optional[np.ndarray] = None,
    gt_left_finger_pose_in_robot_base_frame: Optional[np.ndarray] = None,
    gt_right_finger_pose_in_robot_base_frame: Optional[np.ndarray] = None,
) -> Optional[list[tuple[str, int, np.ndarray, np.ndarray, str, str, Optional[np.ndarray]]]]:
    """One ``<block_name>_pose_comparison.png``: 8 corners, sim vs GT (X/Y/Z),
    a row comparing the sim vs GT physics parameters (block friction,
    restitution, table friction, table height, cube side length, object mass)
    in effect for the episode, and (when finger-link poses were recorded) a row giving
    the (X, Y, Z) of each fingertip's first contact with this block, sim and
    GT computed independently since their contact events can land on
    different frames (or not happen at all) for either side.

    Sim and GT corners are reconstructed from each side's own recorded cube
    scale, so a real size mismatch between the two datasets shows up as a
    difference in corner spread rather than being masked by a shared scale.

    Returns the same ``(label, frame_idx, point_in_robot_base_frame,
    point_in_block_frame, color, marker, corners_at_frame)`` contact list used
    for the contact row here, so callers (``save_eval_plots``) can feed it to
    ``_save_contact_heatmap`` without recomputing first-contact detection.
    ``None`` when neither side had corners to plot at all.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eval_corners = (
        _cube_corners(eval_block_pose_in_robot_base_frame, eval_side_length / 2.0)
        if eval_block_pose_in_robot_base_frame is not None else None
    )
    gt_corners = (
        _cube_corners(gt_block_pose_in_robot_base_frame, gt_side_length / 2.0)
        if gt_block_pose_in_robot_base_frame is not None else None
    )
    if eval_corners is None and gt_corners is None:
        return None

    def _friction_component(props: Optional[np.ndarray], idx: int) -> Optional[float]:
        return float(props[idx]) if props is not None and len(props) > idx else None

    physics_specs = [
        ("Static Friction", _friction_component(eval_friction_properties, 0), _friction_component(gt_friction_properties, 0)),
        ("Dynamic Friction", _friction_component(eval_friction_properties, 1), _friction_component(gt_friction_properties, 1)),
        ("Restitution", _friction_component(eval_friction_properties, 2), _friction_component(gt_friction_properties, 2)),
        ("Table Friction", eval_table_friction, gt_table_friction),
        ("Table Height Delta (m)", eval_table_height_delta, gt_table_height_delta),
        ("Cube Side Length (m)", eval_side_length, gt_side_length),
        ("Object Mass (kg)", eval_object_mass, gt_object_mass),
    ]

    # First-contact point per (source, finger): each entry is only present
    # when that fingertip's signed distance to the block surface reaches zero
    # or below at some frame. Each candidate also carries the same source's
    # corner-position array (in world/robot-base X/Y/Z, computed above for the
    # main 8-corner plots) so the contact panel can draw that source's true
    # cube extent at the contact frame, rather than assuming the block is
    # axis-aligned.
    contact_candidates = [
        ("Sim left finger", eval_left_finger_pose_in_robot_base_frame, eval_block_pose_in_robot_base_frame, eval_side_length / 2.0, "blue", "o", _LEFT_FINGERTIP_POSITION_IN_LEFT_FINGER_LINK_FRAME, eval_corners),
        ("Sim right finger", eval_right_finger_pose_in_robot_base_frame, eval_block_pose_in_robot_base_frame, eval_side_length / 2.0, "blue", "s", _RIGHT_FINGERTIP_POSITION_IN_RIGHT_FINGER_LINK_FRAME, eval_corners),
        ("GT left finger", gt_left_finger_pose_in_robot_base_frame, gt_block_pose_in_robot_base_frame, gt_side_length / 2.0, "red", "o", _LEFT_FINGERTIP_POSITION_IN_LEFT_FINGER_LINK_FRAME, gt_corners),
        ("GT right finger", gt_right_finger_pose_in_robot_base_frame, gt_block_pose_in_robot_base_frame, gt_side_length / 2.0, "red", "s", _RIGHT_FINGERTIP_POSITION_IN_RIGHT_FINGER_LINK_FRAME, gt_corners),
    ]
    has_finger_data = any(fp is not None for _, fp, *_ in contact_candidates)
    contacts: list[tuple[str, int, np.ndarray, np.ndarray, str, str, Optional[np.ndarray]]] = []
    for label, finger_pose, obj_pose, half_size, color, marker, local_offset, corners in contact_candidates:
        result = _first_contact_point(finger_pose, obj_pose, half_size, local_offset)
        if result is not None:
            frame_idx, point, local_point = result
            corners_at_frame = corners[frame_idx] if corners is not None else None
            contacts.append((label, frame_idx, point, local_point, color, marker, corners_at_frame))
    sim_contact_step = _side_first_contact_step(contacts, "Sim")
    gt_contact_step = _side_first_contact_step(contacts, "GT")

    axis_colors = {"X": "tab:red", "Y": "tab:green", "Z": "tab:blue"}
    n_physics_cols = 3
    n_physics_rows = math.ceil(len(physics_specs) / n_physics_cols)
    n_contact_rows = 1 if has_finger_data else 0
    fig = plt.figure(figsize=(14, 26 + 3 * n_physics_rows + 3.5 * n_contact_rows))
    gs = fig.add_gridspec(
        8 + n_physics_rows + n_contact_rows, n_physics_cols, hspace=0.6, wspace=0.4, top=0.975,
        height_ratios=[1] * 8 + [0.5] * n_physics_rows + [0.8] * n_contact_rows,
    )
    axes = [fig.add_subplot(gs[i, :]) for i in range(8)]
    fig.suptitle(
        f"Block '{block_name}' Corner Positions (robot base frame): Sim vs Ground Truth",
        fontsize=14,
        y=0.995,
    )
    for i in range(8):
        ax = axes[i]
        if eval_corners is not None:
            n = min(len(eval_corners), len(steps_arr))
            for axis_idx, axis_name in enumerate(("X", "Y", "Z")):
                ax.plot(
                    steps_arr[:n], eval_corners[:n, i, axis_idx],
                    color=axis_colors[axis_name], linewidth=1.3, linestyle="-",
                    marker=_SIM_MARKER, markersize=_MARKER_SIZE,
                    label=f"Sim {axis_name}",
                )
        if gt_corners is not None:
            n = min(len(gt_corners), len(steps_arr))
            for axis_idx, axis_name in enumerate(("X", "Y", "Z")):
                ax.plot(
                    steps_arr[:n], gt_corners[:n, i, axis_idx],
                    color=axis_colors[axis_name], linewidth=1.3, linestyle="--",
                    marker=_GT_MARKER, markersize=_MARKER_SIZE,
                    label=f"GT {axis_name}",
                )
        for s in gripper_close_steps:
            ax.axvline(x=s, color="red", linestyle="-", alpha=0.5, linewidth=0.8,
                       label="Gripper close" if i == 0 and s == gripper_close_steps[0] else "")
        for s in gripper_open_steps:
            ax.axvline(x=s, color="blue", linestyle="-", alpha=0.5, linewidth=0.8,
                       label="Gripper open" if i == 0 and s == gripper_open_steps[0] else "")
        if sim_contact_step is not None:
            ax.axvline(x=sim_contact_step, color="darkgreen", linestyle="-", alpha=0.7, linewidth=1.0,
                       label="Sim first contact" if i == 0 else "")
        if gt_contact_step is not None:
            ax.axvline(x=gt_contact_step, color="darkorange", linestyle="--", alpha=0.7, linewidth=1.0,
                       label="GT first contact" if i == 0 else "")
        ax.set_xlabel("Step")
        ax.set_ylabel("Position (m)")
        ax.set_title(f"Corner {i + 1} {_CORNER_LABELS[i]}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7, ncol=2)

    for q, (label, eval_value, gt_value) in enumerate(physics_specs):
        ax = fig.add_subplot(gs[8 + q // n_physics_cols, q % n_physics_cols])
        values = [v for v in (eval_value, gt_value) if v is not None]
        if eval_value is not None:
            ax.axhline(y=eval_value, color="blue", linewidth=1.5, linestyle="-", label=f"Sim: {eval_value:.4g}")
        if gt_value is not None:
            ax.axhline(y=gt_value, color="red", linewidth=1.5, linestyle="--", label=f"GT: {gt_value:.4g}")
        if values:
            lo, hi = min(values), max(values)
            pad = max(hi - lo, abs(hi) * 0.1, 1e-3)
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(steps_arr[0] if len(steps_arr) else 0, steps_arr[-1] if len(steps_arr) else 1)
        ax.set_xlabel("Step")
        ax.set_title(label, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7)

    if has_finger_data:
        contact_row = 8 + n_physics_rows
        for axis_idx, axis_name in enumerate(("X", "Y", "Z")):
            ax = fig.add_subplot(gs[contact_row, axis_idx])
            if contacts:
                xs = np.arange(len(contacts))
                ys = [point[axis_idx] for _, _, point, _, _, _, _ in contacts]
                edge_extents: list[float] = []
                for x, y, (label, frame_idx, _, _, color, marker, corners_at_frame) in zip(xs, ys, contacts):
                    if corners_at_frame is not None:
                        edge_lo = float(corners_at_frame[:, axis_idx].min())
                        edge_hi = float(corners_at_frame[:, axis_idx].max())
                        ax.plot(
                            [x, x], [edge_lo, edge_hi],
                            color=color, linewidth=6, alpha=0.3, solid_capstyle="butt",
                            zorder=1, label="Cube edge (this axis)" if x == xs[0] else "",
                        )
                        edge_extents.extend([edge_lo, edge_hi])
                    ax.scatter([x], [y], color=color, marker=marker, s=110, zorder=3, edgecolors="black", linewidths=0.5)
                    ax.annotate(
                        f"{y:.4f}\n(step {frame_idx})",
                        xy=(x, y), xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=7,
                    )
                ax.set_xticks(xs)
                ax.set_xticklabels([label for label, *_ in contacts], fontsize=8, rotation=15)
                ax.set_xlim(-0.5, len(contacts) - 0.5)
                lo, hi = min(ys + edge_extents), max(ys + edge_extents)
                pad = max(hi - lo, 1e-2)
                ax.set_ylim(lo - pad, hi + pad * 1.4)
                if edge_extents:
                    ax.legend(loc="upper right", fontsize=6)
            else:
                ax.text(0.5, 0.5, "No contact detected", ha="center", va="center",
                         transform=ax.transAxes, fontsize=9)
                ax.set_xticks([])
            ax.set_ylabel(f"{axis_name} (m)")
            ax.set_title(f"First Contact Point — {axis_name}")
            ax.grid(True, alpha=0.3, axis="y")

    out_path = os.path.join(output_dir, f"{block_name}_pose_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved block pose comparison plot to: {out_path}")
    return contacts


def _convex_hull_xy(points_xy: np.ndarray) -> np.ndarray:
    """Convex hull of a set of 2-D points (monotone chain), used to draw a
    cube's true top-down footprint when the cube may be tilted — not just
    yawed — in the frame being plotted, so its corners no longer project to an
    axis-aligned square.
    """
    pts = sorted(set(map(tuple, np.asarray(points_xy, dtype=np.float64).tolist())))
    if len(pts) <= 2:
        return np.array(pts, dtype=np.float32)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], dtype=np.float32)


def _save_contact_heatmap(
    *,
    output_dir: str,
    block_name: str,
    # Each tuple: (label, frame_idx, point_in_robot_base_frame [the (x, y)
    # actually plotted], point_in_block_local_frame [unused here], color,
    # marker, corners_at_frame_in_robot_base_frame). See _first_contact_point.
    contacts: list[tuple[str, int, np.ndarray, np.ndarray, str, str, Optional[np.ndarray]]],
) -> None:
    """One ``<block_name>_contact_heatmap.png``: each fingertip's first-contact
    ``(X, Y)`` in the robot base frame (from ``contacts``, as returned by
    ``_save_block_pose_comparison``), overlaid on that side's block footprint —
    the convex hull of its 8 corners projected onto X/Y, at the frame of that
    side's first-listed contact — also in the robot base frame. Sim and GT
    each draw their own footprint, since the block's actual pose (and hence
    its footprint's position, size, and rotation) can differ between the two.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not contacts:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    footprints: list[np.ndarray] = []
    for side_label, color, linestyle in (("Sim", "blue", "-"), ("GT", "red", "--")):
        side_contact = next((c for c in contacts if c[0].startswith(side_label)), None)
        if side_contact is None or side_contact[6] is None:
            continue
        _, frame_idx, _, _, _, _, corners_at_frame = side_contact
        hull = _convex_hull_xy(corners_at_frame[:, :2])
        footprints.append(hull)
        ax.add_patch(plt.Polygon(
            hull, closed=True, fill=False, edgecolor=color, linewidth=1.5,
            linestyle=linestyle, label=f"{side_label} cube footprint (step {frame_idx})",
        ))

    for label, frame_idx, point, _, color, marker, _ in contacts:
        x, y = float(point[0]), float(point[1])
        ax.scatter([x], [y], color=color, marker=marker, s=130, zorder=3,
                   edgecolors="black", linewidths=0.6)
        ax.annotate(
            f"{label}\n(step {frame_idx})",
            xy=(x, y), xytext=(6, 6), textcoords="offset points", fontsize=7,
        )

    # Center + extent must cover every plotted point and footprint — a
    # near-miss contact can legitimately land further from the block than its
    # own footprint, and clipping it out of view would silently hide a
    # detected contact.
    all_xy = np.concatenate(
        [np.array([[float(p[0]), float(p[1])] for _, _, p, *_ in contacts], dtype=np.float32)]
        + footprints
    )
    center = all_xy.mean(axis=0)
    extent = max(float(np.abs(all_xy - center).max()), 1e-3)
    pad = extent * 0.3
    ax.set_xlim(center[0] - extent - pad, center[0] + extent + pad)
    ax.set_ylim(center[1] - extent - pad, center[1] + extent + pad)
    ax.set_aspect("equal")
    # set_aspect alone gets undone by savefig's bbox_inches="tight" below, which
    # crops away the whitespace equal-aspect padding relies on and stretches the
    # plot back out; set_box_aspect fixes the axes box itself to be square so it
    # survives that crop.
    ax.set_box_aspect(1)
    ax.set_xlabel("X (m, robot base frame)")
    ax.set_ylabel("Y (m, robot base frame)")
    ax.set_title(f"Block '{block_name}' First-Contact (X, Y), robot base frame")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=7)

    out_path = os.path.join(output_dir, f"{block_name}_contact_heatmap.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved contact heatmap plot to: {out_path}")


def _save_block_pose_error_plot(
    *,
    output_dir: str,
    block_name: str,
    eval_pose: Optional[np.ndarray],
    gt_pose: Optional[np.ndarray],
    eval_side_length: float,
    gt_side_length: float,
    steps_arr: np.ndarray,
    gripper_close_steps: Sequence[int] = (),
    gripper_open_steps: Sequence[int] = (),
    sim_contact_step: Optional[int] = None,
    gt_contact_step: Optional[int] = None,
) -> None:
    """One ``<block_name>_pose_errors.png``: 8 corners, GT-minus-sim error (X/Y/Z).

    Reconstructing each side's corners from its own cube scale means this
    error includes any genuine scale mismatch between the two datasets, not
    just a pose offset.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if eval_pose is None or gt_pose is None:
        return

    eval_corners = _cube_corners(eval_pose, eval_side_length / 2.0)
    gt_corners = _cube_corners(gt_pose, gt_side_length / 2.0)
    n = min(len(eval_corners), len(gt_corners), len(steps_arr))
    errors = gt_corners[:n] - eval_corners[:n]

    axis_colors = {"X": "tab:red", "Y": "tab:green", "Z": "tab:blue"}
    fig, axes = plt.subplots(8, 1, figsize=(14, 26))
    fig.suptitle(
        f"Block '{block_name}' Corner Position Error (GT - Sim, robot base frame)",
        fontsize=14,
        y=0.995,
    )
    for i in range(8):
        ax = axes[i]
        for axis_idx, axis_name in enumerate(("X", "Y", "Z")):
            ax.plot(
                steps_arr[:n], errors[:, i, axis_idx],
                color=axis_colors[axis_name], linewidth=1.3, linestyle="-",
                marker=_SIM_MARKER, markersize=_MARKER_SIZE,
                label=f"{axis_name} error",
            )
        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        for s in gripper_close_steps:
            ax.axvline(x=s, color="red", linestyle="-", alpha=0.5, linewidth=0.8,
                       label="Gripper close" if i == 0 and s == gripper_close_steps[0] else "")
        for s in gripper_open_steps:
            ax.axvline(x=s, color="blue", linestyle="-", alpha=0.5, linewidth=0.8,
                       label="Gripper open" if i == 0 and s == gripper_open_steps[0] else "")
        if sim_contact_step is not None:
            ax.axvline(x=sim_contact_step, color="darkgreen", linestyle="-", alpha=0.7, linewidth=1.0,
                       label="Sim first contact" if i == 0 else "")
        if gt_contact_step is not None:
            ax.axvline(x=gt_contact_step, color="darkorange", linestyle="--", alpha=0.7, linewidth=1.0,
                       label="GT first contact" if i == 0 else "")
        ax.set_xlabel("Step")
        ax.set_ylabel("Error (m)")
        ax.set_title(f"Corner {i + 1} {_CORNER_LABELS[i]}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7, ncol=2)

    plt.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = os.path.join(output_dir, f"{block_name}_pose_errors.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved block pose error plot to: {out_path}")


def _save_fingertip_comparison(
    *,
    output_dir: str,
    eval_left_pose: Optional[np.ndarray],  # (n, 7) [x, y, z, qw, qx, qy, qz], robot base frame
    eval_right_pose: Optional[np.ndarray],  # (n, 7) [x, y, z, qw, qx, qy, qz], robot base frame
    gt_left_pose: Optional[np.ndarray],  # (n, 7) [x, y, z, qw, qx, qy, qz], robot base frame
    gt_right_pose: Optional[np.ndarray],  # (n, 7) [x, y, z, qw, qx, qy, qz], robot base frame
    steps_arr: np.ndarray,
    eval_gripper_stiffness: Optional[np.ndarray] = None,
    eval_gripper_damping: Optional[np.ndarray] = None,
    eval_gripper_effort_limit: Optional[np.ndarray] = None,
    gt_gripper_stiffness: Optional[np.ndarray] = None,
    gt_gripper_damping: Optional[np.ndarray] = None,
    gt_gripper_effort_limit: Optional[np.ndarray] = None,
    gripper_close_steps: Sequence[int] = (),
    gripper_open_steps: Sequence[int] = (),
    sim_contact_step: Optional[int] = None,
    gt_contact_step: Optional[int] = None,
) -> None:
    """``fingertip_comparison.png``: left/right fingertip position (sim vs GT)
    plus their GT-minus-sim error, the GT-minus-sim error of the raw finger
    LINK pose (i.e. without the ``_{LEFT,RIGHT}_FINGERTIP_POSITION_IN_*_FRAME``
    local offset applied for the fingertip panels above), and the gripper's
    own PD gains (sim vs GT). Both error flavors are in the robot base frame."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sides = [
        ("Left fingertip", eval_left_pose, gt_left_pose, _LEFT_FINGERTIP_POSITION_IN_LEFT_FINGER_LINK_FRAME),
        ("Right fingertip", eval_right_pose, gt_right_pose, _RIGHT_FINGERTIP_POSITION_IN_RIGHT_FINGER_LINK_FRAME),
    ]
    gain_specs = [
        (eval_gripper_stiffness, gt_gripper_stiffness, "Stiffness", "N·m/rad", True),
        (eval_gripper_damping, gt_gripper_damping, "Damping", "N·m·s/rad", True),
        (eval_gripper_effort_limit, gt_gripper_effort_limit, "Effort limit", "N·m", False),
    ]
    has_gains = any(
        (eval_arr is not None and len(eval_arr) > 0) or (gt_arr is not None and len(gt_arr) > 0)
        for eval_arr, gt_arr, *_ in gain_specs
    )
    if all(eval_pose is None and gt_pose is None for _, eval_pose, gt_pose, _ in sides) and not has_gains:
        return

    axis_colors = {"X": "tab:red", "Y": "tab:green", "Z": "tab:blue"}
    n_rows = 7 if has_gains else 6
    fig = plt.figure(figsize=(14, 27 + (3.5 if has_gains else 0)))
    height_ratios = [1, 1, 1, 1, 1, 1, 0.6] if has_gains else None
    gs = fig.add_gridspec(n_rows, 3, hspace=0.6, wspace=0.3, top=0.965, height_ratios=height_ratios)
    axes = [fig.add_subplot(gs[i, :]) for i in range(6)]
    fig.suptitle(
        "Fingertip Position (robot base frame): Sim vs Ground Truth",
        fontsize=14,
        y=0.995,
    )
    for i, (title, eval_pose, gt_pose, local_offset) in enumerate(sides):
        ax = axes[i]
        eval_tip = _fingertip_positions(eval_pose, local_offset) if eval_pose is not None else None
        gt_tip = _fingertip_positions(gt_pose, local_offset) if gt_pose is not None else None
        if eval_tip is not None:
            n = min(len(eval_tip), len(steps_arr))
            for axis_idx, axis_name in enumerate(("X", "Y", "Z")):
                ax.plot(
                    steps_arr[:n], eval_tip[:n, axis_idx],
                    color=axis_colors[axis_name], linewidth=1.3, linestyle="-",
                    marker=_SIM_MARKER, markersize=_MARKER_SIZE,
                    label=f"Sim {axis_name}",
                )
        if gt_tip is not None:
            n = min(len(gt_tip), len(steps_arr))
            for axis_idx, axis_name in enumerate(("X", "Y", "Z")):
                ax.plot(
                    steps_arr[:n], gt_tip[:n, axis_idx],
                    color=axis_colors[axis_name], linewidth=1.3, linestyle="--",
                    marker=_GT_MARKER, markersize=_MARKER_SIZE,
                    label=f"GT {axis_name}",
                )
        for s in gripper_close_steps:
            ax.axvline(x=s, color="red", linestyle="-", alpha=0.5, linewidth=0.8,
                       label="Gripper close" if i == 0 and s == gripper_close_steps[0] else "")
        for s in gripper_open_steps:
            ax.axvline(x=s, color="blue", linestyle="-", alpha=0.5, linewidth=0.8,
                       label="Gripper open" if i == 0 and s == gripper_open_steps[0] else "")
        if sim_contact_step is not None:
            ax.axvline(x=sim_contact_step, color="darkgreen", linestyle="-", alpha=0.7, linewidth=1.0,
                       label="Sim first contact" if i == 0 else "")
        if gt_contact_step is not None:
            ax.axvline(x=gt_contact_step, color="darkorange", linestyle="--", alpha=0.7, linewidth=1.0,
                       label="GT first contact" if i == 0 else "")
        ax.set_xlabel("Step")
        ax.set_ylabel("Position (m)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, ncol=2)

    for i, (title, eval_pose, gt_pose, local_offset) in enumerate(sides):
        ax = axes[2 + i]
        eval_tip = _fingertip_positions(eval_pose, local_offset) if eval_pose is not None else None
        gt_tip = _fingertip_positions(gt_pose, local_offset) if gt_pose is not None else None
        if eval_tip is not None and gt_tip is not None:
            n = min(len(eval_tip), len(gt_tip), len(steps_arr))
            error = gt_tip[:n] - eval_tip[:n]
            for axis_idx, axis_name in enumerate(("X", "Y", "Z")):
                ax.plot(
                    steps_arr[:n], error[:, axis_idx],
                    color=axis_colors[axis_name], linewidth=1.3, linestyle="-",
                    marker=_SIM_MARKER, markersize=_MARKER_SIZE,
                    label=f"{axis_name} error",
                )
        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        for s in gripper_close_steps:
            ax.axvline(x=s, color="red", linestyle="-", alpha=0.5, linewidth=0.8)
        for s in gripper_open_steps:
            ax.axvline(x=s, color="blue", linestyle="-", alpha=0.5, linewidth=0.8)
        if sim_contact_step is not None:
            ax.axvline(x=sim_contact_step, color="darkgreen", linestyle="-", alpha=0.7, linewidth=1.0)
        if gt_contact_step is not None:
            ax.axvline(x=gt_contact_step, color="darkorange", linestyle="--", alpha=0.7, linewidth=1.0)
        ax.set_xlabel("Step")
        ax.set_ylabel("Error (m)")
        ax.set_title(f"{title} error (GT - Sim)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, ncol=2)

    link_error_titles = ["Left finger link", "Right finger link"]
    for i, (_title, eval_pose, gt_pose, _local_offset) in enumerate(sides):
        ax = axes[4 + i]
        if eval_pose is not None and gt_pose is not None:
            n = min(len(eval_pose), len(gt_pose), len(steps_arr))
            error = gt_pose[:n, :3] - eval_pose[:n, :3]
            for axis_idx, axis_name in enumerate(("X", "Y", "Z")):
                ax.plot(
                    steps_arr[:n], error[:, axis_idx],
                    color=axis_colors[axis_name], linewidth=1.3, linestyle="-",
                    marker=_SIM_MARKER, markersize=_MARKER_SIZE,
                    label=f"{axis_name} error",
                )
        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        for s in gripper_close_steps:
            ax.axvline(x=s, color="red", linestyle="-", alpha=0.5, linewidth=0.8)
        for s in gripper_open_steps:
            ax.axvline(x=s, color="blue", linestyle="-", alpha=0.5, linewidth=0.8)
        if sim_contact_step is not None:
            ax.axvline(x=sim_contact_step, color="darkgreen", linestyle="-", alpha=0.7, linewidth=1.0)
        if gt_contact_step is not None:
            ax.axvline(x=gt_contact_step, color="darkorange", linestyle="--", alpha=0.7, linewidth=1.0)
        ax.set_xlabel("Step")
        ax.set_ylabel("Error (m)")
        ax.set_title(f"{link_error_titles[i]} position error (GT - Sim), robot base frame")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, ncol=2)

    if has_gains:
        for q, (eval_arr, gt_arr, qname, qunit, log_scale) in enumerate(gain_specs):
            ax = fig.add_subplot(gs[6, q])
            if eval_arr is not None and len(eval_arr) > 0:
                n = min(len(eval_arr), len(steps_arr))
                ax.plot(steps_arr[:n], eval_arr[:n], color="blue", linewidth=1.3, linestyle="-",
                        marker=_SIM_MARKER, markersize=_MARKER_SIZE, label="Sim")
            if gt_arr is not None and len(gt_arr) > 0:
                n = min(len(gt_arr), len(steps_arr))
                ax.plot(steps_arr[:n], gt_arr[:n], color="red", linewidth=1.3, linestyle="--",
                        marker=_GT_MARKER, markersize=_MARKER_SIZE, label="GT")
            for s in gripper_close_steps:
                ax.axvline(x=s, color="red", linestyle="-", alpha=0.5, linewidth=0.8)
            for s in gripper_open_steps:
                ax.axvline(x=s, color="blue", linestyle="-", alpha=0.5, linewidth=0.8)
            if sim_contact_step is not None:
                ax.axvline(x=sim_contact_step, color="darkgreen", linestyle="-", alpha=0.7, linewidth=1.0)
            if gt_contact_step is not None:
                ax.axvline(x=gt_contact_step, color="darkorange", linestyle="--", alpha=0.7, linewidth=1.0)
            if log_scale:
                ax.set_yscale("log")
            ax.set_xlabel("Step")
            ax.set_ylabel(f"{qname} ({qunit})")
            ax.set_title(f"Gripper — {qname}")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)

    out_path = os.path.join(output_dir, "fingertip_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved fingertip comparison plot to: {out_path}")


def _first_finite_row(values: np.ndarray | None) -> np.ndarray | None:
    """First row of ``values`` with no NaN/Inf; series realigned onto plot steps can start with gaps."""
    if values is None:
        return None
    for row in values:
        if np.all(np.isfinite(row)):
            return row
    return None


def _save_camera_params_comparison(
    *,
    output_dir: str,
    eval_camera_poses: dict,
    eval_camera_intrinsics: dict,
    gt_camera_poses: dict,
    gt_camera_intrinsics: dict,
    steps_arr: np.ndarray,
) -> None:
    """One ``camera_params.png``: for every camera recorded on either side, its
    extrinsic position (X/Y/Z) and orientation (roll/pitch/yaw) over the
    episode, plus a row comparing its pinhole intrinsics (fx, fy, cx, cy, skew)
    — sim solid vs GT dashed throughout, one block of subplots per camera.

    Camera pose is ``(x, y, z, qw, qx, qy, qz)`` in the robot base frame and
    intrinsics is a flattened row-major 3x3 pinhole ``K``, both per the fixed
    schema the recorder writes (see ``lerobot_recorder.py``'s
    ``observation.camera_pose.*`` / ``observation.camera_intrinsics.*``
    features); a wrist-mounted camera's extrinsics can vary within the
    episode, which is why position/orientation are plotted over steps rather
    than as a single value like the intrinsics. Cameras are matched between
    sim and GT by their ``camera:<name>`` key, so a camera missing from one
    side just shows the other side's curve alone.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from blocksuite.utils.wandb_logging import short_camera_name

    camera_names = sorted(
        set(eval_camera_poses)
        | set(gt_camera_poses)
        | set(eval_camera_intrinsics)
        | set(gt_camera_intrinsics)
    )
    if not camera_names:
        return

    axis_colors = {"X": "tab:red", "Y": "tab:green", "Z": "tab:blue"}
    euler_colors = {"Roll": "tab:red", "Pitch": "tab:green", "Yaw": "tab:blue"}
    # (label, index into the flattened row-major 3x3 K = [fx, s, cx, 0, fy, cy, 0, 0, 1]).
    intrinsics_specs = [("fx", 0), ("fy", 4), ("cx", 2), ("cy", 5), ("skew", 1)]
    n_intrinsics_cols = 3
    n_intrinsics_rows = math.ceil(len(intrinsics_specs) / n_intrinsics_cols)
    rows_per_camera = 2 + n_intrinsics_rows
    n_cameras = len(camera_names)

    fig = plt.figure(figsize=(14, (5 + 3 * n_intrinsics_rows) * n_cameras))
    gs = fig.add_gridspec(
        rows_per_camera * n_cameras, n_intrinsics_cols, hspace=0.7, wspace=0.35, top=0.97,
        height_ratios=([1, 1] + [0.6] * n_intrinsics_rows) * n_cameras,
    )
    fig.suptitle("Camera Parameters (robot base frame): Sim vs Ground Truth", fontsize=14, y=0.995)

    for cam_idx, cam_name in enumerate(camera_names):
        row0 = cam_idx * rows_per_camera
        display_name = short_camera_name(cam_name)
        eval_pose = eval_camera_poses.get(cam_name)
        gt_pose = gt_camera_poses.get(cam_name)

        ax_pos = fig.add_subplot(gs[row0, :])
        for pose, linestyle, marker, side in (
            (eval_pose, "-", _SIM_MARKER, "Sim"), (gt_pose, "--", _GT_MARKER, "GT"),
        ):
            if pose is None:
                continue
            n = min(len(pose), len(steps_arr))
            for axis_idx, axis_name in enumerate(("X", "Y", "Z")):
                ax_pos.plot(
                    steps_arr[:n], pose[:n, axis_idx], color=axis_colors[axis_name],
                    linewidth=1.3, linestyle=linestyle, marker=marker, markersize=_MARKER_SIZE,
                    label=f"{side} {axis_name}",
                )
        ax_pos.set_xlabel("Step")
        ax_pos.set_ylabel("Position (m)")
        ax_pos.set_title(f"{display_name} camera — position")
        ax_pos.grid(True, alpha=0.3)
        ax_pos.legend(loc="upper right", fontsize=7, ncol=2)

        ax_rot = fig.add_subplot(gs[row0 + 1, :])
        for pose, linestyle, marker, side in (
            (eval_pose, "-", _SIM_MARKER, "Sim"), (gt_pose, "--", _GT_MARKER, "GT"),
        ):
            if pose is None:
                continue
            n = min(len(pose), len(steps_arr))
            euler = np.degrees(_quat_array_to_euler(pose[:n, 3:7]))
            for axis_idx, axis_name in enumerate(("Roll", "Pitch", "Yaw")):
                ax_rot.plot(
                    steps_arr[:n], euler[:, axis_idx], color=euler_colors[axis_name],
                    linewidth=1.3, linestyle=linestyle, marker=marker, markersize=_MARKER_SIZE,
                    label=f"{side} {axis_name}",
                )
        ax_rot.set_xlabel("Step")
        ax_rot.set_ylabel("Angle (deg)")
        ax_rot.set_title(f"{display_name} camera — orientation")
        ax_rot.grid(True, alpha=0.3)
        ax_rot.legend(loc="upper right", fontsize=7, ncol=2)

        eval_k = eval_camera_intrinsics.get(cam_name)
        gt_k = gt_camera_intrinsics.get(cam_name)
        eval_k0 = _first_finite_row(eval_k)
        gt_k0 = _first_finite_row(gt_k)
        for q, (label, flat_idx) in enumerate(intrinsics_specs):
            ax = fig.add_subplot(gs[row0 + 2 + q // n_intrinsics_cols, q % n_intrinsics_cols])
            eval_value = float(eval_k0[flat_idx]) if eval_k0 is not None else None
            gt_value = float(gt_k0[flat_idx]) if gt_k0 is not None else None
            values = [v for v in (eval_value, gt_value) if v is not None]
            if eval_value is not None:
                ax.axhline(y=eval_value, color="blue", linewidth=1.5, linestyle="-", label=f"Sim: {eval_value:.4g}")
            if gt_value is not None:
                ax.axhline(y=gt_value, color="red", linewidth=1.5, linestyle="--", label=f"GT: {gt_value:.4g}")
            if values:
                lo, hi = min(values), max(values)
                pad = max(hi - lo, abs(hi) * 0.1, 1e-3)
                ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlim(steps_arr[0] if len(steps_arr) else 0, steps_arr[-1] if len(steps_arr) else 1)
            ax.set_xlabel("Step")
            ax.set_title(f"{display_name} — {label} (px)", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=7)

    out_path = os.path.join(output_dir, "camera_params.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved camera params comparison plot to: {out_path}")


def _load_gt_episode(gt_source: str, episode_index: int = 0) -> "pd.DataFrame":
    """Load a single episode from a LeRobot dataset, avoiding full-dataset reads.

    Uses pyarrow predicate pushdown so only matching row-groups are scanned.
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.dataset as ds

    gt_root = Path(gt_source)
    data_dir = gt_root / "data"
    assert data_dir.exists(), f"No data/ directory found at {data_dir}"
    # A merged dataset's fragments do not all carry the same block set, and
    # `ds.dataset` infers its schema from the first fragment alone, which would
    # silently hide every column that fragment happens to lack. Unifying the
    # fragment schemas makes each recorded column visible, reading back as null
    # for the episodes that do not own it.
    fragments = list(ds.dataset(str(data_dir), format="parquet").get_fragments())
    schema = pa.unify_schemas([f.physical_schema for f in fragments])
    dataset = ds.dataset(str(data_dir), format="parquet", schema=schema)
    table = dataset.to_table(filter=ds.field("episode_index") == episode_index)
    assert table.num_rows > 0, (
        f"No rows found for episode_index={episode_index} in {data_dir}"
    )
    return table.to_pandas()


def _gt_rows(gt_df: "pd.DataFrame", col: str) -> list | None:
    """Rows of a ground-truth column, or ``None`` when the episode has none.

    A merged dataset carries the union of every episode's columns, so a column
    recorded only by other episodes reads back all-null here — that block or
    camera is simply not part of this episode. A column that is null for only
    some of the episode's rows is a corrupt recording, not an absent block.
    """
    if col not in gt_df.columns:
        return None
    rows = list(gt_df[col])
    n_null = sum(1 for row in rows if row is None or (isinstance(row, float) and math.isnan(row)))
    if n_null == len(rows):
        return None
    if n_null:
        raise ValueError(
            f"Ground-truth column {col!r} is null for {n_null}/{len(rows)} rows of "
            "this episode; a column must be either absent from the whole episode "
            "or present on every row."
        )
    return rows


def _gt_series(gt_df: "pd.DataFrame", col: str, width: int | None = None) -> np.ndarray | None:
    """Ground-truth column stacked to ``(n, width)``, ``None`` when absent.

    ``width`` truncates each row; leave it unset to keep the recorded width.
    """
    rows = _gt_rows(gt_df, col)
    if rows is None:
        return None
    stacked = [np.atleast_1d(np.asarray(row, dtype=np.float32)) for row in rows]
    return np.array([row[:width] if width is not None else row for row in stacked])


def _gt_scalar(gt_df: "pd.DataFrame", col: str) -> float | None:
    """First value of a per-episode ground-truth column, ``None`` when absent."""
    rows = _gt_rows(gt_df, col)
    if rows is None:
        return None
    return float(np.asarray(rows[0]).flat[0])


def _quat_to_euler(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to euler angles [roll, pitch, yaw]."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float32)


def _quat_array_to_euler(quat_wxyz: np.ndarray) -> np.ndarray:
    """Roll/pitch/yaw (rad), one row per ``[w, x, y, z]`` quaternion in an ``(n, 4)`` array."""
    return np.stack([_quat_to_euler(q) for q in quat_wxyz])


def save_eval_plots(
    *,
    output_dir: str,
    all_joint_positions: Sequence[np.ndarray],
    all_joint_velocities: Sequence[np.ndarray],
    all_action_targets: Sequence[np.ndarray],
    all_gripper_actions: Sequence[float],
    all_gripper_obs: Sequence[float],
    all_pd_modes: Sequence[float],
    all_ee_poses: Sequence[np.ndarray],
    all_joint_stiffness: Sequence[np.ndarray],
    all_joint_damping: Sequence[np.ndarray],
    all_joint_effort_limits: Sequence[np.ndarray],
    gt_dataset_path: Optional[str] = None,
    gt_episode_index: int = 0,
    block_poses: Optional[dict] = None,
    cube_side_length: Optional[float] = None,
    left_finger_pose: Optional[np.ndarray] = None,
    right_finger_pose: Optional[np.ndarray] = None,
    gripper_stiffness: Optional[np.ndarray] = None,
    gripper_damping: Optional[np.ndarray] = None,
    gripper_effort_limit: Optional[np.ndarray] = None,
    friction_properties: Optional[np.ndarray] = None,
    table_friction: Optional[float] = None,
    table_height_delta: Optional[float] = None,
    object_mass: Optional[float] = None,
    camera_poses: Optional[dict] = None,
    camera_intrinsics: Optional[dict] = None,
) -> None:
    """Save the full suite of per-step evaluation plots.

    Produces:
    - ``joint_position_and_pd_gains_comparison.png``: 7 joint subplots + gripper + gripper obs +
      PD mode + per-joint gain grid (stiffness/damping/effort).
    - ``ee_pose_comparison.png``: 6 subplots for EE position and orientation.
    - ``joint_velocity.png``: 7 joint velocity subplots.
    - ``<block_name>_pose_comparison.png``: one per non-table object in
      ``block_poses`` (and/or the GT dataset), 8 subplots (one per cube corner)
      each plotting sim vs GT X/Y/Z in the robot base frame, a row of physics
      parameters in effect for the episode (friction, table height, cube side
      length, object mass), and a row giving the (X, Y, Z) of each fingertip's
      first contact with the object (sim and GT computed independently) when
      finger-link poses were recorded.
    - ``<block_name>_contact_heatmap.png``: those same first-contact points'
      (X, Y) in the robot base frame, overlaid on the block's own footprint
      (its 8 corners projected onto X/Y) at the contact frame.
    - ``fingertip_comparison.png``: left/right gripper fingertip position, sim
      vs GT, derived from the ``robot.poses.{left,right}_inner_finger`` link
      poses when recorded, plus the gripper's own stiffness/damping/effort-limit
      PD gains (the finger_joint slot of the same gain vectors used for the
      per-arm-joint gain grid).
    - ``camera_params.png``: one block per camera in ``camera_poses``/
      ``camera_intrinsics`` (and/or the GT dataset) giving its extrinsic
      position and orientation over the episode plus its pinhole intrinsics
      (fx, fy, cx, cy, skew), sim vs GT, when those columns were recorded.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_eval_steps = len(all_gripper_obs)
    joint_pos_arr = np.array(all_joint_positions) if all_joint_positions else np.full((n_eval_steps, 7), np.nan)
    joint_vel_arr = np.array(all_joint_velocities) if all_joint_velocities else np.full((n_eval_steps, 7), np.nan)
    action_target_arr = np.array(all_action_targets) if all_action_targets else np.full((n_eval_steps, 7), np.nan)
    steps_arr = np.arange(n_eval_steps)
    joint_names = [f"Joint {i+1}" for i in range(7)]

    # --- Load ground truth data if available ---
    gt_joint_targets = None
    gt_joint_positions = None
    gt_gripper_actions = None
    gt_gripper_obs = None
    gt_pd_modes = None
    gt_ee_positions = None
    gt_joint_velocities = None
    gt_stiffness = None
    gt_damping = None
    gt_effort = None
    gt_block_poses: dict = {}
    gt_camera_poses: dict = {}
    gt_camera_intrinsics: dict = {}
    gt_cube_side_length = None
    gt_left_finger_pose = None
    gt_right_finger_pose = None
    gt_gripper_stiffness = None
    gt_gripper_damping = None
    gt_gripper_effort_limit = None
    gt_friction_properties = None
    gt_table_friction = None
    gt_table_height_delta = None
    gt_object_mass = None
    gt_source = gt_dataset_path
    print(
        "[INFO]: ground-truth overlay source: "
        f"{gt_source or 'none (plotting the policy rollout only)'}"
    )
    if gt_source:
        gt_df = _load_gt_episode(gt_source, episode_index=gt_episode_index)
        gt_joint_targets = _gt_series(gt_df, "action.joint_target", 7)
        gt_joint_positions = _gt_series(gt_df, "observation.state.joint_position", 7)
        gripper_action = _gt_series(gt_df, "action.gripper_position", 1)
        if gripper_action is not None:
            gt_gripper_actions = np.where(gripper_action[:, 0] > 0.5, -1.0, 1.0)
        pd_modes = _gt_series(gt_df, "action.pd_mode", 1)
        if pd_modes is not None:
            gt_pd_modes = pd_modes[:, 0]
        # The terminal row stores a cleared action; drawing it would show a fake
        # command drop on the last step.
        action_valid = _gt_series(gt_df, "action.is_valid", 1)
        if action_valid is not None:
            invalid_rows = action_valid[:, 0] == 0
            for series in (gt_joint_targets, gt_gripper_actions, gt_pd_modes):
                if series is not None:
                    series[invalid_rows] = np.nan
        gripper_obs = _gt_series(gt_df, "observation.state.gripper_position", 1)
        if gripper_obs is not None:
            gt_gripper_obs = gripper_obs[:, 0]
        gt_ee_positions = _gt_series(gt_df, "observation.state.ee_position", 6)
        if gt_ee_positions is not None:
            print(f"Loaded {len(gt_ee_positions)} ground truth EE position frames")
        gt_joint_velocities = _gt_series(gt_df, "observation.state.joint_velocity", 7)
        # Gain vectors are padded with one slot per articulation joint; index 7
        # is the gripper's finger_joint, present only when the vector is wider
        # than the 7 arm joints (older runs recorded arm-only gains).
        full_stiffness = _gt_series(gt_df, "observation.state.current_stiffness")
        if full_stiffness is not None:
            gt_stiffness = full_stiffness[:, :_N_ARM_JOINTS]
            if full_stiffness.shape[1] > _N_ARM_JOINTS:
                gt_gripper_stiffness = full_stiffness[:, _N_ARM_JOINTS]
        full_damping = _gt_series(gt_df, "observation.state.current_damping")
        if full_damping is not None:
            gt_damping = full_damping[:, :_N_ARM_JOINTS]
            if full_damping.shape[1] > _N_ARM_JOINTS:
                gt_gripper_damping = full_damping[:, _N_ARM_JOINTS]
        full_effort = _gt_series(gt_df, "observation.state.current_effort_limit")
        if full_effort is not None:
            gt_effort = full_effort[:, :_N_ARM_JOINTS]
            if full_effort.shape[1] > _N_ARM_JOINTS:
                gt_gripper_effort_limit = full_effort[:, _N_ARM_JOINTS]
        for col, name in _block_pose_columns(gt_df.columns).items():
            block_pose = _gt_series(gt_df, col, 7)
            if block_pose is not None:
                gt_block_poses[name] = block_pose
        for col in gt_df.columns:
            if col.startswith(_CAMERA_POSE_PREFIX):
                camera_pose = _gt_series(gt_df, col, 7)
                if camera_pose is not None:
                    gt_camera_poses[col[len(_CAMERA_POSE_PREFIX):]] = camera_pose
            elif col.startswith(_CAMERA_INTRINSICS_PREFIX):
                intrinsics = _gt_series(gt_df, col, 9)
                if intrinsics is not None:
                    gt_camera_intrinsics[col[len(_CAMERA_INTRINSICS_PREFIX):]] = intrinsics
        gt_cube_side_length = _gt_scalar(gt_df, "observation.state.cube_scale")
        friction_rows = _gt_rows(gt_df, "observation.state.friction_properties")
        if friction_rows is not None:
            gt_friction_properties = np.asarray(friction_rows[0], dtype=np.float32).ravel()
        gt_table_friction = _gt_scalar(gt_df, "observation.state.table_friction")
        gt_table_height_delta = _gt_scalar(gt_df, "observation.state.table_height_delta")
        gt_object_mass = _gt_scalar(gt_df, "observation.state.object_mass")
        gt_left_finger_pose = _gt_series(gt_df, f"{_ROBOT_POSE_PREFIX}{_LEFT_FINGER_LINK}", 7)
        gt_right_finger_pose = _gt_series(gt_df, f"{_ROBOT_POSE_PREFIX}{_RIGHT_FINGER_LINK}", 7)
        if gt_joint_targets is not None:
            print(f"Loaded {len(gt_joint_targets)} ground truth action.joint_target frames")

    # --- Detect gripper close/open transitions ---
    gripper_close_steps: list[int] = []
    gripper_open_steps: list[int] = []
    gripper_arr = np.array(all_gripper_actions) if all_gripper_actions else np.full(n_eval_steps, np.nan)
    for s in range(1, len(gripper_arr)):
        if not np.isnan(gripper_arr[s]) and not np.isnan(gripper_arr[s - 1]):
            if gripper_arr[s] < 0 and gripper_arr[s - 1] >= 0:
                gripper_close_steps.append(s)
            elif gripper_arr[s] >= 0 and gripper_arr[s - 1] < 0:
                gripper_open_steps.append(s)

    pd_mode_arr = np.array(all_pd_modes) if all_pd_modes else np.full(n_eval_steps, np.nan)
    stiff_arr = np.array(all_joint_stiffness) if all_joint_stiffness else np.empty((0, 7))
    damp_arr = np.array(all_joint_damping) if all_joint_damping else np.empty((0, 7))
    effort_arr = np.array(all_joint_effort_limits) if all_joint_effort_limits else np.empty((0, 7))

    # --- Main joint position / gain comparison plot ---
    n_top = 10
    n_joints = 7
    fig = plt.figure(figsize=(15, n_top * 3.2 + n_joints * 2.8 + 2))
    gs = fig.add_gridspec(n_top + n_joints, 3, hspace=0.7, wspace=0.35,
                          left=0.06, right=0.98, top=0.975, bottom=0.02)
    axes = [fig.add_subplot(gs[i, :]) for i in range(n_top)]
    axes_gains = [
        [fig.add_subplot(gs[n_top + j, q]) for q in range(3)]
        for j in range(n_joints)
    ]
    fig.suptitle("Joint Position Comparison", fontsize=14, y=0.995)

    for i in range(7):
        ax = axes[i]
        ax.plot(steps_arr, action_target_arr[:, i], color="green", linewidth=1.5, linestyle=":",
                marker=_SIM_MARKER, markersize=_MARKER_SIZE, label="Sim action.joint_target")
        ax.plot(steps_arr, joint_pos_arr[:, i], color="blue", linewidth=1.5,
                marker=_SIM_MARKER, markersize=_MARKER_SIZE, label="Sim joint position")
        if gt_joint_targets is not None:
            n = min(len(gt_joint_targets), len(steps_arr))
            ax.plot(steps_arr[:n], gt_joint_targets[:n, i], color="red", linewidth=1.5, linestyle="--",
                    marker=_GT_MARKER, markersize=_MARKER_SIZE, label="GT action.joint_target")
        if gt_joint_positions is not None:
            n = min(len(gt_joint_positions), len(steps_arr))
            ax.plot(steps_arr[:n], gt_joint_positions[:n, i], color="magenta", linewidth=1.5, linestyle="-.",
                    marker=_GT_MARKER, markersize=_MARKER_SIZE, label="GT joint position")
        for s in gripper_close_steps:
            ax.axvline(x=s, color="red", linestyle="-", alpha=0.5, linewidth=0.8,
                       label="Gripper close" if i == 0 and s == gripper_close_steps[0] else "")
        for s in gripper_open_steps:
            ax.axvline(x=s, color="blue", linestyle="-", alpha=0.5, linewidth=0.8,
                       label="Gripper open" if i == 0 and s == gripper_open_steps[0] else "")
        ax.set_xlabel("Step")
        ax.set_ylabel("Position (rad)")
        ax.set_title(joint_names[i])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    # Gripper action subplot
    ax = axes[7]
    ax.plot(steps_arr, gripper_arr, color="purple", linewidth=1.5,
            marker=_SIM_MARKER, markersize=_MARKER_SIZE, label="Gripper action")
    if gt_gripper_actions is not None:
        n = min(len(gt_gripper_actions), len(steps_arr))
        ax.plot(steps_arr[:n], gt_gripper_actions[:n], color="red", linewidth=1.5, linestyle="--",
                marker=_GT_MARKER, markersize=_MARKER_SIZE, label="GT gripper action")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    for s in gripper_close_steps:
        ax.axvline(x=s, color="red", linestyle="-", alpha=0.5, linewidth=0.8)
    for s in gripper_open_steps:
        ax.axvline(x=s, color="blue", linestyle="-", alpha=0.5, linewidth=0.8)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Gripper cmd")
    ax.set_title("Gripper Action (neg=close, pos=open)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # Raw gripper observation subplot
    ax = axes[8]
    gripper_obs_arr = np.array(all_gripper_obs)
    ax.plot(steps_arr, gripper_obs_arr, color="blue", linewidth=1.5,
            marker=_SIM_MARKER, markersize=_MARKER_SIZE, label="Sim gripper joint pos")
    if gt_gripper_obs is not None:
        n = min(len(gt_gripper_obs), len(steps_arr))
        ax.plot(steps_arr[:n], gt_gripper_obs[:n], color="red", linewidth=1.5, linestyle="--",
                marker=_GT_MARKER, markersize=_MARKER_SIZE, label="GT gripper obs")
    for s in gripper_close_steps:
        ax.axvline(x=s, color="red", linestyle="-", alpha=0.5, linewidth=0.8)
    for s in gripper_open_steps:
        ax.axvline(x=s, color="blue", linestyle="-", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Joint pos (rad)")
    ax.set_title("Raw Gripper Observation (finger_joint position)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # PD mode subplot
    ax = axes[9]
    ax.plot(steps_arr, pd_mode_arr, color="darkorange", linewidth=1.5,
            marker=_SIM_MARKER, markersize=_MARKER_SIZE, label="PD mode")
    if gt_pd_modes is not None:
        n = min(len(gt_pd_modes), len(steps_arr))
        ax.plot(steps_arr[:n], gt_pd_modes[:n], color="red", linewidth=1.5, linestyle="--",
                marker=_GT_MARKER, markersize=_MARKER_SIZE, label="GT PD mode")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    for s in gripper_close_steps:
        ax.axvline(x=s, color="red", linestyle="-", alpha=0.5, linewidth=0.8)
    for s in gripper_open_steps:
        ax.axvline(x=s, color="blue", linestyle="-", alpha=0.5, linewidth=0.8)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Step")
    ax.set_ylabel("PD mode")
    ax.set_title("PD Mode (0=compliant, 1=stiff)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # --- Per-joint PD gain subplots ---
    gain_specs = [
        (stiff_arr, gt_stiffness, "Stiffness", "N·m/rad", True),
        (damp_arr, gt_damping, "Damping", "N·m·s/rad", True),
        (effort_arr, gt_effort, "Effort", "N·m", False),
    ]
    for j in range(n_joints):
        for q, (eval_arr, gt_arr, qname, qunit, log_scale) in enumerate(gain_specs):
            ax = axes_gains[j][q]
            if eval_arr.shape[0] > 0:
                ax.plot(
                    steps_arr, eval_arr[:, j],
                    color="blue", linewidth=1.3, linestyle="-",
                    marker=_SIM_MARKER, markersize=_MARKER_SIZE, label="eval",
                )
            if gt_arr is not None:
                n = min(len(gt_arr), len(steps_arr))
                ax.plot(
                    steps_arr[:n], gt_arr[:n, j],
                    color="red", linewidth=1.3, linestyle="--",
                    marker=_GT_MARKER, markersize=_MARKER_SIZE, label="GT",
                )
            if log_scale:
                ax.set_yscale("log")
            ax.set_xlabel("Step")
            ax.set_ylabel(f"{qname} ({qunit})")
            ax.set_title(f"J{j + 1} — {qname}")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(loc="upper right", fontsize=7)

    os.makedirs(output_dir, exist_ok=True)
    pos_path = os.path.join(output_dir, "joint_position_and_pd_gains_comparison.png")
    plt.savefig(pos_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved joint position plot to: {pos_path}")

    # --- End-effector pose comparison plot ---
    ee_pose_arr = np.array(all_ee_poses)
    if ee_pose_arr.shape[0] > 0 and ee_pose_arr.shape[1] == 6:
        ee_labels = ["X (m)", "Y (m)", "Z (m)", "Roll (rad)", "Pitch (rad)", "Yaw (rad)"]
        ee_titles = ["EE X Position", "EE Y Position", "EE Z Position",
                     "EE Roll", "EE Pitch", "EE Yaw"]
        fig_ee, axes_ee = plt.subplots(6, 1, figsize=(14, 20))
        fig_ee.suptitle("End-Effector Pose: Sim vs Ground Truth", fontsize=14, y=0.995)
        for i in range(6):
            ax = axes_ee[i]
            ax.plot(steps_arr, ee_pose_arr[:, i], color="blue", linewidth=1.5,
                    marker=_SIM_MARKER, markersize=_MARKER_SIZE, label="Sim EE")
            if gt_ee_positions is not None:
                n = min(len(gt_ee_positions), len(steps_arr))
                ax.plot(steps_arr[:n], gt_ee_positions[:n, i], color="red", linewidth=1.5, linestyle="--",
                        marker=_GT_MARKER, markersize=_MARKER_SIZE, label="GT EE")
            for s in gripper_close_steps:
                ax.axvline(x=s, color="red", linestyle="-", alpha=0.5, linewidth=0.8,
                           label="Gripper close" if i == 0 and s == gripper_close_steps[0] else "")
            for s in gripper_open_steps:
                ax.axvline(x=s, color="blue", linestyle="-", alpha=0.5, linewidth=0.8,
                           label="Gripper open" if i == 0 and s == gripper_open_steps[0] else "")
            ax.set_xlabel("Step")
            ax.set_ylabel(ee_labels[i])
            ax.set_title(ee_titles[i])
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout(rect=(0, 0, 1, 0.97))
        ee_path = os.path.join(output_dir, "ee_pose_comparison.png")
        plt.savefig(ee_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved EE pose comparison plot to: {ee_path}")

    # --- Joint velocity plot ---
    fig, axes_vel = plt.subplots(7, 1, figsize=(14, 22))
    fig.suptitle("Joint Velocity", fontsize=14, y=0.995)
    for i in range(7):
        ax = axes_vel[i]
        ax.plot(steps_arr, joint_vel_arr[:, i], color="blue", linewidth=1.5,
                marker=_SIM_MARKER, markersize=_MARKER_SIZE, label="Sim joint vel")
        if gt_joint_velocities is not None:
            n = min(len(gt_joint_velocities), len(steps_arr))
            ax.plot(steps_arr[:n], gt_joint_velocities[:n, i], color="red", linewidth=1.5, linestyle="--",
                    marker=_GT_MARKER, markersize=_MARKER_SIZE, label="GT joint vel")
        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        ax.set_xlabel("Step")
        ax.set_ylabel("Velocity (rad/s)")
        ax.set_title(joint_names[i])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    vel_path = os.path.join(output_dir, "joint_velocity.png")
    plt.savefig(vel_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved joint velocity plot to: {vel_path}")

    # --- Per-block cube corner comparison plots ---
    block_poses = block_poses or {}
    if (block_poses or gt_block_poses) and (cube_side_length is None or gt_cube_side_length is None):
        print(
            "[INFO]: no observation.state.cube_scale recorded for "
            f"{'sim' if cube_side_length is None else 'ground truth'}; defaulting that "
            f"side to {_DEFAULT_CUBE_SIDE_LENGTH} m cube side length for corner plots"
        )
    eval_side_length = cube_side_length if cube_side_length is not None else _DEFAULT_CUBE_SIDE_LENGTH
    gt_side_length = gt_cube_side_length if gt_cube_side_length is not None else _DEFAULT_CUBE_SIDE_LENGTH
    all_sim_contact_steps: list[int] = []
    all_gt_contact_steps: list[int] = []
    for name in sorted(set(block_poses) | set(gt_block_poses)):
        contacts = _save_block_pose_comparison(
            output_dir=output_dir,
            block_name=name,
            eval_block_pose_in_robot_base_frame=block_poses.get(name),
            gt_block_pose_in_robot_base_frame=gt_block_poses.get(name),
            eval_side_length=eval_side_length,
            gt_side_length=gt_side_length,
            steps_arr=steps_arr,
            gripper_close_steps=gripper_close_steps,
            gripper_open_steps=gripper_open_steps,
            eval_friction_properties=friction_properties,
            gt_friction_properties=gt_friction_properties,
            eval_table_friction=table_friction,
            gt_table_friction=gt_table_friction,
            eval_table_height_delta=table_height_delta,
            gt_table_height_delta=gt_table_height_delta,
            eval_object_mass=object_mass,
            gt_object_mass=gt_object_mass,
            eval_left_finger_pose_in_robot_base_frame=left_finger_pose,
            eval_right_finger_pose_in_robot_base_frame=right_finger_pose,
            gt_left_finger_pose_in_robot_base_frame=gt_left_finger_pose,
            gt_right_finger_pose_in_robot_base_frame=gt_right_finger_pose,
        )
        sim_contact_step = _side_first_contact_step(contacts or [], "Sim")
        gt_contact_step = _side_first_contact_step(contacts or [], "GT")
        if sim_contact_step is not None:
            all_sim_contact_steps.append(sim_contact_step)
        if gt_contact_step is not None:
            all_gt_contact_steps.append(gt_contact_step)
        _save_block_pose_error_plot(
            output_dir=output_dir,
            block_name=name,
            eval_pose=block_poses.get(name),
            gt_pose=gt_block_poses.get(name),
            eval_side_length=eval_side_length,
            gt_side_length=gt_side_length,
            steps_arr=steps_arr,
            gripper_close_steps=gripper_close_steps,
            gripper_open_steps=gripper_open_steps,
            sim_contact_step=sim_contact_step,
            gt_contact_step=gt_contact_step,
        )
        _save_contact_heatmap(
            output_dir=output_dir,
            block_name=name,
            contacts=contacts or [],
        )

    # --- Fingertip position comparison plot ---
    # Uses the earliest contact across every block, since this plot covers the
    # gripper as a whole rather than any one block.
    _save_fingertip_comparison(
        output_dir=output_dir,
        eval_left_pose=left_finger_pose,
        eval_right_pose=right_finger_pose,
        gt_left_pose=gt_left_finger_pose,
        gt_right_pose=gt_right_finger_pose,
        eval_gripper_stiffness=gripper_stiffness,
        eval_gripper_damping=gripper_damping,
        eval_gripper_effort_limit=gripper_effort_limit,
        gt_gripper_stiffness=gt_gripper_stiffness,
        gt_gripper_damping=gt_gripper_damping,
        gt_gripper_effort_limit=gt_gripper_effort_limit,
        steps_arr=steps_arr,
        gripper_close_steps=gripper_close_steps,
        gripper_open_steps=gripper_open_steps,
        sim_contact_step=min(all_sim_contact_steps) if all_sim_contact_steps else None,
        gt_contact_step=min(all_gt_contact_steps) if all_gt_contact_steps else None,
    )

    # --- Camera parameters comparison plot ---
    _save_camera_params_comparison(
        output_dir=output_dir,
        eval_camera_poses=camera_poses or {},
        eval_camera_intrinsics=camera_intrinsics or {},
        gt_camera_poses=gt_camera_poses,
        gt_camera_intrinsics=gt_camera_intrinsics,
        steps_arr=steps_arr,
    )
