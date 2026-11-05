#!/usr/bin/env python3
"""Standalone collision-sphere visualizer.

Loads a kinematics URDF and a CuRobo-style collision_spheres YAML, runs
forward kinematics at a given joint configuration (default: zeros), and
renders every sphere in 3-D with matplotlib.

Dependencies: numpy, pyyaml, matplotlib  (no CuRobo / Isaac required).

Usage (from this directory):
    python visualize_collision_spheres.py
    python visualize_collision_spheres.py --spheres fr3_model_full_robotiq_2f85_collision_spheres.yml \
                                          --urdf   fr3_model_full_robotiq_2f85_kinematics.urdf
    python visualize_collision_spheres.py --joints 0 -0.3 0 -2.2 0 2.0 0.78
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml


# ── Minimal FK from URDF ────────────────────────────────────────────────────

def _rpy_to_matrix(rpy: list[float]) -> np.ndarray:
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [  -sp,           cp*sr,           cp*cr  ],
    ])


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    ax = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0, -ax[2], ax[1]],
                  [ax[2], 0, -ax[0]],
                  [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def _make_tf(rot: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = xyz
    return T


def _parse_vec(s: str) -> list[float]:
    return [float(x) for x in s.strip().split()]


class _Joint:
    def __init__(self, name, jtype, parent, child, origin_xyz, origin_rpy, axis):
        self.name = name
        self.jtype = jtype
        self.parent = parent
        self.child = child
        self.origin_xyz = np.array(origin_xyz)
        self.origin_rpy = origin_rpy
        self.axis = np.array(axis) if axis is not None else np.array([0, 0, 1])


def parse_urdf(urdf_path: str | Path) -> list[_Joint]:
    tree = ET.parse(urdf_path)
    joints = []
    for j in tree.getroot().findall("joint"):
        origin = j.find("origin")
        xyz = _parse_vec(origin.get("xyz", "0 0 0"))
        rpy = _parse_vec(origin.get("rpy", "0 0 0"))
        axis_el = j.find("axis")
        axis = _parse_vec(axis_el.get("xyz")) if axis_el is not None else None
        joints.append(_Joint(
            name=j.get("name"),
            jtype=j.get("type"),
            parent=j.find("parent").get("link"),
            child=j.find("child").get("link"),
            origin_xyz=xyz,
            origin_rpy=rpy,
            axis=axis,
        ))
    return joints


def compute_link_transforms(
    joints: list[_Joint],
    joint_values: dict[str, float],
) -> dict[str, np.ndarray]:
    """Return {link_name: 4x4 world transform} for every link in the tree."""
    transforms: dict[str, np.ndarray] = {}

    children_of: dict[str, list[_Joint]] = {}
    for j in joints:
        children_of.setdefault(j.parent, []).append(j)

    all_children = {j.child for j in joints}
    roots = {j.parent for j in joints} - all_children
    for r in roots:
        transforms[r] = np.eye(4)

    def _recurse(link: str):
        for j in children_of.get(link, []):
            R_origin = _rpy_to_matrix(j.origin_rpy)
            T_origin = _make_tf(R_origin, j.origin_xyz)

            if j.jtype == "revolute" or j.jtype == "continuous":
                angle = joint_values.get(j.name, 0.0)
                R_joint = _axis_angle_matrix(j.axis, angle)
                T_joint = _make_tf(R_joint, np.zeros(3))
                transforms[j.child] = transforms[link] @ T_origin @ T_joint
            else:
                transforms[j.child] = transforms[link] @ T_origin

            _recurse(j.child)

    for r in roots:
        _recurse(r)

    return transforms


# ── Load spheres YAML ────────────────────────────────────────────────────────

def load_spheres(yml_path: str | Path) -> dict[str, list[dict]]:
    with open(yml_path) as f:
        data = yaml.safe_load(f)
    return data.get("collision_spheres", data)


# ── Plotting ─────────────────────────────────────────────────────────────────

_LINK_COLORS = {}
_CMAP_ENTRIES = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
]


def _color_for_link(link: str) -> str:
    if link not in _LINK_COLORS:
        _LINK_COLORS[link] = _CMAP_ENTRIES[len(_LINK_COLORS) % len(_CMAP_ENTRIES)]
    return _LINK_COLORS[link]


def plot_spheres(
    link_transforms: dict[str, np.ndarray],
    spheres_by_link: dict[str, list[dict]],
    title: str = "Collision Spheres",
    save_path: str | None = None,
):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import art3d  # noqa: F401

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    all_pts = []

    for link_name, sphere_list in spheres_by_link.items():
        if not sphere_list:
            continue
        T = link_transforms.get(link_name)
        if T is None:
            print(f"  [warn] link '{link_name}' not in URDF, skipping")
            continue

        color = _color_for_link(link_name)
        for s in sphere_list:
            c_local = np.array(s["center"])
            r = s["radius"]
            c_world = (T[:3, :3] @ c_local + T[:3, 3])
            all_pts.append(c_world)

            u = np.linspace(0, 2 * np.pi, 16)
            v = np.linspace(0, np.pi, 12)
            xs = c_world[0] + r * np.outer(np.cos(u), np.sin(v))
            ys = c_world[1] + r * np.outer(np.sin(u), np.sin(v))
            zs = c_world[2] + r * np.outer(np.ones_like(u), np.cos(v))
            ax.plot_surface(xs, ys, zs, color=color, alpha=0.35, linewidth=0)

    if not all_pts:
        print("No spheres to plot.")
        return

    all_pts = np.array(all_pts)
    mid = all_pts.mean(axis=0)
    span = max((all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2, 0.1)
    margin = span * 1.15
    ax.set_xlim(mid[0] - margin, mid[0] + margin)
    ax.set_ylim(mid[1] - margin, mid[1] + margin)
    ax.set_zlim(mid[2] - margin, mid[2] + margin)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)

    # Legend
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=_color_for_link(l), label=l)
               for l in spheres_by_link if spheres_by_link[l]]
    ax.legend(handles=handles, fontsize=7, loc="upper left")

    ax.set_box_aspect([1, 1, 1])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    here = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Visualize CuRobo collision spheres from YAML + URDF")
    parser.add_argument("--spheres", type=str,
                        default=str(here / "xxxx.yml"))
    parser.add_argument("--urdf", type=str,
                        default=str(here / "xxxx.urdf"))
    parser.add_argument("--joints", type=float, nargs="*", default=None,
                        help="Joint values for panda_joint1..7 (default: all zeros)")
    parser.add_argument("--finger", type=float, default=0.0,
                        help="Finger joint value (default: 0)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save figure to this path instead of showing interactively")
    args = parser.parse_args()

    joints = parse_urdf(args.urdf)
    spheres = load_spheres(args.spheres)

    arm_joint_names = [f"panda_joint{i}" for i in range(1, 8)]
    joint_vals = {}
    if args.joints:
        for name, val in zip(arm_joint_names, args.joints):
            joint_vals[name] = val
    joint_vals["finger_joint"] = args.finger

    link_transforms = compute_link_transforms(joints, joint_vals)

    title_parts = [Path(args.spheres).stem]
    if args.joints:
        title_parts.append(f"q={[round(v, 2) for v in args.joints]}")
    plot_spheres(link_transforms, spheres, title=" | ".join(title_parts), save_path=args.save)


if __name__ == "__main__":
    main()
