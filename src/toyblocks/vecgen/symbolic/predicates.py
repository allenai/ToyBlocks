"""Batched predicate evaluation on base-frame poses (same tolerances as the TAMP classes).

- ``on(a, b)``:      |dx| <= 0.02, |dy| <= 0.02, 0 < dz <= 0.09            (predicates/on.py)
- ``on_table(b)``:   xy inside the table top (+0.02), bottom face within 0.02 of the top
- ``ori_of(b, o)``:  check_quat(q, ORI[o], yaw 40 deg, axis 5 deg)        (ori_of.py)
- ``next_to(a, b)``: |dz| <= 0.02 and flush on one axis (+-0.03) centred on the other (+-0.04)
- ``clear(b)``:      no block is ``on`` it
- ``on_right(a, b)``: |dy| = half_a + half_b + 5 mm (+-0.03), |dx| <= 0.04, |dz| <= 0.02
  (on_right.py; like the TAMP class and the env check, the side is not checked)
- ``on_left`` / ``on_back`` / ``on_front(a, b)``: centre distance 0.065 / 0.065 / 0.05 (+-0.01)
  along y / x / x, |dz| <= 0.01, the other axis within 0.04 (on_left.py, on_back.py,
  on_front.py; side not checked)
- ``pick_obstructs(a, b)``: xy distance <= 0.1 and |dz| <= 0.02 (no TAMP class; the env's
  ``pddl_goal_tensor_conversions.goal_pick_obstructs``)

``isclose`` tolerances follow ``torch.isclose`` (atol plus the default rtol), as the TAMP
classes compare.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from toyblocks.vecgen.math import (
    quat_angle,
    quat_apply,
    quat_inverse,
    quat_mul,
    quat_z_axis,
    world_extents,
)
from toyblocks.vecgen.symbolic.pddl import And, Fact, Not, Or

_S = math.sqrt(0.5)
ORI_NAMES = [f"o{i}" for i in range(1, 25)]
ORI_QUATERNIONS = torch.tensor(
    [
        [1, 0, 0, 0],
        [_S, 0, 0, -_S],
        [0, 0, 0, -1],
        [_S, 0, 0, _S],
        [_S, 0, -_S, 0],
        [0.5, -0.5, -0.5, -0.5],
        [0, _S, 0, _S],
        [0.5, 0.5, -0.5, 0.5],
        [0, 0, 1, 0],
        [0, _S, _S, 0],
        [0, -1, 0, 0],
        [0, -_S, _S, 0],
        [_S, 0, _S, 0],
        [0.5, 0.5, 0.5, -0.5],
        [0, -_S, 0, _S],
        [0.5, -0.5, 0.5, 0.5],
        [0.5, 0.5, -0.5, -0.5],
        [0, 0, _S, _S],
        [0.5, 0.5, 0.5, 0.5],
        [_S, _S, 0, 0],
        [0.5, -0.5, 0.5, -0.5],
        [0, 0, _S, -_S],
        [0.5, -0.5, -0.5, 0.5],
        [_S, -_S, 0, 0],
    ],
    dtype=torch.float32,
)

ON_XY_TOL = 0.02
ON_DZ_MAX = 0.09
ON_TABLE_TOL = 0.02
TABLE_TOP_ABOVE_ROOT = 0.01
TABLE_HALF_EXTENTS = (0.3, 0.4)  # sim2real slab is 0.6 x 0.8 after its -90 deg yaw
YAW_TOL_DEG = 40.0
AXIS_TOL_DEG = 5.0
ADJACENCY_GAP = 0.005  # predicates/next_to.py, on_right's gap
SIDE_CUBE_WIDTH = 0.045  # on_left / on_front / on_back assume a 4.5 cm cube
SIDE_OFFSET = 0.02  # on_left.py / on_back.py
FRONT_OFFSET = 0.005  # on_front.py
PICK_OBSTRUCTS_XY = 0.1
PICK_OBSTRUCTS_DZ = 0.02
#: Predicates :func:`goal_facts_satisfied` can evaluate from poses.
GEOMETRIC_PREDICATES: frozenset[str] = frozenset(
    {
        "on",
        "on_table",
        "clear",
        "ori_of",
        "next_to",
        "on_right",
        "on_left",
        "on_front",
        "on_back",
        "pick_obstructs",
    }
)


def ori_index(name: str) -> int:
    return ORI_NAMES.index(name.lower())


def ori_quat(names: list[str], device: torch.device) -> Tensor:
    idx = torch.tensor([ori_index(n) for n in names], dtype=torch.long)
    return ORI_QUATERNIONS.to(device)[idx.to(device)]


def check_quat(q_cur: Tensor, q_tgt: Tensor, yaw_tol_deg: float, axis_tol_deg: float) -> Tensor:
    """``True`` where the error rotation keeps body z within ``axis_tol`` of world z and
    its total angle is within ``yaw_tol`` (``tmp3.Utils.quat_utils.check_quat``)."""
    q_err = quat_mul(q_cur, quat_inverse(q_tgt))
    z_err = quat_z_axis(q_err)
    tilt = torch.acos(z_err[..., 2].clamp(-1.0, 1.0))
    angle = quat_angle(q_err)
    return (tilt <= math.radians(axis_tol_deg)) & (angle <= math.radians(yaw_tol_deg))


def ori_of(q: Tensor, ori_idx: Tensor) -> Tensor:
    tgt = ORI_QUATERNIONS.to(q.device)[ori_idx]
    return check_quat(q, tgt, YAW_TOL_DEG, AXIS_TOL_DEG)


def classify_orientation(q: Tensor) -> Tensor:
    """Index of the canonical orientation nearest to each ``q`` ``[B, 4]`` (always defined)."""
    oris = ORI_QUATERNIONS.to(q.device)  # [24, 4]
    q_err = quat_mul(q[:, None, :].expand(-1, 24, -1), quat_inverse(oris)[None])
    z_err = quat_z_axis(q_err)[..., 2].clamp(-1.0, 1.0)
    tilt = torch.acos(z_err)
    angle = quat_angle(q_err)
    # primary: smallest tilt; secondary: smallest total angle
    score = tilt * 10.0 + angle
    return score.argmin(dim=1)


def on(pose_a: Tensor, pose_b: Tensor) -> Tensor:
    d = pose_a[..., :3] - pose_b[..., :3]
    return (
        (d[..., 0].abs() <= ON_XY_TOL)
        & (d[..., 1].abs() <= ON_XY_TOL)
        & (d[..., 2] > 0)
        & (d[..., 2] <= ON_DZ_MAX)
    )


def on_table(pose: Tensor, size: Tensor, table_pose: Tensor) -> Tensor:
    """``size``: body-frame edge lengths; the bottom face is the world-z extent's half."""
    hx, hy = TABLE_HALF_EXTENTS
    dx = (pose[..., 0] - table_pose[..., 0]).abs()
    dy = (pose[..., 1] - table_pose[..., 1]).abs()
    top = table_pose[..., 2] + TABLE_TOP_ABOVE_ROOT
    bottom = pose[..., 2] - world_extents(pose[..., 3:7], size)[..., 2] / 2
    return (
        (dx <= hx + ON_TABLE_TOL)
        & (dy <= hy + ON_TABLE_TOL)
        & ((bottom - top).abs() <= ON_TABLE_TOL)
    )


def clear(poses: Tensor, j: Tensor) -> Tensor:
    """``poses`` ``[B, K, 7]``, ``j`` ``[B]``: True when no other block is on block ``j``."""
    b, k, _ = poses.shape
    base = poses[torch.arange(b, device=poses.device), j]  # [B, 7]
    above = on(poses, base[:, None, :].expand(-1, k, -1))  # [B, K]
    above[torch.arange(b, device=poses.device), j] = False
    return ~above.any(dim=1)


def next_to(pose_a: Tensor, pose_b: Tensor, size_a: Tensor, size_b: Tensor) -> Tensor:
    """Sizes are body-frame edge lengths, used unrotated as ``next_to.py`` does."""
    d = pose_a[..., :3] - pose_b[..., :3]
    z_ok = d[..., 2].abs() <= 0.02
    x_adj = ((d[..., 0].abs() - (size_a[..., 0] + size_b[..., 0]) / 2).abs() <= 0.03) & (
        d[..., 1].abs() <= 0.04
    )
    y_adj = ((d[..., 1].abs() - (size_a[..., 1] + size_b[..., 1]) / 2).abs() <= 0.03) & (
        d[..., 0].abs() <= 0.04
    )
    return z_ok & (x_adj | y_adj)


def _isclose(a: Tensor, b: Tensor | float, atol: float) -> Tensor:
    return torch.isclose(a, torch.as_tensor(b, dtype=a.dtype, device=a.device).expand_as(a), atol=atol)


def on_right(pose_a: Tensor, pose_b: Tensor, size_a: Tensor, size_b: Tensor) -> Tensor:
    """``a`` beside ``b`` along y, touching with a 5 mm gap (sizes: body-frame edge lengths,
    used unrotated as ``on_right.py`` does)."""
    d = pose_a[..., :3] - pose_b[..., :3]
    sep = (size_a[..., 1] + size_b[..., 1]) / 2 + ADJACENCY_GAP
    return (
        _isclose(d[..., 1].abs(), sep, 0.03)
        & _isclose(pose_a[..., 2], pose_b[..., 2], 0.02)
        & _isclose(pose_a[..., 0], pose_b[..., 0], 0.04)
    )


def _fixed_side(pose_a: Tensor, pose_b: Tensor, axis: int, distance: float) -> Tensor:
    other = 1 - axis
    return (
        _isclose((pose_a[..., axis] - pose_b[..., axis]).abs(), distance, 0.01)
        & _isclose(pose_a[..., 2], pose_b[..., 2], 0.01)
        & _isclose(pose_a[..., other], pose_b[..., other], 0.04)
    )


def on_left(pose_a: Tensor, pose_b: Tensor) -> Tensor:
    return _fixed_side(pose_a, pose_b, 1, SIDE_CUBE_WIDTH + SIDE_OFFSET)


def on_back(pose_a: Tensor, pose_b: Tensor) -> Tensor:
    return _fixed_side(pose_a, pose_b, 0, SIDE_CUBE_WIDTH + SIDE_OFFSET)


def on_front(pose_a: Tensor, pose_b: Tensor) -> Tensor:
    return _fixed_side(pose_a, pose_b, 0, SIDE_CUBE_WIDTH + FRONT_OFFSET)


def pick_obstructs(pose_a: Tensor, pose_b: Tensor) -> Tensor:
    d = pose_a[..., :3] - pose_b[..., :3]
    return (d[..., :2].norm(dim=-1) <= PICK_OBSTRUCTS_XY) & (d[..., 2].abs() <= PICK_OBSTRUCTS_DZ)


def pair_predicate(
    pred: str, pose_a: Tensor, pose_b: Tensor, size_a: Tensor, size_b: Tensor
) -> Tensor:
    """Evaluate a two-block geometric predicate (``on``, ``next_to``, ``on_right``, ...)."""
    if pred == "on":
        return on(pose_a, pose_b)
    if pred == "next_to":
        return next_to(pose_a, pose_b, size_a, size_b)
    if pred == "on_right":
        return on_right(pose_a, pose_b, size_a, size_b)
    if pred == "on_left":
        return on_left(pose_a, pose_b)
    if pred == "on_back":
        return on_back(pose_a, pose_b)
    if pred == "on_front":
        return on_front(pose_a, pose_b)
    if pred == "pick_obstructs":
        return pick_obstructs(pose_a, pose_b)
    raise ValueError(f"{pred!r} is not a two-block geometric predicate")


def goal_atoms(expr: Any) -> list[Fact]:
    """Every :class:`Fact` in a goal expression tree (either polarity)."""
    if isinstance(expr, Fact):
        return [expr]
    if isinstance(expr, Not):
        return goal_atoms(expr.child)
    if isinstance(expr, (And, Or)):
        return [a for c in expr.children for a in goal_atoms(c)]
    raise TypeError(f"unexpected goal node {expr!r}")


def assert_goal_supported(goal: list[Any], what: str = "goal") -> None:
    """Raise unless every goal atom is one of :data:`GEOMETRIC_PREDICATES` (checked up front, so
    a run does not fail at its first episode end)."""
    bad = sorted({str(a) for g in goal for a in goal_atoms(g) if a.pred not in GEOMETRIC_PREDICATES})
    if bad:
        raise ValueError(
            f"{what} uses predicates vecgen cannot check from poses: {', '.join(bad)} "
            f"(supported: {sorted(GEOMETRIC_PREDICATES)})"
        )


def quat_rotate_vec(q: Tensor, v: Tensor) -> Tensor:
    return quat_apply(q, v)


def goal_facts_satisfied(
    facts: list[Any],
    block_index: dict[str, int],
    poses: Tensor,
    sizes: Tensor,
    table_pose: Tensor,
) -> list[Tensor]:
    """Evaluate goal expressions over :data:`GEOMETRIC_PREDICATES`.

    ``poses`` ``[B, K, 7]``, ``sizes`` ``[B, K, 3]``, ``table_pose`` ``[B, 7]``; returns one
    ``[B]`` bool tensor per entry of ``facts``. An entry is a :class:`Fact` or an
    :class:`And` / :class:`Or` / :class:`Not` tree over facts (the grounded goal
    conjuncts of :func:`toyblocks.vecgen.symbolic.pddl.parse_goal`). A predicate that
    cannot be read off poses raises ``ValueError`` rather than counting as satisfied.
    """
    b = poses.shape[0]
    ar = torch.arange(b, device=poses.device)

    def evaluate(f: Any) -> Tensor:
        if isinstance(f, And):
            return torch.stack([evaluate(c) for c in f.children]).all(dim=0)
        if isinstance(f, Or):
            return torch.stack([evaluate(c) for c in f.children]).any(dim=0)
        if isinstance(f, Not):
            return ~evaluate(f.child)
        pred, args = f.pred, f.args
        if pred == "on":
            return on(poses[ar, block_index[args[0]]], poses[ar, block_index[args[1]]])
        if pred == "on_table":
            j = block_index[args[0]]
            return on_table(poses[ar, j], sizes[ar, j], table_pose)
        if pred == "clear":
            return clear(poses, torch.full((b,), block_index[args[0]], device=poses.device))
        if pred == "ori_of":
            j = block_index[args[0]]
            idx = torch.full((b,), ori_index(args[1]), device=poses.device)
            return ori_of(poses[ar, j, 3:7], idx)
        if pred in GEOMETRIC_PREDICATES:
            j, k = block_index[args[0]], block_index[args[1]]
            return pair_predicate(pred, poses[ar, j], poses[ar, k], sizes[ar, j], sizes[ar, k])
        raise ValueError(
            f"goal predicate {f} cannot be evaluated from poses "
            f"(supported: {sorted(GEOMETRIC_PREDICATES)})"
        )

    return [evaluate(f) for f in facts]
