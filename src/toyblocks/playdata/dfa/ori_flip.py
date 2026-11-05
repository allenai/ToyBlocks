"""Turn a *relative* flip/rotate intent into an absolute orientation constant.

The blocks domain names orientations ``o1``..``o24`` (see
vecgen's :data:`~toyblocks.vecgen.symbolic.predicates.ORI_QUATERNIONS`). A DFA move like "flip this
block forward" is a goal ``(ori_of block oN)`` where ``oN`` is the orientation
reached by rotating the block's *current* orientation a quarter turn about a
world axis. This module does that composition and snaps the result back onto the
nearest named constant, reusing the same snapper the symbolic-state reader uses
so the target always names a real ``o`` face.

Direction convention (world-frame, extrinsic quarter turns):

* ``forward`` / ``back``  -> rotate about the world **Y** axis (tips the top face
  toward / away from the robot).
* ``left`` / ``right``    -> rotate about the world **X** axis.
* rotate (``rotate`` move) -> about the world **Z** axis (spins in place).

The labels are conventional: what matters for data collection is that opposite
directions invert each other and repeated flips cycle through faces. The planner
re-checks feasibility for whatever ``oN`` this produces, so an unreachable target
simply becomes a planning failure the orchestrator already handles.
"""

from __future__ import annotations

import math

import torch

from toyblocks.playdata.symbolic_state import nearest_orientation
from toyblocks.vecgen.symbolic.predicates import ORI_NAMES
from toyblocks.vecgen.symbolic.predicates import ORI_QUATERNIONS as _ORI_TENSOR

ORI_QUATERNIONS: dict[str, tuple[float, ...]] = {name: tuple(q) for name, q in zip(ORI_NAMES, _ORI_TENSOR.tolist())}

# direction -> (world axis, sign of the quarter turn)
_FLIP_AXES: dict[str, tuple[str, int]] = {
    "forward": ("y", -1),
    "back": ("y", +1),
    "left": ("x", -1),
    "right": ("x", +1),
}
_ROTATE_AXIS = "z"

FLIP_DIRECTIONS: tuple[str, ...] = tuple(_FLIP_AXES.keys())


def _axis_quat(axis: str, quarter_turns: int) -> list[float]:
    """Quaternion (w, x, y, z) for ``quarter_turns`` * 90 deg about ``axis``."""
    angle = (math.pi / 2.0) * quarter_turns
    half = angle / 2.0
    s = math.sin(half)
    return [
        math.cos(half),
        s if axis == "x" else 0.0,
        s if axis == "y" else 0.0,
        s if axis == "z" else 0.0,
    ]


def _quat_mul(a: list[float], b: list[float]) -> list[float]:
    """Hamilton product of two (w, x, y, z) quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def _snap(quat: list[float]) -> str:
    """Nearest ``o1``..``o24`` constant for a (w, x, y, z) quaternion."""
    name, _exact = nearest_orientation(torch.tensor(quat, dtype=torch.float32))
    return name


def ori_after_flip(current: str, direction: str, quarter_turns: int = 1) -> str:
    """Orientation reached by flipping ``current`` ``quarter_turns`` in ``direction``."""
    if current not in ORI_QUATERNIONS:
        raise ValueError(f"unknown orientation constant {current!r}")
    if direction not in _FLIP_AXES:
        raise ValueError(f"unknown flip direction {direction!r}; use one of {FLIP_DIRECTIONS}")
    axis, sign = _FLIP_AXES[direction]
    delta = _axis_quat(axis, sign * int(quarter_turns))
    # World-frame rotation is a left-multiply of the current orientation.
    return _snap(_quat_mul(delta, ORI_QUATERNIONS[current]))


def ori_after_rotate(current: str, turns: int = 1) -> str:
    """Orientation reached by spinning ``current`` ``turns`` quarter turns about Z."""
    if current not in ORI_QUATERNIONS:
        raise ValueError(f"unknown orientation constant {current!r}")
    delta = _axis_quat(_ROTATE_AXIS, int(turns))
    return _snap(_quat_mul(delta, ORI_QUATERNIONS[current]))
