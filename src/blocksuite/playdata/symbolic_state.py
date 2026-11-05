"""Read a symbolic (PDDL) view of one vectorized environment out of the simulator.

The play-data loop never resets an environment, so the ``(:init ...)`` of the
next problem cannot come from a checked-in PDDL file — it has to be recovered
from the live simulator state.  The same predicate classes that evaluate goals
(``on``, ``on_table``, ``ori_of``) are reused here so that a goal written from
this state is judged by exactly the thresholds that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

# Same order as vecgen's orientation table (blocksuite.vecgen.symbolic.predicates.ORI_NAMES).
ORI_NAMES: tuple[str, ...] = tuple(f"o{i}" for i in range(1, 25))

# Same tolerances ori_of.check_face_up uses, so a state read here agrees with
# the predicate that will later evaluate the goal.
_ORI_YAW_TOL_DEG = 40.0
_ORI_AXIS_TOL_DEG = 5.0


def nearest_orientation(quat: torch.Tensor) -> tuple[str, bool]:
    """Return the closest ``o1``..``o24`` label for ``quat`` (w, x, y, z).

    The second element reports whether the match is a real one (i.e. it passes
    the same ``check_quat`` test ``ori_of`` uses) rather than a nearest-neighbour
    fallback for a block that is currently tilted or mid-flight.
    """
    from blocksuite.vecgen.symbolic.predicates import ORI_QUATERNIONS as ori_table
    from blocksuite.vecgen.symbolic.predicates import check_quat

    ORI_QUATERNIONS = dict(zip(ORI_NAMES, ori_table.tolist()))

    device = quat.device
    best_name = ORI_NAMES[0]
    best_angle = float("inf")
    for name, ref in ORI_QUATERNIONS.items():
        ref_t = torch.tensor(ref, device=device, dtype=quat.dtype)
        if bool(check_quat(quat, ref_t, _ORI_YAW_TOL_DEG, _ORI_AXIS_TOL_DEG)):
            return name, True
        dot = torch.abs(torch.dot(quat / quat.norm(), ref_t / ref_t.norm())).clamp(max=1.0)
        angle = float(2.0 * torch.acos(dot))
        if angle < best_angle:
            best_angle = angle
            best_name = name
    return best_name, False


@dataclass
class SymbolicState:
    """Symbolic snapshot of one environment, ready to render as a PDDL init."""

    blocks: list[str]
    """Blocks declared in the PDDL problem (the planner's universe)."""
    table: str
    gripper: str
    on: dict[str, str] = field(default_factory=dict)
    """Block -> the block it rests on (only blocks with a block supporter)."""
    on_table: list[str] = field(default_factory=list)
    ori: dict[str, str] = field(default_factory=dict)
    """Block -> orientation constant (``o1``..``o24``)."""
    unsupported: list[str] = field(default_factory=list)
    """Blocks with no detected support; treated as ``on_table`` when rendering."""
    tilted: list[str] = field(default_factory=list)
    """Blocks whose orientation had to fall back to a nearest-neighbour match."""
    covered_by_untracked: list[str] = field(default_factory=list)
    """PDDL blocks held down by a scene block that the PDDL does not declare."""
    missing_from_scene: list[str] = field(default_factory=list)
    """PDDL blocks with no pose in the scene; excluded from ``blocks`` entirely."""

    def supporter(self, block: str) -> str | None:
        return self.on.get(block)

    def clear_blocks(self) -> list[str]:
        """PDDL blocks with nothing stacked on them."""
        covered = set(self.on.values()) | set(self.covered_by_untracked)
        return [b for b in self.blocks if b not in covered]

    def render_init(self, indent: int = 8) -> str:
        """Render the ``(:init ...)`` body for this state."""
        pad = " " * indent
        lines: list[str] = []
        for block in self.blocks:
            supporter = self.on.get(block)
            if supporter is not None:
                lines.append(f"(on {block} {supporter})")
            else:
                lines.append(f"(on_table {block} {self.table})")
        for block in self.clear_blocks():
            lines.append(f"(clear {block})")
        lines.append(f"(handempty {self.gripper})")
        lines.append("(gripper_at_init)")
        for block in self.blocks:
            ori = self.ori.get(block)
            if ori is not None:
                lines.append(f"(ori_of {block} {ori})")
        return "\n".join(f"{pad}{line}" for line in lines)

    def describe(self) -> str:
        """One-line human-readable summary, for logs."""
        parts = []
        for block in self.blocks:
            support = self.on.get(block, self.table)
            parts.append(f"{block}@{self.ori.get(block, '?')}/{support}")
        return " ".join(parts)
