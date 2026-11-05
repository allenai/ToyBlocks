"""Batched joint-target segment tables.

Every env owns a row-table of arm targets for the segment it is executing. All tables
live in padded buffers ``[N, capacity, dof]`` (position, and optionally velocity and
acceleration for target feed-forward) so that a tick is a single gather:
``buffer[arange(N), cursor]``. A segment may ask for ``hold`` extra ticks at its last
row (settle), after which the env's segment is reported finished.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

#: A ``settle_tol`` hold also ends after this many ticks without the error shrinking by more
#: than SETTLE_MIN_PROGRESS (rad): the joint has stopped, e.g. in its friction deadband.
SETTLE_STALL_TICKS = 6
SETTLE_MIN_PROGRESS = math.radians(0.1)


@dataclass
class Segment:
    """One env's segment: ``targets`` is ``[T, dof]``; ``velocity``/``acceleration`` optional."""

    targets: Tensor
    velocity: Tensor | None = None
    acceleration: Tensor | None = None
    hold: int = 0
    label: str = ""  # e.g. "grasp_traj"; recorded as the low-level action name
    gripper_closed: bool | None = None  # gripper command issued when the segment starts
    gripper_target: float | None = None  # explicit finger target (overrides gripper_closed)
    # End the ``hold`` early once every joint is within this many radians of the last row
    # (``hold`` becomes a cap). None: hold exactly ``hold`` ticks.
    settle_tol: float | None = None

    @property
    def has_ff(self) -> bool:
        return self.velocity is not None and self.acceleration is not None


class SegmentTable:
    """Padded per-env segment storage with a single gather per tick."""

    def __init__(self, num_envs: int, dof: int, capacity: int, device: torch.device) -> None:
        self.num_envs = num_envs
        self.dof = dof
        self.capacity = capacity
        self.device = device
        self.pos = torch.zeros(num_envs, capacity, dof, device=device)
        self.vel = torch.zeros(num_envs, capacity, dof, device=device)
        self.acc = torch.zeros(num_envs, capacity, dof, device=device)
        self.length = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.hold = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.cursor = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.ff = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.settle_tol = torch.full((num_envs,), float("inf"), device=device)
        # settle bookkeeping: best error seen on the hold rows and ticks since it last improved
        self.best_err = torch.full((num_envs,), float("inf"), device=device)
        self.stalled = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.labels: list[str] = [""] * num_envs
        self._arange = torch.arange(num_envs, device=device)

    def install(self, env_ids: Tensor | list[int], segments: list[Segment]) -> None:
        """Install one segment per env in ``env_ids``."""
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if len(segments) != int(ids.numel()):
            raise ValueError(f"{len(segments)} segments for {int(ids.numel())} envs")
        lengths: list[int] = []
        holds: list[int] = []
        ffs: list[bool] = []
        tols: list[float] = []
        for env_id, seg in zip(ids.tolist(), segments, strict=True):
            rows = int(seg.targets.shape[0])
            if rows > self.capacity or rows == 0:
                raise ValueError(f"segment of {rows} rows (capacity {self.capacity})")
            self.pos[env_id, :rows] = seg.targets.to(self.device, non_blocking=True)
            if seg.has_ff:
                self.vel[env_id, :rows] = seg.velocity.to(self.device, non_blocking=True)
                self.acc[env_id, :rows] = seg.acceleration.to(self.device, non_blocking=True)
            else:
                self.vel[env_id, :rows] = 0.0
                self.acc[env_id, :rows] = 0.0
            self.labels[env_id] = seg.label
            lengths.append(rows)
            holds.append(int(seg.hold))
            ffs.append(seg.has_ff)
            tols.append(float("inf") if seg.settle_tol is None else float(seg.settle_tol))
        self.length[ids] = torch.as_tensor(lengths, dtype=torch.long, device=self.device)
        self.hold[ids] = torch.as_tensor(holds, dtype=torch.long, device=self.device)
        self.ff[ids] = torch.as_tensor(ffs, dtype=torch.bool, device=self.device)
        self.settle_tol[ids] = torch.as_tensor(tols, dtype=torch.float32, device=self.device)
        self.best_err[ids] = float("inf")
        self.stalled[ids] = 0
        self.cursor[ids] = 0
        self.active[ids] = True

    def needs_joint_pos(self) -> bool:
        """True when an active segment ends its hold on convergence (``settle_tol``)."""
        return bool(torch.isfinite(self.settle_tol[self.active]).any())

    def gather(self, joint_pos: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return ``(pos, vel, acc, commanded, finished)`` for this tick and advance.

        ``commanded`` is the mask of envs that had an active segment this tick (their row
        should be written to the sim); ``finished`` flags envs that consumed their last
        hold row on this tick (they are inactive afterwards). Velocity/acceleration are
        zero on hold rows so the feed-forward term vanishes while settling. With
        ``joint_pos`` (``[N, dof]`` measured), an env on its hold rows whose segment has a
        ``settle_tol`` also finishes once every joint is within that tolerance of the target, or
        once the error has stopped shrinking (a joint parked in its friction deadband would
        otherwise burn the whole ``hold`` cap).
        """
        last = (self.length - 1).clamp_(min=0)
        row = torch.minimum(self.cursor, last)
        pos = self.pos[self._arange, row]
        on_hold = (self.cursor > last).unsqueeze(-1)
        vel = torch.where(on_hold, torch.zeros_like(pos), self.vel[self._arange, row])
        acc = torch.where(on_hold, torch.zeros_like(pos), self.acc[self._arange, row])
        commanded = self.active.clone()
        end = self.length + self.hold
        finished = self.active & (self.cursor >= end - 1)
        if joint_pos is not None:
            err = (pos - joint_pos[:, : self.dof]).abs().amax(dim=-1)
            gated = torch.isfinite(self.settle_tol) & self.active & on_hold.squeeze(-1)
            improved = err < self.best_err - SETTLE_MIN_PROGRESS
            self.best_err = torch.where(gated, torch.minimum(self.best_err, err), self.best_err)
            self.stalled = torch.where(gated & ~improved, self.stalled + 1, torch.zeros_like(self.stalled))
            done = (err <= self.settle_tol) | (self.stalled >= SETTLE_STALL_TICKS)
            finished = finished | (gated & done)
        self.cursor = torch.where(self.active, self.cursor + 1, self.cursor)
        self.active = self.active & ~finished
        return pos, vel, acc, commanded, finished

    def cancel(self, env_ids: Tensor | list[int]) -> None:
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.active[ids] = False
        self.cursor[ids] = 0
        self.length[ids] = 0


def interpolate_rows(start: Tensor, end: Tensor, steps: int) -> Tensor:
    """Linear joint-space interpolation ``[B, steps, D]`` from ``start`` to ``end`` (``[B, D]``)."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    alpha = torch.linspace(0.0, 1.0, steps, device=start.device, dtype=start.dtype)
    return start[:, None, :] + (end - start)[:, None, :] * alpha[None, :, None]
