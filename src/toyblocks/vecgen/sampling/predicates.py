"""Stage-1 init-predicate samplers, ported from ``toyblocks.tamp.blocks.predicates``.

Each sampler is the ``sample_vectorized`` / ``evaluate_vectorized`` pair of the
predicate of the same name, unchanged in distribution, rejection logic and order of
random draws (torch global RNG on the sim device), so a seed samples the same states as
the TAMP-backed ``make_superset_env.py`` did. They run on a
:class:`~toyblocks.vecgen.sampling.scene.SceneSim` instead of ``tmp3``'s simulator.

Predicates without a sampler (``clear``, ``handempty``, ...) are not in :data:`SAMPLERS`.
"""

from __future__ import annotations

import ast
import math

import torch
from torch import Tensor

from toyblocks.tamp.blocks.camera_view import blocks_in_front_view
from toyblocks.vecgen.math import quat_mul
from toyblocks.vecgen.sampling.scene import SceneSim
from toyblocks.vecgen.symbolic.predicates import ORI_NAMES, ORI_QUATERNIONS, check_quat

# on_table: spawn window as table-frame offsets (on_table.DEFAULT_SPAWN_*_INTERVALS);
# set_spawn_ranges overrides it.
SPAWN_X_INTERVALS: list[tuple[float, float]] = [(-0.1, 0.25)]
SPAWN_Y_INTERVALS: list[tuple[float, float]] = [(-0.30, 0.30)]


def parse_spawn_intervals(spec, axis: str = "") -> list[tuple[float, float]]:
    """Normalize a spawn-range spec into a list of ``(lo, hi)`` intervals.

    Accepts a single interval (``[-0.2, 0.58]``), a list of disjoint intervals
    (``[[-0.2, -0.1], [0.3, 0.5]]``), or a string holding either -- so the same
    spec works from a PDDL comment, a CLI flag and Python.
    """
    if spec is None:
        return []
    if isinstance(spec, str):
        text = spec.strip()
        if not text:
            return []
        # Bare "[a,b],[c,d]" (no outer brackets) is a tuple once evaluated.
        try:
            spec = ast.literal_eval(text)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"could not parse {axis or 'spawn range'}: {spec!r}: {exc}") from exc

    try:
        items = list(spec)
    except TypeError as exc:
        raise ValueError(f"{axis or 'spawn range'} must be a sequence, got {spec!r}") from exc
    if not items:
        return []

    # Flat [lo, hi] vs nested [[lo, hi], ...].
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in items):
        items = [items]

    intervals: list[tuple[float, float]] = []
    for raw in items:
        try:
            lo, hi = (float(v) for v in raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{axis or 'spawn range'} interval must be a [lo, hi] pair, got {raw!r}"
            ) from exc
        if hi < lo:
            raise ValueError(
                f"{axis or 'spawn range'} interval {raw!r} is inverted (hi < lo)"
            )
        intervals.append((lo, hi))

    intervals.sort()
    for (lo_a, hi_a), (lo_b, _) in zip(intervals, intervals[1:]):
        if lo_b < hi_a:
            raise ValueError(
                f"{axis or 'spawn range'} intervals overlap: "
                f"[{lo_a}, {hi_a}] and [{lo_b}, ...]. Use disjoint intervals."
            )
    return intervals


def set_spawn_ranges(block_x_range: object = None, block_y_range: object = None) -> None:
    """Configure the on_table spawn window (``on_table.set_spawn_ranges``). ``None`` / empty
    keeps the current window; each accepts one ``[lo, hi]`` or a list of disjoint intervals."""
    global SPAWN_X_INTERVALS, SPAWN_Y_INTERVALS
    x = parse_spawn_intervals(block_x_range, "block_x_range")
    y = parse_spawn_intervals(block_y_range, "block_y_range")
    if x:
        SPAWN_X_INTERVALS = x
    if y:
        SPAWN_Y_INTERVALS = y
    if x or y:
        OnTable._feasibility_checked = False
        print(f"[on_table] spawn window (table frame): x={SPAWN_X_INTERVALS} y={SPAWN_Y_INTERVALS}")
MAX_REACH_RADIUS = 0.65
ROBOT_BASE_X = 0.0
ON_TABLE_MAX_ROUNDS = 5000
ON_TABLE_SPACING = 0.07
ON_TABLE_TOL = 0.02
# ori_of
ORI_YAW_PERTURB_DEG = 30.0
ORI_YAW_TOL_DEG = 40.0
ORI_AXIS_TOL_DEG = 5.0
# next_to / on_right
CARDINAL_OFFSETS: dict[str, tuple[int, int]] = {
    "front": (1, 0),
    "back": (-1, 0),
    "right": (0, 1),
    "left": (0, -1),
}
ADJACENCY_GAP = 0.005
# on_left / on_front / on_back (fixed 4.5 cm cubes + 2 cm gap, as in the predicates)
FIXED_CUBE_WIDTH = 0.045
FIXED_SIDE_OFFSET = 0.02
FIXED_FRONT_OFFSET = 0.005  # on_front.py uses a 5 mm offset, on_left / on_back 2 cm


def sample_intervals(intervals: list[tuple[float, float]], n: int, device: torch.device) -> Tensor:
    """``n`` draws uniform over the union of ``intervals`` (``on_table._sample_intervals_torch``)."""
    lo = torch.tensor([a for a, _ in intervals], device=device, dtype=torch.float32)
    hi = torch.tensor([b for _, b in intervals], device=device, dtype=torch.float32)
    widths = hi - lo
    total = float(widths.sum())
    if total <= 0.0:
        k = torch.randint(len(intervals), (n,), device=device)
        return lo[k]
    edges = torch.cumsum(widths, dim=0)
    t = torch.rand(n, device=device) * total
    k = torch.searchsorted(edges, t, right=True).clamp_(max=len(intervals) - 1)
    prev = torch.cat([torch.zeros(1, device=device, dtype=edges.dtype), edges[:-1]])
    return lo[k] + (t - prev[k])


def quat_unique(q: Tensor) -> Tensor:
    """Quaternion with non-negative real part (``isaaclab.utils.math.quat_unique``)."""
    return torch.where(q[..., 0:1] < 0, -q, q)


def sample_world_yaw_perturb(
    q_base: Tensor, yaw_range_deg: float, n: int, device: torch.device, eps: float = 1e-8
) -> Tensor:
    """``n`` copies of ``q_base`` ``[4]`` each left-multiplied by a uniform world-z yaw
    in ``[-range, range]`` (``tmp3.Utils.quat_utils.sample_world_yaw_perturb``)."""
    q_base = q_base / (q_base.norm(dim=-1, keepdim=True) + eps)
    q_base = quat_unique(q_base)
    yaw_range = math.radians(yaw_range_deg)
    dpsi = (2.0 * torch.rand((n,), device=device, dtype=torch.float32) - 1.0) * yaw_range
    q_base = q_base.unsqueeze(0).expand(n, 4)
    q_delta = torch.zeros(n, 4, device=device, dtype=torch.float32)
    q_delta[:, 0] = torch.cos(0.5 * dpsi)
    q_delta[:, 3] = torch.sin(0.5 * dpsi)
    q = quat_mul(q_delta, q_base)
    q = q / (q.norm(dim=-1, keepdim=True) + eps)
    return quat_unique(q)


def bbox_sizes(sim: SceneSim, name: str) -> Tensor:
    """Per-env asset bounding-box extents ``[num_envs, 3]`` (``next_to._get_obj_sizes``)."""
    from toyblocks.utils.obs_utils import get_bounding_box  # imports isaaclab (after launch)

    bbox = get_bounding_box(sim.scene[name])
    return bbox[:, 1] - bbox[:, 0]


class PredicateSampler:
    """One init fact: ``sample`` moves its first block in the given envs, ``evaluate``
    checks the fact there."""

    def __init__(self, args: list[str]) -> None:
        self.args = list(args)

    def sample(self, sim: SceneSim, env_indices: list[int]) -> None:
        raise NotImplementedError

    def evaluate(self, sim: SceneSim, env_indices: list[int]) -> list[bool]:
        raise NotImplementedError


class OnTable(PredicateSampler):
    """``on_table(block, table)``: uniform over the spawn window, kept 7 cm clear of every
    other block, within reach and inside the front camera view."""

    _feasibility_checked = False

    @classmethod
    def _check_spawn_window(
        cls, table_local: Tensor, heights: Tensor, device: torch.device
    ) -> None:
        """``on_table._warn_if_spawn_window_infeasible`` (once per process; it also draws
        from the RNG, so keeping it keeps the random stream of the TAMP sampler)."""
        if cls._feasibility_checked:
            return
        cls._feasibility_checked = True
        n = 4096
        anchor = table_local[0]
        h = heights.mean().expand(n)
        probe = torch.stack(
            [
                anchor[0] + sample_intervals(SPAWN_X_INTERVALS, n, device),
                anchor[1] + sample_intervals(SPAWN_Y_INTERVALS, n, device),
                anchor[2] + h / 2 + 0.001,
            ],
            dim=1,
        )
        in_reach = torch.hypot(probe[:, 0] - ROBOT_BASE_X, probe[:, 1]) <= MAX_REACH_RADIUS
        in_view = blocks_in_front_view(probe, h)
        ok = float((in_reach & in_view).float().mean())
        if ok < 0.20:
            print(
                f"[on_table] WARNING: only {ok * 100:.1f}% of the spawn window "
                f"x={SPAWN_X_INTERVALS} y={SPAWN_Y_INTERVALS} (table frame) is usable "
                f"(reach {float(in_reach.float().mean()) * 100:.1f}%, "
                f"view {float(in_view.float().mean()) * 100:.1f}%)."
            )

    def sample(self, sim: SceneSim, env_indices: list[int]) -> None:
        obj, table = self.args[0], self.args[1]
        device = sim.device
        n = len(env_indices)
        idx = torch.tensor(env_indices, device=device, dtype=torch.long)
        wrapper = sim.get_rigid_object(obj)
        heights = torch.tensor(
            [wrapper.get_size(ei)[2] for ei in env_indices], device=device, dtype=torch.float32
        )
        rel_z = heights / 2 + 0.001
        table_local = sim.get_pose(table, env_idx=None, local=True)[idx, :3]
        self._check_spawn_window(table_local, heights, device)
        others = [
            sim.get_pose(b, env_idx=None, local=True)[idx, :2]
            for b in sim.obj_prim_scene_mapping
            if b != obj
        ]
        current = sim.get_pose(obj, env_idx=None, local=True)[idx, :7]
        final = torch.zeros(n, 3, device=device)
        valid = torch.zeros(n, dtype=torch.bool, device=device)
        for _ in range(ON_TABLE_MAX_ROUNDS):
            if valid.all():
                break
            pending = ~valid
            n_pending = int(pending.sum().item())
            rand_x = sample_intervals(SPAWN_X_INTERVALS, n_pending, device)
            rand_y = sample_intervals(SPAWN_Y_INTERVALS, n_pending, device)
            tl = table_local[pending]
            proposed = torch.stack(
                [tl[:, 0] + rand_x, tl[:, 1] + rand_y, tl[:, 2] + rel_z[pending]], dim=1
            )
            free = torch.ones(n_pending, dtype=torch.bool, device=device)
            for oxy in others:
                o = oxy[pending]
                free &= ~(
                    ((proposed[:, 0] - o[:, 0]).abs() < ON_TABLE_SPACING)
                    & ((proposed[:, 1] - o[:, 1]).abs() < ON_TABLE_SPACING)
                )
            free &= torch.hypot(proposed[:, 0] - ROBOT_BASE_X, proposed[:, 1]) <= MAX_REACH_RADIUS
            free &= blocks_in_front_view(proposed, heights[pending])
            pending_idx = pending.nonzero(as_tuple=True)[0]
            final[pending_idx[free]] = proposed[free]
            valid[pending_idx[free]] = True
        if not valid.all():
            left = (~valid).nonzero(as_tuple=True)[0]
            n_left = int(left.numel())
            tl = table_local[left]
            final[left, 0] = tl[:, 0] + sample_intervals(SPAWN_X_INTERVALS, n_left, device)
            final[left, 1] = tl[:, 1] + sample_intervals(SPAWN_Y_INTERVALS, n_left, device)
            final[left, 2] = tl[:, 2] + rel_z[left]
            print(
                f"[on_table] WARNING: could not place {obj} in {n_left}/{n} envs after "
                f"{ON_TABLE_MAX_ROUNDS} rounds; using unchecked positions "
                f"(reach/camera/collision constraints NOT satisfied)."
            )
        for i, ei in enumerate(env_indices):
            pose = current[i].clone()
            pose[:3] = final[i]
            sim.set_rigid_object_pose(obj, pose, env_idx=ei, local=True)

    def evaluate(self, sim: SceneSim, env_indices: list[int]) -> list[bool]:
        # No z check, as in on_table.evaluate_vectorized (the sampler puts blocks on the
        # table's root frame, half a slab below its top surface).
        obj, table = self.args[0], self.args[1]
        device = sim.device
        idx = torch.tensor(env_indices, device=device, dtype=torch.long)
        wrapper = sim.get_rigid_object(table)
        extents = torch.tensor(
            [[float(s[0]), float(s[1])] for s in (wrapper.get_size(ei) for ei in env_indices)],
            device=device,
            dtype=torch.float32,
        )
        table_local = sim.get_pose(table, env_idx=None, local=True)[idx, :3]
        xy = sim.get_pose(obj, env_idx=None, local=True)[idx, :2]
        half = extents / 2
        within_x = (xy[:, 0] >= table_local[:, 0] - half[:, 0] - ON_TABLE_TOL) & (
            xy[:, 0] <= table_local[:, 0] + half[:, 0] + ON_TABLE_TOL
        )
        within_y = (xy[:, 1] >= table_local[:, 1] - half[:, 1] - ON_TABLE_TOL) & (
            xy[:, 1] <= table_local[:, 1] + half[:, 1] + ON_TABLE_TOL
        )
        return (within_x & within_y).tolist()


class On(PredicateSampler):
    """``on(top, bottom)``: top block centred on the bottom one, resting on it."""

    def sample(self, sim: SceneSim, env_indices: list[int]) -> None:
        top, bottom = self.args[0], self.args[1]
        device = sim.device
        idx = torch.tensor(env_indices, device=device, dtype=torch.long)
        tw, bw = sim.get_rigid_object(top), sim.get_rigid_object(bottom)
        top_half = torch.tensor([tw.get_size(ei)[2] / 2 for ei in env_indices], device=device)
        bottom_half = torch.tensor([bw.get_size(ei)[2] / 2 for ei in env_indices], device=device)
        top_asset, bottom_asset = sim.scene[top], sim.scene[bottom]
        all_top = top_asset.data.root_state_w[:, :7].clone()
        all_bottom = bottom_asset.data.root_state_w[:, :7].clone()
        all_top[idx, :3] = all_bottom[idx, :3]
        all_top[idx, 2] += bottom_half + top_half
        top_asset.write_root_pose_to_sim(all_top)
        bottom_asset.write_root_pose_to_sim(all_bottom)

    def evaluate(self, sim: SceneSim, env_indices: list[int]) -> list[bool]:
        idx = torch.tensor(env_indices, device=sim.device, dtype=torch.long)
        p1 = sim.scene[self.args[0]].data.root_state_w[idx, :3]
        p2 = sim.scene[self.args[1]].data.root_state_w[idx, :3]
        ok = (
            ((p1[:, 0] - p2[:, 0]).abs() <= 0.02)
            & ((p1[:, 1] - p2[:, 1]).abs() <= 0.02)
            & (p1[:, 2] > p2[:, 2])
            & ((p1[:, 2] - p2[:, 2]) <= 0.09)
        )
        return ok.tolist()


class OriOf(PredicateSampler):
    """``ori_of(block, oN)``: canonical orientation with a +-30 deg world yaw, lifted 2 cm."""

    def sample(self, sim: SceneSim, env_indices: list[int]) -> None:
        name, face = self.args[0], self.args[1]
        device = sim.device
        idx = torch.tensor(env_indices, device=device, dtype=torch.long)
        asset = sim.scene[name]
        all_poses = asset.data.root_state_w[:, :7].clone()
        base = ORI_QUATERNIONS[ORI_NAMES.index(face.lower())].to(device)
        quats = sample_world_yaw_perturb(base, ORI_YAW_PERTURB_DEG, len(env_indices), device)
        all_poses[idx, 2] += 0.02
        all_poses[idx, 3:] = quats
        asset.write_root_pose_to_sim(all_poses)

    def evaluate(self, sim: SceneSim, env_indices: list[int]) -> list[bool]:
        idx = torch.tensor(env_indices, device=sim.device, dtype=torch.long)
        q = sim.scene[self.args[0]].data.root_state_w[idx, 3:7]
        tgt = ORI_QUATERNIONS[ORI_NAMES.index(self.args[1].lower())].to(sim.device)
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
        ok = check_quat(q, tgt.expand_as(q), ORI_YAW_TOL_DEG, ORI_AXIS_TOL_DEG)
        return ok.tolist()


class NextTo(PredicateSampler):
    """``next_to(a, b)``: a touching b on a random cardinal side."""

    def sample(self, sim: SceneSim, env_indices: list[int]) -> None:
        a, b = self.args[0], self.args[1]
        ei = torch.tensor(env_indices)
        size_a = bbox_sizes(sim, a)[ei]
        size_b = bbox_sizes(sim, b)[ei]
        poses_b = torch.stack([sim.get_pose(b, env_idx=i) for i in env_indices])
        offsets = torch.tensor(
            list(CARDINAL_OFFSETS.values()), device=poses_b.device, dtype=poses_b.dtype
        )
        pick = torch.randint(len(offsets), (len(env_indices),), device=poses_b.device)
        new = poses_b.clone()
        new[:, 0] += offsets[pick, 0] * (size_a[:, 0] / 2.0 + size_b[:, 0] / 2.0)
        new[:, 1] += offsets[pick, 1] * (size_a[:, 1] / 2.0 + size_b[:, 1] / 2.0)
        for i, env_idx in enumerate(env_indices):
            sim.set_rigid_object_pose(a, new[i], env_idx=env_idx, local=True)

    def evaluate(self, sim: SceneSim, env_indices: list[int]) -> list[bool]:
        a, b = self.args[0], self.args[1]
        ei = torch.tensor(env_indices)
        size_a = bbox_sizes(sim, a)[ei]
        size_b = bbox_sizes(sim, b)[ei]
        pa = torch.stack([sim.get_pose(a, env_idx=i, local=True)[:3] for i in env_indices])
        pb = torch.stack([sim.get_pose(b, env_idx=i, local=True)[:3] for i in env_indices])
        dx = (pa[:, 0] - pb[:, 0]).abs()
        dy = (pa[:, 1] - pb[:, 1]).abs()
        z_ok = torch.isclose(pa[:, 2], pb[:, 2], atol=0.02)
        ex = (size_a[:, 0] / 2.0 + size_b[:, 0] / 2.0).to(dtype=pa.dtype)
        ey = (size_a[:, 1] / 2.0 + size_b[:, 1] / 2.0).to(dtype=pa.dtype)
        zero = torch.zeros_like(dx)
        x_adj = torch.isclose(dx, ex, atol=0.03) & torch.isclose(dy, zero, atol=0.04)
        y_adj = torch.isclose(dy, ey, atol=0.03) & torch.isclose(dx, zero, atol=0.04)
        return (z_ok & (x_adj | y_adj)).tolist()


class OnRight(PredicateSampler):
    """``on_right(a, b)``: a at +y of b, touching (bounding-box halves + 5 mm gap)."""

    @staticmethod
    def offset(sim: SceneSim, a: str, b: str, env_idx: int) -> float:
        half_a = float(bbox_sizes(sim, a)[env_idx, 1]) / 2.0
        half_b = float(bbox_sizes(sim, b)[env_idx, 1]) / 2.0
        return half_a + half_b + ADJACENCY_GAP

    def sample(self, sim: SceneSim, env_indices: list[int]) -> None:
        a, b = self.args[0], self.args[1]
        for env_idx in env_indices:
            pose_b = sim.get_pose(b, env_idx=env_idx, local=True)
            pose_a = sim.get_pose(a, env_idx=env_idx, local=True)
            new = pose_b.clone()
            new[1] += self.offset(sim, a, b, env_idx)
            new[3:] = pose_a[3:]
            sim.set_rigid_object_pose(a, new, env_idx=env_idx, local=True)

    def evaluate(self, sim: SceneSim, env_indices: list[int]) -> list[bool]:
        a, b = self.args[0], self.args[1]
        out: list[bool] = []
        for env_idx in env_indices:
            pa = sim.get_pose(a, env_idx=env_idx)[:3]
            pb = sim.get_pose(b, env_idx=env_idx)[:3]
            target = torch.tensor(self.offset(sim, a, b, env_idx), device=pa.device)
            y_ok = torch.isclose((pa[1] - pb[1]).abs(), target, atol=0.03)
            z_ok = torch.isclose(pa[2], pb[2], atol=0.02)
            x_ok = torch.isclose(pa[0], pb[0], atol=0.04)
            out.append(bool(y_ok and z_ok and x_ok))
        return out


class _FixedSide(PredicateSampler):
    """``on_left`` / ``on_front`` / ``on_back``: a beside b along ``axis`` (sign ``sign``)
    at a fixed centre distance (6.5 cm; 5 cm for ``on_front``), 1 cm higher, with b's
    orientation."""

    axis = 0
    sign = 1.0
    offset = FIXED_SIDE_OFFSET

    def sample(self, sim: SceneSim, env_indices: list[int]) -> None:
        a, b = self.args[0], self.args[1]
        distance = FIXED_CUBE_WIDTH + self.offset
        for env_idx in env_indices:
            new = sim.get_pose(b, env_idx=env_idx, local=True).clone()
            new[self.axis] += self.sign * distance
            new[2] += 0.01
            sim.set_rigid_object_pose(a, new, env_idx=env_idx, local=True)

    def evaluate(self, sim: SceneSim, env_indices: list[int]) -> list[bool]:
        a, b = self.args[0], self.args[1]
        other = 1 - self.axis
        out: list[bool] = []
        for env_idx in env_indices:
            pa = sim.get_pose(a, env_idx=env_idx)[:3]
            pb = sim.get_pose(b, env_idx=env_idx)[:3]
            distance = torch.tensor(FIXED_CUBE_WIDTH + self.offset, device=pa.device)
            d_ok = torch.isclose((pa[self.axis] - pb[self.axis]).abs(), distance, atol=0.01)
            z_ok = torch.isclose(pa[2], pb[2], atol=0.01)
            o_ok = torch.isclose(pa[other], pb[other], atol=0.04)
            out.append(bool(d_ok and z_ok and o_ok))
        return out


class OnLeft(_FixedSide):
    axis = 1
    sign = -1.0


class OnFront(_FixedSide):
    axis = 0
    sign = 1.0
    offset = FIXED_FRONT_OFFSET


class OnBack(_FixedSide):
    axis = 0
    sign = -1.0


SAMPLERS: dict[str, type[PredicateSampler]] = {
    "on_table": OnTable,
    "on": On,
    "ori_of": OriOf,
    "next_to": NextTo,
    "on_right": OnRight,
    "on_left": OnLeft,
    "on_front": OnFront,
    "on_back": OnBack,
}


def make_sampler(name: str, args: list[str]) -> PredicateSampler | None:
    """Sampler for init fact ``name(args)``, or None for facts nothing samples."""
    cls = SAMPLERS.get(name)
    return None if cls is None else cls(args)
