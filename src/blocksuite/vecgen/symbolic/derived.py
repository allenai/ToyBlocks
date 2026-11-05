"""Geometric facts derived from the live scene before the symbolic planner runs.

The old pipeline discovered these only after a refinement failed (``IsValidReOrientPose``,
``IsValidRotatePose``, ``IsValidRotateStartPose`` in partial-trajectory mode) and then
re-ran FF with the fact added. Here they are computed up front for every env with the
same rules, so FF picks a clear flip side or moves an obstructor first.

Facts (domain.pddl):
- ``(cant_flip b d)``: the landing spot (block xy shifted 6 cm along ``d``) has x outside
  [0.2, 0.65] or the block / landing is not in the front camera view.
- ``(reorient_obstructs x b d)``: neighbour ``x`` within 10 cm in z sits in the flip band
  (lateral < 0.10, between 5 cm behind and 10 cm ahead of the landing spot along ``d``).
- ``(rotate_obstructs x b d)``: neighbour within |dx| < 0.07 and |dy| < 0.04 of ``b``.
- ``(cant_rotate b)``: the rotate start pose (above the block) has x outside [0.325, 0.825]
  or y outside [-0.3, 0.3], or the block is not in view.
"""

from __future__ import annotations

import torch
from torch import Tensor

from blocksuite.tamp.blocks.camera_view import blocks_in_front_view
from blocksuite.vecgen.math import world_extents
from blocksuite.vecgen.symbolic.pddl import Fact

# IsValidReOrientPose: clearance a flip needs on the side it flips toward (in-swing and
# lateral band), how far back from the landing spot a neighbour still overlaps the landed
# block, and the world-frame flip direction in the table plane
REORIENT_INSWING_CLEARANCE = 0.10
REORIENT_LATERAL_CLEARANCE = 0.10
REORIENT_BACK_CLEARANCE = 0.05
REORIENT_Z_CLEARANCE = 0.10
FLIP_DIRECTION_XY = {
    "forward": (-1.0, 0.0),
    "backward": (1.0, 0.0),
    "left_flip": (0.0, -1.0),
    "right_flip": (0.0, 1.0),
}
FLIP_DIRS = ("forward", "backward", "left_flip", "right_flip")
ROTATE_DIRS = ("cw", "ccw")
FLIP_LANDING_SHIFT = 0.06
FLIP_X_RANGE = (0.2, 0.65)
ROTATE_X_RANGE = (0.325, 0.825)
ROTATE_Y_RANGE = (-0.3, 0.3)
ROTATE_DX, ROTATE_DY = 0.07, 0.04
# OnTablePlacePoseGenerator (naman/dev 02b9a693): neighbour clearance a relocated block needs
# for its own rotate (RotatePoseGenerator.validate_pose), and 3x the place candidates when ranked
PLACE_ROTATE_CLEARANCE = 0.10
SKILL_AWARE_SAMPLE_FACTOR = 3


def derived_facts(
    blocks: list[str],
    poses: Tensor,
    sizes: Tensor,
) -> list[list[Fact]]:
    """Per-env derived facts; ``poses`` ``[B, K, 7]`` (base frame), ``sizes`` ``[B, K, 3]``.

    ``blocks[j]`` is the PDDL name of block index ``j``.
    """
    b, k, _ = poses.shape
    dev = poses.device
    xy = poses[..., :2]  # [B, K, 2]
    z = poses[..., 2]
    edge = world_extents(poses[..., 3:7], sizes)[..., 2]
    in_view = blocks_in_front_view(poses.reshape(-1, 7)[:, :3], edge.reshape(-1)).to(dev).view(b, k)
    dz_ok = (z[:, :, None] - z[:, None, :]).abs() < REORIENT_Z_CLEARANCE  # [B, X, b]
    eye = torch.eye(k, dtype=torch.bool, device=dev)
    out: list[list[Fact]] = [[] for _ in range(b)]
    # ---- flips
    for d in FLIP_DIRS:
        dx, dy = FLIP_DIRECTION_XY[d]
        dvec = torch.tensor([dx, dy], device=dev)
        landing = xy + FLIP_LANDING_SHIFT * dvec  # [B, K, 2]
        land_view = (
            blocks_in_front_view(
                torch.cat([landing, z[..., None]], dim=-1).reshape(-1, 3), edge.reshape(-1)
            )
            .to(dev)
            .view(b, k)
        )
        cant = (
            (landing[..., 0] < FLIP_X_RANGE[0])
            | (landing[..., 0] > FLIP_X_RANGE[1])
            | ~in_view
            | ~land_view
        )
        rel = (
            xy[:, :, None, :] - landing[:, None, :, :]
        )  # neighbour X relative to b's landing: [B, X, b, 2]
        along = rel[..., 0] * dx + rel[..., 1] * dy
        lateral = (rel[..., 0] * dy - rel[..., 1] * dx).abs()
        obstructs = (
            (lateral < REORIENT_LATERAL_CLEARANCE)
            & (along > -REORIENT_BACK_CLEARANCE)
            & (along < REORIENT_INSWING_CLEARANCE)
            & dz_ok
            & ~eye[None]
        )
        for e in range(b):
            for j in range(k):
                if bool(cant[e, j]):
                    out[e].append(Fact("cant_flip", (blocks[j], d)))
            for x, j in obstructs[e].nonzero().tolist():
                out[e].append(Fact("reorient_obstructs", (blocks[x], blocks[j], d)))
    # ---- rotates
    diff = xy[:, :, None, :] - xy[:, None, :, :]  # [B, X, b, 2]
    rot_obs = (diff[..., 0].abs() < ROTATE_DX) & (diff[..., 1].abs() < ROTATE_DY) & ~eye[None]
    cant_rot = (
        (xy[..., 0] < ROTATE_X_RANGE[0])
        | (xy[..., 0] > ROTATE_X_RANGE[1])
        | (xy[..., 1] < ROTATE_Y_RANGE[0])
        | (xy[..., 1] > ROTATE_Y_RANGE[1])
        | ~in_view
    )
    for e in range(b):
        for j in range(k):
            if bool(cant_rot[e, j]):
                out[e].append(Fact("cant_rotate", (blocks[j],)))
        for x, j in rot_obs[e].nonzero().tolist():
            for d in ROTATE_DIRS:
                out[e].append(Fact("rotate_obstructs", (blocks[x], blocks[j], d)))
    return out




def _band_hit(points_xy: Tensor, landing_xy: Tensor, d: str) -> Tensor:
    """``reorient_obstructed`` as a broadcast tensor op: points inside the flip band of a
    flip in direction ``d`` landing at ``landing_xy``."""
    dx, dy = FLIP_DIRECTION_XY[d]
    rel = points_xy - landing_xy
    along = rel[..., 0] * dx + rel[..., 1] * dy
    lateral = (rel[..., 0] * dy - rel[..., 1] * dx).abs()
    return (
        (lateral < REORIENT_LATERAL_CLEARANCE)
        & (along > -REORIENT_BACK_CLEARANCE)
        & (along < REORIENT_INSWING_CLEARANCE)
    )


def relocation_scores(
    cand: Tensor,
    edge: Tensor,
    others: Tensor,
    other_mask: Tensor,
    own_flip: Tensor,
    own_rotate: Tensor,
    oth_flip_landing: Tensor,
    oth_flip_dir: Tensor,
    oth_flip_valid: Tensor,
    oth_rot_pos: Tensor,
    oth_rot_valid: Tensor,
) -> Tensor:
    """Score candidate place positions for a block that is being relocated (``[B, C]``).

    ``cand`` ``[B, C, 3]`` are candidate block centres (base frame), ``edge`` ``[B]`` the
    held block's edge, ``others`` ``[B, K, 3]`` the other blocks' centres with ``other_mask``
    ``[B, K]`` (False for the held block). The score counts, as in naman/dev's
    OnTablePlacePoseGenerator.skill_score, the skills this block is moved for that the spot
    allows: each flip direction in ``own_flip`` ``[B, 4]`` (landing x in range, landing in
    the front view, no neighbour in the flip band within 10 cm of z) and the rotate in
    ``own_rotate`` ``[B]`` (training x/y range, 10 cm neighbour clearance). It subtracts the
    upcoming flips and rotates of OTHER blocks the spot would obstruct: flips given by their
    landing ``[B, M, 3]``, direction index ``[B, M]`` and validity ``[B, M]``, rotates by
    the block position ``[B, R, 3]`` and validity ``[B, R]``.
    """
    b, c, _ = cand.shape
    dev = cand.device
    score = torch.zeros(b, c, device=dev)
    xy = cand[..., :2]
    z = cand[..., 2]
    for di, d in enumerate(FLIP_DIRS):
        need = own_flip[:, di]
        if bool(need.any()):
            dx, dy = FLIP_DIRECTION_XY[d]
            landing = cand + FLIP_LANDING_SHIFT * torch.tensor([dx, dy, 0.0], device=dev)
            in_x = (landing[..., 0] >= FLIP_X_RANGE[0]) & (landing[..., 0] <= FLIP_X_RANGE[1])
            view = (
                blocks_in_front_view(landing.reshape(-1, 3), edge.repeat_interleave(c))
                .to(dev)
                .view(b, c)
            )
            near_z = (others[:, None, :, 2] - z[:, :, None]).abs() < REORIENT_Z_CLEARANCE
            hit = (
                _band_hit(others[:, None, :, :2], landing[:, :, None, :2], d)
                & near_z
                & other_mask[:, None, :]
            )
            score += (need[:, None] & in_x & view & ~hit.any(dim=2)).float()
        sel = oth_flip_valid & (oth_flip_dir == di)  # [B, M]
        if bool(sel.any()):
            near_z = (z[:, :, None] - oth_flip_landing[:, None, :, 2]).abs() < REORIENT_Z_CLEARANCE
            hit = _band_hit(xy[:, :, None, :], oth_flip_landing[:, None, :, :2], d) & near_z
            score -= (hit & sel[:, None, :]).sum(dim=2).float()
    if bool(own_rotate.any()):
        in_range = (
            (cand[..., 0] >= ROTATE_X_RANGE[0])
            & (cand[..., 0] <= ROTATE_X_RANGE[1])
            & (cand[..., 1] >= ROTATE_Y_RANGE[0])
            & (cand[..., 1] <= ROTATE_Y_RANGE[1])
        )
        close = ((others[:, None, :, :2] - xy[:, :, None, :]).abs() < PLACE_ROTATE_CLEARANCE).all(
            dim=3
        ) & other_mask[:, None, :]
        score += (own_rotate[:, None] & in_range & ~close.any(dim=2)).float()
    if bool(oth_rot_valid.any()):
        diff = (xy[:, :, None, :] - oth_rot_pos[:, None, :, :2]).abs()
        hit = (diff[..., 0] < ROTATE_DX) & (diff[..., 1] < ROTATE_DY) & oth_rot_valid[:, None, :]
        score -= hit.sum(dim=2).float()
    return score
