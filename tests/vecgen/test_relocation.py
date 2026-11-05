"""Placement ranking for relocated blocks (symbolic/derived.relocation_scores)."""

from __future__ import annotations

import torch

from blocksuite.vecgen.symbolic.derived import FLIP_DIRS, relocation_scores

Z = 0.03  # block centre on the table (base frame)


def _score(
    cand_xy: list[tuple[float, float]],
    others_xyz: list[tuple[float, float, float]],
    own_flip: str | None = None,
    own_rotate: bool = False,
    oth_flip: tuple[tuple[float, float, float], str] | None = None,
    oth_rot: tuple[float, float, float] | None = None,
) -> list[float]:
    cand = torch.tensor([[(x, y, Z) for x, y in cand_xy]])
    others = torch.tensor([others_xyz])
    mask = torch.ones(1, len(others_xyz), dtype=torch.bool)
    flips = torch.zeros(1, len(FLIP_DIRS), dtype=torch.bool)
    if own_flip is not None:
        flips[0, FLIP_DIRS.index(own_flip)] = True
    landing = torch.zeros(1, 1, 3)
    fdir = torch.zeros(1, 1, dtype=torch.long)
    fvalid = torch.zeros(1, 1, dtype=torch.bool)
    if oth_flip is not None:
        landing[0, 0] = torch.tensor(oth_flip[0])
        fdir[0, 0] = FLIP_DIRS.index(oth_flip[1])
        fvalid[0, 0] = True
    rpos = torch.zeros(1, 1, 3)
    rvalid = torch.zeros(1, 1, dtype=torch.bool)
    if oth_rot is not None:
        rpos[0, 0] = torch.tensor(oth_rot)
        rvalid[0, 0] = True
    out = relocation_scores(
        cand,
        torch.tensor([0.04]),
        others,
        mask,
        flips,
        torch.tensor([own_rotate]),
        landing,
        fdir,
        fvalid,
        rpos,
        rvalid,
    )
    return out[0].tolist()


def test_own_flip_prefers_a_clear_landing() -> None:
    # forward flips land 6 cm towards -x; a neighbour at (0.45, 0) sits in the band of a
    # block placed at (0.55, 0) but not of one placed at (0.45, 0.15)
    s = _score([(0.55, 0.0), (0.45, 0.15)], [(0.45, 0.0, Z)], own_flip="forward")
    assert s == [0.0, 1.0]


def test_own_flip_rejects_landing_out_of_range() -> None:
    # landing x 0.20 - 0.06 = 0.14 is outside [0.2, 0.65]
    s = _score([(0.20, 0.0), (0.40, 0.0)], [(0.40, 0.30, Z)], own_flip="forward")
    assert s == [0.0, 1.0]


def test_spot_in_another_blocks_flip_band_is_penalised() -> None:
    # another block flips forward and lands at (0.39, 0): a spot 4 cm beyond the landing
    # along the flip is in its band, a spot 30 cm to the side is not
    s = _score([(0.35, 0.0), (0.35, 0.30)], [(0.70, 0.30, Z)], oth_flip=((0.39, 0.0, Z), "forward"))
    assert s == [-1.0, 0.0]


def test_own_rotate_needs_clearance_and_range() -> None:
    s = _score([(0.50, 0.0), (0.50, -0.20), (0.20, -0.20)], [(0.55, 0.05, Z)], own_rotate=True)
    assert s == [0.0, 1.0, 0.0]


def test_spot_next_to_a_block_that_rotates_later_is_penalised() -> None:
    s = _score([(0.50, 0.02), (0.50, 0.20)], [(0.70, 0.30, Z)], oth_rot=(0.50, 0.0, Z))
    assert s == [-1.0, 0.0]
