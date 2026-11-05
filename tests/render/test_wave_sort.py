"""Length-sorted waves (render.py --wave_sort_block).

A wave renders every env until its longest episode ends, so grouping episodes of similar length
cuts the rendered env-frames. The sort is done inside fixed blocks of the render (uid) order, and a
Level-2 fragment loads its prefix through the end of its last block, so the waves a fragment
renders are exactly the full run's waves. render.py launches Isaac at import, so the two pure
helpers are pulled out of its AST, the same way test_split_groups does.
"""

from __future__ import annotations

import ast
import random
import sys
from pathlib import Path
from typing import Any

import pytest

RENDER = Path(__file__).resolve().parents[2] / "scripts" / "render" / "render.py"


def _load() -> dict[str, Any]:
    tree = ast.parse(RENDER.read_text())
    want = {"sort_units_into_waves", "sort_block_cap", "pack_units_by_spawn_slot"}
    nodes = [
        n
        for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in want)
        or (isinstance(n, ast.ImportFrom) and n.module == "typing")
    ]
    ns: dict[str, Any] = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "render.py", "exec"), ns)
    return ns


NS = _load()
sort_units_into_waves = NS["sort_units_into_waves"]
sort_block_cap = NS["sort_block_cap"]
pack_units_by_spawn_slot = NS["pack_units_by_spawn_slot"]


def _units(lengths: list[int]) -> list[dict[str, Any]]:
    return [{"episode_index": i, "length": n} for i, n in enumerate(lengths)]


def _rendered(waves: list[list[dict[str, Any]]]) -> int:
    return sum(max(u["length"] for u in w) * len(w) for w in waves)


def test_off_keeps_source_order() -> None:
    u = _units([5, 9, 1, 7, 3])
    assert [[x["episode_index"] for x in w] for w in sort_units_into_waves(u, 2, 0)] == [
        [0, 1],
        [2, 3],
        [4],
    ]


def test_sorted_within_blocks_and_every_unit_once() -> None:
    rng = random.Random(0)
    u = _units([rng.randint(300, 2000) for _ in range(310)])
    waves = sort_units_into_waves(u, 32, 4)
    flat = [x["episode_index"] for w in waves for x in w]
    assert sorted(flat) == list(range(310))
    assert all(len(w) <= 32 for w in waves)
    # never crosses a block: every wave's uids lie in one block of 4 * 32
    assert all(len({x["episode_index"] // 128 for x in w}) == 1 for w in waves)
    assert _rendered(waves) < _rendered(sort_units_into_waves(u, 32, 0))


def test_ties_keep_uid_order() -> None:
    waves = sort_units_into_waves(_units([4, 4, 4, 4]), 2, 1)
    assert [[x["episode_index"] for x in w] for w in waves] == [[0, 1], [2, 3]]


@pytest.mark.parametrize("last_wanted", [0, 5, 31, 32, 63, 64, 200, 309])
def test_fragment_waves_match_the_full_run(last_wanted: int) -> None:
    """A Level-2 fragment loads episodes [0, cap) of the problem; every wave holding a wanted
    episode must be a wave of the full run, member for member."""
    rng = random.Random(last_wanted)
    lengths = [rng.randint(300, 2000) for _ in range(310)]
    ne, k = 32, 3
    full = {
        x["episode_index"]: tuple(sorted(y["episode_index"] for y in w))
        for w in sort_units_into_waves(_units(lengths), ne, k)
        for x in w
    }
    cap = min(sort_block_cap(0, last_wanted, ne, k), len(lengths))
    assert cap > last_wanted
    frag = sort_units_into_waves(_units(lengths[:cap]), ne, k)
    for w in frag:
        ids = tuple(sorted(x["episode_index"] for x in w))
        if any(i <= last_wanted for i in ids):
            assert ids == full[ids[0]]


def test_cap_with_earlier_sources() -> None:
    # 40 units loaded from an earlier source: the last source's uids start at 40
    assert sort_block_cap(40, 10, 32, 1) == 64 - 40  # uid 50 -> block [32, 64)
    assert sort_block_cap(40, 30, 32, 1) == 96 - 40  # uid 70 -> block [64, 96)
    assert sort_block_cap(0, 10, 32, 0) == 11  # sorting off: plain prefix


# -- spawn-slot packing: an episode only renders in an env spawned at its recorded block size --


def _slotted(lengths: list[int], slots: list[int]) -> list[dict[str, Any]]:
    return [{"episode_index": i, "length": n, "_spawn_slot": s}
            for i, (n, s) in enumerate(zip(lengths, slots))]


@pytest.mark.parametrize("block_waves", [0, 1, 3])
def test_spawn_slot_packing_keeps_every_unit_in_an_env_of_its_slot(block_waves: int) -> None:
    rng = random.Random(block_waves)
    units = _slotted([rng.randint(300, 2000) for _ in range(150)],
                     [rng.choice([0, 1, 1, 2, 5, 7]) for _ in range(150)])
    env_slots = [0] * 3 + [1] * 8 + [2] * 5 + [5] * 6 + [7] * 10
    waves = pack_units_by_spawn_slot(units, env_slots, block_waves)
    real = [(e, u) for w in waves for e, u in enumerate(w) if not u.get("_placeholder")]
    assert sorted(u["episode_index"] for _e, u in real) == list(range(150))
    assert all(len(w) == len(env_slots) for w in waves)
    assert all(env_slots[e] == u["_spawn_slot"] for e, u in real)
    for w in waves:
        assert any(not u.get("_placeholder") for u in w)
        for u in w:
            if u.get("_placeholder"):
                assert u["length"] == 0 and u["_render"] is False


def test_spawn_slot_packing_one_wave_when_the_allocation_fits() -> None:
    # 22 episodes over 5 sizes, envs allocated to cover each size: one wave, like the old packing
    units = _slotted([500] * 22, [0] * 6 + [1] * 5 + [2] * 4 + [3] * 4 + [4] * 3)
    env_slots = [0] * 7 + [1] * 7 + [2] * 6 + [3] * 6 + [4] * 6
    waves = pack_units_by_spawn_slot(units, env_slots, 10)
    assert len(waves) == 1
    assert sum(not u.get("_placeholder") for u in waves[0]) == 22


def test_spawn_slot_packing_rejects_a_slot_without_envs() -> None:
    with pytest.raises(ValueError, match="no env"):
        pack_units_by_spawn_slot(_slotted([5, 5], [0, 3]), [0, 0, 1, 1], 1)


@pytest.mark.parametrize("last_wanted", [0, 40, 95, 96, 250])
def test_spawn_slot_fragment_waves_match_the_full_run(last_wanted: int) -> None:
    rng = random.Random(last_wanted)
    lengths = [rng.randint(300, 2000) for _ in range(260)]
    slots = [rng.randrange(4) for _ in range(260)]
    env_slots = [0] * 8 + [1] * 8 + [2] * 8 + [3] * 8
    k = 3

    def by_uid(waves: list[list[dict[str, Any]]]) -> dict[int, tuple[int, ...]]:
        out = {}
        for w in waves:
            ids = tuple(u["episode_index"] if not u.get("_placeholder") else -1 for u in w)
            for u in w:
                if not u.get("_placeholder"):
                    out[u["episode_index"]] = ids
        return out

    full = by_uid(pack_units_by_spawn_slot(_slotted(lengths, slots), env_slots, k))
    cap = min(sort_block_cap(0, last_wanted, len(env_slots), k), len(lengths))
    frag = by_uid(pack_units_by_spawn_slot(_slotted(lengths[:cap], slots[:cap]), env_slots, k))
    for uid in range(last_wanted + 1):
        assert frag[uid] == full[uid]


@pytest.mark.xfail(
    strict=True,
    reason="plan_block_sizes allocates env_slots from the units a render LOADED "
    "(allocate_env_slots(counts)), so a Level-2 fragment of a problem longer than one sort block "
    "spawns a different slot map than the full run and every one of its waves differs.",
)
def test_spawn_slot_fragment_waves_match_the_full_run_with_allocated_slots() -> None:
    sys.path.insert(0, str(RENDER.parent))
    import scene_swap

    rng = random.Random(1)
    n, ne, k = 200, 32, 3
    lengths = [rng.randint(300, 2000) for _ in range(n)]
    slots = [rng.choice([0, 1, 2, 3, 4, 5, 6, 7, 7, 7]) for _ in range(n)]

    def waves_by_uid(cap: int) -> dict[int, tuple[int, ...]]:
        units = _slotted(lengths[:cap], slots[:cap])
        counts: dict[int, int] = {}
        for u in units:
            counts[u["_spawn_slot"]] = counts.get(u["_spawn_slot"], 0) + 1
        env_slots = scene_swap.allocate_env_slots(counts, ne)
        out: dict[int, tuple[int, ...]] = {}
        for w in pack_units_by_spawn_slot(units, env_slots, k):
            ids = tuple(-1 if u.get("_placeholder") else u["episode_index"] for u in w)
            for u in w:
                if not u.get("_placeholder"):
                    out[u["episode_index"]] = ids
        return out

    full = waves_by_uid(n)
    last_wanted = 40
    frag = waves_by_uid(min(sort_block_cap(0, last_wanted, ne, k), n))
    assert [frag[u] for u in range(last_wanted + 1)] == [full[u] for u in range(last_wanted + 1)]
