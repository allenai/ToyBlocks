"""render_from_params.plan_waves: every replayed episode lands in an env of its recorded block
size, and --reproduce_placement rebuilds the recorded waves exactly or refuses.

render_from_params / render launch Isaac at import, so the pure helpers are pulled out of
their ASTs (as test_wave_sort does).
"""

from __future__ import annotations

import ast
import sys
import textwrap
import types
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2] / "scripts" / "render"
sys.path.insert(0, str(ROOT))
import scene_swap  # noqa: E402


def _extract(path: Path, names: set[str]) -> dict[str, Any]:
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body
             if (isinstance(n, ast.FunctionDef) and n.name in names)
             or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in names for t in n.targets))
             or (isinstance(n, ast.ImportFrom) and n.module == "typing")]
    ns: dict[str, Any] = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns


FP = _extract(ROOT / "render_from_params.py", {"plan_waves", "Placed"})
RR = _extract(ROOT / "render.py", {"pack_units_by_spawn_slot"})
T = types.SimpleNamespace(_scene_swap=scene_swap, pack_units_by_spawn_slot=RR["pack_units_by_spawn_slot"])
plan_waves = FP["plan_waves"]
TABLE = [0.03, 0.035, 0.04, 0.045]


@pytest.fixture
def env_file(tmp_path: Path) -> str:
    p = tmp_path / "env.py"
    sizes = ", ".join(f"[{s}, {s}, {s}]" for s in TABLE)
    p.write_text(textwrap.dedent(f"""
        class E:
            def _setup_scene_objects(self):
                block_names = ['block_a']
                block_specs = {{'block_a': {{'type': 'cuboid', 'usd_path': None}}}}
                randomizations = {{'block_sizes': {{'block_a': [{sizes}]}}}}
        """))
    return str(p)


def _pending(sizes: list[float], recs: list[dict] | None = None) -> list[tuple]:
    out = []
    for k, s in enumerate(sizes):
        rec = {"episode_index": k, "src_episode_index": k, "block_size": s}
        rec.update((recs or [{}] * len(sizes))[k])
        out.append((k, {"length": 10 + k, "cube_scale": 1.0}, rec))
    return out


def test_default_one_episode_per_wave_in_an_env_of_its_size(env_file: str) -> None:
    waves, ne = plan_waves(_pending([0.04, 0.03, 0.045]), env_file, 1, False, T)
    assert ne == len(TABLE)  # at least one env per size slot
    assert [[(env, sidx) for env, sidx, _e, _r in w] for w in waves] == [[(2, 0)], [(0, 1)], [(3, 2)]]


def test_default_packs_by_size_when_num_envs_gt_1(env_file: str) -> None:
    waves, ne = plan_waves(_pending([0.03, 0.03, 0.035, 0.03]), env_file, 8, False, T)
    assert ne == 8
    env_map = scene_swap.round_robin_slots(8, len(TABLE))
    placed = [(env, sidx) for w in waves for env, sidx, _e, _r in w]
    assert sorted(s for _env, s in placed) == [0, 1, 2, 3]
    sizes = [0.03, 0.03, 0.035, 0.03]
    assert all(TABLE[env_map[env]] == sizes[s] for env, s in placed)


def test_falls_back_to_source_cube_scale(env_file: str) -> None:
    p = [(0, {"length": 5, "cube_scale": 0.035}, {"episode_index": 0})]
    waves, _ne = plan_waves(p, env_file, 1, False, T)
    assert waves[0][0][0] == 1


def test_unknown_size_refused(env_file: str) -> None:
    with pytest.raises(SystemExit, match="not in"):
        plan_waves(_pending([0.0333]), env_file, 1, False, T)


def test_reproduce_placement_rebuilds_recorded_waves(env_file: str) -> None:
    # recorded: 8 envs (round-robin over 4 sizes); wave 1 holds episodes 0 and 2, wave 0 episode 1
    recs = [{"wave": 1, "env_slot": 2, "num_envs": 8}, {"wave": 0, "env_slot": 4, "num_envs": 8},
            {"wave": 1, "env_slot": 5, "num_envs": 8}]
    waves, ne = plan_waves(_pending([0.04, 0.03, 0.035], recs), env_file, 1, True, T)
    assert ne == 8
    assert [[(env, sidx) for env, sidx, _e, _r in w] for w in waves] == [[(4, 1)], [(2, 0), (5, 2)]]


def test_reproduce_placement_refuses_records_without_wave(env_file: str) -> None:
    with pytest.raises(SystemExit, match="wave"):
        plan_waves(_pending([0.03], [{"env_slot": 0, "num_envs": 4}]), env_file, 1, True, T)


def test_reproduce_placement_refuses_wrong_size_slot(env_file: str) -> None:
    recs = [{"wave": 0, "env_slot": 1, "num_envs": 4}]  # env 1 spawns 0.035, episode is 0.03
    with pytest.raises(SystemExit, match="size slot"):
        plan_waves(_pending([0.03], recs), env_file, 1, True, T)
