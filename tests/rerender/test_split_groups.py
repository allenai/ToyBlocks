"""--split_groups: a vecgen problem_grp_* dir renders as one work item per member problem.

A vectorized datagen group packs several PDDL problems into one env and one execution dir, its
episodes interleaved across members and tagged by `source_problem`. These tests build such a source
on disk and check the planner's enumeration, the item plan, the per-item episode selection that
rerender.py compiles from it, rerender.py's job expansion and the render_params record -- all on
CPU, without Isaac.
"""

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

RERENDER = Path(__file__).resolve().parents[2] / "scripts" / "rerender"
sys.path.insert(0, str(RERENDER))
import completion as bp  # noqa: E402
import render_sources as rd  # noqa: E402

GROUP = "problem_grp_problem_declarative_10_a"
PLAIN = "problem_declarative_10_z"
# Interleaved like vecgen writes them: episode i belongs to MEMBERS[i % 3] (success split).
MEMBERS = ["problem_declarative_10_m1", "problem_declarative_10_m2", "problem_declarative_10_m3"]
N_SUCCESS, N_FAILURE, N_PLAIN = 9, 4, 5


def _split(d: Path, episodes: list[int], members: list[str] | None) -> None:
    (d / "meta").mkdir(parents=True)
    (d / "data" / "chunk-000").mkdir(parents=True)
    (d / "meta" / "info.json").write_text(json.dumps({"total_episodes": len(episodes)}))
    rows: dict[str, list] = {"episode_index": [], "frame_index": []}
    if members is not None:
        rows["source_problem"] = []
    for e in episodes:
        for f in range(3):                                    # several frames per episode
            rows["episode_index"].append(e)
            rows["frame_index"].append(f)
            if members is not None:
                rows["source_problem"].append(members[e % len(members)])
    pq.write_table(pa.table(rows), d / "data" / "chunk-000" / "file-000.parquet")


@pytest.fixture
def src(tmp_path: Path) -> Path:
    root = tmp_path / "execution"
    _split(root / GROUP / "blocks_success", list(range(N_SUCCESS)), MEMBERS)
    _split(root / GROUP / "blocks_failure", list(range(N_FAILURE)), MEMBERS[::-1])
    _split(root / PLAIN / "blocks_success", list(range(N_PLAIN)), None)
    for p in (GROUP, PLAIN):
        (tmp_path / "envs" / p).mkdir(parents=True)
        (tmp_path / "envs" / p / "env.py").write_text("block_names = ['block_a']\n")
    return root


def test_off_by_default_keeps_the_group_as_one_problem(src: Path) -> None:
    glob = rd.enumerate_source_episodes(str(src))
    assert {p for p, _s, _e in glob} == {GROUP, PLAIN}
    assert len(glob) == N_SUCCESS + N_FAILURE + N_PLAIN


def test_split_expands_members_contiguously_with_group_indices(src: Path) -> None:
    glob = rd.enumerate_source_episodes(str(src), split_groups=True)
    assert len(glob) == N_SUCCESS + N_FAILURE + N_PLAIN
    order = [p for p, _s, _e in glob]
    # members replace the group in its sorted place, each contiguous; the plain problem untouched
    runs = [p for i, p in enumerate(order) if i == 0 or order[i - 1] != p]
    assert runs == [PLAIN] + sorted(MEMBERS)
    m1 = [(s, e) for p, s, e in glob if p == MEMBERS[0]]
    # success episodes 0,3,6 (i % 3 == 0) and failure episodes whose reversed member is m1: 2
    assert m1 == [("blocks_success", 0), ("blocks_success", 3), ("blocks_success", 6),
                  ("blocks_failure", 2)]
    # skip_failures keeps only success rows, still per member
    succ = rd.enumerate_source_episodes(str(src), skip_failures=True, split_groups=True)
    assert len(succ) == N_SUCCESS + N_PLAIN and {s for _p, s, _e in succ} == {"blocks_success"}


def test_group_cache_is_reused(src: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = rd.enumerate_source_episodes(str(src), split_groups=True)
    assert (src / rd.GROUP_CACHE_NAME).exists()

    def boom(sub: Path) -> dict:
        raise AssertionError("re-read the group parquet despite a valid cache")

    monkeypatch.setattr(rd, "_group_split_members", boom)
    assert rd.enumerate_source_episodes(str(src), split_groups=True) == first


def test_member_names_must_be_unique(src: Path) -> None:
    other = src / "problem_grp_problem_declarative_10_b" / "blocks_success"
    _split(other, [0, 1], MEMBERS[:1])                       # m1 claimed by two groups
    with pytest.raises(SystemExit, match="appears in both"):
        rd.group_members(str(src), ("blocks_success",), use_cache=False)


def test_rerender_command_emits_the_flag() -> None:
    kw = dict(python="py", script="rr.py", src="s", envs_dir=None, out_root="o", flags="f", n=1,
              num_envs=32, assets_paths=[])
    item = {"episode_offset": 0, "max_episodes": 3, "problems": [MEMBERS[0]]}
    assert "--split_groups" in bp.rerender_command(item, split_groups=True, **kw)
    assert "--split_groups" not in bp.rerender_command(item, **kw)


def _discover_jobs():
    """rerender.discover_jobs and the helpers it needs, loaded from source (rerender.py launches
    Isaac at import time)."""
    src = (RERENDER / "rerender.py").read_text()
    tree = ast.parse(src)
    want = {"_is_lerobot_dataset", "_base_problem", "_resolve_env_file", "discover_jobs"}
    nodes = [n for n in tree.body
             if (isinstance(n, ast.FunctionDef) and n.name in want)
             or (isinstance(n, ast.ImportFrom) and n.module == "typing")
             or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "_SEED_SUFFIX"
                                                   for t in n.targets))]
    ns: dict[str, Any] = {"Path": Path, "re": re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "rerender.py", "exec"), ns)
    return ns["discover_jobs"]


def test_discover_jobs_expands_members(src: Path, tmp_path: Path) -> None:
    discover = _discover_jobs()
    envs = str(tmp_path / "envs")
    plain = discover(str(src), str(tmp_path / "out"), None, envs_dir=envs)
    assert [Path(j["out_dir"]).name for j in plain] == [PLAIN, GROUP]
    jobs = discover(str(src), str(tmp_path / "out"), None, envs_dir=envs, split_groups=True)
    assert [Path(j["out_dir"]).name for j in jobs] == [PLAIN] + sorted(MEMBERS)
    m2 = jobs[2]
    assert m2["member"] == MEMBERS[1] and m2["group"] == GROUP
    assert m2["env_file"] == str(tmp_path / "envs" / GROUP / "env.py")
    assert [Path(d).name for d, _ok in m2["sources"]] == ["blocks_success", "blocks_failure"]
    assert m2["member_episodes"] == {"blocks_success": [1, 4, 7], "blocks_failure": [1]}
    assert "member" not in jobs[0]


def test_render_params_record_the_member() -> None:
    sys.path.insert(0, str(RERENDER.parents[1] / "src"))
    from blocksuite.domain_randomization.render_params import render_params_payload

    job = {"out_dir": "/o/m1", "sources": [("/s/g/blocks_success", True)], "env_file": None,
           "group": GROUP, "member": MEMBERS[0]}
    rp = render_params_payload(job, {}, 25, False, {}, [], True, {})
    assert rp["split_group"] == {"group": GROUP, "problem": MEMBERS[0]}
    plain = render_params_payload({k: v for k, v in job.items() if k not in ("group", "member")},
                                  {}, 25, False, {}, [], True, {})
    assert "split_group" not in plain
