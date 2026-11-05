"""The task table must be REBUILT under randomize_task_letters, not copied and not owned.

Copying the source table would name the ORIGINAL glyphs. Owning task_index (what the code did
first) takes the RECORDER's index, which points into its own one-row table holding the generic
string "manipulation" -- so a swapped render came out with its entire task table replaced by
["manipulation"] while the unswapped control kept all four real strings. The instruction text was
not rewritten, it was discarded, and nothing downstream could tell.

These exercise the three helpers directly on synthetic parquet: no Isaac, no GPU, milliseconds.
"""
import ast
import glob as _glob
import json
import os as _os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

RERENDER = Path(__file__).resolve().parents[2] / "scripts" / "rerender" / "rerender.py"
WANT = ("_src_task_strings", "_src_task_schema", "_write_task_table",
        "_patch_episode_task_meta")


@pytest.fixture(scope="module")
def helpers():
    """Exec only the helper defs -- rerender.py's module body imports scene_swap and Isaac."""
    tree = ast.parse(RERENDER.read_text(encoding="utf-8"))
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in WANT]
    assert len(defs) == len(WANT), f"missing helpers: {set(WANT) - {d.name for d in defs}}"
    ns = {"os": _os, "glob": _glob}
    exec(compile(ast.Module(body=defs, type_ignores=[]), "rerender", "exec"), ns)
    return ns


def test_reads_the_source_task_strings(helpers, tmp_path):
    src = tmp_path / "src" / "meta"
    src.mkdir(parents=True)
    pq.write_table(pa.table({"task_index": pa.array([0, 1], type=pa.int64()),
                             "__index_level_0__": pa.array(
                                 ["pnp ['robot', 'block_a']", "done"], type=pa.large_string())}),
                   src / "tasks.parquet")
    assert helpers["_src_task_strings"](str(tmp_path / "src")) == {
        0: "pnp ['robot', 'block_a']", 1: "done"}


def test_missing_source_table_is_not_fatal(helpers, tmp_path):
    assert helpers["_src_task_strings"](str(tmp_path / "nope")) == {}


def test_writes_the_rebuilt_table_and_updates_total_tasks(helpers, tmp_path):
    meta = tmp_path / "s" / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({"total_tasks": 1}))
    strings = ["pnp ['robot', 'block_x']", "done"]
    helpers["_write_task_table"](str(tmp_path), "s", strings, verbose=False)
    back = pq.read_table(meta / "tasks.parquet").to_pydict()
    assert back["task_index"] == [0, 1]
    assert back["__index_level_0__"] == strings
    assert json.loads((meta / "info.json").read_text())["total_tasks"] == 2


def test_episode_meta_follows_the_rebuilt_indices(helpers, tmp_path):
    """`tasks` is the instruction text consumers read; the stats must not contradict the data."""
    ep = tmp_path / "s" / "meta" / "episodes" / "chunk-000"
    ep.mkdir(parents=True)
    pq.write_table(pa.table({
        "episode_index": pa.array([0, 1], type=pa.int64()),
        "tasks": pa.array([["manipulation"], ["manipulation"]], type=pa.list_(pa.string())),
        "stats/task_index/min": pa.array([[0.0], [0.0]], type=pa.list_(pa.float64())),
        "stats/task_index/max": pa.array([[0.0], [0.0]], type=pa.list_(pa.float64())),
        "stats/task_index/count": pa.array([[3.0], [2.0]], type=pa.list_(pa.float64())),
    }), ep / "file-000.parquet")
    strings = ["pnp ['robot', 'block_x']", "done"]
    helpers["_patch_episode_task_meta"](str(tmp_path), "s", strings,
                                        {0: [0, 0, 1], 1: [1, 1]}, verbose=False)
    t = pq.read_table(ep / "file-000.parquet").to_pydict()
    assert t["tasks"][0] == sorted(set(strings)), "episode 0 used both tasks"
    assert t["tasks"][1] == ["done"], "episode 1 used only the second"
    assert "manipulation" not in {s for row in t["tasks"] for s in row}, \
        "the recorder's placeholder must not survive next to a rewritten tasks.parquet"
    assert t["stats/task_index/min"][0] == [0.0] and t["stats/task_index/max"][0] == [1.0]
    assert t["stats/task_index/count"][0] == [3.0]


def test_episode_meta_patch_is_a_noop_without_indices(helpers, tmp_path):
    ep = tmp_path / "s" / "meta" / "episodes" / "chunk-000"
    ep.mkdir(parents=True)
    tbl = pa.table({"episode_index": pa.array([0], type=pa.int64()),
                    "tasks": pa.array([["manipulation"]], type=pa.list_(pa.string()))})
    pq.write_table(tbl, ep / "file-000.parquet")
    helpers["_patch_episode_task_meta"](str(tmp_path), "s", ["x"], {}, verbose=False)
    assert pq.read_table(ep / "file-000.parquet").to_pydict()["tasks"] == [["manipulation"]]


def test_rewritten_table_reads_back_like_the_source(helpers, tmp_path):
    """The task string must come back as the pandas INDEX, exactly as every source does.

    Writing a bare schema dropped the pandas metadata (index_columns=["__index_level_0__"]), so
    to_pandas() returned a RangeIndex with the string as an ordinary column. Consumers reading
    df.index would have silently got integers -- a shape change, not an error.
    """
    src = tmp_path / "src" / "meta"
    src.mkdir(parents=True)
    import pandas as pd
    pd.DataFrame({"task_index": [0, 1]},
                 index=["pnp ['robot', 'block_a']", "done"]).to_parquet(src / "tasks.parquet")
    schema = helpers["_src_task_schema"](str(tmp_path / "src"))
    assert schema is not None and schema.metadata, "source schema must carry pandas metadata"

    out = tmp_path / "out" / "s" / "meta"
    out.mkdir(parents=True)
    strings = ["pnp ['robot', 'block_x']", "done"]
    helpers["_write_task_table"](str(tmp_path / "out"), "s", strings,
                                 schema=schema, verbose=False)

    src_df = pd.read_parquet(src / "tasks.parquet")
    new_df = pd.read_parquet(out / "tasks.parquet")
    assert list(new_df.columns) == list(src_df.columns), "same columns as the source"
    assert list(new_df.index) == strings, "task string must be the INDEX"


def test_missing_source_schema_is_not_fatal(helpers, tmp_path):
    assert helpers["_src_task_schema"](str(tmp_path / "nope")) is None
    out = tmp_path / "o" / "s" / "meta"
    out.mkdir(parents=True)
    helpers["_write_task_table"](str(tmp_path / "o"), "s", ["a"], schema=None, verbose=False)
    assert (out / "tasks.parquet").is_file()


def _alignment_check():
    """Compile the LIVE _GLYPH_SHARD check out of rerender.py so the test runs the shipped code.

    A source-grep would pass on a check that compares the wrong things; this executes it.
    """
    tree = ast.parse(RERENDER.read_text(encoding="utf-8"))
    def _is_check(n):
        # the INNER if -- test is exactly flags.get("randomize_task_letters") -- not the
        # enclosing `if args_cli.episode_offset is not None:`, which also contains _tot
        if not (isinstance(n, ast.If) and isinstance(n.test, ast.Call)
                and isinstance(n.test.func, ast.Attribute) and n.test.func.attr == "get"
                and getattr(n.test.func.value, "id", "") == "flags"
                and getattr(n.test.args[0], "value", None) == "randomize_task_letters"):
            return False
        return any(isinstance(d, ast.Assign) and getattr(d.targets[0], "id", "") == "_tot"
                   for d in ast.walk(n))
    nodes = [n for n in ast.walk(tree) if _is_check(n)]
    assert len(nodes) == 1, "expected exactly one _GLYPH_SHARD alignment check"
    mod = ast.Module(body=[nodes[0]], type_ignores=[])
    ast.fix_missing_locations(mod)
    code = compile(mod, "rerender", "exec")

    def run(pairs, counts, filt, flag=True):
        ns = {"flags": {"randomize_task_letters": flag}, "_pairs": pairs, "_counts": counts,
              "_filt": filt, "_off": 0, "_M": None}
        exec(code, ns)
    return run


# (problem, split, src) triples -- the check only reads the first two
P = [("prob_a", "blocks_success", None), ("prob_a", "blocks_failure", None),
     ("prob_b", "blocks_success", None), ("prob_b", "blocks_failure", None)]
COUNTS = [3, 2, 4, 1]


def test_problem_aligned_slice_is_allowed():
    """Whole-problem slices are the normal case, and those must RUN.

    This was the 2026-10-02 outage: the guard refused --episode_offset outright, the worker always
    passes one, and all 123 workers of a 5000-episode run exited having rendered nothing.
    """
    _alignment_check()(P, COUNTS, {("prob_a", "blocks_success"): {0, 1, 2},
                                   ("prob_a", "blocks_failure"): {0, 1}})


def test_mid_problem_slice_is_refused():
    """Half a problem means another worker holds the rest; both would rebuild its tasks.parquet."""
    with pytest.raises(SystemExit) as e:
        _alignment_check()(P, COUNTS, {("prob_a", "blocks_success"): {0, 1}})   # 2 of 3
    assert "PROBLEM-ALIGNED" in str(e.value) and "prob_a/blocks_success: 2 of 3" in str(e.value)


def test_split_boundary_is_refused_because_the_unit_is_the_problem():
    """Owning every episode of ONE split of a problem is not exclusive ownership of the problem.

    replay_job rmtree's the whole problem DIRECTORY when it finds un-finalized output, and
    write_render_params writes into every split's meta dir of a touched problem -- so a fragment
    holding only blocks_success still writes over blocks_failure's metadata. The check therefore
    requires every split of every touched problem, not every touched (problem, split).
    """
    with pytest.raises(SystemExit) as e:
        _alignment_check()(P, COUNTS, {("prob_a", "blocks_success"): {0, 1, 2}})  # failure absent
    assert "prob_a/blocks_failure: 0 of 2" in str(e.value)


def test_untouched_problems_do_not_trip_the_check():
    """prob_b is wholly outside this fragment: that is the normal case, not a violation."""
    _alignment_check()(P, COUNTS, {("prob_a", "blocks_success"): {0, 1, 2},
                                   ("prob_a", "blocks_failure"): {0, 1}})


def test_check_does_not_fire_without_the_flag():
    """Copying the source table (the non-glyph path) is safe under any sharding."""
    _alignment_check()(P, COUNTS, {("prob_a", "blocks_success"): {0}}, flag=False)


def test_episodes_file_is_still_refused_and_before_the_app_launches():
    """A patch renumbers episode_index, so it redraws glyphs tasks.parquet does not describe.

    It must be refused BEFORE AppLauncher: raised from main() the SystemExit is flattened to rc=0
    by Isaac's shutdown path, so the fleet saw a clean exit with no output rather than an error --
    after paying ~245s of Kit startup for it.
    """
    src = RERENDER.read_text(encoding="utf-8")
    assert "_GLYPH_EPISODES_FILE_MSG" in src
    msg_at = src.index("_GLYPH_EPISODES_FILE_MSG = (")
    launch_at = src.index("app_launcher = AppLauncher(args_cli)")
    pre = src.index("raise SystemExit(_GLYPH_EPISODES_FILE_MSG)")
    assert msg_at < pre < launch_at, \
        "the episodes_file refusal must be defined and raised before the app launches"
    assert "args_cli.episode_offset" not in src[pre - 600:pre], \
        "the pre-launch refusal must NOT cover --episode_offset; the planner's slices are valid"


def test_ep_glyphs_is_keyed_by_split_not_by_episode_alone():
    """episode_index is the PER-SPLIT output counter, so both splits number from 0.

    Keyed on the episode alone, blocks_failure's mapping overwrote blocks_success's and the rebuild
    stamped the wrong letters into task_index/pddl_goal/glyph_map for every shared index.
    """
    tree = ast.parse(RERENDER.read_text(encoding="utf-8"))
    writes, reads = [], []
    for n in ast.walk(tree):
        if isinstance(n, ast.Subscript) and getattr(n.value, "id", "") == "ep_glyphs":
            (writes if isinstance(n.ctx, ast.Store) else reads).append(n)
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "get" and getattr(n.func.value, "id", "") == "ep_glyphs"):
            reads.append(n.args[0])
    assert writes, "ep_glyphs is never written -- the test is looking at the wrong name"
    for node in writes:
        assert isinstance(node.slice, ast.Tuple), "ep_glyphs must be keyed by (split, episode)"
    for node in reads:
        key = node.slice if isinstance(node, ast.Subscript) else node
        assert isinstance(key, ast.Tuple), "every ep_glyphs read must use the (split, episode) key"
