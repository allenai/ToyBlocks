"""Regression test for the enum cache's SPLIT COVERAGE contract.

    python tests/rerender/test_enum_cache.py

`--skip_failure_episodes` only ever walks blocks_success, so the cache such a run leaves in the
SOURCE dir is a partial enumeration. That file is shared by every run against that source, so
without a coverage check one skipped run poisons every later full run: it cache-hits, gets the
success-only list, and plans against a global episode list missing ~45% of its rows. The worker
does not inherit the poison (rerender.py requires every (problem, split) pair it needs and rescans
otherwise), so planner and worker then enumerate DIFFERENT lists and every --episode_offset
re-targets -- silently, because both sides hash the same short list and the plan fence passes.

Section E covers the other half of the same contract: the PLANNER
(render_sources.enumerate_source_episodes) and the WORKER (rerender.discover_jobs) must build
the SAME ordered list, because --episode_offset indexes it. A split one side accepts and the other
rejects shifts every later offset with no error.

Runs against the real modules, not a reimplementation.
"""
import contextlib
import io
import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "rerender"))
from render_sources import ENUM_CACHE_NAME, enumerate_source_episodes  # noqa: E402

PROBLEMS = ["problem_000", "problem_001"]
N_SUCCESS, N_FAILURE = 3, 2
FULL = len(PROBLEMS) * (N_SUCCESS + N_FAILURE)
SUCCESS_ONLY = len(PROBLEMS) * N_SUCCESS

_failures = []


def _check(name, got, want):
    if got == want:
        print(f"  PASS  {name}")
        return
    print(f"  FAIL  {name}\n        got : {got}\n        want: {want}")
    _failures.append(name)


def _mksrc(root):
    """Minimal source tree: _is_ds() needs meta/info.json plus a data/ dir per split."""
    for prob in PROBLEMS:
        for split, n in (("blocks_success", N_SUCCESS), ("blocks_failure", N_FAILURE)):
            d = root / prob / split
            (d / "meta").mkdir(parents=True)
            (d / "data").mkdir()
            (d / "meta" / "info.json").write_text(json.dumps({"total_episodes": n}))
    return root


def _cache(root):
    f = root / ENUM_CACHE_NAME
    return json.loads(f.read_text()) if f.exists() else None


def _legacy_cache(root, rows):
    """A schema-1 cache with no `scanned` key -- what is already sitting on weka."""
    (root / ENUM_CACHE_NAME).write_text(json.dumps(
        {"schema": 1, "src": str(root), "problems": PROBLEMS, "splits": rows}))


def main():
    both = ["blocks_success", "blocks_failure"]

    print("A: full, then skip, then full -- a skipped run must not poison the cache")
    with tempfile.TemporaryDirectory() as td:
        r = _mksrc(Path(td) / "a")
        _check("full scan sees both splits", len(enumerate_source_episodes(str(r))), FULL)
        _check("cache records what was walked", _cache(r).get("scanned"), both)
        _check("skip run sees success only",
               len(enumerate_source_episodes(str(r), skip_failures=True)), SUCCESS_ONLY)
        _check("a cache HIT does not overwrite the full cache", _cache(r).get("scanned"), both)
        _check("full run afterwards still sees both", len(enumerate_source_episodes(str(r))), FULL)

    print("B: skip FIRST, then full -- the poisoning case")
    with tempfile.TemporaryDirectory() as td:
        r = _mksrc(Path(td) / "b")
        _check("skip run sees success only",
               len(enumerate_source_episodes(str(r), skip_failures=True)), SUCCESS_ONLY)
        _check("cache admits it is partial", _cache(r).get("scanned"), ["blocks_success"])
        got = enumerate_source_episodes(str(r))
        _check("full run RESCANS instead of eating the partial cache", len(got), FULL)
        _check("and repairs the cache", _cache(r).get("scanned"), both)
        _check("both splits present", sorted({s for _p, s, _e in got}),
               ["blocks_failure", "blocks_success"])

    print("C: legacy caches already on weka (schema 1, no `scanned`)")
    with tempfile.TemporaryDirectory() as td:
        r = _mksrc(Path(td) / "c")
        _legacy_cache(r, [[p, "blocks_success", N_SUCCESS] for p in PROBLEMS])
        _check("legacy POISONED cache is rejected", len(enumerate_source_episodes(str(r))), FULL)
    with tempfile.TemporaryDirectory() as td:
        r = _mksrc(Path(td) / "d")
        _legacy_cache(r, [[p, s, n] for p in PROBLEMS
                          for s, n in (("blocks_success", N_SUCCESS), ("blocks_failure", N_FAILURE))])
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            got = enumerate_source_episodes(str(r))
        # Coverage is derived from the rows when `scanned` is absent, so a good legacy cache must
        # still HIT -- invalidating them all would cost a ~25 min rescan per source on weka.
        _check("legacy GOOD cache still hits", len(got), FULL)
        _check("  ...and it really was a hit", "cache hit" in buf.getvalue(), True)

    print("D: canonical order -- --episode_offset indexes this exact list")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "e"
        for prob in PROBLEMS:
            for split, n in (("blocks_success", 2), ("blocks_failure", 1)):
                d = root / prob / split
                (d / "meta").mkdir(parents=True)
                (d / "data").mkdir()
                (d / "meta" / "info.json").write_text(json.dumps({"total_episodes": n}))
        _check("sorted problems x [success, failure] x episode",
               [tuple(t) for t in enumerate_source_episodes(str(root))],
               [("problem_000", "blocks_success", 0), ("problem_000", "blocks_success", 1),
                ("problem_000", "blocks_failure", 0), ("problem_001", "blocks_success", 0),
                ("problem_001", "blocks_success", 1), ("problem_001", "blocks_failure", 0)])

    print("E: planner and worker must enumerate the SAME ordered list")
    import rerender  # noqa: E402  (heavy: numpy/pandas; only needed for this section)

    def worker_list(src, skip):
        out = []
        for job in rerender.discover_jobs(src, "/tmp/_parity_out", None,
                                          envs_dir=None, skip_failures=skip):
            pname = Path(job["out_dir"]).name
            for sub, succ in job["sources"]:
                split = ("blocks_success" if succ else
                         ("blocks_failure" if succ is False else "blocks"))
                for e in range(rerender._count_source_episodes(sub, None)):
                    out.append((pname, split, e))
        return out

    def mkspec(root, spec):
        for prob, splits in spec.items():
            for split, n in splits.items():
                d = root / prob / split
                (d / "meta").mkdir(parents=True)
                (d / "data").mkdir()
                (d / "meta" / "info.json").write_text(json.dumps({"total_episodes": n}))
        return root

    shapes = {
        "uniform": {f"problem_{i:03d}": {"blocks_success": 3 + i, "blocks_failure": 1 + i}
                    for i in range(4)},
        # a problem with no blocks_failure dir at all, and one whose failure split is empty
        "ragged": {"problem_000": {"blocks_success": 3, "blocks_failure": 2},
                   "problem_001": {"blocks_success": 5},
                   "problem_002": {"blocks_success": 2, "blocks_failure": 0}},
    }
    for label, spec in shapes.items():
        for skip in (False, True):
            with tempfile.TemporaryDirectory() as td:
                r = mkspec(Path(td) / "p", spec)
                planner = [tuple(x) for x in enumerate_source_episodes(str(r), skip_failures=skip)]
                _check(f"{label}, skip_failures={skip}: planner == worker ({len(planner)} eps)",
                       planner, [tuple(x) for x in worker_list(str(r), skip)])

    with tempfile.TemporaryDirectory() as td:
        # `data` as a FILE: _is_ds used is_dir() while _is_lerobot_dataset used exists(), so the
        # worker counted a split the planner did not -- +2 episodes, every later offset shifted.
        r = mkspec(Path(td) / "p", {"problem_000": {"blocks_success": 3},
                                    "problem_001": {"blocks_success": 1}})
        bad = r / "problem_001" / "blocks_failure"
        (bad / "meta").mkdir(parents=True)
        (bad / "meta" / "info.json").write_text(json.dumps({"total_episodes": 2}))
        (bad / "data").write_text("not a directory")
        _check("`data` as a file: predicates still agree",
               [tuple(x) for x in enumerate_source_episodes(str(r))],
               [tuple(x) for x in worker_list(str(r), False)])

    print("G: RERENDER_ENUM_CACHE=0 bypasses the cache on BOTH sides")
    import os
    import subprocess
    here = str(Path(__file__).resolve().parents[2] / "scripts" / "rerender")
    with tempfile.TemporaryDirectory() as td:
        r = _mksrc(Path(td) / "g")
        enumerate_source_episodes(str(r))                       # populate
        code = ("import sys;sys.path.insert(0,%r);import rerender;"
                "print('None' if rerender._enum_counts_from_cache(%r) is None else 'READ')"
                % (here, str(r)))

        def worker_reads(env):
            return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                  env={**os.environ, **env}).stdout.strip()

        _check("worker reads it by default", worker_reads({}), "READ")
        _check("worker honours the bypass", worker_reads({"RERENDER_ENUM_CACHE": "0"}), "None")

    print()
    if _failures:
        print(f"{len(_failures)} FAILURE(S): {_failures}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())


def test_main():
    assert main() == 0
