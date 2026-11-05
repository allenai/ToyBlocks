"""    python tests/rerender/test_partial_selection.py

make_partial_dataset must read the completion flag where rerender.py actually writes it.

render_params.json carries `completed` and `episodes_recorded` at the TOP LEVEL. The codebase
describes the file as "meta/render_params.json (meta.completed=true)", where meta/ is the
DIRECTORY -- read that as a nested JSON key and EVERY problem looks unfinished. That shipped:
on 2026-09-15 two partial merges exited "0 completed problem(s)" against an output that had six
finished problems and ~1000 episodes, because the selector looked for d["meta"]["completed"].

Fixture keys mirror a real render_params.json pulled off weka.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "rerender"))
from make_partial_dataset import completed_problems, select  # noqa: E402

_failures = []


def _check(name, got, want):
    if got == want:
        print(f"  PASS  {name}")
        return
    print(f"  FAIL  {name}\n        got : {got}\n        want: {want}")
    _failures.append(name)


def _mk(root, name, eps, completed, nested=False, empty_reason=None):
    m = root / name / "blocks_success" / "meta"
    m.mkdir(parents=True, exist_ok=True)
    (root / name / "blocks_success" / "data").mkdir(parents=True, exist_ok=True)
    # the real shape: flat keys alongside args/flags/episodes/...
    d = {"args": {}, "flags": {}, "episodes": [], "dr_seeding": "hash",
         "completed": completed, "episodes_recorded": eps, "episodes_total": eps}
    if empty_reason:
        d["empty_reason"] = empty_reason
    if nested:                      # tolerated fallback shape
        d = {"meta": {"completed": completed, "episodes_recorded": eps}}
    (m / "render_params.json").write_text(json.dumps(d))
    (m / "info.json").write_text(json.dumps({"total_episodes": eps, "total_frames": eps * 1900}))


def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _mk(root, "problem_a", 46, True)
        _mk(root, "problem_b", 40, True)
        _mk(root, "problem_c", 52, True)
        _mk(root, "problem_d", 12, False)                                   # still rendering
        _mk(root, "problem_e", 0, True, empty_reason="every parquet unreadable")  # unrenderable
        _mk(root, "problem_f", 33, True, nested=True)                       # legacy nested shape

        print("A: completion is read from the TOP LEVEL, where rerender.py writes it")
        got = completed_problems(root)
        names = [n for n, _, _ in got]
        _check("flat-shape completed problems are found", 
               [n for n in names if n in ("problem_a", "problem_b", "problem_c")],
               ["problem_a", "problem_b", "problem_c"])
        _check("a problem still rendering is EXCLUDED", "problem_d" in names, False)
        _check("completed-but-zero-episodes is EXCLUDED", "problem_e" in names, False)
        _check("a nested shape is still accepted", "problem_f" in names, True)
        _check("episode counts come from render_params",
               {n: e for n, e, _ in got if n == "problem_a"}, {"problem_a": 46})

        print("\nB: selection overshoots the target, never splits a problem")
        picked, tot = select([(n, e, d) for n, e, d in got if n != "problem_f"], 110)
        _check("reaches the target", tot >= 110, True)
        _check("by whole problems only", tot, 138)
        _check("stops as soon as it is reached", len(picked), 3)

        print("\nC: selection is deterministic and supersetting")
        small, _ = select(got, 50)
        big, _ = select(got, 130)
        _check("a larger target is a superset of a smaller one",
               [n for n, _, _ in small] == [n for n, _, _ in big][:len(small)], True)

        print("\nD: --all_completed takes everything finished")
        allp, alltot = select(got, None)
        _check("all completed problems", len(allp), len(got))
        _check("total episodes", alltot, sum(e for _, e, _ in got))

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
