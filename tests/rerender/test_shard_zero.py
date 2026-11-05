"""    python tests/rerender/test_shard_zero.py

rerender must always write to the problem dir itself -- shard 0 -- never a sibling __sNNN.

The recorder picks its dir with next_free_shard(out_dir). Everything downstream of rerender (its
own output verifier, --resume, the orchestrator's completion oracle, merge, annotate) reads
out_dir, so a sharded write renders fully and then fails verification forever. That is a wedge,
and it bit rerender once already: rerender.py mkdirs out_dir immediately before constructing the
recorder, and the pre-fix next_free_shard counted that empty dir as taken.

Worst case is a source problem whose own name already ends in __sNNN (seed variants: 1185 of the
2382 problems in sep3_ai2_reorient_fixed). There the bug produces <problem>__s001__s001.
"""
import ast
import importlib.machinery
import importlib.util
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parents[2] / "scripts" / "rerender"
sys.path.insert(0, str(HERE.parents[1] / "src"))


def _stub_sim() -> None:
    specs = {
        "isaaclab": (),
        "isaaclab.scene": (),
        "isaaclab.scene.interactive_scene": ("InteractiveScene",),
        "isaaclab.assets": ("Articulation", "RigidObject"),
        "isaaclab.sensors": ("ContactSensor",),
        "isaaclab.sensors.camera": ("Camera",),
    }
    for name, attrs in specs.items():
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        for attr in attrs:
            setattr(module, attr, type(attr, (), {}))
        sys.modules[name] = module
    # PyAV / LeRobot get stand-ins only when missing (the CPU lane has neither); nothing here
    # encodes or writes. Installed ones are left alone.
    optional = {
        "av": (),
        "lerobot": (),
        "lerobot.datasets": (),
        "lerobot.datasets.lerobot_dataset": ("LeRobotDataset", "embed_images"),
    }
    installed = {top: importlib.util.find_spec(top) is not None for top in ("av", "lerobot")}
    for name, attrs in optional.items():
        if name in sys.modules or installed[name.split(".")[0]]:
            continue
        module = types.ModuleType(name)
        # A spec, so a later importlib.util.find_spec(name) in the same session (other recorder
        # tests probe for the real package) reads it instead of raising on __spec__ None.
        module.__spec__ = importlib.machinery.ModuleSpec(name, None)
        for attr in attrs:
            setattr(module, attr, type(attr, (), {}))
        sys.modules[name] = module


_stub_sim()
from blocksuite.vecgen.recording.lerobot_recorder import (  # noqa: E402
    next_free_shard, shard_dir_for,
)

_failures = []


def _check(name, got, want):
    if got == want:
        print(f"  PASS  {name}")
        return
    print(f"  FAIL  {name}\n        got : {got}\n        want: {want}")
    _failures.append(name)


def main():
    print("A: shard 0 is the problem dir itself")
    _check("plain name", Path(shard_dir_for("/d/problem_x", 0)).name, "problem_x")
    _check("__s001 source name is NOT re-suffixed",
           Path(shard_dir_for("/d/problem_x__s001", 0)).name, "problem_x__s001")
    _check("shard 1 is a sibling", Path(shard_dir_for("/d/problem_x", 1)).name, "problem_x__s001")

    print("B: what rerender actually does -- mkdir, then construct the recorder")
    for base in ("problem_x", "problem_letter_color_10_20260903__s001"):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / base
            out.mkdir(parents=True)                      # rerender.py does exactly this
            d, i = next_free_shard(str(out))
            _check(f"empty pre-created {base[:28]!r} -> shard 0", (i, Path(d).name), (0, base))

    print("C: a NON-empty dir still shards -- this is why rerender clears it first")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "problem_x__s001"
        (out / "meta").mkdir(parents=True)
        d, i = next_free_shard(str(out))
        _check("leftover content -> shard 1 (the doubled name)",
               (i, Path(d).name), (1, "problem_x__s001__s001"))
        # clearing it, as rerender now does regardless of --resume, restores shard 0
        import shutil
        shutil.rmtree(out)
        out.mkdir(parents=True)
        d, i = next_free_shard(str(out))
        _check("  after the clear -> shard 0", (i, Path(d).name), (0, "problem_x__s001"))

    print("D: sharding is OPT-OUT, and rerender opts out")
    rec_src = (HERE.parents[1] / "src" / "blocksuite" / "vecgen" / "recording"
               / "lerobot_recorder.py").read_text()
    _check("recorder takes enable_sharding", "enable_sharding: bool = True" in rec_src, True)
    _check("  default is ON, so datagen is unchanged",
           "enable_sharding: bool = True" in rec_src, True)
    _check("  disabled -> shard 0 without consulting the filesystem",
           "shard_dir, self._shard_index = Path(self._shard_base_dir), 0" in rec_src, True)
    rr = (HERE / "rerender.py").read_text()
    _check("rerender passes enable_sharding=False", "enable_sharding=False," in rr, True)

    # the real guard, called on a minimal self: it is the first statement in roll_shard
    from blocksuite.vecgen.recording.lerobot_recorder import IsaacDataRecorderLeRobot as _R

    class _Fake:
        _enable_sharding = False
    try:
        _R.roll_shard(_Fake())
        _check("roll_shard refuses when sharding is off", "no raise", "RuntimeError")
    except RuntimeError as e:
        _check("roll_shard refuses when sharding is off", "enable_sharding=False" in str(e), True)
    except Exception as e:                                    # noqa: BLE001
        _check("roll_shard refuses when sharding is off", f"{type(e).__name__}", "RuntimeError")

    print("E: rerender clears only NON-finalized output, either way")
    tree = ast.parse(rr)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "replay_job")
    clean = mkdir = raise_ln = None
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            f = node.func
            if getattr(f, "id", "") == "_clean_incomplete_output" and clean is None:
                clean = node.lineno
            if getattr(getattr(f, "value", None), "id", "") == "os" \
               and getattr(f, "attr", "") == "makedirs" and mkdir is None:
                mkdir = node.lineno
        if isinstance(node, ast.Raise) and "shard 0" in ast.dump(node):
            raise_ln = node.lineno
    _check("clears stale output", clean is not None, True)
    _check("  before mkdir + recorder", clean is not None and mkdir is not None and clean < mkdir, True)
    _check("  guarded on partial, NOT on --resume",
           "_has_partial_output(out_dir)" in rr and
           "args_cli.resume" not in rr[rr.index("clearing partial") - 300:rr.index("clearing partial")],
           True)
    _check("  a FINALIZED dir is never deleted here",
           "_job_completed(out_dir)" not in rr[rr.index("Stale PARTIAL output"):rr.index("os.makedirs(out_dir")],
           True)
    _check("backstop raise still present", raise_ln is not None, True)

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
