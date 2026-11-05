"""The LOCAL env-stage argv: every sizing flag the generator needs must actually reach it.

A dropped flag here is silent and expensive. `--num-scene-profiles` was omitted from this argv
while the pipeline's own pre-flight check validated `--num-envs` against the REQUESTED profile
count, so a run could pass validation for 4 profiles and then build 8-profile packages -- the
count is baked into the env package at codegen time (MAX_NON_SIM_STATE_RANDOMIZATIONS) and every
later stage inherits it, so nothing downstream can notice.

Everything here is CPU-only and sub-second: the pipeline imports no Isaac, and the generator's
accepted-flag set is read by AST rather than by running it (importing it would launch Isaac Sim).

Lives under tests/rerender/ because that is where the other run_datagen_pipeline local-path argv
tests live, and it is one of the directories ci-unit runs.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "scripts/datagen/run_datagen_pipeline.py"
GENERATOR = ROOT / "scripts/generate_task_from_pddl.py"
BASE_ARGS = ROOT / "src/toyblocks/cli/base_args.py"


def _pipeline_module():
    """Import the pipeline WITHOUT running it (it is a script, not a package module)."""
    spec = importlib.util.spec_from_file_location("_rdp_env", PIPELINE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_rdp_env"] = mod
    spec.loader.exec_module(mod)
    return mod


def _dataclass_fields(path: Path) -> set[str]:
    """Field names of every dataclass in `path`, read by AST.

    generate_task_from_pddl.py calls launch_app() at module scope, so importing it to ask tyro
    what it accepts would boot Isaac Sim. The field names are what tyro turns into flags.
    """
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    names.add(stmt.target.id)
    return names


def _accepted_flags() -> set[str]:
    """Flags the generator accepts, as tyro renders them, in both spellings."""
    fields = _dataclass_fields(GENERATOR) | _dataclass_fields(BASE_ARGS)
    out: set[str] = set()
    for f in fields:
        out.add("--" + f)
        out.add("--" + f.replace("_", "-"))
    return out


@pytest.fixture(scope="module")
def argv():
    """The real builder, driven by the real parser, with a non-default profile count."""
    mod = _pipeline_module()
    args = mod.build_parser().parse_args([
        "--experiment-id", "x", "--num-scene-profiles", "4",
        "--num-envs", "4",
        "--env-per-problem", "16",
    ])
    return mod.local_env_argv(args, Path("/tasks/problem_x.pddl"), "/b/envs")


def test_num_scene_profiles_reaches_the_generator(argv):
    """The regression: it used to be absent, so the package silently baked the default 8."""
    assert "--num_scene_profiles" in argv, (
        "the local env argv does not forward --num-scene-profiles; the generated package will "
        "bake the generator's own default of 8 regardless of what the run asked for"
    )


def test_num_scene_profiles_carries_the_requested_value(argv):
    """Presence is not enough -- a wrong value bakes the wrong number of profiles."""
    assert argv[argv.index("--num_scene_profiles") + 1] == "4"


@pytest.mark.parametrize("flag", [
    "--pddl", "--domain_path", "--output_dir",
    "--num_reset_states", "--num_envs", "--num_scene_profiles", "--seed",
])
def test_every_sizing_flag_is_present(argv, flag):
    """These decide what the env package contains; any of them going missing is silent."""
    assert flag in argv


def test_reset_states_come_from_env_per_problem(argv):
    """--num-reset-states is dead on every path; --env-per-problem is what is forwarded."""
    assert argv[argv.index("--num_reset_states") + 1] == "16"


def test_every_flag_is_one_the_generator_accepts(argv):
    """An unknown flag would be a hard parse error ~1 Isaac startup into the run."""
    accepted = _accepted_flags()
    unknown = [a for a in argv if a.startswith("--") and a not in accepted]
    assert not unknown, f"generate_task_from_pddl.py would reject: {unknown}"


# ------------------------------------------------------- the floor the fix makes reachable
# Validation runs before any stage; skipping every stage and writing to a throwaway root stops a
# run that passes validation right there instead of launching a real datagen.
_STOP = ("--skip-task-gen", "--skip-env", "--skip-execution", "--skip-merge", "--skip-annotate")


def _run_pipeline(*flags: str):
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as root:
        r = subprocess.run(
            [sys.executable, str(PIPELINE), "--experiment-id", "_guard", "--output-root", root,
             *_STOP, *flags],
            capture_output=True, text=True, cwd=ROOT, timeout=600,
        )
    return r.returncode, r.stdout + r.stderr


CAMTEST = "src/toyblocks/domain_randomization/flags/rerender_flags_camtest.yaml"


def test_profiles_above_the_rerender_floor_are_refused():
    """Honouring --num-scene-profiles made this reachable: the re-render maps env i to slot
    i % len(table), so fewer envs than profiles leaves the high slots unspawned and
    plan_block_sizes skips EVERY problem. Refuse up front rather than render nothing."""
    rc, out = _run_pipeline(
        "--rerender-local", "--rerender-flags", CAMTEST,
        "--num-scene-profiles", "16", "--num-envs", "16", "--exec-num-envs", "16",
    )
    assert rc != 0
    assert "--rerender-local-num-envs" in out and "at least" in out


def test_the_default_pairing_is_not_refused():
    """8 profiles against the default 8 rerender envs is the common case; it must not trip."""
    rc, out = _run_pipeline("--rerender-local", "--rerender-flags", CAMTEST)
    assert "must be at least --num-scene-profiles" not in out
