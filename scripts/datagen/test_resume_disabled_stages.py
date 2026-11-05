"""--resume must not let a stage the run disables decide where it resumes.

A disabled stage never writes a completion marker (--vecgen used to turn off env and the since
removed planning stage; --vecgen-problems-per-env still turns env off). apply_resume used to take
the first stage without a marker as the resume point, so every vecgen --resume restarted at 'env'
and deleted a finished execution/ (oct26_stack3_ori_vg_1000x32, 2026-10-27).
"""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent / "run_datagen_pipeline.py"


@pytest.fixture(scope="module")
def rdp():
    spec = importlib.util.spec_from_file_location("run_datagen_pipeline_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _resume(
    rdp, tmp_path: Path, markers: list[str], disabled: set[str]
) -> tuple[set[str], set[str]]:
    outputs = {key: [str(tmp_path / f"out_{key}")] for key, _ in rdp.STEPS}
    for paths in outputs.values():
        Path(paths[0]).mkdir()
    for key in markers:
        rdp.mark_stage_complete(str(tmp_path), key)
    enabled = {key: key not in disabled for key, _ in rdp.STEPS}
    rdp.apply_resume(str(tmp_path), enabled, outputs)
    still_enabled = {k for k, v in enabled.items() if v}
    kept = {k for k, paths in outputs.items() if Path(paths[0]).exists()}
    return still_enabled, kept


VECGEN_OFF = {"env", "render", "annotate"}


def test_vecgen_finished_exec_is_kept(rdp, tmp_path):
    enabled, kept = _resume(rdp, tmp_path, ["task_gen", "exec"], VECGEN_OFF)
    assert enabled == {"merge"}
    assert "exec" in kept


def test_vecgen_unfinished_exec_reruns(rdp, tmp_path):
    enabled, kept = _resume(rdp, tmp_path, ["task_gen"], VECGEN_OFF)
    assert enabled == {"exec", "merge"}
    assert "exec" not in kept


def test_all_stages_enabled_unchanged(rdp, tmp_path):
    enabled, kept = _resume(rdp, tmp_path, ["task_gen", "env"], set())
    assert enabled == {"exec", "render", "merge", "annotate"}
    assert {"task_gen", "env"} <= kept and "exec" not in kept


def test_finished_run_is_untouched(rdp, tmp_path):
    enabled, kept = _resume(rdp, tmp_path, ["task_gen", "env", "exec", "merge", "annotate"], {"render"})
    assert enabled == set()
    assert {"task_gen", "env", "exec", "merge"} <= kept
