"""CPU tests for replaying a rendered dataset's recorded renders in eval.

Covers the parts of :mod:`toyblocks.eval.render_replay` that decide *which* record an
episode gets -- the merged-episode -> (problem, local index) mapping, the per-problem
count check, the per-episode field check, the flags hash -- plus the empty-local-lights
fix and the ``--wave_resets`` bookkeeping. Each dataset is a tmp_path LeRobot skeleton
(one data shard, ``meta/render_params.json`` or ``meta/provenance.json``); the USD
appliers are replaced by recorders, so no Isaac Lab is needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from toyblocks.eval import render_replay as rr
from toyblocks.eval.render_replay import (
    STATUS_APPLIED,
    STATUS_COUNT_MISMATCH,
    STATUS_FIELD_MISMATCH,
    STATUS_NO_RECORD,
    RecordedRenderReplay,
    WaveGate,
)

FRONT = [0.95, 0.0, 0.48, 0.95, 0.0, 0.3, 0.0]
FLAGS: dict[str, Any] = {"carb_settings": {"/rtx/rendermode": "RealTimePathTracing"}}


def _write_shard(root: Path, problems: list[str | None]) -> None:
    """One data shard: episode ``i`` belongs to ``problems[i]``, two frames each."""
    data = root / "data" / "chunk-000"
    data.mkdir(parents=True)
    eps = [i for i in range(len(problems)) for _ in range(2)]
    cols: dict[str, list[Any]] = {
        "episode_index": eps,
        "frame_index": [f for _ in problems for f in (0, 1)],
    }
    if any(p is not None for p in problems):
        cols["source_problem"] = [problems[e] for e in eps]
    pq.write_table(pa.table(cols), data / "file-000.parquet")


def _record(local: int, *, x: float = FRONT[0], delta: float = 0.01, **extra: Any) -> dict:
    return {
        "episode_index": local,
        "table_height_delta": delta,
        "cameras": {"camera:front_camera": {"position": [x, FRONT[1], FRONT[2]]}},
        **extra,
    }


def _write_render_params(render_dir: Path, episodes: list[dict], sha: str = "abc") -> None:
    meta = render_dir / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    payload = {"flags": FLAGS, "flags_sha256": sha, "args": {}, "episodes": episodes}
    (meta / "render_params.json").write_text(json.dumps(payload))


def _merged(tmp_path: Path, per_problem: dict[str, int], records: dict[str, list[dict]],
            shas: dict[str, str] | None = None) -> Path:
    """Merged root whose episodes are grouped per problem, in ``per_problem`` order."""
    root = tmp_path / "merged"
    problems = [p for p, n in per_problem.items() for _ in range(n)]
    _write_shard(root, problems)
    sources = []
    for name, recs in records.items():
        src = tmp_path / "unmerged" / name / "blocks_success"
        _write_render_params(src, recs, sha=(shas or {}).get(name, "abc"))
        sources.append({"problem_name": name, "source_dataset": str(src)})
    (root / "meta").mkdir(exist_ok=True)
    (root / "meta" / "provenance.json").write_text(json.dumps({"sources": sources}))
    return root


def _setup(root: Path, ep: int, *, x: float = FRONT[0], delta: float = 0.01) -> dict:
    return {
        "episode_index": ep,
        "source_dataset_path": str(root),
        "table_height_delta": delta,
        "camera_poses": {"front_camera": [x, *FRONT[1:]]},
    }


def test_no_render_params_means_replay_off(tmp_path: Path) -> None:
    _write_shard(tmp_path / "plain", [None, None])
    assert RecordedRenderReplay.from_dataset(tmp_path / "plain") is None


def test_merged_episode_maps_to_its_rank_within_the_problem(tmp_path: Path) -> None:
    root = _merged(
        tmp_path,
        {"pa": 2, "pb": 3},
        {"pa": [_record(0), _record(1)],
         "pb": [_record(0), _record(1, marker="pb1"), _record(2)]},
    )
    replay = RecordedRenderReplay.from_dataset(root)
    assert replay is not None
    assert replay.flags == FLAGS
    rec, status = replay.record_for(_setup(root, 3))
    assert status == STATUS_APPLIED
    assert rec is not None and rec["marker"] == "pb1"


def test_count_mismatch_disables_the_problem_without_raising(tmp_path: Path) -> None:
    root = _merged(
        tmp_path, {"pa": 1, "pb": 3},
        {"pa": [_record(0)], "pb": [_record(0), _record(1)]},
    )
    replay = RecordedRenderReplay.from_dataset(root)
    assert replay is not None
    assert replay.record_for(_setup(root, 2)) == (None, STATUS_COUNT_MISMATCH)
    assert replay.record_for(_setup(root, 0))[1] == STATUS_APPLIED


@pytest.mark.parametrize(
    "setup_kw",
    [{"x": FRONT[0] + 0.018}, {"delta": 0.02}],
    ids=["front_camera_position", "table_height_delta"],
)
def test_field_mismatch_rejects_the_record(tmp_path: Path, setup_kw: dict) -> None:
    root = _merged(tmp_path, {"pa": 1}, {"pa": [_record(0)]})
    replay = RecordedRenderReplay.from_dataset(root)
    assert replay is not None
    assert replay.record_for(_setup(root, 0, **setup_kw)) == (None, STATUS_FIELD_MISMATCH)


def test_fields_within_tolerance_are_accepted(tmp_path: Path) -> None:
    root = _merged(tmp_path, {"pa": 1}, {"pa": [_record(0)]})
    replay = RecordedRenderReplay.from_dataset(root)
    assert replay is not None
    setup = _setup(root, 0, x=FRONT[0] + 5e-5, delta=0.01 + 5e-6)
    assert replay.record_for(setup)[1] == STATUS_APPLIED


def test_unknown_episode_and_missing_source_are_no_record(tmp_path: Path) -> None:
    root = _merged(tmp_path, {"pa": 1, "pb": 1}, {"pa": [_record(0)]})
    replay = RecordedRenderReplay.from_dataset(root)
    assert replay is not None
    assert replay.record_for(_setup(root, 1)) == (None, STATUS_NO_RECORD)
    assert replay.record_for(_setup(root, 7)) == (None, STATUS_NO_RECORD)
    assert replay.record_for(None) == (None, STATUS_NO_RECORD)


def test_flags_hash_mismatch_raises_when_the_problem_loads(tmp_path: Path) -> None:
    root = _merged(
        tmp_path, {"pa": 1, "pb": 1},
        {"pa": [_record(0)], "pb": [_record(0)]},
        shas={"pa": "abc", "pb": "def"},
    )
    replay = RecordedRenderReplay.from_dataset(root)
    assert replay is not None
    assert replay.record_for(_setup(root, 0))[1] == STATUS_APPLIED
    with pytest.raises(ValueError, match="flags_sha256"):
        replay.record_for(_setup(root, 1))


def test_unmerged_render_output_reads_its_own_render_params(tmp_path: Path) -> None:
    root = tmp_path / "render"
    _write_shard(root, [None, None])
    _write_render_params(root, [_record(0), _record(1, marker="e1")])
    replay = RecordedRenderReplay.from_dataset(root)
    assert replay is not None
    rec, status = replay.record_for(_setup(root, 1))
    assert status == STATUS_APPLIED and rec is not None and rec["marker"] == "e1"


def test_stage_writes_the_status_into_the_setup(tmp_path: Path) -> None:
    root = _merged(tmp_path, {"pa": 1}, {"pa": [_record(0)]})
    replay = RecordedRenderReplay.from_dataset(root)
    assert replay is not None
    setup = _setup(root, 0, delta=0.5)
    assert replay.stage(0, setup) == STATUS_FIELD_MISMATCH
    assert setup["render_replay_status"] == STATUS_FIELD_MISMATCH
    assert replay.status_counts == {STATUS_FIELD_MISMATCH: 1}


def _bound_replay(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[RecordedRenderReplay, dict]:
    """A replay "bound" to a fake 2-env scene, with every USD applier recorded."""
    from toyblocks.domain_randomization import interpreter, lights

    calls: dict[str, list] = {"apply_record": [], "local_lights": [], "cameras": []}
    monkeypatch.setattr(
        interpreter, "apply_record",
        lambda prims, sim, record, *, active=(0,): calls["apply_record"].append(
            (dict(record), tuple(active))
        ),
    )
    monkeypatch.setattr(
        lights, "_apply_local_lights",
        lambda prims, wave, active, ep_lights, origins, eff_dome=None, enabled=True:
            calls["local_lights"].append((list(active), dict(ep_lights), enabled)),
    )
    replay = RecordedRenderReplay.from_dataset(root)
    assert replay is not None
    replay._prims = SimpleNamespace(local_prims=[["L0"], ["L0"]])
    replay._base_env = SimpleNamespace(scene=SimpleNamespace(env_origins=torch.zeros(2, 3)))
    monkeypatch.setattr(
        replay, "_apply_cameras", lambda i, rec, setup: calls["cameras"].append(i)
    )
    monkeypatch.setattr(
        replay, "_apply_default_cameras", lambda i: calls["cameras"].append(("default", i))
    )
    return replay, calls


def test_empty_local_lights_record_zeroes_that_envs_lights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _merged(
        tmp_path, {"pa": 1},
        {"pa": [_record(0, local_lights=[], skybox={"texture": "sky.hdr"})]},
    )
    replay, calls = _bound_replay(root, monkeypatch)
    replay.stage(1, _setup(root, 0))
    replay.apply_staged([1], apply_globals=False)
    assert calls["local_lights"] == [([1], {}, False)]
    (per_env, active), = calls["apply_record"]
    assert active == (1,) and "skybox" not in per_env and per_env["local_lights"] == []
    assert calls["cameras"] == [1]


def test_globals_come_from_the_lowest_env_with_a_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _merged(
        tmp_path, {"pa": 2},
        {"pa": [_record(0, skybox={"texture": "a.hdr"}),
                _record(1, skybox={"texture": "b.hdr"})]},
    )
    replay, calls = _bound_replay(root, monkeypatch)
    replay.stage(0, _setup(root, 0, delta=0.9))   # rejected -> default cameras, no record
    replay.stage(1, _setup(root, 1))
    replay.apply_staged([0, 1], apply_globals=True)
    assert calls["cameras"] == [("default", 0), 1]
    per_env, globals_only = calls["apply_record"]
    assert "skybox" not in per_env[0] and per_env[1] == (1,)
    assert globals_only[0]["skybox"] == {"texture": "b.hdr"} and globals_only[1] == (1,)
    assert "materials" not in globals_only[0] and "cameras" not in globals_only[0]


def test_globals_reuse_records_applied_earlier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _merged(tmp_path, {"pa": 1}, {"pa": [_record(0, exposure={"iso": 100.0})]})
    replay, calls = _bound_replay(root, monkeypatch)
    replay.stage(0, _setup(root, 0))
    replay.apply_staged([0], apply_globals=False)       # what apply_setup does with 2 envs
    replay.apply_staged([0, 1], apply_globals=True)     # the first-reset refresh
    assert calls["apply_record"][-1][0]["exposure"] == {"iso": 100.0}


def test_globals_pass_keeps_the_recorded_local_sun_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from toyblocks.domain_randomization import interpreter

    sun: dict[str, Any] = {"ltype": "distant", "intensity": 577.0, "_slot": 0}
    root = _merged(tmp_path, {"pa": 1}, {"pa": [_record(0, local_lights=[sun])]})
    replay = RecordedRenderReplay.from_dataset(root)
    assert replay is not None
    applied: list[tuple[list[int], dict]] = []
    monkeypatch.setattr(
        interpreter, "_apply_local_lights",
        lambda prims, wave, active, ep_lights, origins, eff_dome=None, enabled=True:
            applied.append((list(active), dict(ep_lights))),
    )
    replay._prims = SimpleNamespace(
        assets=SimpleNamespace(hdri_dirs=[]), local_spec={"lights": [{"type": "distant"}]},
        local_prims=[["L0"]], sun_prim=None, exp_spec=None, object_specs={},
        env_sphere_spec=None,
    )
    replay._base_env = SimpleNamespace(scene=SimpleNamespace(env_origins=torch.zeros(1, 3)))
    monkeypatch.setattr(replay, "_apply_cameras", lambda i, rec, setup: None)
    replay.stage(0, _setup(root, 0))
    replay.apply_staged([0], apply_globals=True)
    assert applied == [([0], {0: [sun]})]
    assert replay._prims.local_spec is not None


def test_apply_staged_is_a_noop_until_bound(tmp_path: Path) -> None:
    root = _merged(tmp_path, {"pa": 1}, {"pa": [_record(0)]})
    replay = RecordedRenderReplay.from_dataset(root)
    assert replay is not None
    replay.stage(0, _setup(root, 0))
    replay.apply_staged([0], apply_globals=True)        # no prims: nothing to write to
    assert not replay.bound


def test_kit_args_match_drsession() -> None:
    from toyblocks.domain_randomization.session import DRSession

    flags = {"carb_settings": {"/rtx/rendermode": "RealTimePathTracing", "a/b": True}}
    assert rr.kit_args_from_flags(flags) == DRSession.kit_args(flags)
    assert rr.kit_args_from_flags({}) == DRSession.kit_args({}) == ""


def test_merge_kit_args_keeps_user_args_and_skips_duplicates() -> None:
    merged = rr.merge_kit_args("--ext-folder=/x --/rtx/rendermode=RT", "--/rtx/rendermode=RT --/a=1")
    assert merged == "--ext-folder=/x --/rtx/rendermode=RT --/a=1"
    assert rr.merge_kit_args("", "--/a=1") == "--/a=1"


def test_balanced_kit_collisions_are_reported() -> None:
    flags = {"carb_settings": {"/rtx/post/dlss/execMode": 2, "/rtx/rendermode": "x"}}
    assert rr.balanced_kit_collisions(flags) == ["/rtx/post/dlss/execMode"]


def test_wave_gate_waits_for_every_active_env() -> None:
    gate = WaveGate(3)
    assert not gate.ready([0, 1, 2])
    gate.park([1])
    assert gate.parked == [1] and not gate.ready([0, 1, 2])
    gate.park([0, 2])
    assert gate.ready([0, 1, 2])
    assert gate.release() == [0, 1, 2] and gate.parked == []


def test_wave_gate_drained_envs_do_not_block_the_wave() -> None:
    gate = WaveGate(3)
    gate.park([0])
    assert gate.ready([0])        # envs 1 and 2 have no setup left
    assert not WaveGate(2).ready([])
