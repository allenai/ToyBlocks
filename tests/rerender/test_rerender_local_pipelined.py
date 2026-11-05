"""`--rerender`: the scheduler that overlaps execution and re-render on one slot pool.

Everything here is CPU-only and sub-second. The scheduler is exercised with fake commands
(``true`` / ``false``) over a real ``run_local_stage``, so the concurrency is real while Isaac
is not: the expensive failure this guards against is a dispatch bug that either drops a
problem (it silently vanishes from the merge) or renders one twice (``rerender.py --resume``
``shutil.rmtree``s the live writer's output dir).

No sleeps in the scheduler path and no clock: it is purely event-driven, so these are
deterministic.
"""
from __future__ import annotations

import importlib.util
import inspect
import subprocess
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "scripts/datagen/run_datagen_pipeline.py"


def _pipeline_module():
    spec = importlib.util.spec_from_file_location("_rdp_pipe", PIPELINE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_rdp_pipe"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _pipeline_module()


def _args(mod, *extra: str):
    return mod.build_parser().parse_args([
        "--experiment-id", "x", "--rerender",
        "--rerender-flags", "src/toyblocks/domain_randomization/flags/rerender_flags_camtest.yaml",
        *extra,
    ])


def _executed_problem(execution_dir: Path, slug: str) -> None:
    """The on-disk shape stage_rerender_source looks for: <split>/meta/info.json."""
    for split in ("blocks_success", "blocks_failure"):
        meta = execution_dir / slug / split / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        (meta / "info.json").write_text("{}")


def _rendered_problem(out_dir: Path, slug: str) -> None:
    """A render the merge will actually ship: render_params AND a blocks_success info.json."""
    for split in ("blocks_success", "blocks_failure"):
        meta = out_dir / slug / split / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        (meta / "render_params.json").write_text('{"completed": true, "episodes_recorded": 4, "episodes_total": 4}')
    (out_dir / slug / "blocks_success" / "meta" / "info.json").write_text("{}")


def _unrenderable_problem(out_dir: Path, slug: str) -> None:
    """What rerender.py writes for 'bad_size' / 'every source parquet unreadable'.

    completed=true means TERMINAL, not HAS DATA: render_params is written, info.json is not.
    """
    meta = out_dir / slug / "blocks_success" / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "render_params.json").write_text(
        '{"completed": true, "episodes_recorded": 0, "episodes_total": 0,'
        ' "empty_reason": "bad_size"}'
    )


class _RecordingScheduler:
    """The real scheduler with the render command swapped for a no-op, recording dispatch order."""

    def __init__(self, mod, tmp_path, *, fail=frozenset(), ok_render=True):
        self.mod = mod
        self.execution = tmp_path / "execution"
        self.rerender_dir = tmp_path / "rerender"
        self.execution.mkdir(parents=True, exist_ok=True)
        self.order: list[str] = []
        self.lock = threading.Lock()
        self._fail = fail
        outer = self

        class Sched(mod.LocalRerenderScheduler):
            def render_cmd(self, slug):
                # Finalize the output the way rerender.py would, so report() is meaningful.
                target = outer.rerender_dir / "out" / mod.LOCAL_RERENDER_JOB / slug
                mk = (
                    "mkdir -p " + str(target) + "/blocks_success/meta && "
                    "printf '{\"completed\": true, \"episodes_recorded\": 4}' > "
                    + str(target) + "/blocks_success/meta/render_params.json && "
                    "printf '{}' > " + str(target) + "/blocks_success/meta/info.json"
                )
                return ["bash", "-c", mk if ok_render else "false"]

        self.sched = Sched(_args(mod), "/envs", self.execution, self.rerender_dir)

    def exec_cmd(self, slug: str) -> list[str]:
        return ["bash", "-c", "false" if slug in self._fail else "true"]

    def record(self, name: str, ok: bool) -> None:
        with self.lock:
            self.order.append(name)
        self.sched.on_done(name, ok)

    def run(self, slugs, slots=(None,)):
        jobs = [(s, self.exec_cmd(s)) for s in slugs]
        feed = self.sched.arm(jobs, list(slugs))
        self.mod.run_local_stage(
            "Execution", feed, list(slots), self.rerender_dir / "logs", on_done=self.record
        )
        return self.order


# ------------------------------------------------------------------------------- JobFeed
def test_job_feed_drains_and_terminates(mod):
    feed = mod.JobFeed([("a", ["true"]), ("b", ["true"])])
    feed.close()
    assert [n for n, _ in feed] == ["a", "b"]
    assert feed.total == 2


def test_job_feed_serves_rerenders_before_queued_executions(mod):
    """Render-first keeps at most one un-rendered execution dataset on disk at a time."""
    feed = mod.JobFeed([("exec1", ["true"]), ("exec2", ["true"])])
    feed.add("rerender:exec1", ["true"], first=True)
    feed.close()
    assert [n for n, _ in feed] == ["rerender:exec1", "exec1", "exec2"]


def test_renders_interleave_with_executions_through_the_real_stage(mod, tmp_path):
    """Ordering must hold through run_local_stage, not just when iterating the feed directly.

    This is the test that was missing. The original feeder loop was
    `for job in feed: pool.submit(...)`, which drains the WHOLE feed into the pool's own FIFO
    queue before anything runs -- so a render queued later by on_done landed behind every
    execution, and the actual order was P0 P1 P2 P3 then all four renders. At one slot that
    holds N un-rendered execution datasets on disk instead of 1, which is the opposite of what
    the mode is for. test_job_feed_serves_rerenders_before_queued_executions passed throughout,
    because it iterates the JobFeed directly and never exercises the pool.
    """
    order = []
    lock = threading.Lock()
    feed = mod.JobFeed([(f"P{i}", ["true"]) for i in range(4)])
    pending = {"n": 4}

    def on_done(name, _ok):
        with lock:
            order.append(name)
            if not name.startswith("rr:"):
                feed.add("rr:" + name, ["true"], first=True)
                pending["n"] += 1
            pending["n"] -= 1
            if pending["n"] == 0:
                feed.close()

    mod.run_local_stage("T", feed, [None], tmp_path, on_done=on_done)
    assert order == ["P0", "rr:P0", "P1", "rr:P1", "P2", "rr:P2", "P3", "rr:P3"], order


def test_a_slot_pulls_only_when_free(mod, tmp_path):
    """Back-pressure: the stage must not hand every job to the pool up front."""
    src = PIPELINE.read_text()
    assert "for name, build in feed:" in src and "pool.submit(worker, i)" in src, \
        "run_local_stage no longer uses per-slot worker loops; priority ordering dies without them"


def test_job_feed_refuses_work_added_after_close(mod):
    """A late add would be silently dropped; it must be refused instead."""
    feed = mod.JobFeed()
    feed.close()
    assert feed.add("late", ["true"]) is False


# ----------------------------------------------------------------------- run_local_stage
def test_run_local_stage_still_takes_four_positional_args(mod):
    """run_playdata_pipeline.py calls this with four positionals; on_done must stay kw-only."""
    sig = inspect.signature(mod.run_local_stage)
    assert list(sig.parameters)[:4] == ["label", "jobs", "slots", "log_dir"]
    assert sig.parameters["on_done"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["on_done"].default is None


def test_run_local_stage_survives_a_job_that_cannot_be_built(mod, tmp_path):
    """build(slot) raising must count as a failure, not UnboundLocalError out of the pool."""
    def explode(_slot):
        raise RuntimeError("cannot build")

    seen = []
    mod.run_local_stage(
        "T", [("boom", explode), ("fine", ["true"])], [None], tmp_path,
        on_done=lambda n, ok: seen.append((n, ok)),
    )
    assert ("boom", False) in seen and ("fine", True) in seen


def test_run_local_stage_does_not_die_on_a_raising_callback(mod, tmp_path):
    """A raising on_done would otherwise abort the stage through future.result()."""
    def bad(_name, _ok):
        raise RuntimeError("callback blew up")

    mod.run_local_stage("T", [("a", ["true"])], [None], tmp_path, on_done=bad)


def test_isolation_is_on_whenever_two_items_can_be_live():
    """The per-slot Kit cache gate must key on the slot count, not on how many jobs exist now.

    It used to be min(len(jobs), len(slots)) > 1, which is False for a growing feed and for a
    two-GPU box running one problem -- shipping concurrent Kit processes with a SHARED
    KIT_CACHE_ROOT and OMNI_USER_CONFIG_DIR.
    """
    src = PIPELINE.read_text()
    assert "parallel = len(slots) > 1 and (isinstance(jobs, JobFeed) or len(jobs) > 1)" in src, \
        "the isolation gate no longer covers a growing feed"


# ---------------------------------------------------------------------------- scheduling
def test_every_executed_problem_gets_rendered(mod, tmp_path):
    slugs = ["problem_a", "problem_b", "problem_c"]
    r = _RecordingScheduler(mod, tmp_path)
    for s in slugs:
        _executed_problem(r.execution, s)
    order = r.run(slugs)
    for s in slugs:
        assert f"{mod.RERENDER_ITEM_PREFIX}{s}" in order, f"{s} was never re-rendered"
    shipped, unrenderable, absent = r.sched.report()
    assert sorted(shipped) == slugs and unrenderable == [] and absent == []


def test_a_problem_is_never_dispatched_twice(mod, tmp_path):
    """Two renders on one output dir: --resume rmtree()s the live writer. Must be impossible."""
    slugs = ["problem_a", "problem_b"]
    r = _RecordingScheduler(mod, tmp_path)
    for s in slugs:
        _executed_problem(r.execution, s)
    order = r.run(slugs)
    renders = [n for n in order if n.startswith(mod.RERENDER_ITEM_PREFIX)]
    assert len(renders) == len(set(renders)) == 2


def test_a_render_is_queued_only_after_its_execution_succeeds(mod, tmp_path):
    """Coupled mode gets this from `exec && rerender`; pipelined mode must gate explicitly."""
    r = _RecordingScheduler(mod, tmp_path, fail={"problem_bad"})
    for s in ("problem_ok", "problem_bad"):
        _executed_problem(r.execution, s)
    order = r.run(["problem_ok", "problem_bad"])
    assert f"{mod.RERENDER_ITEM_PREFIX}problem_ok" in order
    assert f"{mod.RERENDER_ITEM_PREFIX}problem_bad" not in order


def test_a_render_follows_its_own_execution(mod, tmp_path):
    slugs = ["problem_a", "problem_b"]
    r = _RecordingScheduler(mod, tmp_path)
    for s in slugs:
        _executed_problem(r.execution, s)
    order = r.run(slugs)
    for s in slugs:
        assert order.index(s) < order.index(f"{mod.RERENDER_ITEM_PREFIX}{s}")


def test_problems_executed_by_an_earlier_run_are_still_rendered(mod, tmp_path):
    """The exec stage SKIPS them, and merge then reads only rerender/out -- so they would vanish."""
    r = _RecordingScheduler(mod, tmp_path)
    _executed_problem(r.execution, "problem_new")
    _executed_problem(r.execution, "problem_old")      # on disk, not in this run's job list
    jobs = [("problem_new", r.exec_cmd("problem_new"))]
    feed = r.sched.arm(jobs, ["problem_new", "problem_old"])
    mod.run_local_stage("Execution", feed, [None], tmp_path / "logs", on_done=r.record)
    assert f"{mod.RERENDER_ITEM_PREFIX}problem_old" in r.order


def test_an_already_rendered_problem_is_not_rendered_again(mod, tmp_path):
    r = _RecordingScheduler(mod, tmp_path)
    _executed_problem(r.execution, "problem_a")
    _rendered_problem(r.rerender_dir / "out" / mod.LOCAL_RERENDER_JOB, "problem_a")
    order = r.run(["problem_a"])
    assert f"{mod.RERENDER_ITEM_PREFIX}problem_a" not in order


def test_the_stage_terminates_with_nothing_to_do(mod, tmp_path):
    """An empty feed that is never closed would hang the feeder loop forever."""
    r = _RecordingScheduler(mod, tmp_path)
    feed = r.sched.arm([], [])
    mod.run_local_stage("Execution", feed, [None], tmp_path / "logs", on_done=r.record)


def test_two_slots_run_two_items_at_once(mod, tmp_path):
    """The point of oversubscription: concurrency must actually reach the slot count."""
    live = 0
    peak = 0
    lock = threading.Lock()

    def build(_slot):
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
        return ["bash", "-c", "sleep 0.4"]

    def done(_name, _ok):
        nonlocal live
        with lock:
            live -= 1

    jobs = [(f"j{i}", build) for i in range(4)]
    mod.run_local_stage("T", jobs, [None, None], tmp_path, on_done=done)
    assert peak >= 2, f"two slots never ran two items concurrently (peak={peak})"


def test_shard_siblings_are_not_dispatched_as_their_own_problem(mod, tmp_path):
    """problem_x__s001 belongs to problem_x's staging set; dispatching it too renders it twice.

    Both renders target one output dir, and rerender.py --resume rmtree()s whichever has not
    finalized -- so the second one deletes the first's live output. Only the local sweep can
    hit this, because it globs problem_* off disk; a job list never contains shard names.
    """
    assert mod.base_problem_slugs(["problem_x", "problem_x__s001", "problem_y"]) == [
        "problem_x", "problem_y",
    ]
    # ...and the sweep must use it.
    r = _RecordingScheduler(mod, tmp_path)
    _executed_problem(r.execution, "problem_x")
    _executed_problem(r.execution, "problem_x__s001")
    r.sched.sweep(["problem_x", "problem_x__s001"], [None], tmp_path / "logs")
    staged = r.sched._staged
    assert staged.count("problem_x__s001") <= 1, f"shard staged twice: {staged}"


def test_an_execution_that_failed_is_reported_as_missing(mod, tmp_path):
    """Skipped-on-purpose and failed-to-render are the same event to merge: absent episodes.

    A problem whose execution finalized its dataset and then exited non-zero (routine Isaac
    teardown) used to land in _seen only -- never staged, so absent from BOTH done and missing.
    report() then claimed success, the stage marker was set, and merge dropped it silently.
    """
    r = _RecordingScheduler(mod, tmp_path, fail={"problem_bad"})
    for s_ in ("problem_ok", "problem_bad"):
        _executed_problem(r.execution, s_)
    r.run(["problem_ok", "problem_bad"])
    shipped, _unrenderable, absent = r.sched.report()
    assert shipped == ["problem_ok"]
    assert "problem_bad" in absent, f"a failed execution vanished from the report: {absent}"


def test_report_does_not_double_count(mod, tmp_path):
    r = _RecordingScheduler(mod, tmp_path)
    _executed_problem(r.execution, "problem_a")
    r.run(["problem_a"])
    r.sched.sweep(["problem_a"], [None], tmp_path / "logs2")
    shipped, unrenderable, absent = r.sched.report()
    assert shipped == ["problem_a"] and unrenderable == [] and absent == []


# ------------------------------------------------- what counts as rendered (both modes)
def test_a_finalized_but_empty_render_is_not_counted_as_shipped(mod, tmp_path):
    """rerender.py finalizes a problem EMPTY on bad_size / unreadable source, so completed=true
    means terminal, not has-data. Counting it as rendered set the marker and let merge -- which
    needs blocks_success/meta/info.json -- drop every episode of that problem at exit 0.
    """
    out = tmp_path / "out"
    _rendered_problem(out, "problem_real")
    _unrenderable_problem(out, "problem_empty")
    assert mod.local_render_state(out, "problem_real") == mod.RENDER_SHIPPED
    assert mod.local_render_state(out, "problem_empty") == mod.RENDER_UNRENDERABLE
    assert mod.local_render_state(out, "problem_never") == mod.RENDER_ABSENT
    # The plain completion predicate cannot tell the first two apart; that is the bug.
    assert mod.rerender_problem_complete(out, "problem_empty") is True


def _real_render_params(episode_records, *, total_episodes, empty_reason=None) -> dict:
    """A render_params payload from the REAL producer, not a hand-written literal.

    Fixtures that invent their own keys agree with whatever the consumer happens to read, so a
    consumer reading a key the producer never writes stays green forever. That is exactly how
    local_render_state came to branch on "total_episodes" -- a key render_params_payload has
    never emitted (it serializes that argument as "episodes_total").
    """
    from toyblocks.domain_randomization.render_params import render_params_payload

    return render_params_payload(
        {"env_file": None, "sources": []}, {}, 30, False, {}, episode_records, True, {},
        total_episodes=total_episodes, empty_reason=empty_reason,
    )


def test_the_gate_only_branches_on_keys_the_real_payload_writes(mod):
    """Pin the gate to the PRODUCER so a rename on either side fails here, loudly."""
    import json as _json

    payload = _real_render_params([], total_episodes=0, empty_reason="bad_size")
    for key in ("completed", "empty_reason", "episodes_recorded"):
        assert key in payload, f"gate branches on {key!r}; payload has {sorted(payload)}"
    assert "total_episodes" not in payload, (
        "render_params_payload serializes total_episodes as 'episodes_total' -- a gate reading "
        "'total_episodes' always gets None"
    )
    src = inspect.getsource(mod.local_render_state)
    assert '"total_episodes"' not in src, "local_render_state reads a key the payload never writes"
    _json.dumps(payload)  # the gate reads this off disk, so it must round-trip


def _write_params(out: Path, slug: str, payload: dict, split: str = "blocks_success") -> None:
    import json as _json

    meta = out / slug / split / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "render_params.json").write_text(_json.dumps(payload))


def test_a_real_empty_payload_is_unrenderable(mod, tmp_path):
    """The genuinely terminal case, written by the real producer rather than a literal."""
    out = tmp_path / "out"
    _write_params(out, "problem_empty", _real_render_params(
        [], total_episodes=0, empty_reason="recorded block size not in the problem's size table"))
    assert mod.local_render_state(out, "problem_empty") == mod.RENDER_UNRENDERABLE


def test_a_render_with_episodes_but_no_dataset_is_retried_not_written_off(mod, tmp_path):
    """completed + episodes recorded + NO info.json: the render ran but never shipped a dataset.

    That is transient (a crash between finalize and publish), so the sweep must RETRY it. While
    the gate read the non-existent "total_episodes" this landed in UNRENDERABLE -- a permanent,
    acknowledged outcome -- so the run proceeded and merge shipped without those episodes. Silent
    data loss, which is the one thing this classifier exists to prevent.
    """
    out = tmp_path / "out"
    _write_params(out, "problem_partial", _real_render_params(
        [{"episode_index": i, "split": "blocks_success"} for i in range(4)], total_episodes=4))
    assert mod.local_render_state(out, "problem_partial") == mod.RENDER_ABSENT


def test_a_failure_only_problem_is_permanent_not_retried(mod, tmp_path):
    """Every episode failed, so rerender finalizes blocks_failure and never creates blocks_success.

    merge reads blocks_success only, so the problem can never ship -- and re-running cannot change
    that, because the SOURCE has no successful episodes. Calling it ABSENT asks the sweep to retry
    something that can never satisfy it. Failure-only problems are routine (18 of 161 in
    datasets/june_10_rotate_3b_data_complete), not an exotic edge.
    """
    out = tmp_path / "out"
    _write_params(out, "problem_failonly", _real_render_params(
        [{"episode_index": 0, "split": "blocks_failure"}], total_episodes=4),
        split="blocks_failure")
    assert not (out / "problem_failonly" / "blocks_success").exists()
    assert mod.local_render_state(out, "problem_failonly") == mod.RENDER_UNRENDERABLE


def test_unrenderable_is_reported_separately_and_never_as_missing(mod, tmp_path):
    """It is permanent: failing the run on it would wedge the pipeline on every later re-run."""
    r = _RecordingScheduler(mod, tmp_path)
    _executed_problem(r.execution, "problem_empty")
    _unrenderable_problem(r.sched.out_dir, "problem_empty")
    r.sched.sweep(["problem_empty"], [None], tmp_path / "logs")
    shipped, unrenderable, absent = r.sched.report()
    assert unrenderable == ["problem_empty"], (shipped, unrenderable, absent)
    assert absent == [] and shipped == []


def test_merge_shipped_problems_is_merges_own_discovery(mod, tmp_path):
    """The gate's universe must be what merge ships, not a problem_* glob.

    The execution stage mkdirs each output dir BEFORE running its job, so a crashed execution
    leaves an empty problem_* dir for good. Quantified over the glob, the gate would refuse to
    merge every other problem forever over a directory merge would never have read.
    """
    execution = tmp_path / "execution"
    _executed_problem(execution, "problem_real")
    (execution / "problem_crashed").mkdir(parents=True)          # the mkdir-before-job stub
    failure_only = execution / "problem_failed_only" / "blocks_failure" / "meta"
    failure_only.mkdir(parents=True)
    (failure_only / "info.json").write_text("{}")                # merge never ships this split

    assert sorted(d.name for d in execution.iterdir()) == [
        "problem_crashed", "problem_failed_only", "problem_real",
    ]
    assert mod.merge_shipped_problems(execution) == ["problem_real"]
    assert mod.merge_shipped_problems(tmp_path / "nope") == []


def test_a_slug_that_exists_only_as_a_shard_is_still_swept(mod, tmp_path):
    """base_problem_slugs collapses problem_z__s001 to problem_z, which may have no dir at all
    when the first execution attempt was preempted. The .is_dir() guard then dropped it: never
    staged, never rendered, never reported."""
    r = _RecordingScheduler(mod, tmp_path)
    _executed_problem(r.execution, "problem_z__s001")
    assert not (r.execution / "problem_z").exists()
    r.sched.sweep(["problem_z__s001"], [None], tmp_path / "logs")
    assert "problem_z__s001" in r.sched._staged, r.sched._staged


def test_shards_are_collapsed_for_dispatch_but_checked_individually(mod, tmp_path):
    """problem_x rendering says nothing about whether problem_x__s001 did."""
    r = _RecordingScheduler(mod, tmp_path)
    for s_ in ("problem_x", "problem_x__s001"):
        _executed_problem(r.execution, s_)
    _rendered_problem(r.sched.out_dir, "problem_x")              # base only
    r.sched.sweep(["problem_x", "problem_x__s001"], [None], tmp_path / "logs")
    shipped, _unrenderable, absent = r.sched.report()
    assert "problem_x__s001" in shipped + absent, (shipped, absent)
    assert "problem_x__s001" not in mod.base_problem_slugs(["problem_x", "problem_x__s001"])


# ----------------------------------------------------- the gate, and the wiring behind it
def test_the_coupled_report_block_is_gone(mod):
    """It quantified done/missing over exec_jobs -- this invocation's job list -- so a problem
    skipped on .execution_complete was in neither, while `if done:` latched the marker on ANY
    success and merge then dropped the rest."""
    src = PIPELINE.read_text()
    assert "Hand the output to the merge stage exactly as the Beaker path does" not in src
    assert "done = [s_ for s_, _ in exec_jobs" not in src
    assert src.count('mark_stage_complete(base, "rerender")') == 1,         "the local marker must be set in exactly one place (the sweep)"


def test_the_local_rerender_is_a_real_stage_enabled_after_prompt_steps(mod):
    """Flipped before prompt_steps, an interactive run could untick the stage while the exec
    chains still carry the render: full GPU cost, then the un-rendered tree merged at exit 0."""
    src = PIPELINE.read_text()
    flip = src.index('        enabled["rerender"] = True')
    assert src.index("enabled = prompt_steps(enabled)") < flip, "flip must follow prompt_steps"
    assert src.index('"--rerender and --rerender-local are alternatives') < flip,         "flip must follow the mutual-exclusion guard, which still keys on enabled['rerender']"


def test_a_coupled_run_can_resume_into_the_rerender(mod, tmp_path):
    """The state a coupled run leaves when its executions finalize but the renders do not.

    With 'rerender' disabled, apply_resume skipped straight to 'merge' and shipped the raw
    un-rendered execution tree: there was no --resume that could ever finish the render.
    """
    base = tmp_path / "base"
    base.mkdir()
    for stage in ("task_gen", "env", "exec"):
        mod.mark_stage_complete(str(base), stage)

    def resume_point(rerender_enabled):
        enabled = {"task_gen": True, "env": True, "exec": True,
                   "rerender": rerender_enabled, "merge": True, "annotate": True}
        mod.apply_resume(str(base), enabled, {k: [] for k in enabled})
        return next((k for k, v in enabled.items() if v), None)

    assert resume_point(False) == "merge", "the old behaviour; kept here as the contrast"
    assert resume_point(True) == "rerender", "--rerender-local must be resumable"


def test_coupled_mode_never_arms_the_scheduler(mod):
    """Its render already runs in each job's shell chain; arming too would put two rerender.py
    on one output dir, and --resume rmtree()s the one that has not finalized."""
    assert "if local_rerender is not None and not args.rerender_local:" in PIPELINE.read_text()


def test_the_gate_fails_on_missing_problems_by_default(mod):
    src = PIPELINE.read_text()
    assert "if not args.merge_allow_missing_problems:" in src
    assert "args.rerender_merge_incomplete" not in src.split("def main")[-1] or         src.count("args.rerender_merge_incomplete") == 1,         "the local gate must not key on the Beaker per-episode knob"
    assert mod.build_parser().get_default("merge_allow_missing_problems") is False,         "a dataset must never silently drop problems by default"


# --------------------------------------------------------------------------------- flags
def test_worker_count_multiplies_the_slots_per_gpu(mod):
    assert "[g for g in local_gpus for _ in range(workers)]" in PIPELINE.read_text()
    assert mod.build_parser().get_default("local_gpu_workers") == 1, "default must be a no-op"


def test_local_rerender_is_refused_alongside_pipelined_rerender():
    r = subprocess.run(
        [sys.executable, str(PIPELINE), "--experiment-id", "_g", "--rerender", "--rerender-local"],
        capture_output=True, text=True, cwd=ROOT, timeout=600,
    )
    assert r.returncode != 0 and "alternatives" in (r.stdout + r.stderr)


def test_pipelined_mode_checks_the_flags_yaml_too():
    """This guard used to sit inside `if args.rerender_local`, so --rerender skipped it
    and burned ~245s of Kit startup per problem before dying."""
    r = subprocess.run(
        [sys.executable, str(PIPELINE), "--experiment-id", "_g", "--rerender",
         "--rerender-flags", "/root/toyblocks/src/toyblocks/domain_randomization/flags/x.yaml"],
        capture_output=True, text=True, cwd=ROOT, timeout=600,
    )
    assert r.returncode != 0 and "readable on THIS machine" in (r.stdout + r.stderr)


# ------------------------------------------------- multiplex: a problem owes EVERY pass
def test_a_problem_is_not_shipped_until_every_pass_has_landed(mod, tmp_path):
    """merge globs rerender/out/job_*, so a problem whose ORIGINAL pass failed would contribute
    N copies where (N+1) were promised -- and nothing would report it."""
    r = _RecordingScheduler(mod, tmp_path)
    # Force the two-pass shape onto the already-built scheduler.
    orig = r.rerender_dir / "out" / mod.LOCAL_RERENDER_JOB_NONRAND
    r.sched._out_dirs = [r.sched.out_dir, orig]
    _executed_problem(r.execution, "problem_a")
    _rendered_problem(r.sched.out_dir, "problem_a")          # randomized pass done
    assert r.sched._state_of("problem_a") == mod.RENDER_ABSENT, "original pass is missing"
    _rendered_problem(orig, "problem_a")                      # original pass done too
    assert r.sched._state_of("problem_a") == mod.RENDER_SHIPPED


def test_the_original_job_dir_is_a_sibling_merge_already_globs(mod):
    """No merge change needed: merge_input_dirs is (rerender/out).glob('job_*')."""
    assert mod.LOCAL_RERENDER_JOB_NONRAND.startswith("job_")
    assert mod.LOCAL_RERENDER_JOB_NONRAND != mod.LOCAL_RERENDER_JOB
    assert 'glob("job_*")' in PIPELINE.read_text()
