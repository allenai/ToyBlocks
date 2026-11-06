"""`--render-pass`: the guards, and the argv it hands render.py.

Everything here is CPU-only and sub-second. The expensive failure this guards against is a typo in
the composed argv: render.py uses strict parse_args(), so one bad flag dies ~245s into Kit
startup, after the GPU is already busy.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "scripts/datagen/run_datagen_pipeline.py"
RENDER = ROOT / "scripts/render/render.py"
#: A real, locally-readable flags YAML. --render-pass has no default, so every run that
#: renders has to name one.
CAMTEST = ROOT / "src/toyblocks/domain_randomization/flags/render_flags_camtest.yaml"


def _pipeline_module():
    """Import the pipeline WITHOUT running it (it is a script, not a package module)."""
    spec = importlib.util.spec_from_file_location("_rdp", PIPELINE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_rdp"] = mod
    spec.loader.exec_module(mod)
    return mod


# Validation runs before any stage; skipping every stage and writing to a throwaway root stops a
# run that passes validation right there instead of launching a real datagen.
_STOP = ("--skip-task-gen", "--skip-env", "--skip-execution", "--skip-merge", "--skip-annotate")


def _run(*flags: str):
    """Run the pipeline through validation only, and return (rc, combined output)."""
    with tempfile.TemporaryDirectory() as root:
        r = subprocess.run(
            [sys.executable, str(PIPELINE), "--experiment-id", "_guard", "--output-root", root,
             *_STOP, *flags],
            capture_output=True, text=True, cwd=ROOT, timeout=600,
        )
    return r.returncode, r.stdout + r.stderr


def test_rejects_in_physics_video():
    """Recording video in the physics loop and then rendering over it is pure waste."""
    rc, out = _run("--render-pass", str(CAMTEST), "--vecgen-record-videos")
    assert rc != 0 and "--vecgen-record-videos" in out


def test_rejects_a_flags_yaml_that_is_not_local():
    """A YAML that exists only inside the render image cannot drive a local render."""
    rc, out = _run("--render-pass",
                   "/root/toyblocks/src/toyblocks/domain_randomization/flags/x.yaml")
    assert rc != 0 and "readable on THIS machine" in out


def test_render_pass_has_no_default_and_the_shipped_yamls_exist():
    """The look is always named explicitly: no render runs on an implicit YAML."""
    mod = _pipeline_module()
    assert mod.build_parser().get_default("render_pass") is None
    for name in ("render_flags_plain.yaml",
                 "render_flags_allrand_texcache_glyphs_trainsplit_symmetric.yaml"):
        f = ROOT / "src/toyblocks/domain_randomization/flags" / name
        assert f.is_file(), f


#: Every spelling the cleanup removed. None may be SILENTLY accepted: argparse abbreviates, so a
#: removed flag that is a unique prefix of a surviving one is absorbed by it. --render-n 16 was
#: taken as --render-num-envs 16, rendering one copy per episode with 16 envs per wave -- a run
#: that looks successful and ships the wrong number of episodes.
REMOVED_SPELLINGS = [
    "--render-randomized", "--rerender", "--render-flags", "--rerender-flags",
    "--render-n", "--rerender-n", "--render-non-randomized",
    "--render-non-randomized-flags", "--render", "--render-local", "--rerender-local",
    "--render-local-num-envs", "--render-local-stall-timeout",
    "--render-local-fail-on-video-defects", "--rerender-schedule", "--rerender-extra",
    "--rerender-skip-cameras", "--rerender-skip-failure-episodes",
]


@pytest.mark.parametrize("flag", REMOVED_SPELLINGS)
def test_a_removed_spelling_is_never_silently_accepted(flag):
    mod = _pipeline_module()
    with pytest.raises(SystemExit):
        mod.build_parser().parse_args(
            ["--experiment-id", "x", "--render-pass", str(CAMTEST), flag, "16"])


def test_render_n_says_what_replaced_it(capsys):
    """It is a unique prefix of --render-num-envs, so without an exact option it is absorbed."""
    mod = _pipeline_module()
    with pytest.raises(SystemExit):
        mod.build_parser().parse_args(
            ["--experiment-id", "x", "--render-pass", str(CAMTEST), "--render-n", "3"])
    err = capsys.readouterr().err
    assert "--render-n has been removed" in err and "--render-pass FLAGS.yaml:3" in err


def test_no_render_pass_means_no_render_stage():
    """--render-pass is the only thing that turns the stage on."""
    mod = _pipeline_module()
    args = mod.build_parser().parse_args(["--experiment-id", "x"])
    assert mod.resolve_render_passes(args) == []


def test_render_pass_names_a_look_directly():
    mod = _pipeline_module()
    args = mod.build_parser().parse_args(
        ["--experiment-id", "x", "--render-pass", str(CAMTEST), "--render-pass", f"{ALLRAND}:3"])
    assert mod.resolve_render_passes(args) == [(str(CAMTEST), 1), (f"{ALLRAND}", 3)]
    assert [j for j, _ in mod.local_render_passes(args, "/b")] == ["job_camtest", "job_allrand_texcache"]


# ------------------------------------------------------------------------------- argv
@pytest.fixture(scope="module")
def argv():
    mod = _pipeline_module()
    args = mod.build_parser().parse_args([
        "--experiment-id", "x", "--render-pass", "src/toyblocks/domain_randomization/flags/render_flags_camtest.yaml",
    ])
    return mod, mod.local_render_argv(args, Path("/b/rr/src/problem_x"),
                                        Path("/b/rr/out/job_camtest"), "/b/envs",
                                        flags_path=CAMTEST, renders_per_episode=1)


def test_argv_is_accepted_by_render_argparse(argv):
    """check_render_args validates by AST, so this costs no GPU and no Kit startup."""
    _mod, cmd = argv
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts/render/check_render_args.py"),
         "--source", str(RENDER), "--", *cmd[3:]],
        capture_output=True, text=True, cwd=ROOT, timeout=600,
    )
    assert r.returncode == 0, f"render.py would reject this argv:\n{r.stdout}\n{r.stderr}"


@pytest.mark.parametrize("flag,value", [
    ("--env_spacing", "60"),        # render.py default 5.0 -> neighbours leak into the cameras
    ("--camera_base", "scene"),     # render.py default "data" -> different pose source
    ("--rtx_warmup_frames", "16"),  # render.py default 1 -> different exposure
])
def test_carries_the_chain_defaults_that_differ_from_render_defaults(argv, flag, value):
    """Omitting any of these silently changes the pixels, because the pipeline overrides them."""
    _mod, cmd = argv
    assert flag in cmd and cmd[cmd.index(flag) + 1] == value


def test_carries_the_hardcoded_literals(argv):
    """completion.render_command() hardcodes these; no CLI above it sets them."""
    _mod, cmd = argv
    assert "--skip_initial_step" in cmd and "--resume" in cmd


def test_runs_render_by_absolute_path_never_dash_m(argv):
    """render.py sibling-imports scene_swap/render_sources, so -m or an import breaks it."""
    _mod, cmd = argv
    assert "-m" not in cmd
    assert cmd[1] == "-u" and Path(cmd[2]).is_absolute() and Path(cmd[2]).name == "render.py"


def test_num_envs_covers_every_block_size_slot(argv):
    """Below the slot count plan_block_sizes returns 'error' and SKIPS the problem entirely."""
    _mod, cmd = argv
    assert int(cmd[cmd.index("--num_envs") + 1]) >= 8


def test_every_skipped_camera_survives_render_argparse():
    """One --skip_cameras flag carrying every camera, never the flag repeated.

    render.py:76 declares --skip_cameras as a plain nargs="*" store, so a repeated flag
    OVERWRITES rather than accumulating: `--skip_cameras a --skip_cameras b` parses to ["b"]
    and camera `a` is silently RENDERED. Nothing downstream errors -- the dataset just carries
    a camera the run meant to drop. This passed unnoticed while exactly one camera was skipped.
    """
    import argparse

    mod = _pipeline_module()
    args = mod.build_parser().parse_args([
        "--experiment-id", "x", "--render-pass", "src/toyblocks/domain_randomization/flags/render_flags_camtest.yaml",
        "--render-skip-cameras", "left_camera", "gripper_camera",
    ])
    cmd = mod.local_render_argv(args, Path("/b/rr/src/p"), Path("/b/rr/out/job_camtest"),
                                "/b/envs", flags_path=CAMTEST, renders_per_episode=1)
    assert cmd.count("--skip_cameras") == 1, "repeated --skip_cameras drops all but the last camera"

    # Parse it the way render.py does, so this fails on behaviour and not on spelling.
    probe = argparse.ArgumentParser()
    probe.add_argument("--skip_cameras", type=str, nargs="*", default=None)
    parsed, _ = probe.parse_known_args(cmd[3:])
    assert set(parsed.skip_cameras) == {"left_camera", "gripper_camera"}


# ------------------------------------------------------------------------------ steps
def test_staged_dir_is_named_problem_star():
    """stage_render_source only picks up problem_* dirs, and merge_datasets assumes the same."""
    mod = _pipeline_module()
    args = mod.build_parser().parse_args([
        "--experiment-id", "x", "--render-pass", "src/toyblocks/domain_randomization/flags/render_flags_camtest.yaml",
    ])
    steps = mod.local_render_steps(args, "/b", "/b/envs", "problem_abc")
    joined = " ".join(steps)
    assert "/b/render/src/problem_abc" in joined
    assert "/b/execution/problem_abc" in joined


def test_output_job_dir_is_globbable_by_merge():
    """Merge globs render/out/job_*; a name not starting with job_ is silently never merged."""
    mod = _pipeline_module()
    for flags in (CAMTEST, ALLRAND, "render_flags_plain.yaml", "whatever.yaml"):
        assert mod.render_job_name(flags).startswith("job_"), flags


def test_each_pass_gets_its_own_job_dir():
    """Two passes writing to one directory would make the second render nothing and exit 0."""
    mod = _pipeline_module()
    assert mod.render_job_name(CAMTEST) != mod.render_job_name(ALLRAND)


# --------------------------------------------------------------------- the teardown hang
def _steps():
    mod = _pipeline_module()
    args = mod.build_parser().parse_args([
        "--experiment-id", "x", "--render-pass", "src/toyblocks/domain_randomization/flags/render_flags_camtest.yaml", "--render-schedule", "coupled",
    ])
    return mod, mod.local_render_steps(args, "/b", "/b/envs", "problem_abc")


def test_render_runs_through_the_hang_guard_not_render_directly():
    """render.py prints 'all jobs done' and then hangs in Isaac teardown.

    Inside the `bash -c a && b` execution chain that hang blocks run_local_stage forever and the
    pipeline never reaches merge -- observed as 16/16 episodes finalized then 53 minutes of
    nothing. The render must therefore be launched via run_local_render.py.
    """
    _mod, steps = _steps()
    render = steps[-1]
    assert "run_local_render.py" in render, "render is not bounded by the hang guard"
    assert "--stall-timeout" in render
    # render.py is still what ultimately runs, just underneath the guard
    assert "render.py" in render and render.index("run_local_render.py") < render.index(
        "/render.py"
    )


def test_hang_guard_treats_the_marker_as_success_not_the_exit_code():
    """The guard SIGKILLs after the marker, so it must not then report that kill as a failure."""
    src = (ROOT / "scripts/render/run_local_render.py").read_text()
    assert 'MARKER = "all jobs done"' in src
    assert "start_new_session=True" in src   # so the whole process group can be killed
    assert "killpg" in src


# ------------------------------------------------------------------- provenance in output
def test_render_params_records_the_dr_yaml():
    """Which YAML produced a dataset must be recoverable FROM the dataset.

    For a glyph/colour run the yaml IS the experiment, so a dataset whose randomization config
    cannot be recovered is not reproducible. It is recorded as 'flags_file' (not 'flags').
    """
    src = RENDER.read_text()
    assert '"flags_file": a.flags' in src, "_rp_args no longer records the DR yaml path"


def test_render_params_records_the_pixel_affecting_args():
    """These differ from render.py's defaults; at least the recorded ones must stay recorded."""
    src = RENDER.read_text()
    for key in ('"camera_base"', '"rtx_warmup_frames"', '"num_envs"', '"dr_seeding"',
                '"env_spacing"'):
        assert key in src, f"_rp_args no longer records {key}"


def test_render_params_records_env_spacing_when_the_run_sets_it():
    """A too-small env_spacing puts neighbor envs in frame, and the local argv always passes 60 --
    so the value has to reach the output. It is also what fast_patch rebuilds an argv from
    (rargs.get("env_spacing", 60)), which until now never found the key and always used 60."""
    src = RENDER.read_text()
    assert '"env_spacing": float(a.env_spacing)' in src, "_rp_args no longer records env_spacing"
    # Conditional on purpose: recording None would make fast_patch DROP --env_spacing (the key
    # would exist, so its 60 default never applies) rather than keep the documented fallback.
    assert 'if a.env_spacing is not None else {}' in src, \
        "env_spacing must stay recorded-only-when-set (see fast_patch rargs.get default)"


# ------------------------------------------------------------------- --render-extra
def _argv(*extra_flags: str):
    mod = _pipeline_module()
    a = ["--experiment-id", "x", "--render-pass", str(CAMTEST),
         "--render-skip-cameras", "left_camera", *extra_flags]
    ns = mod.build_parser().parse_args(a)
    return mod, mod.local_render_argv(ns, Path("/b/src/p"), Path("/b/out"), "/b/envs",
                                      flags_path=CAMTEST, renders_per_episode=1)


#: Every pass re-applies its own identity after --render-extra, so the extra is never the last
#: thing on the line. Anything after it must be exactly that identity and nothing else.
_PASS_IDENTITY = 4  # --flags <yaml> -N <n>


def _tail(v, n):
    """The n tokens the extra contributed, i.e. the ones before the pass identity."""
    assert v[-_PASS_IDENTITY] == "--flags" and v[-2] == "-N", v[-_PASS_IDENTITY:]
    return v[-(n + _PASS_IDENTITY):-_PASS_IDENTITY]


def test_render_extra_reaches_the_local_render():
    """Without it a local dataset silently lacks the sidecar streams."""
    _, v = _argv("--render-extra", "--extra_rgb 224 --extra_depth g12_224")
    assert _tail(v, 4) == ["--extra_rgb", "224", "--extra_depth", "g12_224"], v[-8:]


def test_render_extra_can_beat_the_pipeline_hardcodes():
    """render.py has no append/extend/count actions, so a repeat is last-wins."""
    _, v = _argv("--render-extra", "--env_spacing 999")
    assert _tail(v, 2) == ["--env_spacing", "999"], v[-6:]
    assert v.count("--env_spacing") == 2, "the pipeline's own value must still be emitted first"


def test_a_pass_identity_cannot_be_overridden_by_render_extra():
    """--flags/-N in an extra would otherwise rewrite one pass into another."""
    _, v = _argv("--render-extra", "--flags /tmp/other.yaml -N 9")
    assert v[-4:-2] == ["--flags", str(CAMTEST)], v[-4:]
    assert v[-2:] == ["-N", "1"], v[-2:]


def test_skip_cameras_is_never_the_last_option_before_the_extra():
    """--skip_cameras is nargs="*": a stray bare token in the extra would be absorbed into the
    camera list instead of erroring, silently re-enabling a camera the run meant to skip."""
    import argparse
    _, v = _argv("--render-extra", "--extra_rgb 224")
    i = v.index("--skip_cameras")
    assert i + 2 <= len(v) - 1, f"--skip_cameras is last: {v[i:]}"
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("--skip_cameras", nargs="*")
    for f in ("--extra_rgb", "--extra_depth", "--extra_rgb_GOP", "--rgb_GOP", "--output",
              "--envs_dir", "--flags", "-N", "--num_envs", "--rtx_warmup_frames", "--depth_codec",
              "--depth_format", "--depth_gop", "--dr_seeding", "--camera_base", "--env_spacing"):
        p.add_argument(f)
    p.add_argument("--skip_initial_step", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip_failure_episodes", action="store_true")
    ns = p.parse_args(v[3:])
    assert ns.skip_cameras == ["left_camera"], ns.skip_cameras
    assert ns.extra_rgb == "224"


def test_render_extra_cannot_steer_a_problem_out_of_the_merge():
    """--output would move it out of render/out/job_local, where merge globs. Dropped, not fatal:
    dropping restores the pipeline's own value, which is the correct one."""
    for bad, rest in (("--output /tmp/evil", "/tmp/evil"),
                      ("--max_episodes=3", "--max_episodes"),
                      ("--episodes_file /tmp/eps.json", "/tmp/eps.json")):
        _, v = _argv("--render-extra", f"{bad} --extra_rgb 224")
        assert rest not in v, (bad, v[-6:])
        assert "--extra_rgb" in v, "the rest of the extra must survive"
        assert v.count("--output") == 1, "the pipeline's own --output must remain"


# ------------------------------------------------------------------ multiplex / original pass
ALLRAND = ROOT / "src/toyblocks/domain_randomization/flags/render_flags_allrand_texcache.yaml"


def _mx_args(mod, *extra: str, flags=None):
    return mod.build_parser().parse_args([
        "--experiment-id", "x", "--render-pass", str(flags or CAMTEST),
        "--render-schedule", "coupled", *extra,
    ])


def test_one_pass_by_default(tmp_path):
    """One --render-pass is one look: no second pass appears unless it is asked for."""
    mod = _pipeline_module()
    passes = mod.local_render_passes(_mx_args(mod), str(tmp_path))
    assert [j for j, _ in passes] == ["job_camtest"], passes


def test_each_look_is_a_sibling_job_dir_with_its_own_copy_count(tmp_path):
    """A second pass into the SAME dir renders nothing: --resume skips a completed problem."""
    mod = _pipeline_module()
    args = mod.build_parser().parse_args([
        "--experiment-id", "x",
        "--render-pass", f"{ALLRAND}:3",
        "--render-pass", str(CAMTEST),
    ])
    passes = mod.local_render_passes(args, str(tmp_path))
    assert [j for j, _ in passes] == ["job_allrand_texcache", "job_camtest"]
    assert passes[0][1]["renders_per_episode"] == 3
    assert passes[1][1]["renders_per_episode"] == 1
    assert passes[0][1]["flags_path"].name == ALLRAND.name
    assert passes[1][1]["flags_path"].name == CAMTEST.name


def test_a_randomized_pass_beside_a_plain_one_is_the_n_plus_one_multiplex(tmp_path):
    """Three randomized copies plus one plain pass is (N+1)x the source."""
    mod = _pipeline_module()
    plain = ROOT / "src/toyblocks/domain_randomization/flags/render_flags_plain.yaml"
    args = mod.build_parser().parse_args([
        "--experiment-id", "x",
        "--render-pass", f"{ALLRAND}:3",
        "--render-pass", str(plain),
    ])
    passes = mod.local_render_passes(args, str(tmp_path))
    assert [j for j, _ in passes] == ["job_allrand_texcache", "job_plain"]
    assert passes[0][1]["renders_per_episode"] == 3
    assert passes[1][1]["renders_per_episode"] == 1, "the plain pass is rendered once"


def test_two_passes_cannot_collide_on_one_job_dir(tmp_path):
    """Same YAML twice would make the second render nothing and exit 0, so it is dropped."""
    mod = _pipeline_module()
    args = mod.build_parser().parse_args([
        "--experiment-id", "x", "--render-pass", str(CAMTEST), "--render-pass", f"{CAMTEST}:2",
    ])
    assert [j for j, _ in mod.local_render_passes(args, str(tmp_path))] == ["job_camtest"]


def test_a_pass_spec_parses_its_copy_count(tmp_path):
    mod = _pipeline_module()
    assert mod.parse_render_pass("a/b.yaml") == ("a/b.yaml", 1)
    assert mod.parse_render_pass("a/b.yaml:4") == ("a/b.yaml", 4)
    assert mod.parse_render_pass("/abs/p ath/b.yaml") == ("/abs/p ath/b.yaml", 1)


def test_every_pass_is_one_dispatch_item(tmp_path):
    """Exactly-once dispatch is per PROBLEM; a problem owes every pass before it is finished."""
    mod = _pipeline_module()
    plain = ROOT / "src/toyblocks/domain_randomization/flags/render_flags_plain.yaml"
    args = _mx_args(mod, "--render-pass", str(plain))
    steps = mod.local_render_steps(args, str(tmp_path), str(tmp_path / "envs"), "problem_x")
    renders = [s for s in steps if "run_local_render.py" in s]
    assert len(renders) == 2, steps
    assert "job_plain" in renders[1] and "-N 1" in renders[1]


def test_camera_base_data_is_refused_locally():
    """--camera_base data means "use the pose the DATASET records", and the local execution stage
    records none: vecgen is parquet-only and spawns no cameras, so its output has no
    observation.camera_pose.* at all. recorded_static_bases then collects nothing and silently
    falls back to the scene mount -- so asking for it locally is never what the caller meant.
    Dropped with a warning, never fatal; `scene` (what the pipeline passes) still works.
    """
    mod = _pipeline_module()

    def eff(extra: str):
        ns = mod.build_parser().parse_args([
            "--experiment-id", "x", "--render-pass", str(CAMTEST),
            f"--render-extra={extra}",
        ])
        v = mod.local_render_argv(ns, Path("/b/src/p"), Path("/b/out"), "/b/envs",
                                  flags_path=CAMTEST, renders_per_episode=1)
        i = [k for k, x in enumerate(v) if x == "--camera_base"]
        assert not any(x.startswith("--camera_base=") for x in v), v
        return v[i[-1] + 1], v

    for spelling in ("--camera_base data", "--camera_base=data"):
        val, _ = eff(spelling)
        assert val == "scene", (spelling, val)
    # scene is a harmless no-op and must still pass through
    val, _ = eff("--camera_base scene")
    assert val == "scene"
    # dropping it must not eat the rest of the extra
    val, v = eff("--camera_base data --extra_rgb 224")
    assert val == "scene" and v[-6:-4] == ["--extra_rgb", "224"], v[-6:]


@pytest.mark.parametrize(("flags", "refused"), [
    ((), False),
    (("--skip-annotate",), False),
    (("--render-pass=",), True),
])
def test_an_unreadable_render_pass_is_refused_only_when_rendering(flags, refused):
    """An explicitly empty YAML is refused exactly when a render will happen.

    The guard keys on enabled["render"], not on argv, because it has to sit AFTER
    prompt_steps (interactive mode can tick the stage on without the flag). A run that does
    not render must be completely unaffected.
    """
    rc, out = _run(*flags)
    assert ("readable on THIS machine" in out) is refused, (flags, out[:400])
