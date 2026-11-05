"""`--rerender-local`: the guards, and the argv it hands rerender.py.

Everything here is CPU-only and sub-second. The expensive failure this guards against is a typo in
the composed argv: rerender.py uses strict parse_args(), so one bad flag dies ~245s into Kit
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
RERENDER = ROOT / "scripts/rerender/rerender.py"
#: A real, locally-readable flags YAML. --rerender-flags has no default, so every run that
#: re-renders has to name one.
CAMTEST = ROOT / "src/blocksuite/domain_randomization/flags/rerender_flags_camtest.yaml"


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
    """Recording video in the physics loop and then re-rendering over it is pure waste."""
    rc, out = _run("--rerender-local", "--vecgen-record-videos",
                   "--rerender-flags", str(CAMTEST))
    assert rc != 0 and "--vecgen-record-videos" in out


def test_rejects_pipelined_rerender_too():
    rc, out = _run("--rerender-local", "--rerender")
    assert rc != 0 and "alternatives" in out


def test_rejects_a_flags_yaml_that_is_not_local():
    """A YAML that exists only inside the rerender image cannot drive a local render."""
    rc, out = _run("--rerender-local", "--rerender-flags",
                   "/root/blocksuite/src/blocksuite/domain_randomization/flags/x.yaml")
    assert rc != 0 and "readable on THIS machine" in out


def test_the_default_randomization_yaml_is_ri_s():
    """The default flags are the RI training set's, and the file is checked in."""
    mod = _pipeline_module()
    default = mod.build_parser().get_default("rerender_flags")
    assert default == mod.RERENDER_FLAGS_DEFAULT
    assert default.endswith("rerender_flags_allrand_texcache_glyphs_trainsplit_symmetric.yaml")
    assert (ROOT / default).is_file(), default


# ------------------------------------------------------------------------------- argv
@pytest.fixture(scope="module")
def argv():
    mod = _pipeline_module()
    args = mod.build_parser().parse_args([
        "--experiment-id", "x", "--rerender-local",
        "--rerender-flags", "src/blocksuite/domain_randomization/flags/rerender_flags_camtest.yaml",
    ])
    return mod, mod.local_rerender_argv(args, Path("/b/rr/src/problem_x"),
                                        Path("/b/rr/out/job_local"), "/b/envs")


def test_argv_is_accepted_by_rerender_argparse(argv):
    """check_rerender_args validates by AST, so this costs no GPU and no Kit startup."""
    _mod, cmd = argv
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts/rerender/check_rerender_args.py"),
         "--source", str(RERENDER), "--", *cmd[3:]],
        capture_output=True, text=True, cwd=ROOT, timeout=600,
    )
    assert r.returncode == 0, f"rerender.py would reject this argv:\n{r.stdout}\n{r.stderr}"


@pytest.mark.parametrize("flag,value", [
    ("--env_spacing", "60"),        # rerender.py default 5.0 -> neighbours leak into the cameras
    ("--camera_base", "scene"),     # rerender.py default "data" -> different pose source
    ("--rtx_warmup_frames", "16"),  # rerender.py default 1 -> different exposure
])
def test_carries_the_chain_defaults_that_differ_from_rerender_defaults(argv, flag, value):
    """Omitting any of these silently changes the pixels, because the pipeline overrides them."""
    _mod, cmd = argv
    assert flag in cmd and cmd[cmd.index(flag) + 1] == value


def test_carries_the_hardcoded_literals(argv):
    """completion.rerender_command() hardcodes these; no CLI above it sets them."""
    _mod, cmd = argv
    assert "--skip_initial_step" in cmd and "--resume" in cmd


def test_runs_rerender_by_absolute_path_never_dash_m(argv):
    """rerender.py sibling-imports scene_swap/render_sources, so -m or an import breaks it."""
    _mod, cmd = argv
    assert "-m" not in cmd
    assert cmd[1] == "-u" and Path(cmd[2]).is_absolute() and Path(cmd[2]).name == "rerender.py"


def test_num_envs_covers_every_block_size_slot(argv):
    """Below the slot count plan_block_sizes returns 'error' and SKIPS the problem entirely."""
    _mod, cmd = argv
    assert int(cmd[cmd.index("--num_envs") + 1]) >= 8


def test_every_skipped_camera_survives_rerender_argparse():
    """One --skip_cameras flag carrying every camera, never the flag repeated.

    rerender.py:76 declares --skip_cameras as a plain nargs="*" store, so a repeated flag
    OVERWRITES rather than accumulating: `--skip_cameras a --skip_cameras b` parses to ["b"]
    and camera `a` is silently RENDERED. Nothing downstream errors -- the dataset just carries
    a camera the run meant to drop. This passed unnoticed while exactly one camera was skipped.
    """
    import argparse

    mod = _pipeline_module()
    args = mod.build_parser().parse_args([
        "--experiment-id", "x", "--rerender-local",
        "--rerender-flags", "src/blocksuite/domain_randomization/flags/rerender_flags_camtest.yaml",
        "--rerender-skip-cameras", "left_camera", "gripper_camera",
    ])
    cmd = mod.local_rerender_argv(args, Path("/b/rr/src/p"), Path("/b/rr/out/job_local"), "/b/envs")
    assert cmd.count("--skip_cameras") == 1, "repeated --skip_cameras drops all but the last camera"

    # Parse it the way rerender.py does, so this fails on behaviour and not on spelling.
    probe = argparse.ArgumentParser()
    probe.add_argument("--skip_cameras", type=str, nargs="*", default=None)
    parsed, _ = probe.parse_known_args(cmd[3:])
    assert set(parsed.skip_cameras) == {"left_camera", "gripper_camera"}


# ------------------------------------------------------------------------------ steps
def test_staged_dir_is_named_problem_star():
    """stage_rerender_source only picks up problem_* dirs, and merge_datasets assumes the same."""
    mod = _pipeline_module()
    args = mod.build_parser().parse_args([
        "--experiment-id", "x", "--rerender-local",
        "--rerender-flags", "src/blocksuite/domain_randomization/flags/rerender_flags_camtest.yaml",
    ])
    steps = mod.local_rerender_steps(args, "/b", "/b/envs", "problem_abc")
    joined = " ".join(steps)
    assert "/b/rerender/src/problem_abc" in joined
    assert "/b/execution/problem_abc" in joined


def test_output_job_dir_is_globbable_by_merge():
    """Merge globs rerender/out/job_*; a name not starting with job_ is silently never merged."""
    mod = _pipeline_module()
    assert mod.LOCAL_RERENDER_JOB.startswith("job_")


# --------------------------------------------------------------------- the teardown hang
def _steps():
    mod = _pipeline_module()
    args = mod.build_parser().parse_args([
        "--experiment-id", "x", "--rerender-local",
        "--rerender-flags", "src/blocksuite/domain_randomization/flags/rerender_flags_camtest.yaml",
    ])
    return mod, mod.local_rerender_steps(args, "/b", "/b/envs", "problem_abc")


def test_render_runs_through_the_hang_guard_not_rerender_directly():
    """rerender.py prints 'all jobs done' and then hangs in Isaac teardown.

    Inside the `bash -c a && b` execution chain that hang blocks run_local_stage forever and the
    pipeline never reaches merge -- observed as 16/16 episodes finalized then 53 minutes of
    nothing. The render must therefore be launched via run_local_render.py.
    """
    _mod, steps = _steps()
    render = steps[-1]
    assert "run_local_render.py" in render, "render is not bounded by the hang guard"
    assert "--stall-timeout" in render
    # rerender.py is still what ultimately runs, just underneath the guard
    assert "rerender.py" in render and render.index("run_local_render.py") < render.index(
        "/rerender.py"
    )


def test_hang_guard_treats_the_marker_as_success_not_the_exit_code():
    """The guard SIGKILLs after the marker, so it must not then report that kill as a failure."""
    src = (ROOT / "scripts/rerender/run_local_render.py").read_text()
    assert 'MARKER = "all jobs done"' in src
    assert "start_new_session=True" in src   # so the whole process group can be killed
    assert "killpg" in src


# ------------------------------------------------------------------- provenance in output
def test_render_params_records_the_dr_yaml():
    """Which YAML produced a dataset must be recoverable FROM the dataset.

    For a glyph/colour run the yaml IS the experiment, so a dataset whose randomization config
    cannot be recovered is not reproducible. It is recorded as 'flags_file' (not 'flags').
    """
    src = RERENDER.read_text()
    assert '"flags_file": a.flags' in src, "_rp_args no longer records the DR yaml path"


def test_render_params_records_the_pixel_affecting_args():
    """These differ from rerender.py's defaults; at least the recorded ones must stay recorded."""
    src = RERENDER.read_text()
    for key in ('"camera_base"', '"rtx_warmup_frames"', '"num_envs"', '"dr_seeding"',
                '"env_spacing"'):
        assert key in src, f"_rp_args no longer records {key}"


def test_render_params_records_env_spacing_when_the_run_sets_it():
    """A too-small env_spacing puts neighbor envs in frame, and the local argv always passes 60 --
    so the value has to reach the output. It is also what fast_patch rebuilds an argv from
    (rargs.get("env_spacing", 60)), which until now never found the key and always used 60."""
    src = RERENDER.read_text()
    assert '"env_spacing": float(a.env_spacing)' in src, "_rp_args no longer records env_spacing"
    # Conditional on purpose: recording None would make fast_patch DROP --env_spacing (the key
    # would exist, so its 60 default never applies) rather than keep the documented fallback.
    assert 'if a.env_spacing is not None else {}' in src, \
        "env_spacing must stay recorded-only-when-set (see fast_patch rargs.get default)"


# ------------------------------------------------------------------- --rerender-extra
def _argv(*extra_flags: str):
    mod = _pipeline_module()
    a = ["--experiment-id", "x", "--rerender-local",
         "--rerender-flags", str(CAMTEST), "--rerender-skip-cameras", "left_camera", *extra_flags]
    ns = mod.build_parser().parse_args(a)
    return mod, mod.local_rerender_argv(ns, Path("/b/src/p"), Path("/b/out"), "/b/envs")


def test_rerender_extra_reaches_the_local_render():
    """Without it a local dataset silently lacks the HVLA sidecar streams."""
    _, v = _argv("--rerender-extra", "--extra_rgb 224 --extra_depth g12_224")
    assert v[-4:] == ["--extra_rgb", "224", "--extra_depth", "g12_224"], v[-4:]


def test_rerender_extra_can_beat_the_pipeline_hardcodes():
    """rerender.py has no append/extend/count actions, so a repeat is last-wins."""
    _, v = _argv("--rerender-extra", "--env_spacing 999")
    assert v[-2:] == ["--env_spacing", "999"], v[-2:]
    assert v.count("--env_spacing") == 2, "the pipeline's own value must still be emitted first"


def test_skip_cameras_is_never_the_last_option_before_the_extra():
    """--skip_cameras is nargs="*": a stray bare token in the extra would be absorbed into the
    camera list instead of erroring, silently re-enabling a camera the run meant to skip."""
    import argparse
    _, v = _argv("--rerender-extra", "--extra_rgb 224")
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


def test_rerender_extra_cannot_steer_a_problem_out_of_the_merge():
    """--output would move it out of rerender/out/job_local, where merge globs. Dropped, not fatal:
    dropping restores the pipeline's own value, which is the correct one."""
    for bad, rest in (("--output /tmp/evil", "/tmp/evil"),
                      ("--max_episodes=3", "--max_episodes"),
                      ("--episodes_file /tmp/eps.json", "/tmp/eps.json")):
        _, v = _argv("--rerender-extra", f"{bad} --extra_rgb 224")
        assert rest not in v, (bad, v[-6:])
        assert "--extra_rgb" in v, "the rest of the extra must survive"
        assert v.count("--output") == 1, "the pipeline's own --output must remain"


# ------------------------------------------------------------------ multiplex / original pass
ALLRAND = ROOT / "src/blocksuite/domain_randomization/flags/rerender_flags_allrand_texcache.yaml"


def _mx_args(mod, *extra: str, flags=None):
    return mod.build_parser().parse_args([
        "--experiment-id", "x", "--rerender-local",
        "--rerender-flags", str(flags or CAMTEST), *extra,
    ])


def test_original_pass_is_off_by_default(tmp_path):
    mod = _pipeline_module()
    assert mod.build_parser().get_default("render_non_randomized") is False
    passes = mod.local_rerender_passes(_mx_args(mod), str(tmp_path))
    assert [j for j, _ in passes] == [mod.LOCAL_RERENDER_JOB], passes


def test_original_pass_is_a_sibling_job_dir_at_n_1(tmp_path):
    """A second pass into the SAME dir renders nothing: --resume skips a completed problem."""
    mod = _pipeline_module()
    args = _mx_args(mod, "--rerender-n", "3", "--render-non-randomized", flags=ALLRAND)
    passes = mod.local_rerender_passes(args, str(tmp_path))
    assert [j for j, _ in passes] == [mod.LOCAL_RERENDER_JOB, mod.LOCAL_RERENDER_JOB_NONRAND]
    assert passes[0][1] == {}, "the randomized pass keeps every CLI value"
    assert passes[1][1]["renders_per_episode"] == 1, "the original is rendered once, not N times"
    assert passes[1][1]["enable_gi"] is True, "allrand has a skybox block -> GI must stay on"


def test_derive_non_randomized_flags_keeps_the_scene_and_drops_the_draws(tmp_path):
    """'No DR' must not mean 'no yaml': dropping it brings the backdrop back, turns GI off and
    reverts render_mode/env_spacing -- a different scene, not the same scene unrandomized."""
    import yaml
    mod = _pipeline_module()
    dest = tmp_path / "orig.yaml"
    had_skybox = mod.derive_non_randomized_flags(ALLRAND, dest)
    src = yaml.safe_load(ALLRAND.read_text())
    got = yaml.safe_load(dest.read_text()) or {}
    assert had_skybox is True
    for random_key in ("skybox", "randomize", "materials", "scenarios", "scene_lights",
                       "env_sphere", "exposure"):
        assert random_key in src, f"fixture drifted: {random_key} not in the source yaml"
        assert random_key not in got, f"{random_key} is a draw and must be dropped"
    for scene_key in ("carb_settings", "env_spacing"):
        assert got.get(scene_key) == src.get(scene_key), scene_key
    # hide_objects is the exception: it is only scene-describing while something still BACKS the
    # scene. allrand pairs `hide_objects: [backdrop]` with skybox/env_sphere -- the backdrop is
    # hidden because an HDRI replaces it -- so keeping it once those are dropped would render the
    # baseline against nothing at all.
    assert src.get("hide_objects") and ({"skybox", "env_sphere"} & set(src))
    assert "hide_objects" not in got, "hide_objects must go when no background survives"


def test_hide_objects_survives_when_nothing_was_backing_the_scene(tmp_path):
    """Only drop it when the thing it existed for is gone."""
    import yaml
    mod = _pipeline_module()
    src = tmp_path / "src.yaml"
    src.write_text("hide_objects: [backdrop]\nenv_spacing: 60\n")
    dest = tmp_path / "out.yaml"
    mod.derive_non_randomized_flags(src, dest)
    got = yaml.safe_load(dest.read_text()) or {}
    assert got.get("hide_objects") == ["backdrop"], got


def test_an_explicit_original_flags_yaml_wins(tmp_path):
    mod = _pipeline_module()
    mine = tmp_path / "mine.yaml"
    mine.write_text("cameras: {}\n")
    args = _mx_args(mod, "--render-non-randomized", "--render-non-randomized-flags", str(mine))
    passes = mod.local_rerender_passes(args, str(tmp_path))
    assert passes[1][1]["flags_path"] == mine
    assert not (tmp_path / "rerender" / "non_randomized_flags.yaml").exists(), "must not derive one"


def test_both_passes_are_one_dispatch_item(tmp_path):
    """Exactly-once dispatch is per PROBLEM; a problem owes every pass before it is finished."""
    mod = _pipeline_module()
    args = _mx_args(mod, "--render-non-randomized")
    steps = mod.local_rerender_steps(args, str(tmp_path), str(tmp_path / "envs"), "problem_x")
    renders = [s for s in steps if "run_local_render.py" in s]
    assert len(renders) == 2, steps
    assert mod.LOCAL_RERENDER_JOB_NONRAND in renders[1] and "-N 1" in renders[1]


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
            "--experiment-id", "x", "--rerender-local",
            "--rerender-flags", str(CAMTEST), f"--rerender-extra={extra}",
        ])
        v = mod.local_rerender_argv(ns, Path("/b/src/p"), Path("/b/out"), "/b/envs")
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
    assert val == "scene" and v[-2:] == ["--extra_rgb", "224"], v[-4:]


@pytest.mark.parametrize(("flags", "refused"), [
    (("--rerender-flags="), False),
    (("--skip-annotate", "--rerender-flags="), False),
    (("--rerender", "--rerender-flags="), True),
    (("--rerender-local", "--rerender-flags="), True),
])
def test_empty_rerender_flags_refused_only_when_rerendering(flags, refused):
    """An explicitly empty YAML is refused exactly when a re-render will happen.

    The guard keys on enabled["rerender"], not on argv, because it has to sit AFTER
    prompt_steps (interactive mode can tick the stage on without either flag) and after the
    --rerender-local flip. A run that does not re-render must be completely unaffected.
    """
    rc, out = _run(*flags)
    assert ("--rerender-flags is empty" in out) is refused, (flags, out[:400])
