"""The DR package must reproduce the parameters a REAL render run recorded.

This is the parity metric that does not need a second renderer: render writes every sampled
value into meta/render_params.json, so driving the package with the same (flags, salt, episode
keys, seeding) must reproduce them exactly. A gym wrapper that supplies those four inputs gets
the same scene by construction -- which is the whole claim the dagger integration rests on.

Pure: reads a recorded artifact, calls pure samplers. No Isaac, no GPU.

THE SALT IS THE TRAP. render derives it as _stable_seed_salt(Path(job["out_dir"]).name) --
the basename of the OUTPUT directory, which is whatever the dataset discovery named the job,
NOT the problem or env directory. A caller with no output dir (a gym wrapper) must be handed
the same string explicitly or every sampler keys differently and nothing matches.
"""
import json
import os

import pytest

from toyblocks.domain_randomization import interpreter as I
from toyblocks.domain_randomization.camera import _sample_episode_offsets
from toyblocks.domain_randomization.seeds import _stable_seed_salt

# The reference render runs, cheapest rung first, are subdirectories of $TOYBLOCKS_PARITY_RENDERS
# (one render output per run); without it the tests skip. A run writes TWO render_params -- an
# empty `blocks/` split and the real one -- so name the real path explicitly.
_RENDERS_ROOT = os.environ.get("TOYBLOCKS_PARITY_RENDERS", "")
_REAL_PARAMS = os.path.join("blocks_success", "blocks_failure", "meta", "render_params.json")
RECORDINGS = {
    name: os.path.join(_RENDERS_ROOT, run, _REAL_PARAMS) if _RENDERS_ROOT else ""
    for name, run in (
        ("rung0_PINNED_repro", "rung0_base"),
        ("allrand_full", "allrand_ref"),
        ("local_lights_only", "locallights_ref"),
    )
}
_NO_ROOT = "set TOYBLOCKS_PARITY_RENDERS to the directory holding the reference render runs"
SALT_SOURCE = "blocks_success"          # basename of job["out_dir"] for these runs


@pytest.fixture(params=sorted(RECORDINGS), ids=sorted(RECORDINGS))
def recording(request):
    path = RECORDINGS[request.param]
    if not path:
        pytest.skip(_NO_ROOT)
    if not os.path.exists(path):
        pytest.skip(f"no recorded render at {path}")
    R = json.load(open(path))
    if not R.get("episodes"):
        pytest.skip(f"{request.param}: no episode records")
    return R


def _recorded():
    """The rung-0 run, for the tests that pin the salt behaviour specifically."""
    p = RECORDINGS["rung0_PINNED_repro"]
    if not p:
        pytest.skip(_NO_ROOT)
    if not os.path.exists(p):
        pytest.skip(f"no recorded render at {p}; re-run the rung-0 baseline")
    return json.load(open(p))


def test_camera_offsets_reproduce_the_recorded_run_exactly(recording):
    R = recording
    _require(R, "camera_offsets")
    eps = R["episodes"]
    cam_keys = sorted(eps[0]["camera_offsets"])
    keys = [e["episode_index"] for e in eps]

    ranges, rseed_base = I._resolve_cam_ranges(R["flags"], cam_keys)
    rseed = int(rseed_base) + int(_stable_seed_salt(SALT_SOURCE))
    got = _sample_episode_offsets(ranges, keys, rseed, indexed=R["dr_seeding"])

    checked = 0
    for e in eps:
        mine = got[e["episode_index"]]
        for ck, rec in e["camera_offsets"].items():
            for field, rv in rec.items():
                mv = mine[ck][field]
                if isinstance(rv, list):
                    assert len(mv) == len(rv)
                    for a, b in zip(mv, rv):
                        assert abs(a - b) < 1e-12, f"ep{e['episode_index']} {ck}.{field}"
                else:
                    assert mv == rv, f"ep{e['episode_index']} {ck}.{field}"
                checked += 1
    assert checked >= 30, f"only {checked} fields compared"


def test_a_wrong_salt_does_not_accidentally_match():
    """Guards the test itself: if any salt reproduced the values, it would prove nothing."""
    R = _recorded()
    eps = R["episodes"]
    cam_keys = sorted(eps[0]["camera_offsets"])
    keys = [e["episode_index"] for e in eps]
    ranges, rseed_base = I._resolve_cam_ranges(R["flags"], cam_keys)

    wrong = _stable_seed_salt("problem_declarative_1_20260824_195954_228514_2386")
    got = _sample_episode_offsets(ranges, keys, int(rseed_base) + int(wrong),
                                  indexed=R["dr_seeding"])
    rec = eps[0]["camera_offsets"][cam_keys[0]]["pitch_deg"]
    assert got[eps[0]["episode_index"]][cam_keys[0]]["pitch_deg"] != rec


def _salt_and_base(R):
    """rseed_base comes from the yaml's `randomize` block; a yaml without one resolves to 0,
    which is exactly what setup() would use."""
    cams = sorted(R["episodes"][0].get("camera_offsets", {})) or ["camera:front_camera"]
    _, rseed_base = I._resolve_cam_ranges(R["flags"], cams)
    return int(_stable_seed_salt(SALT_SOURCE)), int(rseed_base)


def _require(R, concern):
    if not any(concern in e for e in R["episodes"]):
        pytest.skip(f"this run does not exercise {concern}")


def test_local_lights_reproduce_the_recorded_run(recording):
    from toyblocks.domain_randomization import lights as LI
    R = recording; eps = R["episodes"]
    spec = (R["flags"].get("scene_lights") or {}).get("local_lights")
    if not spec:
        pytest.skip("this yaml has no scene_lights.local_lights")
    salt, rseed_base = _salt_and_base(R)
    got = LI._sample_episode_local_lights(spec, [e["episode_index"] for e in eps],
                                          int(spec.get("seed", rseed_base)) + salt,
                                          indexed=R["dr_seeding"])
    present = [e for e in eps if "local_lights" in e]
    if not present:
        pytest.skip("not recorded in this run -- render omits it when the wave's scenario "
                    "disables the sun (rec['local_lights'] is only set when non-empty)")
    for e in present:
        assert got[e["episode_index"]] == e["local_lights"], e["episode_index"]


def test_materials_reproduce_the_recorded_run(recording):
    """render_params records the APPLIED form (rec['materials'] = wave_materials, straight off
    _apply_episode_materials), which flattens the sampler's {material_name, props} into
    {name, **props}. Values must match through that mapping."""
    from toyblocks.domain_randomization import materials as MAT
    from toyblocks.domain_randomization.assets import AssetPaths
    R = recording; _require(R, "materials"); eps = R["episodes"]
    salt, _ = _salt_and_base(R)
    specs = MAT._resolve_material_specs(R["flags"], AssetPaths.default().table_materials)
    if not specs:
        pytest.skip("this yaml has no materials block")
    specs = {k: {**v, "seed": int(v["seed"]) + salt} for k, v in specs.items()}
    got = MAT._sample_episode_materials(specs, [e["episode_index"] for e in eps],
                                        indexed=R["dr_seeding"])
    checked = 0
    for e in eps:
        for label, rec in e["materials"].items():
            mine = got[e["episode_index"]][label]
            flat = {"name": mine.get("material_name"), **(mine.get("props") or {})}
            for k, rv in rec.items():
                if k in flat:
                    assert flat[k] == rv, f"ep{e['episode_index']} {label}.{k}"
                    checked += 1
    assert checked >= 10, checked


def test_env_sphere_reproduces_the_recorded_run(recording):
    """Same post-apply mapping: the record keeps the texture BASENAME and calls the rotation
    `rotation_deg`, where the sampler returns a full path and `rotation`."""
    import os as _os
    from toyblocks.domain_randomization import env_sphere as ES
    from toyblocks.domain_randomization.assets import AssetPaths
    R = recording; _require(R, "env_sphere"); eps = R["episodes"]
    spec = R["flags"].get("env_sphere")
    if not spec:
        pytest.skip("this yaml has no env_sphere block")
    salt, rseed_base = _salt_and_base(R)
    got = ES._sample_episode_env_spheres(spec, [e["episode_index"] for e in eps],
                                         int(spec.get("seed", rseed_base)) + salt,
                                         AssetPaths.default().hdri_dirs,
                                         indexed=R["dr_seeding"])
    checked = 0
    for e in eps:
        mine = got[e["episode_index"]]
        mapped = {"texture": _os.path.basename(mine.get("texture", "")),
                  "rotation_deg": mine.get("rotation")}
        for k, rv in e["env_sphere"].items():
            if mapped.get(k) is not None:
                assert mapped[k] == rv, f"ep{e['episode_index']} env_sphere.{k}"
                checked += 1
    assert checked >= 4, checked


def test_recorded_run_used_hash_seeding():
    """Everything above assumes prefix-free seeding; if the baseline was stream, the whole
    incremental story would not apply to it."""
    assert _recorded()["dr_seeding"] == "hash"
