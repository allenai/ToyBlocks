"""interpreter.sample_episodes() must give a gym wrapper the same numbers a bulk call gives.

setup() samples the whole episode list up front because rerender knows its dataset; a training
env does not, so it samples per reset. Those two paths only agree under prefix-free seeding --
under `stream` an episode's draw depends on how many preceded it. These tests pin that contract.
Pure: no Isaac, no GPU.
"""
from types import SimpleNamespace

import pytest
import yaml

from toyblocks.domain_randomization import interpreter as I
from toyblocks.domain_randomization.assets import AssetPaths
from toyblocks.domain_randomization.resolve import resolve_skybox_textures

KEYS = [0, 1, 2, 3, 4, 5, 6, 7]
FIELDS = ("ep_offsets", "ep_mats", "ep_spheres", "ep_local")
FLAGS_YAML = "src/toyblocks/domain_randomization/flags/rerender_flags_open_hybrid_camera_rand_only.yaml"
ENV_SPHERE_SPEC: dict = {"sets": [{"set": "polyhaven/all"}], "seed": 7}


def _require_skybox_pool() -> None:
    """Skip unless the env_sphere spec resolves at least two skyboxes.

    The defect the xfails below pin only shows when there is a choice to make: with zero or one
    texture every call draws the same thing, bulk and incremental agree, and strict xfail turns
    that trivial pass into a failure. CI installs no HDRI pool (asset_manifest.json has no
    environments entry), so there the tests would verify nothing.
    """
    hdri_dirs: tuple = AssetPaths.default().hdri_dirs
    textures: list[str] = [
        t for s in ENV_SPHERE_SPEC["sets"] for t in resolve_skybox_textures(s, hdri_dirs)
    ]
    if len(textures) < 2:
        pytest.skip(f"needs >= 2 env_sphere skyboxes under {hdri_dirs}, found {len(textures)}")


def _ir(mode):
    """A minimal `ir` carrying exactly the fields sample_episodes reads."""
    ranges, _ = I._resolve_cam_ranges(
        yaml.safe_load(open(FLAGS_YAML)), ["camera:front_camera", "camera:gripper_camera"])
    return SimpleNamespace(
        indexed_seeding=mode, randomize_cams=True, cam_ranges=ranges, rseed=1234,
        object_specs={"block_a": {"seed": 11, "randomize": {}, "overrides": {},
                                  "materials": [{"name": "m1"}, {"name": "m2"}, {"name": "m3"}]}},
        # env_sphere MUST be exercised here: setup() reaches this sampler through a LOCAL
        # closure that injects assets.hdri_dirs, so a module-level call in sample_episodes
        # raises NameError. A None spec hid that until a gym wrapper hit it for real.
        env_sphere_spec=ENV_SPHERE_SPEC,
        env_sphere_seed=7, local_spec=None, local_rng_seed=99,
        assets=AssetPaths.default(),
        ep_offsets={}, ep_mats={}, ep_spheres={}, ep_local={})


@pytest.mark.xfail(reason="env_sphere is NOT prefix-free: _sample_episode_env_spheres rebuilds its SkyBuckets queue per call, so a one-key call always draws the queue's first item. See the DEFECT section in the working brief -- needs an operator decision.", strict=True)
@pytest.mark.parametrize("mode", ["hash", "indexed"])
def test_one_at_a_time_matches_bulk(mode):
    """A gym wrapper sampling one episode per reset lands on the bulk values."""
    _require_skybox_pool()
    bulk = _ir(mode)
    I.sample_episodes(bulk, KEYS)
    inc = _ir(mode)
    for k in KEYS:
        I.sample_episodes(inc, [k])
    for f in FIELDS:
        assert getattr(bulk, f) == getattr(inc, f), f


@pytest.mark.xfail(reason="env_sphere is NOT prefix-free: _sample_episode_env_spheres rebuilds its SkyBuckets queue per call, so a one-key call always draws the queue's first item. See the DEFECT section in the working brief -- needs an operator decision.", strict=True)
@pytest.mark.parametrize("mode", ["hash", "indexed"])
def test_arrival_order_does_not_matter(mode):
    """Envs finish in whatever order they finish; the values must not depend on it."""
    _require_skybox_pool()
    bulk = _ir(mode)
    I.sample_episodes(bulk, KEYS)
    rev = _ir(mode)
    for k in reversed(KEYS):
        I.sample_episodes(rev, [k])
    for f in FIELDS:
        assert getattr(bulk, f) == getattr(rev, f), f


def test_stream_seeding_is_refused():
    """Under `stream` a per-episode call would silently return a DIFFERENT number than the bulk
    call, so it must raise rather than quietly produce a wrong scene."""
    with pytest.raises(ValueError, match="prefix-free"):
        I.sample_episodes(SimpleNamespace(indexed_seeding=False), [0])


def test_empty_keys_is_a_noop():
    ir = _ir("hash")
    I.sample_episodes(ir, [])
    assert all(getattr(ir, f) == {} for f in FIELDS)
