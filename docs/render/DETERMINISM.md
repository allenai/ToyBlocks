# Determinism in render and `toyblocks.domain_randomization`

What is reproducible, what is not, and which changes moved the line. Every number here was
measured on this repo (aug24_bigblocks_flip_a, one problem, RTX 5090), not asserted.

## The contract

**Same params, same output — going forward.** `--dr_seeding hash` is the anchor and the main
supported mode. Under it every per-episode value derives from `md5(seed | concern |
episode_index)`, so an episode's draw does not depend on how many episodes preceded it.

**Backward compatibility with already-rendered datasets is explicitly NOT a goal.** A bug fix may
change what a given yaml renders. When it does, it is listed below.

## What is bit-reproducible, and what is not

Two runs of the same renderer with byte-identical inputs:

| artifact | reproducible |
|---|---|
| every `.parquet` (all recorded DR parameters) | **yes, bit-exact** |
| every `observation.depth.*` video | **yes, bit-exact** |
| every `observation.images.*` (RGB) video | **no** |

RGB self-noise, front camera: **mean 1.8/255, max 55, ~87% of pixels differ** — between render
and *itself*. So "pixel-exact RGB" is not a meaningful acceptance criterion for anything.

Useful reference scales, all measured the same way:

| comparison | mean abs diff |
|---|---|
| renderer vs itself, identical inputs | 1.8 |
| same scene, two different construction paths | 2.1 |
| same scene, DR off vs full `allrand` | 21.0 |
| different scenes entirely | 86.7 |

Anything at ~2 is noise. Anything at ~20+ is a real difference.

**Consequence for verification:** compare parquet + depth, never RGB. Concretely: hash every
`*.parquet` and every `observation.depth.*` mp4 under the output root and compare those digests
between two runs. An RGB diff proves nothing, and an RGB *match* is luck. Guard the comparison
against empty inputs -- comparing two failed renders that produced no files reports a false
match.

## Seeding modes

| mode | per-episode value depends on | prefix-free |
|---|---|---|
| `hash` (default, supported) | `md5(seed \| concern \| episode_index)` | yes |
| `indexed` | `SeedSequence(seed, episode_index)` | yes |
| `stream` (legacy) | position in one advancing stream | **no** |

Under `stream`, sampling a subset returns different values than sampling the whole list.
`interpreter.sample_episodes()` therefore refuses `stream` rather than silently returning numbers
that do not match a bulk call.

## Statefulness that is NOT covered by seeding

`SkyBuckets` (skybox and `env_sphere` texture choice) is a **without-replacement queue**. It is
stateful by design — that is what stops an HDRI repeating before the pool is exhausted — and no
seeding mode makes it order-independent. Sampling one episode at a time restarts the queue and
returns its first entry every time.

Both callers must therefore sample the FULL episode list in one call:

* **render** does this already via `--fragment_invariant` (default ON): a fragment keeps the
  whole per-problem episode list so its DR matches a single-machine run.
* **a gym wrapper** must pass `episode_keys=` up front (see
  `toyblocks_dagger/.../dr_wrapper.py`), after which each reset only *applies* values already
  drawn. Without it the wrapper warns.

This is pinned as a strict `xfail` in
`tests/env_randomization/test_sample_episodes_incremental.py` — a documented property, not a
regression.

## Per-wave levers

Dome/skybox intensity, exposure ISO, sun, fill and the scenario pick are drawn once per wave from
rngs seeded in `setup()` (`rseed+12345` light, `+4242` fill, `+777` exposure,
`sky.seed|rseed_base +999+prob_salt`, scenario `_sseed+31337`). They depend on wave grouping, so a
caller whose batching differs from render's waves gets different globals even at `hash`.

## The salt

`prob_salt = _stable_seed_salt(Path(job["out_dir"]).name)` — the basename of the **output**
directory, not the problem or env directory. A caller with no output directory (a gym wrapper)
must be handed the same string; nothing in the env or the yaml reveals it. Getting it wrong
produces a completely different scene with no error.

## Changes that affect output

| change | breaks old-dataset determinism? | rendered pixels |
|---|---|---|
| `setup()` no longer dies on a yaml without `scenarios:` (67 of 106 yamls) | no | those yamls previously **crashed**; there is no old output to differ from |
| same for `missing` / `desc` | no | as above |
| `apply_wave(active=)`, `setup(salt=/episode_keys=)`, `sample_episodes()`, `apply_wave_cameras()`, `DRSession` | no | verified byte-identical on parquet + depth |
| removal of `orchestrator.py` / `randomizers.py` / `scene_context.py` | no | 516 lines with zero importers, and they crashed on first call |
| `--material_baseline_revert` | **only if enabled** | OFF by default = today's behaviour. ON reverts every library-writable key to the asset before each draw, so materials stop inheriting the previous episode's values. Applies 12 keys where OFF applies 8; the four added (`enabled`, `metallic_constant`, `reflection_roughness_constant`, `specular_level`) are exactly the ones that used to carry over |
| `table_height_delta` now recorded per episode | no | additive metadata; pixels byte-identical. Fixes `render_from_params`, which read it back with a `0.0` fallback and so replayed a **flat table** |
| `--dr_seeding hash` as the default | **yes, vs pre-hash datasets** | intentional: hash is the anchor. Datasets rendered under `stream` will not reproduce under the default |
| `--camera_base data` as the default | **yes, vs datasets rendered before it** | intentional: extrinsics now come from the recorded data rather than `SceneConfig` constants |

## Known non-determinism, unresolved

`render_from_params.py` does **not** reproduce the render it replays: depth differs by mean
2076.7 (16-bit), 76% of pixels, on the aug24 reference. Eliminated as causes: camera pose
(verified live against the record), `--reproduce_placement`, episode pairing, depth encoding,
recorded trajectory state, build args, `skip_initial_step`. The divergence is invariant across all
of those, which points at a structural difference between the two replay loops rather than data.
Since this tool backs the orchestrator's patch path, it matters for future runs.
