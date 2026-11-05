# Randomization yaml reference

The yaml is the single description of *what* gets randomized. It is the only place
randomization is configured, and the file is recorded in the output's `render_params.json` as the
dataset's visual provenance.

In this repository it is consumed by:

| Consumer | How it is named |
|---|---|
| `scripts/rerender/rerender.py` | `--flags` |
| `scripts/rerender/rerender_from_params.py` | reads the recorded flags back |
| `scripts/datagen/run_datagen_pipeline.py`, `scripts/datagen/run_playdata_pipeline.py` | `--rerender-flags` (default: `rerender_flags_allrand_texcache_glyphs_trainsplit_symmetric.yaml`, the RI training set's) |
| the library itself | `DRSession`, for in-process consumers |

The presets live in `src/blocksuite/domain_randomization/flags/`:

| preset | use |
|---|---|
| `rerender_flags_allrand_texcache_glyphs_trainsplit_symmetric.yaml` | the RI training set's render and the pipeline default: `allrand_texcache` plus the train glyph split (symmetry-preserving) and per-env block colours |
| `rerender_flags_allrand_texcache.yaml` | the production all-random lighting / materials / camera render, without glyph or colour swaps |
| `rerender_flags_camtest.yaml` | fixed camera overrides for quick local tests of the camera rig; randomizes nothing |
| `rerender_flags_open_hybrid_camera_rand_only.yaml` | camera randomization only |

The `allrand` presets sample backdrops from `polyhaven` and the 44 GB `indoor-2018` pool, installed
under `assets/environments/` (or `RERENDER_HDRI_DIR`). Where to get each pool and what happens when
one is missing: [HDRI backdrops](../../../docs/rerender/README_local.md#hdri-backdrops).

For what the renderer does with this file, see
[the re-rendering guide](../../../docs/source/user_guide/workflows/datagen/rerender.md);
for reproducibility guarantees, see
[DETERMINISM.md](../../../docs/rerender/DETERMINISM.md).

**Beware `rerender_flags_camtest.yaml`:** it is the one shipped preset with no random section at
all, so an `-N 4` run with it produces four pixel-identical copies while every size-based check
still reports 4x. Use it to test the camera rig, never to validate that randomization works.

### Validation

`validate_flags()` returns `(errors, warnings)`. An unknown key is normally a **warning, not an
error**, so a typo silently does nothing — check the startup log. The exceptions are
`randomize_task_letters` and `randomize_block_colors`, where an unknown or malformed key is a hard
**error** (`randomize_task_letters: unknown key(s) ...; only 'glyphs'`).

`config.py::TOP_LEVEL` is the allowlist, but it is **not** a complete list of what the code reads.
Three keys are read by the interpreter/build layer yet missing from it, so every file using them
warns on load even though they work:

| Key | Status |
|---|---|
| `couple_gi_to_env_sphere` | read, documented below, warns |
| `randomized_colored_lighting` | read at build time, warns |
| `dr_schema` | used by `canonicalize_flags`, ships in both `allrand` presets, warns |

`skybox.intensity_dist` is in the same position: it is read, but is absent from the skybox
allowlist, so a file that uses it warns.

## Value forms

Almost every numeric field accepts:

| form | meaning |
|---|---|
| `5` | fixed |
| `[min, max]` | sampled uniformly |
| `<field>_dist: log` | sample that range log-uniformly. Spelled `intensity_dist` **inside** the `skybox:` block, and `skybox_intensity_dist` at scenario level. |
| `auto` | derived from another quantity (only `fill_intensity`) |
| a probability in (0, 1) | a per-wave Bernoulli draw (only `skybox.visible_background`) |
| `intensity_auto_ratio` | on a local light: a scalar or `[min, max]` ratio resolved at wave time against the effective dome intensity |

## When each thing is drawn — this matters

| scope | concerns | keyed by |
|---|---|---|
| **per episode** | `materials`, `env_sphere`, `local_lights`, camera offsets (`randomize`) | `md5(seed \| concern \| episode_index)` under `hash` seeding — order-independent |
| **per episode, own RNG** | block glyphs (`randomize_task_letters`), cuboid colours (`randomize_block_colors`) | their own fixed per-episode RNGs — **not** affected by `--dr_seeding` at all |
| **per wave** (one batch of envs) | `scenarios`, dome/`skybox` intensity + rotation, `exposure`, `lighting_intensity` (`[min,max]` form), `scene_lights.fill_intensity`, `scene_lights.sun`, `skybox.visible_background` as a probability | a sequential RNG — **depends on how many waves have been drawn** |

Per-wave draws are *not* hash-keyed. Offline that is handled by `--fragment_invariant` (on by
default), which replays the full episode prefix so wave N really is wave N. In training it simply
means the scenario sequence depends on reset count.

## Top-level keys

### `carb_settings`
Raw carb settings applied at launch, e.g. the RT2 render mode:
```yaml
carb_settings:
  "/persistent/rtx/modes/rt2/enabled": true
  "/rtx/rendermode": RealTimePathTracing
```
⚠️ These become **launch kit_args**, and IsaacLab re-applies the `balanced.kit` preset *after*
launch (`simulation_context.py:742-745`), overriding ~15 keys — including the denoiser and
`dlss/execMode`. None of the shipped yamls collide, but if you set one of those keys here it is
silently discarded. Use `run_parity_ab --carb` (which writes `RenderCfg.carb_settings`) instead.

### `hide_objects`, `env_spacing`, `render_mode`, `lighting_intensity`
```yaml
hide_objects: ["backdrop"]   # prims removed before rendering
env_spacing: 500             # metres between envs; the YAML WINS over --env_spacing
```
`env_spacing` must exceed `2 × env_sphere.radius`, or a neighbour's backdrop sphere intrudes into
the side cameras.

`lighting_intensity` takes three forms, and they do **not** go through the same path:

| form | behaviour |
|---|---|
| `[min, max]` | sampled **per wave**, the intended form |
| a scalar | set **once at build time** on `/World/Light` |
| `"random"` | set once at build time from `uniform(200, 5000)` drawn from bare `np.random` — **not reproducible from any seed in this file** |

`validate_flags` only warns on a non-range value; it does not reject one.

### `skybox` — the dome light (global illumination)
```yaml
skybox:
  set: polyhaven/rooms          # or: sets / names / textures / textures_dir
  texture_format: latlong
  rotation_deg: [0, 360]
  intensity: [500, 1000]
  visible_background: false     # false = lights the scene but is not drawn
  seed: 0
  shuffle: true
  exhaust_before_repeat: true   # serve every unique texture before repeating
```
One dome for the whole wave — it is global, so it cannot differ per env.

### `env_sphere` — the visible backdrop
A large emissive sphere per env, so **each env can show a different background** in the same batch:
```yaml
env_sphere:
  sets: [{set: polyhaven/all}, {set: indoor-2018/all}]   # as in the allrand presets
                                                        # drops the second entry -- 1.8 GB not 46 GB
  exhaust_before_repeat: true
  shuffle: true
  seed: 7
  material: usdpreview
  emissive: true
  emissive_normalize: true          # scale each HDRI to a target median luminance
  emissive_intensity: [0.15, 0.28]  # the target when normalising
  emissive_clamp: [0.05, 100.0]
  cast_shadows: false
  backdrop_only: true               # visible to camera rays only, not to secondary rays
  radius: 200
  far_clip: 800                     # cameras' far clip is raised to see the sphere
  rotation_deg: 0
```
`emissive_normalize` is why dim HDRIs stay visible. Its queue is **stateful** — picks advance a
without-replacement queue.

### `scene_lights` — fill, local lights and the shared sun

`scene_lights` has three sub-keys: `fill_intensity` (with `fill_auto_ratio`), `local_lights`, and
`sun`.

`sun` is the single shared `DistantLight` for the whole scene, sampled **per wave**. For a
multi-env render this is the one to use: a per-env `local_lights` distant light is authored
per environment, which is not what a sun is. None of the four shipped presets uses it.

```yaml
scene_lights:
  fill_intensity: 0        # number | [min,max] | "auto"
  fill_auto_ratio: 0.35    # with "auto": fill = ratio × dome intensity
  sun:                     # ONE shared distant light for the whole scene, drawn per wave
    intensity: [300, 700]
    elevation: [28, 58]
    azimuth: [0, 360]
    angle: [0.1, 0.3]
  local_lights:
    link_per_env: true     # each env gets its own light, not one shared
    seed: 5
    count: 1
    lights:
      - type: distant                      # a sun
        intensity: [300, 700]
        color: {temperature: [4300, 6700]}
        angle: [0.1, 0.3]                  # angular size in deg — SMALLER = sharper shadows
        elevation: [28, 58]
        azimuth: [0, 360]
```

### `scenarios` — coordinated "looks", one per wave
The highest-leverage block. Each entry **overrides** the global levers for that wave:
```yaml
scenarios:
  seed: 123
  mix:
    - name: open_hybrid                 # soft and bright, no hard shadow
      prob: 0.4                         # relative weight; need not sum to 1
      skybox_intensity: [500, 1100]
      skybox_intensity_dist: log
      iso: [90, 145]
      fill_intensity: auto
      local_lights: false               # suppress local_lights this wave (see note below)
    - name: sun_shadow                  # darker, hard shadows
      prob: 0.6
      skybox_intensity: [150, 400]
      iso: [70, 110]
      fill_intensity: 0                 # no fill => high contrast
      sun: true
```
`prob`, `probability` and `weight` are synonyms (first wins). Contrast comes from **low GI + no
fill**, not from ISO.

**The key you type is not the key the interpreter sees.** Under `dr_schema: 3`,
`canonicalize_flags` renames `mix[].local_lights` to `sun` (and the top-level `global_distant` to
`sun`) before interpretation. So you author `local_lights: false`, and the key named `sun` is what
gates local lights internally. Older files still spell it `sun:` directly; both work, but
`local_lights` is what the shipped `allrand` presets use.

### `exposure` — camera response
```yaml
exposure:
  auto: false      # disable auto-exposure so scenario ISO actually takes effect
  iso: [70, 145]   # also fstop, shutter, iso_dist
  seed: 7
```
Scenarios that set `iso` force this path on automatically.

### `materials` — per-object surface randomization
```yaml
materials:
  table: {object: workspace_table, seed: 11}
```
A materials entry takes five keys, not two:

| key | default | meaning |
|---|---|---|
| `object` | the entry's own label | the scene prim to apply to |
| `library` | `blocksuite.TABLE_MATERIALS_FILEPATH` (override with the `RERENDER_TABLE_MATERIALS` env var) | the materials JSON to draw from |
| `seed` | `0` | per-entry seed |
| `randomize` | — | a dict of ± jitter ranges, restricted to the eight jitter keys |
| `overrides` | — | per-material-**name** jitter overrides, mirroring `randomize.cameras.<cam>` |

`table` is a label; `object` is the scene prim. Materials come from the object's
`materials.json` library — texture, `albedo_add`, `albedo_brightness`, `texture_scale`,
`texture_rotate`, `texture_translate`, roughness, metallic. Each env has its own shader prim, so
every env in a batch can differ.

⚠️ A key a draw omits keeps the **previous episode's** value (materials bleed). `rerender.py
--material_baseline_revert` reverts to the asset's baseline first; it is opt-in because it changes
existing output.

### `randomize` — camera jitter
```yaml
randomize:
  position_offset_cm: [3, 3, 3]     # per-axis magnitude
  pitch_deg: 5
  yaw_deg: 5
  roll_deg: 2
  vertical_fov_deg: 3
  seed: 0
  cameras:                          # per-camera overrides
    gripper_camera: {position_offset_cm: [0.5, 0.5, 0.5], pitch_deg: 1}
```
⚠️ Offsets are drawn in **sorted camera order**, so adding a camera whose name sorts earlier
shifts every later camera's draw.

### `randomize_task_letters` — swap the glyph on every block
```yaml
randomize_task_letters: true      # default false; every block draws from all 36
```
The vocabulary is **36 glyphs: `A`–`Z`, `1`–`9` and `10`**. There is no `0` tile. `10` is the one
glyph whose token is two characters, so `[1-10]` spans ten glyphs and `block_10` is a valid
object name (the task generators never author one; blocks can only be swapped TO it).

The value takes four forms:
```yaml
randomize_task_letters: true                     # unconstrained: all 36, no repeats in an episode
randomize_task_letters: [C, D, E, 1, 3]          # a LIST (or a string): every block draws from it
randomize_task_letters:
  glyphs: [A-F, 1-5]                             # the same, spelled as a mapping with `glyphs`
randomize_task_letters:
  glyphs:                                        # a MAPPING under `glyphs`: per authored character
    A: [A-Z]
    7: [1-9]
```

**Two draws, not one.** `true` keeps the historical draw: one rng per episode picks all of that
episode's blocks *without replacement* from the 36, so no two blocks share a glyph. Every other form
is *constrained*: each block draws **independently** from the set its authored character allows,
so two blocks may land on the same glyph — routine once a set is small. They stay distinct PDDL
objects because copy indices are renumbered on rewrite: three blocks drawing `X` become `block_x`,
`block_x_2`, `block_x_3`, the convention `glyph_blocks.block_name` uses. `glyphs: "*"` therefore
allows the same 36 as `true` but draws differently.

A mapping is per authored character and acts as an **allowlist**:
```yaml
randomize_task_letters:
  glyphs:
    A: [A-Z]                      # A-blocks become letters
    7: [1-9]                      # 7-blocks become digits
                                  # every other character KEEPS ITS GLYPH

randomize_task_letters:
  glyphs:
    "*": [A-Z]                    # catch-all for characters not named
    I: [I]                        # pinned (confusable with 1)
    7: [1-9]                      # escapes the letters-only default
```
A rule keys on the **authored character**, so `A: [X-Z]` covers `block_a`, `block_a_2` and
`block_a_3` alike, each drawing on its own. Writing `F: [F]` and omitting `F` mean the same thing.

Entry grammar is three shapes — a character `A`, an inclusive range `A-F`, or `"*"` — and a list
unions them. Rejected at load with a specific message: `[0]` and `[0-5]` (no 0 tile), `[A-9]`
(mixes kinds), `[Z-A]` and `[10-1]` (backwards), and anything resolving to nothing.

⚠️ `*` is YAML's alias marker, so quote it in **both** positions: `glyphs: "*"` and `"*": [A-Z]`.

#### Train and test glyph splits

`rerender_flags_allrand_texcache_glyphs_trainsplit_symmetric.yaml` restricts the swap to the task
generator's TRAIN split (the 20 letters A-Z minus A, B, I, O, R, T, and the digits 1, 3, 5) and maps
every glyph only to a glyph of its own symmetry class, so goals that accept several orientations stay
correct; it also randomizes block colours over the train palette. The classes are the ones
`scripts/task_gen/stack_three.py` builds goals from:

| class | glyphs |
|---|---|
| upside-down and lower-case symmetric | `S X Z` |
| upside-down symmetric only | `H N` |
| lower-case symmetric only | `C P U V W` |
| neither ("plain") | `D E F G J K L M Q Y 1 3 5` |

and the colour palette is red, green, blue and white. Characters outside the train split keep their
glyph.

The symmetric split exists because a glyph's symmetry class decides which orientations read as
"the same character", so swapping across classes silently changes what an orientation goal means.

The preset is `rerender_flags_allrand_texcache.yaml` plus those keys, so it differs from an
unswapped re-render only in the block faces and the text. A list preset is constrained, so two
blocks can show the same glyph.

#### What gets rewritten and recorded

Per env, each block's glyph is moved on the shared decal atlas, and both `pddl_goal` and the
LeRobot `task` string are rewritten so the instruction still describes what the camera sees. One
"stack A on I on 2" dataset becomes many relabelled variants without re-planning or re-simulating
anything — the trajectory, the poses and the physics are untouched.

How it works: every block in `train_blocks_large` is the SAME USD with the decal mesh's `st`
shifted to a different 3×2 quadrant of one 16×16 atlas, and the decals material is `OmniPBR`,
which exposes `inputs:texture_translate`. So the glyph is one shader input per env — no USD
swapping, no extra materials. Applied in the same pre-render window as camera DR, so the RTX
warm-up sees the final letters.

| carrier | what happens |
|---|---|
| `pddl_goal` | rewritten, e.g. `(on block_a block_i)` → `(on block_x block_7)` |
| `task` (→ `meta/tasks.parquet`, `task_index`) | rewritten the same way |
| `observation.state.object_pose.block_*` | **unchanged** — parquet schema is per file, not per episode |
| `glyph_map` column | forward `{original_block: glyph}` per episode (the shape shipped datasets carry) |
| `block_variable_map` column | inverse, keyed by the GOAL token: `{"block_8": {"from": "block_a", "kind": "glyph", ...}}` — covers glyph blocks and colour cuboids; use it to find a goal token's pose column |
| `render_params.json` | `episodes[].block_randomization`, same keying, with the pose column |

Notes:
- Keyed on the episode's own `episode_index` (salted per concern), not the env slot or wave, so a
  shard renders the same glyphs as a full run.
- **Sharding must be problem-aligned.** Each process rebuilds the `meta/tasks.parquet` of every
  problem it touches; two processes holding parts of one problem would each write it, and the last
  writer wins. `rerender.py` refuses a slice that splits a problem (`randomize_task_letters needs
  PROBLEM-ALIGNED sharding`). The pipeline renders whole problems, so its slices are always
  problem-aligned.
- Widening the pool from 35 to 36 (when `10` became assignable) changed the unconstrained draw for
  a given seed; datasets rendered before it do not re-render with identical letters. Each episode
  is still self-consistent, because its recorded map drives both the pixels and the text.
- Only `type: usd` blocks with an authored glyph are swapped. If a block's decal shader is not
  found the run logs a **text/pixel mismatch** naming the blocks, because the strings are rewritten
  regardless and the dataset would otherwise claim a glyph it never drew.

Verify a run with `scripts/datagen/verify_glyph_swap.py` (labels each block with the glyph the
dataset claims) and `scripts/datagen/compare_glyph_runs.py` (asserts OFF is a no-op and that
only text changed).

### `randomize_block_colors` — recolour the procedural cuboids
```yaml
randomize_block_colors: true          # default false; the six dataset colours below
randomize_block_colors:
  colors:                             # or an explicit palette; HSV floats in 0-1
    - {name: red,    hsv: [0.0,                1.0, 1.0]}
    - {name: green,  hsv: [0.3333333333333333, 1.0, 1.0]}
    - {name: blue,   hsv: [0.6666666666666666, 1.0, 1.0]}
    - {name: yellow, hsv: [0.1666666666666667, 1.0, 1.0]}
    - {name: purple, hsv: [0.75,               1.0, 1.0]}
    - {name: orange, hsv: [0.0833333333333333, 1.0, 1.0]}
```
The colour sibling of `randomize_task_letters`. It owns the blocks whose env spec is
`type: cuboid` (spawned by `create_cuboid`); glyphs own `type: usd` blocks with an authored glyph.
The partition is by spec type, never by name (`block_b` is a letter block, `block_blue` a cuboid),
and a hard assert stops a run if both ever claim one block, so the two keys combine safely.

- The default palette is the six colours the generated env packages author, so a recoloured "red"
  is byte-identical to an authored red. HSV converts to linear RGB with no gamma, the units
  `UsdPreviewSurface.diffuseColor` takes. The mapping takes only `colors`; each entry needs exactly
  `name` and `hsv`, and one name cannot carry two colours.
- Drawn **with replacement**, independently per cuboid and per episode (keyed on `episode_index`),
  so two cuboids may share a colour ("the red block" can be ambiguous in a scene).
- `pddl_goal` and `task` are rewritten so the colour word matches the pixels, with tokens numbered
  in the dataset's style (`block_red1`, `block_red2`). A rewritten token can collide with another
  block's authored name — `block_orange` recoloured red becomes `block_red1` while the block
  authored `block_red1` turned blue — so look objects up through `block_variable_map` (keyed by the
  goal token), not by matching names to pose columns.
- Recorded per episode in the `cuboid_color_map` column and in `render_params.json`
  `episodes[].block_randomization` (`from`, `pose_column`, `color`, `hsv`, `rgb`).

### `couple_gi_to_env_sphere`
When only `env_sphere` defines a texture pool, take the wave's dome GI from the batch's **first**
env-sphere pick. If `skybox` also defines a pool the two stay independent and this is ignored.

## Debugging

- `[dr] WARNING skybox bucket resolved 0 textures` → your HDRI root is wrong; the backdrop will be
  black. Fix it with `rerender.py --assets_paths <root> [<root> ...]` (ordered extra roots, the
  repo default searched last) or the `RERENDER_HDRI_DIR` environment variable. Each root may be an
  assets-root or an environments-root; the resolver auto-detects which.
- An unknown key is only a warning in most blocks — if a section seems to do nothing, check its
  spelling. Inside `randomize_task_letters` and `randomize_block_colors` it is an error instead,
  so those fail loudly.
- Every applied value is recorded per episode in the output's `meta/render_params.json`. The
  `episodes[]` entries carry `materials`, `skybox`, `env_sphere`, `exposure`, `scenario`,
  `camera_offsets`, and also `sun`, `local_lights`, `fill_intensity`, `lighting_intensity`,
  `split` and `block_randomization`. That file, from the run that produced your reference frames,
  is the ground truth for any comparison.
- To compare two shards, use the digests the file carries rather than diffing the YAML:
  `flags_sha256` identifies the literal file as written on disk, and `flags_resolved_sha256`
  identifies the **resolved** settings. Group shards by the resolved digest — two shards with the
  same one were rendered under identical settings even if the YAML was reformatted or moved, and
  they differ if a code-side default changed under the same file.
- `dr_seeding` is recorded top-level. Pass the recorded mode back when reproducing or patching a
  dataset; never mix two modes within one dataset.
