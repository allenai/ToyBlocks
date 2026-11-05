"""Per-env colour randomization for procedural CUBOID blocks (``randomize_block_colors``).

The sibling of :mod:`glyphs`, for the other half of the block vocabulary. A generated env spawns
each block one of two ways (``codegen.py`` ``Block.from_names``): a block with a USD override is
``type="usd"`` and carries an atlas glyph; everything else is ``type="cuboid"`` and is built by
``rigid_objects.create_cuboid(..., diffuse_color=spec["color"], metallic=0.2)``. So a cuboid's
colour is a single ``UsdPreviewSurface`` diffuse input -- the colour analogue of the glyph's one
OmniPBR ``texture_translate``, but a DIFFERENT material path. Do not reuse the glyph applier.

OWNERSHIP IS THE SPEC ``type``, NEVER A NAME TEST. ``block_b`` is a USD letter block, ``block_blue``
and ``block_brown`` are cuboids; any prefix/substring rule gets that wrong in the direction that
still renders. glyphs.py:98 ``_BLOCK_GLYPH_RE`` returns None for every cuboid, so the two owners are
disjoint by construction and the partition is enforced, not assumed.

Why the text is rewritten and not merely recorded: a goal reading
``(on block_l block_red1) (on block_orange block_l)`` is the sentence "put L on the RED block, put
the ORANGE block on L". Recolour those cuboids and record the mapping only in render_params.json --
which no training code reads -- and ``pddl_goal``, the LeRobot ``task`` string and
``prompt_resolver``'s "the red block" all still say red while the camera shows cyan. That is the
same contradiction glyphs.py:13-15 exists to prevent, in a worse channel: for a cuboid the colour
word IS the referring expression.

This module is the PURE half: palette parsing, HSV->RGB, seeded assignment, token rewriting. No
pxr, no torch, no Isaac -- so it is unit-testable on a CPU box.
"""
from __future__ import annotations

import colorsys
import re
from typing import Dict, Iterable, List, Sequence, Tuple

#: The palette the oct06_stack3 datasets actually use, taken from the generated env packages'
#: ``block_specs`` literals (identical to ``codegen._NAMED_COLORS``). HSV is stored because that is
#: the authoring surface; RGB is derived. Every one of these round-trips EXACTLY through
#: ``colorsys``, which is what lets a recoloured "red" be the same red as an authored one -- a
#: near-miss red is a silent wrong colour, the signature failure of this feature family.
DATASET_COLORS: Tuple[Tuple[str, Tuple[float, float, float]], ...] = (
    ("red",    (0.0,            1.0, 1.0)),
    ("green",  (1.0 / 3.0,      1.0, 1.0)),
    ("blue",   (2.0 / 3.0,      1.0, 1.0)),
    ("yellow", (1.0 / 6.0,      1.0, 1.0)),
    ("purple", (0.75,           1.0, 1.0)),
    ("orange", (1.0 / 12.0,     1.0, 1.0)),
)

#: A cuboid token: ``block_<colour><n>``. The trailing index is NOT underscore-separated -- the
#: datasets author ``block_red1`` beside ``block_blue``, so this is the existing convention and
#: glyphs' ``block_a_2`` shape would be a new one.
_COLOR_TOKEN_RE = re.compile(r"^block_([a-z]+)(\d*)$")

#: Any block token, for simultaneous rewriting. Mirrors glyphs._TOKEN_RE so the two rewrites can
#: run over the same text without one corrupting the other's targets.
_TOKEN_RE = re.compile(r"\bblock_([A-Za-z0-9]+(?:_\d+)?)\b")

_HSV_NAMES = ("h", "s", "v")


def hsv_to_rgb(hsv: Sequence[float]) -> Tuple[float, float, float]:
    """HSV (each 0-1) -> linear RGB, the units ``create_cuboid``'s ``diffuse_color`` takes.

    NO sRGB transform. ``UsdPreviewSurface.diffuseColor`` is linear and the authored cuboid colours
    are plain ``_NAMED_COLORS`` triples handed straight through, so applying a gamma here would make
    every recoloured red a different red from every authored red -- plausible-looking and wrong.
    """
    h, s, v = (float(x) for x in hsv)
    return tuple(round(c, 12) for c in colorsys.hsv_to_rgb(h % 1.0, s, v))  # type: ignore[return-value]


def _parse_entry(entry, where: str) -> Tuple[str, Tuple[float, float, float]]:
    """One palette entry -> (name, hsv). Requires exactly ``name`` and ``hsv``."""
    if not isinstance(entry, dict):
        raise ValueError(f"{where}: each colour must be a mapping with 'name' and 'hsv', "
                         f"got {entry!r}")
    unknown = set(entry) - {"name", "hsv"}
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {sorted(unknown)}; only 'name' and 'hsv'")
    name = str(entry.get("name", "")).strip().lower()
    if not name:
        raise ValueError(f"{where}: missing 'name'")
    if not name.isalpha():
        # The name becomes a PDDL token (block_<name><n>) and an English word in the instruction,
        # so digits/punctuation would produce an unreadable or ambiguous object.
        raise ValueError(f"{where}: colour name {name!r} must be letters only -- it becomes the "
                         f"PDDL token block_{name}1 and the word in the instruction")
    hsv = entry.get("hsv")
    if not isinstance(hsv, (list, tuple)) or len(hsv) != 3:
        raise ValueError(f"{where}: {name!r} needs 'hsv' as three numbers, got {hsv!r}")
    vals: List[float] = []
    for key, raw in zip(_HSV_NAMES, hsv):
        try:
            f = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{where}: {name!r} hsv {key}={raw!r} is not a number") from None
        if not 0.0 <= f <= 1.0:
            # Loud, because the other convention (H 0-360, S/V 0-100) also exists in this repo
            # (lights.py:73-81 uses degrees while materials.py:118-122 uses turns). A degrees
            # triple read as turns lands on a real colour, so it would never crash -- it would
            # just render the wrong one.
            raise ValueError(
                f"{where}: {name!r} hsv {key}={f} is out of range. HSV here is 0-1 FLOATS "
                f"(materials.py convention), not H 0-360 / S,V 0-100 -- write "
                f"[{f / 360.0:.4g}, ...] if you meant degrees")
        vals.append(f)
    return name, (vals[0], vals[1], vals[2])


def parse_palette(spec) -> Tuple[Tuple[str, Tuple[float, float, float]], ...]:
    """``randomize_block_colors``'s value -> the ordered palette it allows.

    * ``True`` -- the six :data:`DATASET_COLORS`.
    * a mapping ``{colors: [...]}`` -- that list, each entry ``{name, hsv}``.

    Parsed at startup so a typo is one loud line, not a mis-labelled dataset.
    """
    if spec is True or spec is None:
        return DATASET_COLORS
    if isinstance(spec, dict):
        unknown = set(spec) - {"colors"}
        if unknown:
            raise ValueError(f"randomize_block_colors: unknown key(s) {sorted(unknown)}; "
                             f"only 'colors'")
        entries = spec.get("colors")
        if entries is None:
            raise ValueError("randomize_block_colors: mapping needs a 'colors' list")
    else:
        entries = spec
    if not isinstance(entries, (list, tuple)) or not entries:
        raise ValueError(f"randomize_block_colors: 'colors' must be a non-empty list, "
                         f"got {entries!r}")
    out: List[Tuple[str, Tuple[float, float, float]]] = []
    seen: Dict[str, Tuple[float, float, float]] = {}
    for i, e in enumerate(entries):
        name, hsv = _parse_entry(e, f"randomize_block_colors.colors[{i}]")
        if name in seen and seen[name] != hsv:
            raise ValueError(f"randomize_block_colors: {name!r} is listed twice with different "
                             f"hsv ({seen[name]} and {hsv}); one name is one colour")
        if name not in seen:
            seen[name] = hsv
            out.append((name, hsv))
    return tuple(out)


def is_cuboid(spec) -> bool:
    """True when a ``block_specs`` entry spawns via ``create_cuboid``.

    The authoritative partition, read from the same field the spawn loop branches on
    (codegen.py: ``if spec["type"] == "cuboid"``). Anything else -- including the legacy
    ``block_o1``/``block_o2``, which are ``type="usd"`` with no authored glyph -- is not ours.
    """
    return isinstance(spec, dict) and str(spec.get("type", "")).strip().lower() == "cuboid"


def cuboid_names(block_specs: Dict[str, dict]) -> List[str]:
    """The recolourable blocks in a generated env, sorted so the draw order is stable."""
    return sorted(n for n, s in (block_specs or {}).items() if is_cuboid(s))


def assign_colors(names: Sequence[str], rng, palette) -> Dict[str, str]:
    """Map each cuboid to a colour NAME, deterministically from ``rng``.

    WITH REPLACEMENT and independently per cuboid: two cuboids may draw the same colour. That is
    intended -- the glyph draw is without-replacement only because two blocks collapsing to one
    PDDL token yields ``(on block_x block_x)``, and :func:`output_tokens` removes that hazard here
    by numbering every token. The visible consequence is that "the red block" can be ambiguous
    within one scene, which ``distractors.py`` already documents as a real property of these tasks.

    Blocks are visited in sorted order so the draw never depends on dict insertion order.
    """
    pal = tuple(palette or ()) or DATASET_COLORS
    out: Dict[str, str] = {}
    for name in sorted(names):
        out[name] = pal[int(rng.integers(0, len(pal)))][0]
    return out


def token_for_color(color: str, index: int) -> str:
    """PDDL/task token for the *index*-th block of a colour, 1-based.

    DELEGATES to ``distractors.cube_name``, the namer datagen itself uses, rather than restating
    the rule -- which is not the obvious one. Red is ALWAYS numbered (``block_red1``) while every
    other colour leaves copy 1 bare (``block_blue``, then ``block_blue2``). Authored PDDL carries
    exactly that mix, and a rewritten token that does not match it would name an object the
    generators would never have produced.
    """
    from toyblocks.tamp.blocks.distractors import cube_name
    return cube_name(str(color).strip().lower(), int(index))


def _authored_sort_key(block: str) -> Tuple[str, int]:
    """(authored colour, authored index) so block_red10 sorts after block_red2."""
    m = _COLOR_TOKEN_RE.match(str(block).strip().lower())
    if not m:
        return (str(block).lower(), 0)
    return (m.group(1), int(m.group(2) or 1))


def output_tokens(mapping: Dict[str, str]) -> Dict[str, str]:
    """block name -> its rewritten PDDL token, with per-colour indices RENUMBERED.

    Numbering follows ``distractors.cube_name`` exactly, so a rewritten token is one the
    generators could have authored: red is always numbered (``block_red1``), other colours leave
    copy 1 bare (``block_blue``, ``block_blue2``). Two cuboids drawing one colour still become two
    distinct objects, which is what the renumbering is for.

    Order is the blocks' own (authored colour, authored index), so it is stable and independent of
    dict insertion order.
    """
    by_color: Dict[str, List[str]] = {}
    for name in sorted(mapping, key=_authored_sort_key):
        by_color.setdefault(mapping[name], []).append(name)
    out: Dict[str, str] = {}
    for color, names in by_color.items():
        for idx, name in enumerate(names, start=1):
            out[name] = token_for_color(color, idx)
    return out


def rewrite_tokens(text: str, mapping: Dict[str, str]) -> str:
    """Rewrite every cuboid token in ``text`` through ``mapping`` (block name -> colour name).

    One simultaneous pass: rewriting sequentially would chain (red->blue then blue->green leaves
    the first block green). Tokens with no entry are untouched, so glyph blocks, ``workspace_table``
    and ``robot`` pass through -- which is what lets this run over the same text as the glyph
    rewrite without either corrupting the other's targets.
    """
    if not text or not mapping:
        return text
    lower = {k.lower(): v for k, v in output_tokens(mapping).items()}

    def sub(m: re.Match) -> str:
        return lower.get(m.group(0).lower()) or m.group(0)

    return _TOKEN_RE.sub(sub, text)


def block_randomization_record(mapping: Dict[str, str], palette) -> Dict[str, dict]:
    """The ``render_params.json`` ``block_randomization`` entry for one episode.

    Keyed by the GOAL TOKEN, carrying the block it came from, that block's pose column, the
    colour it became and the colour's hsv/rgb -- so the record alone answers "the goal says X,
    which object is that and what colour was rendered?" in one lookup.
    """
    hsv_by_name = {n: h for n, h in (tuple(palette or ()) or DATASET_COLORS)}
    tokens = output_tokens(mapping)
    out: Dict[str, dict] = {}
    for name in sorted(mapping):
        color = mapping[name]
        hsv = hsv_by_name.get(color)
        # Keyed by the GOAL TOKEN, not the original name -- same inverse convention as
        # glyph_map_json, and for the same reason: "pddl_goal says block_red1, whose poses are
        # those?" must be one lookup, and the forward direction is ambiguous by name whenever a
        # rewritten token collides with another block's authored name (observed: block_orange
        # recoloured to red takes the token block_red1 while the block AUTHORED block_red1 became
        # blue, so object_pose.block_red1 is a different object than the goal's block_red1).
        out[tokens[name]] = {
            "from": name,
            "pose_column": f"observation.state.object_pose.{name}",
            "color": color,
            "hsv": list(hsv) if hsv is not None else None,
            "rgb": list(hsv_to_rgb(hsv)) if hsv is not None else None,
        }
    return out


def cuboid_color_map_json(mapping: Dict[str, str], palette) -> str:
    """Compact, stable serialisation for the per-episode ``cuboid_color_map`` column."""
    import json
    return json.dumps(block_randomization_record(mapping, palette),
                      separators=(",", ":"), sort_keys=True)


# ---------------------------------------------------------------------------------------------
# Sampling + USD application. pxr imports stay function-local so this module still imports on a
# CPU box (same convention as glyphs.py / camera.py / materials.py).
# ---------------------------------------------------------------------------------------------


def block_prim_path(env_idx: int, object_name: str) -> str:
    """Where a block's prim lives at runtime.

    Pinned against the generated env, which spawns every object with
    ``prim_path=f"/World/envs/env_.*/{name}"``. Deliberately duplicated from
    ``glyphs.block_prim_path`` rather than imported, so the colour path does not depend on the
    glyph module -- ``test_block_colors`` asserts the two agree, which catches drift without
    coupling them.
    """
    return f"/World/envs/env_{env_idx}/{object_name}"


def find_surface_shader(env_idx: int, object_name: str):
    """The Shader prim carrying a cuboid's diffuse colour, or None.

    A cuboid is spawned by ``create_cuboid`` as a ``CuboidCfg`` with a single
    ``PreviewSurfaceCfg``, so unlike a glyph block (Decals_Mat + Base_Mat + PhysicsMaterial) there
    is exactly one visual Shader and the first one found is the right one. Taking the first Shader
    would be WRONG for a glyph block, which is why this is a separate function from
    ``glyphs.find_decal_shader`` instead of a shared helper with a flag.
    """
    from pxr import Usd
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(block_prim_path(env_idx, object_name))
    if not (root and root.IsValid()):
        return None
    for p in Usd.PrimRange(root):
        if p.GetTypeName() == "Shader":
            return p
    return None


def sample_wave_block_colors(wave, active, names_by_slot, base_seed: int, palette,
                             *, rng_for_episode=None, indexed_seeding=None) -> Dict[int, Dict[str, str]]:
    """{env slot: {block: colour name}} for one wave.

    Keyed on each episode's own ``episode_index``, never on the env slot, so a shard that renders
    episode 300 alone draws what the full run drew for it. Each slot samples from ITS OWN block
    list: a wave-wide union would both make the draw depend on who shared the wave and assign
    colours to blocks an episode does not have.
    """
    import numpy as _np

    out: Dict[int, Dict[str, str]] = {}
    for i in active:
        names = list(names_by_slot.get(i) or ())
        if not names:
            continue
        ep = int((wave[i] or {}).get("episode_index", i))
        if rng_for_episode is not None:
            rng = rng_for_episode(indexed_seeding, base_seed, ep, concern="block_colors")
        else:
            rng = _np.random.default_rng((int(base_seed) ^ (ep * 0x9E3779B1)) & 0xFFFFFFFF)
        out[i] = assign_colors(names, rng, palette)
    return out


def apply_wave_block_colors(assignment: Dict[int, Dict[str, str]], palette,
                            *, verbose: bool = True) -> Dict[int, Dict[str, str]]:
    """Set ``inputs:diffuseColor`` on each cuboid's surface shader. Returns what was APPLIED.

    Returns the applied map (dropping blocks whose shader was not found) rather than the sampled
    one, so the caller records what actually happened. Recording the intent instead is how a
    dataset ends up with a map that describes pixels nobody produced.
    """
    from pxr import Gf, Sdf

    rgb_by_name = {n: hsv_to_rgb(h) for n, h in (tuple(palette or ()) or DATASET_COLORS)}
    applied: Dict[int, Dict[str, str]] = {}
    missing = 0
    for env_idx, mapping in sorted(assignment.items()):
        got: Dict[str, str] = {}
        for name in sorted(mapping):
            color = mapping[name]
            rgb = rgb_by_name.get(color)
            if rgb is None:
                continue
            shader = find_surface_shader(env_idx, name)
            if shader is None:
                missing += 1
                continue
            attr = shader.GetAttribute("inputs:diffuseColor")
            if not attr:
                attr = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).GetAttr()
            attr.Set(Gf.Vec3f(*(float(c) for c in rgb)))
            got[name] = color
        if got:
            applied[env_idx] = got
    if missing and verbose:
        # Not fatal: the record and the text rewrite still describe the draw. But the pixels do
        # not, so say it loudly -- this is precisely the shape that renders cleanly and is wrong.
        print(f"[rerender] randomize_block_colors: {missing} cuboid shader(s) NOT FOUND; their "
              f"pixels keep the authored colour while the text was rewritten", flush=True)
    return applied
