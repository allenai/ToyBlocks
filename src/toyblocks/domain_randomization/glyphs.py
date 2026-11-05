"""Per-env block glyph randomization (flags.yaml ``randomize_task_letters``).

Every block in ``assets/objects/train_blocks_large`` is the SAME template with the decal mesh's
``st`` primvar shifted to a different 3x2 quadrant of one 16x16 atlas, so a block's glyph is
purely a UV offset. The decals material is ``OmniPBR.mdl``, which exposes
``inputs:texture_translate``; that makes the glyph a single per-env shader input rather than a
USD swap, and per-env Shader prims are already how ``materials.py`` randomizes the table.

This module is the PURE half: vocabulary, atlas math, seeded assignment, and the text rewriting
that keeps the symbolic goal and the language instruction describing what the camera sees. No
pxr, no torch, no Isaac -- so it is unit-testable on a CPU box.

Why the text must be rewritten: a dataset that swaps the pixels but keeps ``pddl_goal`` saying
"stack A on I" teaches a language-conditioned policy the opposite of what it sees. Both the
symbolic goal and the LeRobot ``task`` string carry block tokens and both are rewritten.
"""
from __future__ import annotations

import re
import string
from typing import Dict, Iterable, List, Sequence, Tuple

# Order is load-bearing: the atlas anchor is derived from the INDEX, so this must stay
# A-Z, then 1-9, then 10 -- the order dataset_description.json was generated in.
ATLAS_GLYPHS: Tuple[str, ...] = (
    tuple(string.ascii_uppercase) + tuple(str(n) for n in range(1, 10)) + ("10",)
)

#: Everything that may be ASSIGNED -- all 36, "10" included. It ships as a USD and has a valid
#: atlas anchor, so it RENDERS; the only thing that ever excluded it was name parsing, and
#: ``_BLOCK_GLYPH_RE`` below now matches ``block_10`` explicitly (the ``10|`` alternative is
#: first, so it wins over ``1`` + a stray ``0``).
#:
#: "10" is two characters, so it breaks the one-character-per-glyph shorthand. That is simply how
#: the atlas is: there is a 10 tile and there is no 0 tile. Everything here treats a glyph as a
#: TOKEN, never as a char -- do not reintroduce ``len(g) == 1`` assumptions.
#:
#: NOTE: toyblocks.tamp.blocks.glyph_blocks still declares DIGITS = "123456789", so the task
#: GENERATORS never author a block_10; this only widens what a block may be swapped TO.
GLYPHS: Tuple[str, ...] = ATLAS_GLYPHS

GRID_W = GRID_H = 16      # atlas is 16x16 tiles; the last x column is unused (5 blocks x 3 = 15)
QUAD_W, QUAD_H = 3, 2     # one block occupies a 3x2 quadrant
COLS = 5                  # blocks per atlas row

# The optional `_<n>` tail is the COPY INDEX (glyph_blocks.block_name: the 2nd "a" block is
# `block_a_2`). Without it the token stopped at the underscore, `block_a_2` matched nothing,
# and a second-copy block was swapped in the IMAGE while pddl_goal kept its old name -- the
# same text/pixel mismatch this module exists to prevent, just in the other direction.
_TOKEN_RE = re.compile(r"\bblock_([A-Za-z0-9]+(?:_\d+)?)\b")


def anchor_for(glyph: str) -> Tuple[int, int]:
    """Atlas quadrant anchor (col, row) in TILES for one glyph."""
    g = glyph.strip().upper()
    try:
        i = ATLAS_GLYPHS.index(g)          # anchors exist for every shipped block, incl. "10"
    except ValueError:
        raise ValueError(f"unknown glyph {glyph!r}; expected one of {len(ATLAS_GLYPHS)} "
                         f"in {ATLAS_GLYPHS[:5]}...")
    return (QUAD_W * (i % COLS), QUAD_H * (i // COLS))


def uv_offset(glyph: str) -> Tuple[float, float]:
    """UV translation that moves the decal lookup onto ``glyph``'s quadrant.

    Mirrors ``proc_blocks/simplified/alphabet_large.build_block``, which bakes exactly this into
    the mesh: ``du = ax / 16``, ``dv = -ay / 16``. The v term is NEGATIVE because atlas rows run
    top-to-bottom while UV v runs bottom-to-top.

    NOTE: this is the offset in MESH-UV space. Whether OmniPBR's ``texture_translate`` adds or
    subtracts it is a renderer convention that must be settled against an actual render --
    see ``TEXTURE_TRANSLATE_SIGN``.
    """
    ax, ay = anchor_for(glyph)
    return (ax / GRID_W, -ay / GRID_H)


# +1 because OmniPBR ADDS the translation to the texture coordinate, which is the same
# operation the generator bakes into the mesh `st`. Confirmed from the MDL source rather than
# assumed -- isaacsim/kit/mdl/core/Base/OmniPBR_ClearCoat.mdl:544 builds the UV transform as
#
#     base::transform_coordinate(
#         transform: rotation_translation_scale(
#             scaling: texture_scale, rotation: texture_rotate,
#             translation: float3(texture_translate.x, texture_translate.y, 0.0)),
#         coordinate: uvw)
#
# and rotation_translation_scale applies translation AFTER scale/rotation, i.e. uv + translate.
# The opposite sign would mirror the atlas and select a real-but-wrong glyph, so this is pinned
# by test_texture_translate_sign_matches_the_mdl_convention rather than left to a comment.
TEXTURE_TRANSLATE_SIGN = 1.0


# `block_a` -> "a", `block_2` -> "2", `block_a_2` -> "a". Same shape as glyph_blocks'
# _GLYPH_BLOCK_PATTERN; duplicated rather than imported so this module stays free of
# toyblocks.tamp (it is imported inside the Isaac render loop).
_BLOCK_GLYPH_RE = re.compile(r"^block_(10|[a-z1-9])(?:_(\d+))?$")


#: The ONLY asset library this module's atlas maths describe. Every anchor here is measured from
#: train_blocks_large's 16x16 decal atlas. The `train_blocks` library is a different asset with a
#: different decal layout, so the same UV shift lands on an arbitrary glyph -- observed: blocks
#: authored B/H/R asked for U/6/A and rendered M and friends, cleanly and undetectably.
SUPPORTED_LIBRARY = "train_blocks_large"


def library_supported(usd_path: str | None) -> bool:
    """True only for assets from :data:`SUPPORTED_LIBRARY`.

    Unknown/unparseable paths return False on purpose: rendering a confidently wrong letter is
    worse than skipping the block, because the wrong letter is a real glyph that looks correct.
    """
    return bool(usd_path) and SUPPORTED_LIBRARY in str(usd_path)


def authored_glyph(block: str) -> str | None:
    """The glyph a block's MESH is authored with, recovered from its PDDL name.

    None for anything that is not a glyph block (colour cuboids like ``block_red1``), which the
    applier treats as "cannot swap" rather than guessing.
    """
    m = _BLOCK_GLYPH_RE.match(str(block).strip().lower())
    return m.group(1).upper() if m else None


def texture_translate(glyph: str, authored: str) -> Tuple[float, float]:
    """Value for the decals shader's ``inputs:texture_translate`` to show ``glyph``.

    It is a DELTA, not an absolute offset. The generator BAKES the authored glyph's anchor into
    the mesh UVs (``alphabet_large.build_block``: ``du = ax/16``, ``dv = -ay/16``) and OmniPBR
    then ADDS ``texture_translate`` on top, so the rendered tile is ``authored + written``.

    Writing the target's absolute offset therefore lands on ``authored + target``, which is a
    real-but-wrong tile and looks like a plausible render. Observed before this was fixed:
    ``block_d`` ("D" at 9,0) asked for "6" (+3,-12) and rendered "9" (12,12); ``block_i``
    ("I" at 9,2) asked for "A" (0,0), a zero shift, and stayed "I"; ``block_2`` ("2" at 6,10)
    asked for "I" and landed on column 15, the atlas's unused 16th column, rendering a blank
    face. Only ``block_a`` looked right, because "A" is at the origin (0,0) so delta == absolute.

    ``authored`` is REQUIRED on purpose. It defaulted to "no subtraction" while this was
    broken, and an omitted argument is exactly how the absolute value got written -- a mistake
    with no visible symptom, since the wrong tile is a real glyph that renders perfectly.
    """
    du, dv = uv_offset(glyph)
    au, av = uv_offset(authored)
    return (TEXTURE_TRANSLATE_SIGN * (du - au), TEXTURE_TRANSLATE_SIGN * (dv - av))


#: In a glyph-set spec, "*" means every assignable glyph; as a MAPPING KEY it means
#: "every character not named explicitly". YAML treats a bare * as an alias marker, so it has
#: to be quoted in both positions -- "*" as a value, "*": as a key.
WILDCARD = "*"
_LETTERS = frozenset(string.ascii_uppercase)
#: Digit TOKENS, not characters: "10" is one glyph. There is no "0".
_DIGITS = frozenset(str(n) for n in range(1, 11))


def _parse_entry(entry, where: str) -> set:
    """One spec entry: "*", a single character ("A", 7), or an inclusive range ("A-F")."""
    tok = str(entry).strip().upper()
    if not tok:
        raise ValueError(f"{where}: empty entry")
    if tok == WILDCARD:
        return set(GLYPHS)
    if "-" in tok[1:]:                                   # a dash after position 0 => a range
        lo, _, hi = tok.partition("-")
        if lo in _LETTERS and hi in _LETTERS:
            if lo > hi:
                raise ValueError(f"{where}: {entry!r} runs backwards; write {hi}-{lo}")
            return {chr(c) for c in range(ord(lo), ord(hi) + 1)}
        if lo in _DIGITS and hi in _DIGITS:
            # Numeric, not ordinal: 1-10 spans ten glyphs because "10" is one of them, which is
            # also what anyone writing [1-10] means.
            if int(lo) > int(hi):
                raise ValueError(f"{where}: {entry!r} runs backwards; write {hi}-{lo}")
            return {str(n) for n in range(int(lo), int(hi) + 1)} & set(GLYPHS)
        for end in (lo, hi):
            if end not in GLYPHS:
                extra = (" (there is no 0 glyph; digits run 1-10)" if end == "0" else "")
                raise ValueError(f"{where}: {end!r} in {entry!r} is not an assignable glyph"
                                 f"{extra}; the vocabulary is A-Z and 1-10")
        raise ValueError(f"{where}: {entry!r} mixes letters and digits; "
                         f"use two entries, e.g. [A-Z, 1-10]")
    if tok in GLYPHS:
        return {tok}
    extra = " (there is no 0 glyph; digits run 1-10)" if tok == "0" else ""
    raise ValueError(f"{where}: {entry!r} is not an assignable glyph{extra}; "
                     f"the vocabulary is A-Z and 1-10")


def parse_glyph_set(spec, where: str = "glyphs") -> frozenset:
    """A glyph-set spec -> the glyphs it allows.

    ``"*"``, a single entry (``"A-Z"``), or a LIST whose entries are unioned
    (``[A, B, C]``, ``[A-F, 1-9]``, ``[A-C, X-Z, 7]``). YAML hands back ints for bare digits,
    so every entry is coerced with ``str``.
    """
    entries = list(spec) if isinstance(spec, (list, tuple)) else [spec]
    out: set = set()
    for e in entries:
        out |= _parse_entry(e, where)
    if not out:
        raise ValueError(f"{where}: resolves to no glyphs")
    return frozenset(out)


def resolve_glyph_sets(spec) -> Dict[str, frozenset]:
    """``randomize_task_letters``'s value -> {authored glyph: glyphs it may become}, for all 36.

    * ``True`` -- every block draws from all 36.
    * a string or list -- that one set applies to every block.
    * a MAPPING -- keyed by single characters, with an optional ``"*"`` catch-all. A character
      with no rule and no catch-all keeps its glyph, expressed as the one-element set ``{g}``
      so "leave it alone" needs no special case anywhere downstream.
    """
    if spec is True or spec is None:
        return {g: frozenset(GLYPHS) for g in GLYPHS}
    if not isinstance(spec, dict):
        allowed = parse_glyph_set(spec)
        return {g: allowed for g in GLYPHS}
    default = (parse_glyph_set(spec[WILDCARD], 'glyphs."*"')
               if WILDCARD in spec else None)
    out: Dict[str, frozenset] = {}
    for key, value in spec.items():
        if str(key).strip() == WILDCARD:
            continue
        k = str(key).strip().upper()
        if k not in GLYPHS:
            raise ValueError(f"glyphs: {key!r} is not an assignable glyph to key on; "
                             f"the vocabulary is A-Z and 1-10")
        out[k] = parse_glyph_set(value, f"glyphs.{k}")
    for g in GLYPHS:
        out.setdefault(g, default if default is not None else frozenset((g,)))
    return out


def assign_glyphs(block_names: Sequence[str], rng, charsets=None) -> Dict[str, str]:
    """Map each block to a glyph, deterministically from ``rng``.

    UNCONSTRAINED (``charsets is None``) draws without replacement over the whole vocabulary, in
    the caller's order.

    REPRODUCIBILITY: the pool WIDTH is part of the rng stream, so widening GLYPHS from 35 to 36
    (when "10" became assignable) changed this draw -- measured 97.5% of (n, seed) pairs differ.
    A dataset rendered before that commit will NOT render identically from its own seed. Each
    episode is still self-consistent, because its map is recorded and drives both the pixels and
    the text; what changed is only which letters a given seed produces. Do not "fix" this by
    quietly trimming the pool here -- that would break the runs made since just as silently.

    CONSTRAINED: every block draws INDEPENDENTLY from the set its authored character allows, so
    two blocks may land on the same glyph -- routine once a character is restricted to a handful
    of targets, and harmless because ``output_tokens`` renumbers copy indices so they stay
    distinct PDDL objects. Blocks are visited in their own (glyph, copy index) order so the draw
    does not depend on how the caller happened to order them.
    """
    names = list(block_names)
    if not names:
        return {}
    if charsets is None:
        if len(names) <= len(GLYPHS):
            picked = list(rng.choice(len(GLYPHS), size=len(names), replace=False))
        else:
            picked = list(rng.integers(0, len(GLYPHS), size=len(names)))
        return {n: GLYPHS[int(i)] for n, i in zip(names, picked)}
    out: Dict[str, str] = {}
    for name in sorted(names, key=_copy_sort_key):
        src = authored_glyph(name)
        pool = sorted(charsets.get(src) or (src,)) if src else None
        if not pool:
            continue                       # not a glyph block: the caller already filtered these
        out[name] = pool[int(rng.integers(0, len(pool)))]
    return out


def token_for(glyph: str) -> str:
    """PDDL/task token for a glyph. Tokens are lowercase (``block_a``, ``block_10``)."""
    return f"block_{glyph.strip().lower()}"


def _copy_sort_key(block: str) -> Tuple[str, int]:
    """(authored glyph, copy index) -- so block_a_10 sorts after block_a_2, not before."""
    m = _BLOCK_GLYPH_RE.match(str(block).strip().lower())
    if not m:
        return (str(block).lower(), 0)
    return (m.group(1), int(m.group(2) or 1))


def output_tokens(mapping: Dict[str, str]) -> Dict[str, str]:
    """block name -> its rewritten PDDL token, with copy indices RENUMBERED.

    ``token_for`` alone drops the copy index, so two blocks that drew the same glyph both became
    ``block_x`` and two distinct objects collapsed into one name -- ``(on block_x block_x)``,
    which is the unsatisfiable goal the old without-replacement draw existed to avoid. That made
    the draw, not the rewrite, carry the uniqueness, and it is what blocked per-character glyph
    sets (restrict four blocks to [A-C] and duplicates are unavoidable).

    Numbering mirrors ``glyph_blocks.block_name``: the first block to claim a glyph is
    ``block_x``, the rest are ``block_x_2``, ``block_x_3``. Order is the blocks' own
    (authored glyph, copy index), so it is stable and independent of dict insertion order.
    """
    by_glyph: Dict[str, List[str]] = {}
    for name in sorted(mapping, key=_copy_sort_key):
        by_glyph.setdefault(mapping[name], []).append(name)
    out: Dict[str, str] = {}
    for glyph, names in by_glyph.items():
        base = token_for(glyph)
        for idx, name in enumerate(names, start=1):
            out[name] = base if idx == 1 else f"{base}_{idx}"
    return out


def rewrite_tokens(text: str, mapping: Dict[str, str]) -> str:
    """Rewrite every ``block_<x>`` token in ``text`` through ``mapping`` (block name -> glyph).

    Replacement is SIMULTANEOUS via one pass: rewriting sequentially would chain (a->b then
    b->c leaves the first block as c). Word boundaries keep ``block_1`` from matching inside
    ``block_10``. Tokens with no entry in ``mapping`` are left untouched -- ``workspace_table``
    and ``robot`` are not blocks.

    Targets come from ``output_tokens``, so blocks sharing a glyph stay DISTINCT objects.
    """
    if not text or not mapping:
        return text
    lower = {k.lower(): v for k, v in output_tokens(mapping).items()}

    def sub(m: re.Match) -> str:
        new = lower.get(m.group(0).lower())
        return new if new else m.group(0)

    return _TOKEN_RE.sub(sub, text)


def glyph_map_json(mapping: Dict[str, str]) -> str:
    """Compact, stable serialisation for the per-episode ``glyph_map`` column.

    FORWARD ``{original_block: glyph}`` -- unchanged, deliberately. Shipped datasets carry this
    shape (verified on sep3_ai2_stack_glyphs_allrand_N1: ``{"block_a":"8","block_l":"A"}``) and
    nothing gains from breaking them. The inverse lookup a consumer actually wants lives in the
    unified ``block_variable_map`` column instead, which covers glyph blocks AND colour cuboids.
    """
    import json
    return json.dumps({k: mapping[k] for k in sorted(mapping)}, separators=(",", ":"))


# ---------------------------------------------------------------------------------------------
# Sampling + USD application. pxr imports stay function-local so this module still imports on a
# CPU box (same convention as camera.py / materials.py).
# ---------------------------------------------------------------------------------------------

DECAL_MATERIAL_HINT = "Decals"     # block USDs carry Decals_Mat, Base_Mat and PhysicsMaterial


def sample_wave_glyphs(wave, active, block_names, base_seed: int,
                       charsets=None) -> Dict[int, Dict[str, str]]:
    """{env_slot: {block_name: glyph}} for one wave.

    Keyed on the episode's OWN index, not the env slot, so a shard that renders episode 300 alone
    produces the same glyphs as a full run -- the same requirement that drives ``episode_rng``.
    One rng per episode draws all of that episode's blocks together, because without-replacement
    couples them; a per-block rng would re-introduce collisions.

    ``block_names`` is a MAPPING {env_slot: [names]} -- each episode's OWN blocks. A flat sequence
    is still accepted (same list for every slot) for callers that genuinely share one block set.

    Why per-slot matters: the caller used to union ``objects`` across the wave's active envs, so
    the draw SIZE depended on who shared the wave. rng.choice(len(GLYPHS), size=n, replace=False)
    does not
    agree with itself across n, so episode 300 rendered beside a 4-block episode drew different
    letters than episode 300 rendered alone -- breaking the shard-invariance this function's own
    episode keying exists to provide. It also assigned glyphs to blocks the episode did not have.
    """
    from .seeds import _stable_seed_salt, episode_rng

    salt = _stable_seed_salt("randomize_task_letters")
    per_slot = isinstance(block_names, dict)
    out: Dict[int, Dict[str, str]] = {}
    for i in active:
        ep = int(wave[i]["episode_index"])
        rng = episode_rng(int(base_seed) ^ salt, ep)
        names = block_names[i] if per_slot else block_names
        out[i] = assign_glyphs(names, rng, charsets)
    return out


def block_prim_path(env_idx: int, object_name: str) -> str:
    """Where a block's prim lives at runtime.

    Pinned against the generated env, which spawns every object with
    ``prim_path=f"/World/envs/env_.*/{name}"`` -- so the per-env path is this. Kept separate from
    the USD lookup so the format is unit-testable without Isaac: if the scene layout ever changes,
    the shader lookup silently finds nothing, glyph_map is still recorded, the strings are still
    rewritten, and only the pixels stay wrong.
    """
    return f"/World/envs/env_{env_idx}/{object_name}"


def find_decal_shader(env_idx: int, object_name: str):
    """The DECALS Shader prim for one block in one env, or None.

    Not ``materials._find_shader_prim``: that returns the first Shader under the root, and a block
    carries Decals_Mat, Base_Mat and PhysicsMaterial. Binding the base material's shader would
    tint the wood and leave the glyph untouched -- wrong in a way that still renders.
    """
    from pxr import Usd
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(block_prim_path(env_idx, object_name))
    if not (root and root.IsValid()):
        return None
    fallback = None
    for p in Usd.PrimRange(root):
        if p.GetTypeName() != "Shader":
            continue
        if DECAL_MATERIAL_HINT.lower() in str(p.GetPath()).lower():
            return p
        fallback = fallback or p
    return fallback


def apply_wave_glyphs(assignment: Dict[int, Dict[str, str]], *, verbose: bool = True) -> Dict[int, Dict[str, str]]:
    """Set ``inputs:texture_translate`` on each block's decals shader. Returns what was applied.

    Returns the APPLIED map (dropping blocks whose shader was not found) so the caller records
    what actually happened rather than what was intended -- the same reason the camera path
    records applied rather than sampled values.
    """
    from pxr import Gf, Sdf

    applied: Dict[int, Dict[str, str]] = {}
    no_shader = 0        # the decals shader prim could not be found
    no_authored = 0      # the block's own glyph could not be recovered from its name
    for env_idx, per_block in sorted(assignment.items()):
        got: Dict[str, str] = {}
        for block, glyph in sorted(per_block.items()):
            prim = find_decal_shader(env_idx, block)
            if prim is None:
                no_shader += 1
                continue
            auth = authored_glyph(block)
            if auth is None:
                # No authored glyph to subtract -> any value we wrote would shift from an
                # unknown origin and render a real-but-wrong letter. Leave it alone; reconcile()
                # reports it and the wave log names it. Unreachable via the normal path -- the
                # sampler only offers blocks whose authored glyph resolves -- so this firing
                # means the two filters have drifted apart.
                no_authored += 1
                continue
            du, dv = texture_translate(glyph, auth)
            attr = prim.GetAttribute("inputs:texture_translate")
            if not attr:
                attr = prim.CreateAttribute("inputs:texture_translate", Sdf.ValueTypeNames.Float2)
            attr.Set(Gf.Vec2f(float(du), float(dv)))
            got[block] = glyph
        applied[env_idx] = got
    if verbose:
        # Positive confirmation, not just warnings. This applier only ever spoke up on failure,
        # so when it was never called at all the log looked exactly like a clean run: zero
        # SHADER NOT FOUND, zero MISMATCH, and original letters in every frame. One line per
        # wave makes "did the swap actually run" readable straight off the log.
        # NOTE this fires TWICE per wave when camera DR is configured: render applies at t==0
        # before the render branch, and apply_wave_cameras applies again inside the camera arm.
        # Both write the same value (a pure function of the glyph), so it is a no-op rewrite --
        # but anything COUNTING these lines sees 2x the wave count on an allrand run.
        pairs = sum(len(m) for m in applied.values())
        sample = next((f" e.g. env{e} {sorted(m.items())[:3]}"
                       for e, m in sorted(applied.items()) if m), "")
        print(f"[render] randomize_task_letters: set {pairs} block-env glyph(s) across "
              f"{len(applied)} env(s){sample}")
    if (no_shader or no_authored) and verbose:
        # Loud, not a footnote: pddl_goal and task are rewritten from the SAMPLED map, so a block
        # whose shader was not found keeps its authored glyph while the text claims the new one.
        # That combination is undetectable downstream -- the parquet is self-consistent and only
        # the pixels disagree -- so name the blocks rather than just counting them.
        stale = sorted({b for e, m in assignment.items()
                        for b in m if b not in (applied.get(e) or {})})
        cause = (f"SHADER NOT FOUND for {no_shader}" if no_shader else "")
        if no_authored:
            cause += (" and " if cause else "") + f"NO AUTHORED GLYPH for {no_authored}"
        print(f"[render] randomize_task_letters: {cause} block-env "
              f"pair(s) {stale[:8]}; those blocks keep their authored glyph while pddl_goal/task "
              f"claim the new one. The dataset's text will NOT match its pixels.")
    return applied


def reconcile(sampled: Dict[int, Dict[str, str]],
              applied: Dict[int, Dict[str, str]]) -> Dict[int, list]:
    """{env_slot: [blocks sampled but not applied]} -- empty when the render matches the text."""
    out = {}
    for env_idx, per_block in (sampled or {}).items():
        got = (applied or {}).get(env_idx) or {}
        missed = sorted(b for b in per_block if b not in got)
        if missed:
            out[env_idx] = missed
    return out


def rewrite_episode_text(pddl_goal: str, task: str, mapping: Dict[str, str]) -> Tuple[str, str]:
    """Rewrite both letter-carrying strings for one episode with one mapping.

    Kept together so the two can never diverge: a dataset whose ``pddl_goal`` and ``task``
    disagree about which block is which is worse than one that is uniformly stale.
    """
    return rewrite_tokens(pddl_goal or "", mapping), rewrite_tokens(task or "", mapping)
