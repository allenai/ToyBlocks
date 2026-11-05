"""Pure-Python codegen for the blocks domain.

Turns ``(block_names, goal_texts)`` into the env.py / env_cfg.py / __init__.py
sources written into ``src/blocksuite/tasks/autogen/<slug>/`` by
``scripts/generate_task_from_pddl.py`` and ``scripts/generate_task_from_dataset.py``.

Both scripts pre-launch Isaac (``simulation_app = launch_app()``) before
importing this module because :mod:`blocksuite.tamp.blocks.pddl_goal_tensor_conversions`
transitively pulls in ``isaaclab``. No Isaac calls are made during codegen
itself; the dependency is purely an import-time chain through ``filter_goal_atoms`` /
``assert_goal_atoms_supported``. Block-spawn shape (cuboid vs USD), diffuse color,
and USD asset overrides are derived per block name by :meth:`Block.from_names`:
a name listed in :data:`USD_OVERRIDES`, or a glyph name (``block_a`` ..
``block_z``, ``block_1`` .. ``block_9``, plus ``block_<glyph>_<copy>`` for
repeats) resolved via
:func:`~blocksuite.tamp.blocks.glyph_blocks.glyph_usd_subpath`, spawns that USD
asset; everything else is a cuboid whose color comes from a color word in the
name (via :func:`_base_color_for_name`) or a rotating palette. Nothing is
hardcoded per block, so arbitrarily many blocks of a color or glyph need no new
table entries.
"""

from __future__ import annotations

import inspect
import itertools
import json
import os
import re
from typing import Dict, List, Optional, Tuple
import numpy as np

AUTOGEN_PACKAGE_VERSION = 4
"""Format version of the generated env package, stamped into ``profiles.json``.

Bump this whenever generated env *semantics* change in a way an old on-disk
package would silently keep. Autogen dirs are gitignored, so a stale package is
indistinguishable from a fresh one by inspection, and every difference between
the two shows up as a changed success rate rather than an error. Readers
(:func:`blocksuite.utils.eval_helpers._load_task_profiles`) compare against this
and refuse to run.

History:

1. Pre-stamp packages (no ``codegen_version`` key at all): goal selected from
   ``goal_atoms_list`` with a silent goal-0 default, no settle gate, fixed 8
   cube-scale slots.
2. Goal selected from ``goal_conjuncts_list`` with ``set_episode_goal`` raising on
   an unknown goal, ``scene_settled_and_gripper_open`` success gate, one
   cube-scale slot per distinct recorded scale.
3. ``scene_preset_id`` / ``robot_id`` in the generated ``env_cfg.py`` are the
   values the caller asked for rather than always the template's defaults:
   ``generate_task_from_dataset.py`` bakes the dataset's recorded provenance and
   ``generate_task_from_pddl.py`` exposes ``--scene_preset_id`` / ``--robot_id``.
   A version-2 package silently keeps the template defaults of its time
   (``sim2real`` / ``fr3_robotiq_gripper_softer``; the template now defaults to
   ``fr3_robotiq_gripper_sysid``), which changes the table baseline every
   ``table_height_delta`` is relative to.
4. Env ``i`` spawns at slot ``cfg.env_slots[i]`` when the cfg gives a per-env slot
   map (eval allocates envs to cube-scale slots by queue workload), else at
   ``i % MAX_NON_SIM_STATE_RANDOMIZATIONS`` as before; ``num_envs`` need not be a
   multiple of the slot count when the map is given. A version-3 package ignores
   the map and spawns ``i % MAX``, putting blocks at the wrong size for the
   episodes the queue routes to that env.
"""

_USD_ORIG_SIDE_LENGTH = 0.0296
_USD_DEFAULT_SCALE = 1.65

PREDICATE_PATTERN = re.compile(r'\(([^\s()]+)((?:\s+[^()\s]+)*)\)', re.MULTILINE)

# Init predicates the sampler in generate_task_from_pddl.py skips outright.
SKIP_PREDICATES = frozenset({"at", "clear"})

# Sampling order for predicate dependencies (frontface/faceup before on_table before on).
PREDICATE_ORDER: Tuple[str, ...] = ("frontface", "faceup", "on_table", "on")


class Block:
    """Spawn descriptor for one block: its name, spawn shape (``"cuboid"`` vs
    ``"usd"``), diffuse color, and (for USD blocks) an optional asset-override
    path + scale.
    """

    def __init__(
        self,
        name: str,
        object_type: str,
        color: Tuple[float, float, float],
        usd_path: Optional[str] = None,
        usd_scale: Optional[Tuple[float, float, float]] = None,
    ):
        assert object_type in ["usd", "cuboid"]
        self.name = name
        self.type = object_type
        self.color = color
        self.usd_path = usd_path
        self.usd_scale = usd_scale

    @classmethod
    def from_names(
        cls,
        block_names: List[str],
        color_jitter: float = 0.0,
        usd_library: Optional[str] = None,
    ) -> List["Block"]:
        """Build one :class:`Block` per name (in order), deriving type + color.

        A :data:`USD_OVERRIDES` entry wins; otherwise a glyph name resolves to its
        letter / digit USD by name, so the A-Z / 1-9 assets need no table entries.

        ``usd_library`` selects which glyph asset library glyph names resolve
        against (see :data:`blocksuite.tamp.blocks.glyph_blocks.GLYPH_LIBRARIES`);
        it defaults to ``train_blocks_large``. Explicit :data:`USD_OVERRIDES`
        entries (only the legacy ``block_o1`` / ``block_o2`` aliases) carry their
        own library-qualified sub-path.

        ``color_jitter`` perturbs each cuboid's RGB channels by a uniform
        ``+/- color_jitter`` (clamped to ``[0, 1]``) so that several blocks that
        share a color word get slightly different shades. Pass ``0.0`` to
        disable and get the exact base color.
        """
        library = usd_library or _DEFAULT_GLYPH_LIBRARY
        blocks: List["Block"] = []
        palette_idx = 0
        for name in block_names:
            usd_override = USD_OVERRIDES.get(name)
            if usd_override is None:
                glyph_subpath = _glyph_usd_subpath(name, library)
                if glyph_subpath is not None:
                    usd_override = (
                        glyph_subpath,
                        _GLYPH_LIBRARY_USD_SCALE.get(library, _USD_BASE_SCALE),
                    )
            if usd_override is not None:
                usd_path, usd_scale = usd_override
                base_color, _ = _base_color_for_name(name, palette_idx)
                # Color is unused for USD spawns; keep it for completeness only.
                blocks.append(cls(name, "usd", base_color, usd_path, usd_scale))
                continue
            base_color, used_palette = _base_color_for_name(name, palette_idx)
            if used_palette:
                # Only name-less blocks draw from the rotating palette; advance
                # so consecutive ones get distinct base colors.
                palette_idx += 1
            color = _jitter_color(base_color, color_jitter)
            blocks.append(cls(name, "cuboid", color))
        return blocks

    @property
    def spec(self) -> Dict:
        """Plain-dict form baked into the generated ``env.py``.

        The generated env module can't import this class, so the spawn loop in
        ``_setup_scene_objects`` consumes these dicts (keyed by block name).
        """
        return {
            "type": self.type,
            "color": self.color,
            "usd_path": self.usd_path,
            "usd_scale": self.usd_scale,
        }


# --- block color inference -------------------------------------------------
# Colors are derived from the block name at codegen time (see Block.from_names),
# so no per-block color/type table has to be maintained. A color word in the
# name wins; name-less blocks cycle through this rotating palette.

_COLOR_PALETTE: Tuple[Tuple[float, float, float], ...] = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 0.0),
    (1.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
    (0.5, 0.5, 0.5),
    (1.0, 0.5, 0.0),
    (0.5, 0.0, 1.0),
    (0.0, 0.5, 1.0),
)

# Name-fragment -> RGB for inferring colors of dynamically-named blocks.
_NAMED_COLORS: Dict[str, Tuple[float, float, float]] = {
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
    "orange": (1.0, 0.5, 0.0),
    "purple": (0.5, 0.0, 1.0),
    "cyan": (0.0, 1.0, 1.0),
    "gray": (0.5, 0.5, 0.5),
    "pink": (1.0, 0.4, 0.7),
    "brown": (0.6, 0.3, 0.0),
    "teal": (0.0, 0.5, 0.5),
    "white": (1.0, 1.0, 1.0),
}


def _base_color_for_name(name: str, palette_idx: int) -> Tuple[Tuple[float, float, float], bool]:
    """Infer a base color from the block name.

    Returns ``(rgb, used_palette)``. A color word in the name (``red``,
    ``green``, ...) wins and yields ``used_palette=False``; otherwise a rotating
    palette color keyed by ``palette_idx`` is returned with ``used_palette=True``.
    """
    lower = name.lower()
    for color_name, rgb in _NAMED_COLORS.items():
        if color_name in lower:
            return rgb, False
    return _COLOR_PALETTE[palette_idx % len(_COLOR_PALETTE)], True


def _jitter_color(
    color: Tuple[float, float, float], magnitude: float
) -> Tuple[float, float, float]:
    """Perturb each RGB channel by uniform ``+/- magnitude``, kept within [0, 1].

    Out-of-range channels are *reflected* back into ``[0, 1]`` rather than
    clamped, so channels sitting at ``0.0`` / ``1.0`` (pure colors like red) still
    vary instead of sticking to the boundary. ``magnitude <= 0`` returns the color
    unchanged. Used to give blocks that share a color word (``block_red1`` ...
    ``block_redN``) slightly different shades instead of an identical color.
    """
    if magnitude <= 0.0:
        return color
    jittered = []
    for c in color:
        v = c + float(np.random.uniform(-magnitude, magnitude))
        if v < 0.0:
            v = -v
        elif v > 1.0:
            v = 2.0 - v
        jittered.append(min(1.0, max(0.0, v)))
    return tuple(jittered)

# USD-block override sub-paths (relative to OBJECT_ASSET_DIR) and uniform scale.
# _USD_BASE_SCALE targets the ~3-5 cm toy range for train_blocks, whose glyph
# assets have a ~3.1 cm base extent (3.1 cm * 1.65 ~= 5.1 cm).
_USD_BASE_SCALE: Tuple[float, float, float] = (1.65, 1.65, 1.65)

# Per-library uniform scale for glyph USD blocks in the legacy fixed-scale spawn
# path. Each library's asset base extent differs, so the scale is chosen to keep
# the spawned block in the same ~3-5 cm toy range. train_blocks_large (AlphabetBlockLarge)
# has a ~4.5 cm base extent, so ~0.9 -> ~4 cm. Libraries not listed fall back to
# _USD_BASE_SCALE. (The cube-scale-recording path ignores this and derives scale
# from each episode's size instead.)
_GLYPH_LIBRARY_USD_SCALE: Dict[str, Tuple[float, float, float]] = {
    "train_blocks": _USD_BASE_SCALE,
    "train_blocks_large": (0.9, 0.9, 0.9),
}
# Names that are not plain glyph names but still spawn a glyph asset: the legacy
# block_o1 / block_o2 aliases for the O block. block_r / block_t / block_b used to be
# pinned here to train_blocks, which overrode --usd-library and mixed asset sets in
# one scene; they now resolve by name through the selected library like every glyph.
USD_OVERRIDES: Dict[str, Tuple[str, Tuple[float, float, float]]] = {
    "block_o1": ("train_blocks_large/usd/O.usda", _GLYPH_LIBRARY_USD_SCALE["train_blocks_large"]),
    "block_o2": ("train_blocks_large/usd/O.usda", _GLYPH_LIBRARY_USD_SCALE["train_blocks_large"]),
}

# Glyph blocks (letters A-Z, digits 1-9) are resolved by name instead of being
# enumerated in USD_OVERRIDES; see blocksuite.tamp.blocks.glyph_blocks.
from blocksuite.tamp.blocks.glyph_blocks import (
    DEFAULT_GLYPH_LIBRARY as _DEFAULT_GLYPH_LIBRARY,
    glyph_usd_subpath as _glyph_usd_subpath,
)


# ----------------------------- text parsing -----------------------------

def extract_block_names(pddl_text: str) -> List[str]:
    """Return objects whose declared type starts with ``block`` from ``(:objects ...)``.

    Handles both single-type and multi-type declarations:
        block_r block_t - block
        block_r - block_r_e
        block_o1 block_b - block_o_b
    """
    match = re.search(r'\(:objects(.*?)\)', pddl_text, re.DOTALL)
    if not match:
        raise ValueError("No (:objects ...) section found in PDDL text.")
    objects_block = match.group(1)
    names: List[str] = []
    for type_decl in re.finditer(r'((?:[\w]+\s+)+)-\s+([\w]+)', objects_block):
        obj_names_str = type_decl.group(1).strip()
        obj_type = type_decl.group(2).strip()
        if obj_type.startswith('block'):
            names.extend(obj_names_str.split())
    return names


def _extract_balanced_block(text: str, opener_pat: str) -> str:
    """Find ``opener_pat`` in ``text`` and return the body of its parenthesized form.

    ``opener_pat`` is matched case-insensitively (``\\(:init`` / ``\\(:goal``);
    the returned string is the inner text between the matching open / close parens,
    excluding both.
    """
    match = re.search(opener_pat, text, re.IGNORECASE)
    if not match:
        raise ValueError(f"No {opener_pat!r} section found in PDDL text.")
    i = match.end()
    while i < len(text) and text[i].isspace():
        i += 1
    depth = 1
    block = ''
    while i < len(text):
        c = text[i]
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0:
                break
        block += c
        i += 1
    if depth > 0:
        raise ValueError(
            f"unbalanced parentheses in {opener_pat!r} block "
            f"(depth={depth} at end of input); PDDL text is truncated or malformed"
        )
    return block


def extract_init_block(text: str) -> str:
    """Body inside ``(:init ...)`` (balanced-paren aware, handles nested forms)."""
    return _extract_balanced_block(text, r'\(:init')


def extract_goal_block(text: str) -> str:
    """Body inside ``(:goal ...)`` (balanced-paren aware)."""
    return _extract_balanced_block(text, r'\(:goal')


def parse_predicates_from_block(block: str) -> List[Dict]:
    """Extract flat ``(name args...)`` literals from a predicate body.

    Logical connectives (``and``, ``or``, ``not``) and any other parens-containing
    forms are ignored because :data:`PREDICATE_PATTERN` doesn't allow nested parens
    in its arg group.
    """
    block = re.sub(r';.*', '', block)
    block = re.sub(r'\s+', ' ', block)
    out: List[Dict] = []
    for match in PREDICATE_PATTERN.finditer(block):
        pred_name = match.group(1).strip()
        args = [arg for arg in match.group(2).strip().split() if arg]
        out.append({'name': pred_name, 'args': args})
    return out


_PDDL_SKIP_FORMS = frozenset({"and", "or", "forall", "exists", "when", "imply"})


def _parse_goal_predicates_with_negation(block: str) -> List[Dict]:
    """Parse a PDDL goal body into predicate dicts that carry a ``negated`` field.

    Unlike :func:`parse_predicates_from_block`, this walks the s-expression tree
    so that ``(not (foo a b))`` emits ``{'name': 'foo', 'args': ['a', 'b'],
    'negated': True}`` rather than a bare ``{'name': 'foo', 'args': ['a', 'b']}``.
    Connective forms (``and``, ``or``, ``forall``, …) are recursed into without
    generating their own predicate entries.
    """
    block = re.sub(r';.*', '', block)
    block = re.sub(r'\s+', ' ', block).strip()
    tokens = re.findall(r'[()]|[^\s()]+', block)
    out: List[Dict] = []
    idx = [0]

    def peek() -> str | None:
        return tokens[idx[0]] if idx[0] < len(tokens) else None

    def consume() -> str:
        t = tokens[idx[0]]
        idx[0] += 1
        return t

    def parse_form(negated: bool = False) -> None:
        if peek() != '(':
            consume()
            return
        consume()  # '('
        if peek() is None:
            return
        name = consume()
        name_lower = name.lower()
        if name_lower == 'not':
            while peek() is not None and peek() != ')':
                if peek() == '(':
                    parse_form(negated=not negated)
                else:
                    consume()
        elif name_lower in _PDDL_SKIP_FORMS:
            while peek() is not None and peek() != ')':
                if peek() == '(':
                    parse_form(negated=negated)
                else:
                    consume()
        else:
            args: List[str] = []
            while peek() is not None and peek() != ')':
                t = peek()
                if t == '(':
                    # nested parens in args (e.g. typed vars) — skip subform
                    consume()
                    depth = 1
                    while depth > 0 and peek() is not None:
                        t2 = consume()
                        if t2 == '(':
                            depth += 1
                        elif t2 == ')':
                            depth -= 1
                else:
                    args.append(consume())
            out.append({'name': name, 'args': args, 'negated': negated})
        if peek() == ')':
            consume()

    while idx[0] < len(tokens):
        if peek() == '(':
            parse_form()
        else:
            consume()

    return out


def _sort_predicates(predicates: List[Dict]) -> List[Dict]:
    order_index = {name: i for i, name in enumerate(PREDICATE_ORDER)}
    return sorted(predicates, key=lambda p: order_index.get(p["name"], len(PREDICATE_ORDER)))


def parse_pddl_init_from_text(pddl_text: str) -> List[Dict]:
    """Parse ``(:init ...)`` from a PDDL string and return sorted predicate dicts."""
    return _sort_predicates(parse_predicates_from_block(extract_init_block(pddl_text)))


def extract_clear_blocks(predicates: List[Dict]) -> set:
    """Return the set of blocks marked ``(clear ...)`` in the init predicates.

    Used during reset-state sampling to filter ``(on a b)`` predicates whose ``b``
    is supposed to stay clear (no block on top).
    """
    clear_blocks = set()
    for pred in predicates:
        if pred["name"] == "clear" and len(pred["args"]) >= 1:
            clear_blocks.add(pred["args"][0])
    return clear_blocks


def reorder_predicates_by_dependencies(
    predicates: List[Dict], clear_blocks: Optional[set] = None
) -> List[Dict]:
    """Reorder predicates so each ``(on b1 b2)`` only runs after ``b2`` is supported.

    A block ``b2`` is supported if some earlier predicate is ``(on_table b2 ...)``
    or ``(on b2 ...)``. Also drops ``(on b1 b2)`` whose ``b2`` is in ``clear_blocks``.
    """
    if clear_blocks is None:
        clear_blocks = set()

    non_on_preds = [p for p in predicates if p["name"] != "on"]
    on_preds = [p for p in predicates if p["name"] == "on"]

    filtered_on_preds = []
    for pred in on_preds:
        if len(pred["args"]) >= 2:
            block2 = pred["args"][1]
            if block2 in clear_blocks:
                print(
                    f"Warning: Skipping (on {pred['args'][0]} {block2}) because "
                    f"{block2} is marked as 'clear'"
                )
                continue
        filtered_on_preds.append(pred)
    on_preds = filtered_on_preds

    supported_blocks = set()
    for pred in non_on_preds:
        if pred["name"] == "on_table" and len(pred["args"]) >= 1:
            supported_blocks.add(pred["args"][0])

    ordered_on_preds = []
    remaining_on_preds = on_preds.copy()
    while remaining_on_preds:
        processable = []
        for pred in remaining_on_preds:
            if len(pred["args"]) >= 2 and pred["args"][1] in supported_blocks:
                processable.append(pred)
        if not processable:
            print(
                f"Warning: Could not resolve dependencies for "
                f"{len(remaining_on_preds)} 'on' predicates. Processing them anyway."
            )
            ordered_on_preds.extend(remaining_on_preds)
            break
        for pred in processable:
            ordered_on_preds.append(pred)
            remaining_on_preds.remove(pred)
            if len(pred["args"]) >= 1:
                supported_blocks.add(pred["args"][0])

    return non_on_preds + ordered_on_preds


def parse_pddl_goal_from_text(text: str) -> List[Dict]:
    """Parse a goal expression from text into sorted predicate dicts.

    Tolerant of multiple shapes — accepts a full PDDL containing ``(:goal ...)``,
    just the ``(:goal ...)`` s-expression, or a bare goal body like
    ``(and (on a b))`` / ``(on a b)`` (the dataset's ``pddl_goal`` column may
    store any of these depending on the datagen pipeline). Negation is tracked:
    ``(not (foo a b))`` emits ``{'name': 'foo', 'args': ['a', 'b'], 'negated': True}``.
    """
    if re.search(r'\(:goal', text, re.IGNORECASE):
        block = extract_goal_block(text)
    else:
        block = text
    return _sort_predicates(_parse_goal_predicates_with_negation(block))


# ----------------------------- quantified goal compiler -----------------------------
#
# The flat parser above only handles a conjunction of grounded literals. Newer
# datagen emits quantified, typed goals (e.g. ``(exists (?b1 ?b2 - block_red)
# (and (not (= ?b1 ?b2)) (on ?b1 ?b2)))``). :func:`_compile` handles these in one
# recursive walk: it grounds lifted vars over the scene blocks, expands
# ``exists``/``forall`` to OR/AND over groundings, and folds (in)equality
# literals. It returns a Python ``bool`` for branches that fold to a constant
# (so an enclosing ``not``/``and`` can drop them) or a ``str`` tensor expression
# reusing ``batched_goal_conjunct``.
#
# URGENT TODO (post-rebase): move this parse/ground/compile logic into
# ``pddl_goal_tensor_conversions.py`` (which already owns
# ``SUPPORTED_GOAL_PREDICATES`` + the flat parser) so codegen stays a thin emitter
# and there's one goal path. Likely conflict point if main reworks goal handling —
# unify it while resolving that conflict.

_QUANT_OPS = {"exists": "or", "forall": "and"}
_CONNECTIVES = frozenset({"and", "or", "not", "=", *_QUANT_OPS})


def _tokenize_sexpr(text: str) -> List[str]:
    """Split a PDDL s-expression body into ``(`` / ``)`` / atom tokens."""
    text = re.sub(r";.*", "", text)
    return re.findall(r"[()]|[^\s()]+", text)


def _read_sexpr(tokens: List[str], i: int):
    """Read one (possibly nested) parenthesized form; returns ``(node, next_i)``.

    ``node`` is a ``str`` for atoms or a ``list`` for parenthesized forms.
    """
    if tokens[i] != "(":
        return tokens[i], i + 1
    i += 1
    items: list = []
    while i < len(tokens) and tokens[i] != ")":
        sub, i = _read_sexpr(tokens, i)
        items.append(sub)
    if i >= len(tokens):
        raise ValueError("unbalanced parentheses in goal expression")
    return items, i + 1


def _parse_typed_vars(decl: list) -> List[Tuple[str, str]]:
    """Parse ``(?b1 - block_red ?b2 - block_red)`` style declarations.

    Also supports the shared-type form ``(?b1 ?b2 - block_red)``. Every variable
    must be typed (fails loudly otherwise — an untyped var can't be grounded).
    """
    out: List[Tuple[str, str]] = []
    pending: List[str] = []
    i = 0
    while i < len(decl):
        tok = decl[i]
        if not isinstance(tok, str):
            raise ValueError(f"nested form inside quantifier variable list: {decl}")
        if tok == "-":
            if i + 1 >= len(decl):
                raise ValueError(f"missing type after '-' in quantifier vars: {decl}")
            typ = decl[i + 1]
            for v in pending:
                out.append((v, typ))
            pending = []
            i += 2
        else:
            pending.append(tok)
            i += 1
    if pending:
        raise ValueError(f"untyped variables {pending} in quantifier declaration {decl}")
    return out


def _parse_goal_sexpr(text: str):
    """Parse a goal (full PDDL, ``(:goal ...)``, or bare body) to a raw s-expr.

    Returns a ``str`` (bare 0-arity atom) or nested ``list`` of the same, as
    produced by :func:`_read_sexpr`. No grounding or validation happens here.
    """
    body = extract_goal_block(text) if re.search(r"\(:goal", text, re.IGNORECASE) else text
    tokens = _tokenize_sexpr(body)
    if not tokens:
        raise ValueError(f"empty goal expression: {text!r}")
    if tokens[0] != "(":
        raise ValueError(f"goal expression must start with '(': {body!r}")
    sexpr, _ = _read_sexpr(tokens, 0)
    return sexpr


def infer_block_types(block_names: List[str]) -> Dict[str, List[str]]:
    """Map each PDDL block *type* to the concrete scene objects of that type.

    The dataset doesn't carry a ``(:objects ...)`` type declaration, so the type
    of a block is inferred by stripping trailing digits from its name (the datagen
    naming convention): ``block_red1 ... block_red8 -> block_red``. A name with no
    trailing digits maps to itself (``block_b -> block_b``).
    """
    out: Dict[str, List[str]] = {}
    for name in block_names:
        type_name = re.sub(r"\d+$", "", name)
        out.setdefault(type_name, []).append(name)
    for type_name in out:
        out[type_name].sort()
    return out


def _collect_predicate_names(sexpr, out: set) -> None:
    """Collect lowercased predicate names from a raw goal s-expression.

    Walks connectives/quantifiers (``and``/``or``/``not``/``=``/``exists``/
    ``forall``) without treating them as predicates; everything else with a
    symbol head is a relational literal whose head is the predicate name. Used
    to gate against :data:`SUPPORTED_GOAL_PREDICATES` before compiling.
    """
    if isinstance(sexpr, str):
        out.add(sexpr.lower())
        return
    if not sexpr:
        return
    head = sexpr[0]
    if not isinstance(head, str):
        return
    h = head.lower()
    if h == "=":
        # ``(= a b)`` operands are object terms / variables, not predicates.
        return
    if h in _CONNECTIVES:
        # Quantifier var-lists (sexpr[1] for exists/forall) carry no predicates.
        rest = sexpr[2:] if h in _QUANT_OPS else sexpr[1:]
        for child in rest:
            _collect_predicate_names(child, out)
        return
    out.add(h)


def _args_tuple_repr(args: Tuple[str, ...]) -> str:
    inner = ", ".join(repr(a) for a in args)
    if len(args) == 1:
        inner += ","
    return "(" + inner + ")"


def _emit_literal(name: str, args: Tuple[str, ...], subst: Dict[str, str]) -> str:
    """Substitute bound vars into a literal and emit its tensor expression."""
    resolved = tuple(subst.get(a, a) for a in args)
    free = [a for a in resolved if a.startswith("?")]
    if free:
        raise ValueError(
            f"goal literal ({name} {' '.join(resolved)}) contains unbound "
            f"variable(s) {free}; every variable must be bound by an enclosing "
            "exists/forall."
        )
    return f"batched_goal_conjunct(self, {name!r}, {_args_tuple_repr(resolved)})"


def _fold(op: str, results: List) -> "bool | str":
    """Fold ``and``/``or`` over compiled operands (``bool`` consts or expr strings).

    ``and``: a ``False`` operand short-circuits to ``False``; ``True`` operands
    drop out. ``or``: a ``True`` operand short-circuits to ``True``; ``False``
    drops out. The surviving expression strings are ``&``/``|``-joined (a lone
    survivor is returned unwrapped, matching the legacy emitter).
    """
    short, drop, joiner = (False, True, " & ") if op == "and" else (True, False, " | ")
    exprs: List[str] = []
    for r in results:
        if r is short:
            return short
        if r is drop:
            continue
        exprs.append(r)
    if not exprs:
        return drop
    if len(exprs) == 1:
        return exprs[0]
    return "(" + joiner.join(exprs) + ")"


def _compile(sexpr, subst: Dict[str, str], type_to_objs: Dict[str, List[str]]) -> "bool | str":
    """Recursively ground + compile a goal s-expression to a bool const or expr string."""
    if isinstance(sexpr, str):
        return _emit_literal(sexpr, (), subst)
    if not sexpr:
        raise ValueError("empty s-expression in goal")
    head = sexpr[0]
    if not isinstance(head, str):
        raise ValueError(f"goal sub-expression must start with a symbol, got {head!r}")
    h = head.lower()
    if h in ("and", "or"):
        return _fold(h, [_compile(c, subst, type_to_objs) for c in sexpr[1:]])
    if h == "not":
        if len(sexpr) != 2:
            raise ValueError(f"(not ...) takes exactly one argument: {sexpr}")
        child = _compile(sexpr[1], subst, type_to_objs)
        return (not child) if isinstance(child, bool) else f"(~{child})"
    if h == "=":
        if len(sexpr) != 3 or not all(isinstance(a, str) for a in sexpr[1:]):
            raise ValueError(f"(= a b) malformed: {sexpr}")
        return subst.get(sexpr[1], sexpr[1]) == subst.get(sexpr[2], sexpr[2])
    if h in _QUANT_OPS:
        if len(sexpr) != 3 or not isinstance(sexpr[1], list):
            raise ValueError(f"({h} (vars) body) malformed: {sexpr}")
        typed_vars = _parse_typed_vars(sexpr[1])
        var_names = [v for v, _ in typed_vars]
        domains: List[List[str]] = []
        for var, type_name in typed_vars:
            objs = type_to_objs.get(type_name)
            if not objs:
                raise ValueError(
                    f"quantifier variable {var} has type {type_name!r} which matches no "
                    f"scene blocks. Known types (inferred from block names by stripping "
                    f"trailing digits): {sorted(type_to_objs)}."
                )
            domains.append(objs)
        results = []
        for combo in itertools.product(*domains):
            child_subst = dict(subst)
            child_subst.update(zip(var_names, combo))
            results.append(_compile(sexpr[2], child_subst, type_to_objs))
        return _fold(_QUANT_OPS[h], results)
    args: List[str] = []
    for a in sexpr[1:]:
        if not isinstance(a, str):
            raise ValueError(f"predicate {head!r} has a nested argument {a!r}; not supported")
        args.append(a)
    return _emit_literal(head, tuple(args), subst)


def _top_level_conjuncts(sexpr) -> list:
    """Flatten a top-level (possibly nested) ``and`` into its operand s-exprs.

    A non-``and`` top-level form (single literal, ``or``, ``exists``, …) yields
    a one-element list. Keeps per-literal conjunct granularity so the generated
    env emits one ``g{g}_conj_{c}`` stat per literal, matching the legacy path.
    """
    if isinstance(sexpr, list) and sexpr and isinstance(sexpr[0], str) and sexpr[0].lower() == "and":
        out: list = []
        for child in sexpr[1:]:
            out.extend(_top_level_conjuncts(child))
        return out
    return [sexpr]


def _sexpr_to_text(sexpr) -> str:
    """Render a raw goal s-expression back to single-line PDDL text."""
    if isinstance(sexpr, str):
        return sexpr
    return "(" + " ".join(_sexpr_to_text(c) for c in sexpr) + ")"


def compile_goal_text(goal_text: str, block_names: List[str]) -> List[str]:
    """Parse + ground + compile one goal into top-level conjunct tensor expressions.

    Returns one Python expression string per top-level conjunct (a flat goal yields
    one expression per literal, exactly like the legacy path; a quantified goal
    yields the expanded OR/AND expression). Raises if the goal carries an
    unsupported predicate, an ungroundable type, or no checkable literal at all.
    """
    return [expr for expr, _ in _compile_goal_conjuncts(goal_text, block_names)]


def goal_conjunct_labels(goal_text: str, block_names: List[str]) -> List[str]:
    """PDDL text of each compiled conjunct of one goal, aligned with :func:`compile_goal_text`.

    Entry ``c`` is the source literal (or quantified form) behind the env's
    ``goal_conj_<c>`` stat, e.g. ``"(on block_x block_w)"``.
    """
    return [label for _, label in _compile_goal_conjuncts(goal_text, block_names)]


def _compile_goal_conjuncts(goal_text: str, block_names: List[str]) -> List[Tuple[str, str]]:
    """``(tensor_expr, pddl_label)`` per kept top-level conjunct; see :func:`compile_goal_text`."""
    sexpr = _parse_goal_sexpr(goal_text)

    names: set = set()
    _collect_predicate_names(sexpr, names)
    from blocksuite.tamp.blocks.goal_predicates import SUPPORTED_GOAL_PREDICATES
    unsupported = sorted(n for n in names if n not in SUPPORTED_GOAL_PREDICATES)
    if unsupported:
        raise ValueError(
            f"goal {goal_text!r} uses predicate(s) without a vectorized goal check: "
            f"{unsupported}. Supported: {sorted(SUPPORTED_GOAL_PREDICATES)}. Predicates "
            "that depend on IK, motion validation, or contact aren't expressible as "
            "pure world-state tensors."
        )

    type_to_objs = infer_block_types(block_names)
    top_level = _top_level_conjuncts(sexpr)
    results = [_compile(c, {}, type_to_objs) for c in top_level]
    # A ``False`` conjunct makes the whole conjunction unsatisfiable, so there's
    # nothing checkable left (mirrors the legacy simplify-then-collect behavior).
    # Otherwise keep the tensor-expression (str) conjuncts and drop folded-out
    # ``True`` constants.
    unsatisfiable = any(r is False for r in results)
    conjuncts: List[Tuple[str, str]] = (
        []
        if unsatisfiable
        else [(r, _sexpr_to_text(c)) for r, c in zip(results, top_level) if isinstance(r, str)]
    )
    if not conjuncts:
        raise ValueError(
            f"goal {goal_text!r} has no checkable predicate after grounding. Expected "
            "an expression like '(and (on a b))' or a quantified goal over blocks."
        )
    return conjuncts


def goal_objects_from_conjuncts(conjuncts: List[str], block_names: List[str]) -> List[str]:
    """Block names referenced by a compiled goal's conjunct expressions.

    The compiled conjuncts are ``batched_goal_conjunct(self, 'pred', ('a', 'b'))``
    strings we emit ourselves, so the grounded object names are exactly the
    single-quoted tokens (quantified goals are already expanded, so this covers
    them too). Intersecting with ``block_names`` drops predicate names and the
    table. Used to scope the settle gate to the objects the goal is *about*,
    instead of every rigid object in the scene. Raises when a goal grounds to no
    block, which would leave the gate with nothing to check.
    """
    blocks = set(block_names)
    found: set = set()
    for expr in conjuncts:
        found.update(t for t in re.findall(r"'([^']*)'", expr) if t in blocks)
    if not found:
        raise ValueError(
            f"compiled goal conjuncts reference no scene block: {conjuncts}. The "
            "settle gate would have nothing to check, so success could fire while "
            "the goal blocks are still moving."
        )
    return sorted(found)


def assert_goal_text_supported(goal_text: str, block_names: List[str]) -> None:
    """Raise (before any output is written) if ``goal_text`` can't be compiled.

    Thin wrapper over :func:`compile_goal_text` so callers can validate early.
    """
    compile_goal_text(goal_text, block_names)


def parse_pddl_randomization_ranges_from_text(text: str) -> Dict:
    # NOTE (stao): in the future randomizations and PDDL will just be defined in python instead of having to make parsers
    """Parse the randomization ranges from the PDDL text."""
    randomization_ranges = {}
    for line in text.split("\n"):
        if (";block_size_range = [" in line):
            randomization_ranges["block_size_range"] = eval(line.split("=")[1].strip())
        if (";table_height_range = [" in line):
            randomization_ranges["table_height_range"] = eval(line.split("=")[1].strip())
        if (";friction_range = [" in line):
            randomization_ranges["friction_range"] = eval(line.split("=")[1].strip())
        # Where on the table blocks may spawn, as table-frame offsets. Either a
        # single interval (";block_x_range = [-0.2, 0.58]") or a list of disjoint
        # intervals (";block_x_range = [[-0.2, -0.1], [0.3, 0.5]]"), which spawns
        # nothing in the gap. Unset means the on_table sampler's default window.
        # Consumed by on_table.set_spawn_ranges, not by the env-cfg codegen.
        if (";block_x_range = [" in line):
            randomization_ranges["block_x_range"] = eval(line.split("=", 1)[1].strip())
        if (";block_y_range = [" in line):
            randomization_ranges["block_y_range"] = eval(line.split("=", 1)[1].strip())
    return randomization_ranges


# ----- file-based wrappers (kept for the existing pddl-driven script) -----

def parse_pddl_init(pddl_path: str) -> Tuple[List[Dict], List[str]]:
    with open(pddl_path, 'r') as f:
        content = f.read()
    return parse_pddl_init_from_text(content), extract_block_names(content)


def parse_pddl_goal(pddl_path: str) -> Tuple[List[Dict], List[str]]:
    with open(pddl_path, 'r') as f:
        content = f.read()
    return parse_pddl_goal_from_text(content), extract_block_names(content)


def task_slug_from_pddl_path(pddl_path: str) -> str:
    """PDDL filename stem with ``-`` -> ``_`` (output package + class prefix)."""
    stem = os.path.splitext(os.path.basename(pddl_path))[0]
    return stem.replace("-", "_")


def task_slug_from_dataset_path(dataset_path: str) -> str:
    """LeRobot dataset directory basename with ``-`` -> ``_`` (output package + class prefix)."""
    name = os.path.basename(os.path.normpath(dataset_path))
    return name.replace("-", "_")


# ----------------------------- goal codegen -----------------------------

def build_goal_compute_method(
    *,
    goal_conjuncts_list: List[List[str]],
    goal_labels_list: List[List[str]],
    goal_objects_list: List[List[str]],
    goal_texts: List[str],
    goal_display_name: str,
) -> Tuple[str, int, int]:
    """Return ``(compute_method_source, max_goal_conjuncts, max_goal_objects)`` for the goal block.

    ``goal_conjuncts_list[g]`` is the list of per-conjunct tensor expression
    strings (from :func:`compile_goal_text`) for goal ``g``,
    ``goal_labels_list[g]`` their PDDL text (from :func:`goal_conjunct_labels`),
    and ``goal_objects_list[g]`` the block names it references (from
    :func:`goal_objects_from_conjuncts`), which scope that goal's settle gate.

    Each step evaluates only the goals some env is currently assigned
    (``_per_env_goal_list``, kept in step with ``_per_env_goal_idx`` by
    ``set_episode_goal``), and each distinct conjunct expression once even when
    several goals share it: a dataset eval compiles every goal in the episode
    set (hundreds) while at most ``num_envs`` of them are live at a time. Each
    env's own goal conjuncts land in ``_goal_conjuncts`` (``num_envs x
    max_goal_conjuncts``, padded with True past the end of a shorter goal) and
    its result in ``_goal_success``. The settle gate's terms for that goal land in
    ``_goal_obj_lin_vel`` / ``_goal_obj_ang_vel`` (``num_envs x max_goal_objects``,
    column k = k-th block of ``_GOAL_OBJECTS[g]``, zero past the end of a shorter
    goal), ``_goal_finger_pos``, ``_goal_blocks_settled`` and ``_goal_gripper_open``,
    so eval can log what the gate saw. Also emits ``_GOAL_TEXTS``,
    ``_GOAL_CONJUNCT_LABELS``, ``_init_goal_state``, ``set_episode_goal``, and
    ``goal_conjunct_labels``. Works for both single-goal and
    multi-goal datasets (a single goal is just the degenerate case where
    ``_per_env_goal_idx`` is always 0).
    """
    n_goals = len(goal_conjuncts_list)
    if len(goal_objects_list) != n_goals:
        raise ValueError(
            f"goal_objects_list has {len(goal_objects_list)} entries but there are "
            f"{n_goals} goal(s); every goal needs its own settle-gate object set."
        )
    if len(goal_labels_list) != n_goals or any(
        len(labels) != len(exprs) for labels, exprs in zip(goal_labels_list, goal_conjuncts_list)
    ):
        raise ValueError(
            f"goal_labels_list must hold one label per compiled conjunct of each of the "
            f"{n_goals} goal(s); a misaligned label would name the wrong goal_conj_<c>."
        )

    # Distinct conjunct expressions across all goals; goals refer to them by index.
    conjunct_index: Dict[str, int] = {}
    goal_conjunct_ids: List[Tuple[int, ...]] = []
    for g_idx, conjunct_exprs in enumerate(goal_conjuncts_list):
        if not conjunct_exprs:
            # compile_goal_text raises when a goal grounds to no checkable
            # conjunct, so this is unreachable; keep it loud rather than
            # emitting a trivially-true goal check.
            raise ValueError(
                f"goal {g_idx} in {goal_display_name!r} compiled to zero conjuncts; "
                "a goal with nothing to check would report every episode successful."
            )
        goal_conjunct_ids.append(
            tuple(conjunct_index.setdefault(expr, len(conjunct_index)) for expr in conjunct_exprs)
        )
    max_conjuncts = max(len(ids) for ids in goal_conjunct_ids)
    goal_objects = [tuple(objs) for objs in goal_objects_list]
    max_objects = max(len(objs) for objs in goal_objects)

    goal_texts_repr = repr(goal_texts)
    preamble: List[str] = [
        f"    _GOAL_TEXTS: list = {goal_texts_repr}",
        "    # Per goal: indices of its conjuncts (_goal_conj_<k> methods), in goal order.",
        f"    _GOAL_CONJUNCT_IDS: tuple = {tuple(goal_conjunct_ids)!r}",
        "    # Per goal: PDDL text of each conjunct, i.e. what goal_conj_<c> checks.",
        f"    _GOAL_CONJUNCT_LABELS: tuple = {tuple(tuple(labels) for labels in goal_labels_list)!r}",
        "    # Per goal: the blocks its settle gate checks.",
        f"    _GOAL_OBJECTS: tuple = {tuple(goal_objects)!r}",
        f"    _MAX_GOAL_CONJUNCTS: int = {max_conjuncts}",
        f"    _MAX_GOAL_OBJECTS: int = {max_objects}",
        "",
        "    @staticmethod",
        "    def _normalize_goal_text(text: str) -> str:",
        '        """Whitespace/case-insensitive form used to match a queue goal to a compiled one."""',
        "        return ' '.join(text.split()).lower()",
        "",
        "    def _init_goal_state(self):",
        "        self._per_env_goal_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)",
        "        # Host-side copy of _per_env_goal_idx, so picking the live goals each step needs no device sync.",
        "        self._per_env_goal_list: list[int] = [0] * self.num_envs",
        "",
        "    def set_episode_goal(self, env_idx: int, pddl_goal: str) -> None:",
        "        normalized = self._normalize_goal_text(pddl_goal)",
        "        for i, gt in enumerate(self._GOAL_TEXTS):",
        "            if normalized == self._normalize_goal_text(gt):",
        "                self._per_env_goal_idx[env_idx] = i",
        "                self._per_env_goal_list[env_idx] = i",
        "                return",
        "        raise ValueError(",
        '            "set_episode_goal env=%d: goal %r is not one of the goals this task "',
        '            "was compiled for (%s). Falling back to another goal would grade the "',
        '            "episode against the wrong target and fabricate the result; "',
        '            "regenerate the task from the same episode set the queue serves."',
        "            % (env_idx, pddl_goal, self._GOAL_TEXTS)",
        "        )",
        "",
        "    def goal_conjunct_labels(self, pddl_goal: str) -> list[str]:",
        '        """PDDL text of ``pddl_goal``\'s conjuncts; entry c is what goal_conj_<c> checks."""',
        "        normalized = self._normalize_goal_text(pddl_goal)",
        "        for i, gt in enumerate(self._GOAL_TEXTS):",
        "            if normalized == self._normalize_goal_text(gt):",
        "                return list(self._GOAL_CONJUNCT_LABELS[i])",
        "        raise ValueError(",
        '            "goal_conjunct_labels: goal %r is not one of the goals this task was "',
        '            "compiled for (%s)." % (pddl_goal, self._GOAL_TEXTS)',
        "        )",
        "",
    ]
    for expr, k in conjunct_index.items():
        preamble += [
            f"    def _goal_conj_{k}(self) -> torch.Tensor:",
            "        from blocksuite.tamp.blocks.pddl_goal_tensor_conversions import batched_goal_conjunct",
            f"        return {expr}",
            "",
        ]

    parts: List[str] = [
        "    def _compute_intermediate_values(self):",
        f"        # Auto-generated PDDL goal success: {goal_display_name}",
        f"        # {n_goals} goal(s), {len(conjunct_index)} distinct conjunct(s); per-env selection via _per_env_goal_idx",
        "        from blocksuite.tamp.blocks.pddl_goal_tensor_conversions import goal_settle_terms",
        "",
        "        if not hasattr(self, '_per_env_goal_idx'):",
        "            self._init_goal_state()",
        "",
        "        # Only the goals some env is assigned, each shared conjunct computed once.",
        "        conj_cache: dict[int, torch.Tensor] = {}",
        "        pad = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)",
        "        goal_conjuncts = pad.unsqueeze(1).repeat(1, self._MAX_GOAL_CONJUNCTS)",
        "        goal_settled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)",
        "        # Settle-gate terms of each env's own goal, logged as eval stats.",
        "        goal_obj_lin_vel = torch.zeros(self.num_envs, self._MAX_GOAL_OBJECTS, device=self.device)",
        "        goal_obj_ang_vel = torch.zeros(self.num_envs, self._MAX_GOAL_OBJECTS, device=self.device)",
        "        goal_finger_pos = torch.zeros(self.num_envs, device=self.device)",
        "        goal_blocks_settled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)",
        "        goal_gripper_open = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)",
        "        for g in sorted(set(self._per_env_goal_list)):",
        "            values: list[torch.Tensor] = []",
        "            for k in self._GOAL_CONJUNCT_IDS[g]:",
        "                if k not in conj_cache:",
        "                    conj_cache[k] = getattr(self, f'_goal_conj_{k}')()",
        "                values.append(conj_cache[k])",
        "            values += [pad] * (self._MAX_GOAL_CONJUNCTS - len(values))",
        "            on_goal = self._per_env_goal_idx == g",
        "            goal_conjuncts = torch.where(on_goal.unsqueeze(1), torch.stack(values, dim=-1), goal_conjuncts)",
        "            # Gate each goal on its own objects settling and the gripper releasing, so",
        "            # 'on' can't fire while a block is still grasped or mid-flight. Scoped to",
        "            # the blocks this goal references: an unrelated distractor rolling",
        "            # somewhere else must not make success unreachable.",
        "            terms = goal_settle_terms(self, self._GOAL_OBJECTS[g])",
        "            goal_settled = torch.where(on_goal, terms.settled & terms.gripper_open, goal_settled)",
        "            n_pad = self._MAX_GOAL_OBJECTS - terms.lin.shape[-1]",
        "            on_goal_col = on_goal.unsqueeze(1)",
        "            goal_obj_lin_vel = torch.where(",
        "                on_goal_col, torch.nn.functional.pad(terms.lin, (0, n_pad)), goal_obj_lin_vel",
        "            )",
        "            goal_obj_ang_vel = torch.where(",
        "                on_goal_col, torch.nn.functional.pad(terms.ang, (0, n_pad)), goal_obj_ang_vel",
        "            )",
        "            goal_finger_pos = torch.where(on_goal, terms.finger, goal_finger_pos)",
        "            goal_blocks_settled = torch.where(on_goal, terms.settled, goal_blocks_settled)",
        "            goal_gripper_open = torch.where(on_goal, terms.gripper_open, goal_gripper_open)",
        "        self._goal_conjuncts = goal_conjuncts",
        "        self._goal_success = goal_conjuncts.all(dim=-1) & goal_settled",
        "        self._goal_obj_lin_vel = goal_obj_lin_vel",
        "        self._goal_obj_ang_vel = goal_obj_ang_vel",
        "        self._goal_finger_pos = goal_finger_pos",
        "        self._goal_blocks_settled = goal_blocks_settled",
        "        self._goal_gripper_open = goal_gripper_open",
    ]

    return "\n".join(preamble) + "\n" + "\n".join(parts) + "\n", max_conjuncts, max_objects


def inject_goal_into_generated_env_py(
    env_source: str,
    *,
    goal_conjuncts_list: List[List[str]],
    goal_labels_list: List[List[str]],
    goal_objects_list: List[List[str]],
    goal_texts: List[str],
    goal_display_name: str,
) -> str:
    """Replace ``_compute_intermediate_values``, wire ``evaluate`` success, add per-conjunct stats.

    The stats are each env's own goal: ``goal_conj_<c>`` for its c-th conjunct
    (1.0 past the end of a goal shorter than the longest) and ``goal_success``,
    so their number does not grow with the number of goals compiled in. The
    generated ``goal_conjunct_labels(pddl_goal)`` names each ``goal_conj_<c>``.

    The settle gate's terms are stats too: ``goal_obj_<k>_lin_vel`` /
    ``goal_obj_<k>_ang_vel`` for the k-th block of the env's goal (its
    alphabetically sorted ``_GOAL_OBJECTS`` entry; 0.0 past the end of a shorter
    goal), ``goal_finger_pos``, ``goal_blocks_settled`` and ``goal_gripper_open``.
    """
    new_compute, max_goal_conjuncts, max_goal_objects = build_goal_compute_method(
        goal_conjuncts_list=goal_conjuncts_list,
        goal_labels_list=goal_labels_list,
        goal_objects_list=goal_objects_list,
        goal_texts=goal_texts,
        goal_display_name=goal_display_name,
    )

    compute_pat = re.compile(
        r"    def _compute_intermediate_values\(self\):.*?(?=\n    def _get_obs_robot)",
        re.DOTALL,
    )
    if not compute_pat.search(env_source):
        raise RuntimeError(
            "Could not find _compute_intermediate_values / _get_obs_robot in generated env; "
            "template layout may have changed."
        )
    env_source = compute_pat.sub(lambda _: new_compute.rstrip("\n") + "\n", env_source, count=1)

    old_success = (
        'result["success"] = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)'
    )
    new_success = 'result["success"] = self._goal_success'
    if old_success not in env_source:
        raise RuntimeError(
            "Could not find default success=zeros line in generated env.py; "
            "update inject_goal_into_generated_env_py for the current template."
        )
    env_source = env_source.replace(old_success, new_success, 1)

    if max_goal_conjuncts > 0:
        eval_insert_pat = re.compile(
            r'(        result\["success"\] = self\._goal_success\n)(        return result\n)',
        )
        if not eval_insert_pat.search(env_source):
            raise RuntimeError(
                "Could not find evaluate() block "
                '(result["success"] = self._goal_success followed by return result) '
                "to inject goal conjunction stats."
            )
        stats_lines = '        stats = result.setdefault("stats", {})\n'
        for c_idx in range(max_goal_conjuncts):
            stats_lines += f'        stats["goal_conj_{c_idx}"] = self._goal_conjuncts[:, {c_idx}].float()\n'
        stats_lines += '        stats["goal_success"] = self._goal_success.float()\n'
        for k_idx in range(max_goal_objects):
            stats_lines += f'        stats["goal_obj_{k_idx}_lin_vel"] = self._goal_obj_lin_vel[:, {k_idx}]\n'
            stats_lines += f'        stats["goal_obj_{k_idx}_ang_vel"] = self._goal_obj_ang_vel[:, {k_idx}]\n'
        stats_lines += '        stats["goal_finger_pos"] = self._goal_finger_pos\n'
        stats_lines += '        stats["goal_blocks_settled"] = self._goal_blocks_settled.float()\n'
        stats_lines += '        stats["goal_gripper_open"] = self._goal_gripper_open.float()\n'
        env_source = eval_insert_pat.sub(r"\1" + stats_lines + r"\2", env_source, count=1)
    return env_source


def inject_instance_layout_json_into_generated_env_py(
    env_source: str, *, block_names: List[str]
) -> str:
    """Replace the ``# --- autogen: instance layout json ---`` stub with merge / reset implementations."""
    sorted_names = sorted(block_names)
    frozenset_inner = ", ".join(repr(n) for n in sorted_names)
    replacement = (
        "    def _merge_instance_layout_json_into_cfg(self, cfg: BaseEnvCfg) -> None:\n"
        "        from blocksuite.utils import instance_layout_json\n"
        "        self._instance_json_blocks = instance_layout_json.merge_instance_json_into_env_cfg(\n"
        "            cfg,\n"
        f"            block_names=frozenset(({frozenset_inner})),\n"
        "        )\n"
        "\n"
        "    def _write_instance_layout_json_block_poses(self, env_ids: Sequence[int] | None) -> None:\n"
        "        from blocksuite.utils import instance_layout_json\n"
        "        layout = getattr(self, \"_instance_json_blocks\", None)\n"
        "        if layout is None or env_ids is None:\n"
        "            return\n"
        "        instance_layout_json.write_block_layout_to_sim(self, layout, env_ids)\n"
        "\n"
    )
    pat = re.compile(
        r"    # --- autogen: instance layout json ---\n"
        r"    def _merge_instance_layout_json_into_cfg\(self, cfg: BaseEnvCfg\) -> None:\n"
        r"        pass\n\n"
        r"    def _write_instance_layout_json_block_poses\(self, env_ids: Sequence\[int\] \| None\) -> None:\n"
        r"        pass\n\n"
        r"    # --- end autogen: instance layout json ---\n",
    )
    if not pat.search(env_source):
        raise RuntimeError(
            "Could not find instance layout autogen markers "
            "('# --- autogen: instance layout json ---' ... '# --- end autogen: instance layout json ---'); "
            "update template_env.py or inject_instance_layout_json_into_generated_env_py."
        )
    return pat.sub(replacement, env_source, count=1)


# ----------------------------- env package codegen -----------------------------

def _override_env_cfg_fields(
    env_cfg_code: str,
    template_cfg_path: str,
    *,
    scene_preset_id: Optional[str] = None,
    robot_id: Optional[str] = None,
) -> str:
    """Rewrite ``scene_preset_id`` / ``robot_id`` assignments in the config source.

    Each non-``None`` value replaces the template's assignment, preserving its
    indentation and any type annotation. The template must declare the field
    exactly once. Neither zero nor several is recoverable: with none the
    generated package keeps the template's scene, and with several only the
    first would be rewritten while the last one is what the class body actually
    evaluates to. Either way the package runs the wrong table baseline and
    invalidates every ``table_height_delta`` comparison downstream while every
    other check still passes, so refuse to emit it.
    """
    for field, value in (("scene_preset_id", scene_preset_id), ("robot_id", robot_id)):
        if value is None:
            continue

        def _replacement(match: re.Match, field: str = field, value: str = value) -> str:
            annotation = (match.group(2) or "").rstrip()
            return f'{match.group(1)}{field}{annotation} = "{value}"'

        pattern = re.compile(
            rf"^([ \t]*){field}([ \t]*:[^=\n]+)?[ \t]*=[^\n]*$", flags=re.M
        )
        n = len(pattern.findall(env_cfg_code))
        if n != 1:
            raise ValueError(
                f"could not override {field}={value!r}: expected exactly one "
                f"assignment to {field} in {template_cfg_path}, found {n}."
            )
        env_cfg_code = pattern.sub(_replacement, env_cfg_code, count=1)
    return env_cfg_code


def generate_environment_code(
    domain_path: str,
    task_name: str,
    block_names: List[str],
    goal_texts: List[str],
    goal_display_name: str,
    randomization_ranges: dict = {},
    table_root_z_offset_bake: Optional[float] = None,
    cube_scales: "List[float] | None" = None,
    color_jitter: float = 0.0,
    scene_preset_id: Optional[str] = None,
    robot_id: Optional[str] = None,
    max_non_sim_state_randomizations: int = 8,
    usd_library: Optional[str] = None,
) -> dict:
    """Render env.py / env_cfg.py source for ``task_name``.

    Returns a dict with ``env_class_name``, ``env_cfg_class_name``,
    ``env_code``, ``env_cfg_code``. ``goal_atoms_list[i]`` is the parsed
    predicate list for ``goal_texts[i]``; single-goal is just length-1.

    Block-spawn shape, color, and USD overrides are derived from the block name
    by :meth:`Block.from_names`: a name with a :data:`USD_OVERRIDES` entry or a
    glyph name (``block_a``, ``block_z_2``, ``block_7``, ...) spawns that USD
    asset, everything else is a cuboid colored from a color word in the
    name (e.g. ``block_green3`` -> green), or a rotating palette color when the
    name has none. ``color_jitter`` perturbs each cuboid color by ``+/-`` that
    amount so same-color blocks get slightly different shades (``0.0`` disables).

    ``scene_preset_id`` / ``robot_id`` override the corresponding fields of
    ``template_env_cfg.py`` in the generated ``env_cfg.py``; ``None`` keeps the
    template default. Datagen bakes the scene in (the template is copied
    verbatim otherwise), so this is how a task is generated for a scene other
    than the template's.

    ``usd_library`` selects the glyph asset library that glyph block names
    resolve against (``train_blocks_large`` by default, or ``train_blocks``); see
    :meth:`Block.from_names`. ``None`` uses the default library.
    """
    blocks = Block.from_names(
        block_names,
        color_jitter=color_jitter,
        usd_library=usd_library or _DEFAULT_GLYPH_LIBRARY,
    )
    block_specs = {block.name: block.spec for block in blocks}

    template_path = os.path.join(domain_path, "template", "template_env.py")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template environment file not found at {template_path}")
    template_cfg_path = os.path.join(domain_path, "template", "template_env_cfg.py")
    if not os.path.exists(template_cfg_path):
        raise FileNotFoundError(f"Template environment config file not found at {template_cfg_path}")
    with open(template_path, 'r') as f:
        template = f.read()


    # NOTE (stao): Because we cannot reconfigure in isaac lab / change non-sim state randomizations such as cube scales, at runtime, we hardcode a max number of options
    # and make sure during task gen and data gen, these MAX_NON_SIM_STATE_RANDOMIZATIONS are always used.
    # Also means number of parallel envs has to be a multiple of MAX_NON_SIM_STATE_RANDOMIZATIONS.

    # When the dataset records per-episode cube scales, every distinct value gets
    # its own spawn slot so eval can reproduce each episode's block size exactly
    # (the queue routes episodes to the env whose slot matches). Those slots have
    # to win: the slot <-> scale correspondence is what makes replay exact.
    # Otherwise the count is whatever stage 1 asked for (8 by default, the legacy
    # scheme). Either way it is baked into the generated env package, so changing
    # it only affects packages generated from here on; stages that load an
    # existing package read its value.
    from blocksuite.utils.episode_dataset import distinct_cube_scales

    cube_scale_slots: List[float] = distinct_cube_scales(cube_scales) if cube_scales else []
    derive_block_scale_from_size = bool(cube_scale_slots)
    MAX_NON_SIM_STATE_RANDOMIZATIONS = (
        len(cube_scale_slots) if cube_scale_slots else max_non_sim_state_randomizations
    )
    randomizations: dict = dict(
        block_sizes=dict(),
        table_heights=[],
        friction=dict(static_friction=[], dynamic_friction=[], restitution=[]),
        cube_scale_slots=cube_scale_slots,
    )
    if cube_scale_slots:
        # One isotropic slot per recorded distinct cube scale, in the shared
        # distinct_cube_scales() order so env slot i == queue slot i.
        shared_sizes = [[s, s, s] for s in cube_scale_slots]
        for name in block_names:
            randomizations["block_sizes"][name] = shared_sizes
    elif "block_size_range" in randomization_ranges:
        # code below is for the case where we want to randomize block sizes for each env individually
        # for name in block_names:
        #     randomizations["block_sizes"][name] = np.random.uniform(
        #         randomization_ranges["block_size_range"][0], 
        #         randomization_ranges["block_size_range"][1], 
        #         size=(MAX_NON_SIM_STATE_RANDOMIZATIONS)
        #     ).reshape(-1, 1).repeat(3, axis=1).tolist()

        # code below is for the case where we want to share the same block sizes for each instance of envs, which is desired.
        shared_sizes = np.random.uniform(
            randomization_ranges["block_size_range"][0],
            randomization_ranges["block_size_range"][1],
            size=(MAX_NON_SIM_STATE_RANDOMIZATIONS)
        ).reshape(-1, 1).repeat(3, axis=1).tolist()
        for name in block_names:
            randomizations["block_sizes"][name] = shared_sizes
    else:
        for name in block_names:
            randomizations["block_sizes"][name] = np.array([0.04, 0.04, 0.04]).reshape(1, -1).repeat(MAX_NON_SIM_STATE_RANDOMIZATIONS, axis=0).tolist()

    if "table_height_range" in randomization_ranges:
        randomizations["table_height"] = (
            np.random.uniform(
                randomization_ranges["table_height_range"][0],
                randomization_ranges["table_height_range"][1],
                size=(MAX_NON_SIM_STATE_RANDOMIZATIONS),
            )
            + float(table_root_z_offset_bake or 0.0)
        ).tolist()
    else:
        bake = float(table_root_z_offset_bake or 0.0)
        randomizations["table_height"] = [bake] * MAX_NON_SIM_STATE_RANDOMIZATIONS

    if "friction_range" in randomization_ranges:
        lo, hi = randomization_ranges["friction_range"]
        static_samples = np.random.uniform(lo, hi, size=(MAX_NON_SIM_STATE_RANDOMIZATIONS)).tolist()
        dynamic_samples = np.random.uniform(lo, hi, size=(MAX_NON_SIM_STATE_RANDOMIZATIONS)).tolist()
        dynamic_samples = [min(s, d) for s, d in zip(static_samples, dynamic_samples)]
        randomizations["friction"] = dict(
            static_friction=static_samples,
            dynamic_friction=dynamic_samples,
            restitution=[0.0] * MAX_NON_SIM_STATE_RANDOMIZATIONS,
        )
    else: 
        randomizations["friction"] = dict(
            static_friction=[0.3] * MAX_NON_SIM_STATE_RANDOMIZATIONS,
            dynamic_friction=[0.3] * MAX_NON_SIM_STATE_RANDOMIZATIONS,
            restitution=[0.0] * MAX_NON_SIM_STATE_RANDOMIZATIONS,
        )

    def _setup_scene_objects(self):
        from blocksuite.utils import rigid_objects
        from blocksuite import OBJECT_ASSET_DIR
        #$block_names
        #$block_specs
        #$randomizations
        #$MAX_NON_SIM_STATE_RANDOMIZATIONS
        #$derive_block_scale_from_size
        self.randomizations = randomizations
        self.MAX_NON_SIM_STATE_RANDOMIZATIONS = MAX_NON_SIM_STATE_RANDOMIZATIONS
        # Spawn slot of every env: the cfg's per-env map when given (eval sizes each
        # cube-scale slot's env count to its queue workload), else round-robin.
        env_slots = [int(s) for s in getattr(self.cfg, "env_slots", ()) or ()]
        if env_slots:
            assert len(env_slots) == self.num_envs, f"cfg.env_slots has {len(env_slots)} entries for {self.num_envs} envs"
            assert all(0 <= s < MAX_NON_SIM_STATE_RANDOMIZATIONS for s in env_slots), f"cfg.env_slots {sorted(set(env_slots))} out of range for {MAX_NON_SIM_STATE_RANDOMIZATIONS} slots"
        else:
            env_slots = [i % MAX_NON_SIM_STATE_RANDOMIZATIONS for i in range(self.num_envs)]
        self.env_slots = env_slots
        self.env_slots_t = torch.tensor(env_slots, device=self.device, dtype=torch.long)
        # NOTE (stao): I think we might want to just get rid most of the codegen code here and stick most of the code into the template_env.py file
        # Round-robin slots need num_envs to cover them evenly; an explicit map
        # spawns exactly the slots the queue needs.
        if getattr(self.cfg, "env_slots", ()):
            pass
        elif derive_block_scale_from_size:
            # Exact per-episode cube-scale matching: every distinct recorded
            # scale is a spawn slot, so num_envs must cover all of them (a
            # positive multiple of MAX) or some episodes have no size-matched env.
            assert self.num_envs % MAX_NON_SIM_STATE_RANDOMIZATIONS == 0, f"num_envs ({self.num_envs}) must be a positive multiple of MAX_NON_SIM_STATE_RANDOMIZATIONS ({MAX_NON_SIM_STATE_RANDOMIZATIONS} = number of distinct recorded cube scales) so every cube-scale slot is spawned for exact matching."
        else:
            assert self.num_envs % MAX_NON_SIM_STATE_RANDOMIZATIONS == 0 or self.num_envs < MAX_NON_SIM_STATE_RANDOMIZATIONS, f"Number of parallel envs has to be a multiple of {MAX_NON_SIM_STATE_RANDOMIZATIONS} due to isaac asset randomization limitations"
        block_sizes_list = dict()

        replay_side = getattr(self.cfg, "replay_block_side_length", None)
        if replay_side is not None and self.num_envs == 1:
            side_vec = [float(replay_side), float(replay_side), float(replay_side)]
            for name in block_names:
                if len(randomizations["block_sizes"][name]) > 0:
                    block_sizes_list[name] = [side_vec]
        else:
            # randomize block sizes if needed and format the list
            for name in block_names:
                if len(randomizations["block_sizes"][name]) > 0:
                    assert isinstance(randomizations["block_sizes"][name], list), "block_size randomization should be given as a list"
                    block_sizes_list[name] = [randomizations["block_sizes"][name][s] for s in env_slots]
        # randomize block frictions
        fric = randomizations["friction"]
        sf_list = [fric["static_friction"][s] for s in env_slots]
        df_list = [fric["dynamic_friction"][s] for s in env_slots]
        rest_list = [fric["restitution"][s] for s in env_slots]

        for i, name in enumerate(block_names):
            spec = block_specs[name]
            if spec["type"] == "cuboid":
                block = rigid_objects.create_cuboid(
                    prim_path=f"/World/envs/env_.*/{name}",
                    size=block_sizes_list[name],
                    init_pos=(i + 1, 0.0, 0.05 / 2),
                    init_quat=(1, 0, 0, 0),
                    diffuse_color=spec["color"],
                    metallic=0.2,
                    static_friction=sf_list,
                    dynamic_friction=df_list,
                    restitution=rest_list,
                )
            else:
                usd_kwargs: dict = {
                    "usd_path": str(OBJECT_ASSET_DIR / spec["usd_path"]),
                }
                if derive_block_scale_from_size:
                    # When matching recorded per-episode sizes, let create_toy_block
                    # derive each env's scale from its size, and disable size
                    # bucketing so every distinct recorded scale is its own USD
                    # prototype (exact match; more prototypes to load).
                    usd_kwargs["max_unique_assets"] = None
                else:
                    # Legacy default: fixed scale from the USD override registry.
                    usd_kwargs["scale"] = spec["usd_scale"]
                block = rigid_objects.create_toy_block(
                    prim_path=f"/World/envs/env_.*/{name}",
                    size=block_sizes_list[name],
                    init_pos=(i + 1, 0.0, 0.05 / 2),
                    init_quat=(1, 0, 0, 0),
                    static_friction=sf_list,
                    dynamic_friction=df_list,
                    restitution=rest_list,
                    **usd_kwargs,
                )
            self.scene.rigid_objects[name] = block
        for k in self.randomizations["friction"].keys():
            self.randomizations["friction"][k] = torch.tensor(self.randomizations["friction"][k], device=self.device)
        self.randomizations["table_height"] = torch.tensor(self.randomizations["table_height"], device=self.device)
        return

    _setup_scene_objects_code = inspect.getsource(_setup_scene_objects)
    _setup_scene_objects_code = _setup_scene_objects_code.replace(
        "#$block_names", f"block_names = {block_names}"
    )
    _setup_scene_objects_code = _setup_scene_objects_code.replace(
        "#$block_specs", f"block_specs = {block_specs}"
    )
    _setup_scene_objects_code = _setup_scene_objects_code.replace(
        "#$MAX_NON_SIM_STATE_RANDOMIZATIONS", f"MAX_NON_SIM_STATE_RANDOMIZATIONS = {MAX_NON_SIM_STATE_RANDOMIZATIONS}"
    )
    _setup_scene_objects_code = _setup_scene_objects_code.replace(
        "#$derive_block_scale_from_size", f"derive_block_scale_from_size = {derive_block_scale_from_size}"
    )
    _setup_scene_objects_code = _setup_scene_objects_code.replace(
        "#$randomizations", f"randomizations = {randomizations}"
    )
    
    environment_code = template.replace(
        "    def _setup_scene_objects(self):\n        pass", _setup_scene_objects_code
    )
    environment_code = environment_code.replace(
        "from .template_env_cfg import TemplateEnvCfg",
        "from .env_cfg import TemplateEnvCfg",
    )
    env_class_name = task_name.replace("-", "_") + "_Env"
    environment_code = environment_code.replace("TemplateEnv", env_class_name)
    goal_conjuncts_list = [compile_goal_text(gt, block_names) for gt in goal_texts]
    goal_labels_list = [goal_conjunct_labels(gt, block_names) for gt in goal_texts]
    goal_objects_list = [
        goal_objects_from_conjuncts(c, block_names) for c in goal_conjuncts_list
    ]
    environment_code = inject_goal_into_generated_env_py(
        environment_code,
        goal_conjuncts_list=goal_conjuncts_list,
        goal_labels_list=goal_labels_list,
        goal_objects_list=goal_objects_list,
        goal_texts=goal_texts,
        goal_display_name=goal_display_name,
    )
    environment_code = inject_instance_layout_json_into_generated_env_py(
        environment_code, block_names=block_names
    )
    environment_code = f"# GENERATED CODE\n{environment_code}\n"

    with open(template_cfg_path, 'r') as f:
        environment_cfg_code = f.read()
    env_cfg_class_name = task_name.replace("-", "_") + "_EnvCfg"
    environment_cfg_code = environment_cfg_code.replace("TemplateEnvCfg", env_cfg_class_name)
    environment_cfg_code = _override_env_cfg_fields(
        environment_cfg_code,
        template_cfg_path,
        scene_preset_id=scene_preset_id,
        robot_id=robot_id,
    )
    environment_cfg_code = f"# GENERATED CODE\n{environment_cfg_code}\n"

    profiles = []
    for i in range(MAX_NON_SIM_STATE_RANDOMIZATIONS):
        profile = {
            "profile_id": i,
            "block_sizes": {
                name: randomizations["block_sizes"][name][i]
                for name in block_names
            },
            "table_height": randomizations["table_height"][i],
            "static_friction": randomizations["friction"]["static_friction"][i],
            "dynamic_friction": randomizations["friction"]["dynamic_friction"][i],
            "restitution": randomizations["friction"]["restitution"][i],
        }
        profiles.append(profile)

    return dict(
        env_class_name=env_class_name,
        env_cfg_class_name=env_cfg_class_name,
        env_code=environment_code,
        env_cfg_code=environment_cfg_code,
        non_sim_state_randomizations=randomizations,
        profiles={
            # Read back by eval_helpers._load_task_profiles, which refuses to run a
            # package whose stamp does not match this checkout's codegen.
            "codegen_version": AUTOGEN_PACKAGE_VERSION,
            "num_profiles": MAX_NON_SIM_STATE_RANDOMIZATIONS,
            # True when every distinct recorded cube_scale is its own spawn slot,
            # so eval must spawn an env per slot the queue needs (an explicit
            # cfg.env_slots map, else a positive multiple of num_profiles).
            # Consumed at eval time by eval_helpers.reconcile_num_envs_with_task.
            "derive_block_scale_from_size": derive_block_scale_from_size,
            "block_names": block_names,
            "profiles": profiles,
        },
    )


def write_env_package(env_folder: str, *, env_info: dict, env_id: str) -> None:
    """Write env.py / env_cfg.py / __init__.py into ``env_folder`` (created if missing)."""
    os.makedirs(env_folder, exist_ok=True)
    with open(os.path.join(env_folder, 'env.py'), 'w') as f:
        f.write(env_info['env_code'])
    with open(os.path.join(env_folder, 'env_cfg.py'), 'w') as f:
        f.write(env_info['env_cfg_code'])
    init_code = (
        "\n# GENERATED CODE\n"
        "import gymnasium as gym\n\n"
        "gym.register(\n"
        f"    id=\"{env_id}\",\n"
        f"    entry_point=f\"{{__name__}}.env:{env_info['env_class_name']}\",\n"
        "    disable_env_checker=True,\n"
        "    kwargs={\n"
        f"        \"env_cfg_entry_point\": f\"{{__name__}}.env_cfg:{env_info['env_cfg_class_name']}\",\n"
        "    },\n"
        ")\n"
    )
    with open(os.path.join(env_folder, '__init__.py'), 'w') as f:
        f.write(init_code)
    with open(os.path.join(env_folder, 'non_sim_state_randomizations.json'), 'w') as f:
        json.dump(env_info['non_sim_state_randomizations'], f, indent=2)
    with open(os.path.join(env_folder, 'profiles.json'), 'w') as f:
        json.dump(env_info['profiles'], f, indent=2)
