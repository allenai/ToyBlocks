"""Letter and digit ("glyph") block registry for the blocks domain.

The ``train_blocks`` asset library ships one USD per letter ``A``-``Z`` and per
digit ``1``-``9`` (there is no ``0.usd``); ``train_blocks_large`` additionally
ships a ``10`` block (:data:`MULTI_DIGIT_GLYPHS`). Glyph blocks are named
``block_<glyph>``, with a ``_<copy>`` suffix when the same glyph appears more
than once in a scene (``block_z`` / ``block_z_2`` for the two Zs in "JAZZ").

All glyph assets in the current ``train_blocks`` batch share ``block_r``'s
orientation convention, so a single :data:`READABLE_ORI` describes "this glyph
is upright and readable" for the whole set.

Pure-Python and Isaac-free: imported both by the PDDL task generators in
``scripts/task_gen`` and by ``template/codegen.py`` for USD spawn resolution.
"""

from __future__ import annotations

import re

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "123456789"
#: Number glyphs longer than one character. ``10`` exists only in
#: ``train_blocks_large`` (``block_10`` -> ``train_blocks_large/usd/10.usda``);
#: the text helpers below work per character and do not produce it.
MULTI_DIGIT_GLYPHS: tuple[str, ...] = ("10",)

#: Known glyph asset libraries. Each ships one USD per glyph under
#: ``<library>/usd/<GLYPH>.usd`` (relative to ``toyblocks.OBJECT_ASSET_DIR``).
#: ``train_blocks`` is the original set; ``train_blocks_large`` is the larger
#: AlphabetBlockLarge set. Kept in sync with the ``objects`` sources listed in
#: ``asset_manifest.json``.
GLYPH_LIBRARIES: tuple[str, ...] = ("train_blocks", "train_blocks_large")

#: Library used when a caller does not select one. All datagen / eval scenes use the
#: AlphabetBlockLarge set; pass ``train_blocks`` explicitly for the original assets.
DEFAULT_GLYPH_LIBRARY = "train_blocks_large"

#: Default glyph library for dataset-driven eval codegen (``run_eval_pipeline.sh``,
#: ``launch_eval.py``). Current datagen uses AlphabetBlockLarge; pass
#: ``--usd-library train_blocks`` for legacy small-block datasets.
EVAL_DEFAULT_GLYPH_LIBRARY = "train_blocks_large"

#: Glyph USD file extension per library. ``train_blocks`` ships binary ``.usd``;
#: ``train_blocks_large`` (AlphabetBlockLarge) ships ASCII ``.usda``. Libraries
#: not listed default to ``usd``.
_GLYPH_LIBRARY_EXT: dict[str, str] = {
    "train_blocks": "usd",
    "train_blocks_large": "usda",
}

#: Orientation in which a glyph reads upright from the front camera. ``block_r``
#: is the reference the asset batch was authored against.
READABLE_ORI = "o2"

LETTER_TYPE = "block_letter"
DIGIT_TYPE = "block_digit"

#: Maximum copies of one glyph in a single scene. Keep in sync with
#: ``tmp3.data_gen.task_gen.blocks.MAX_GLYPH_COPIES``.
MAX_GLYPH_COPIES = 4

#: ``10`` is listed FIRST so it wins over ``1`` followed by a stray ``0``. The atlas ships a 10
#: tile (and a 10 USD), so a block may legitimately be named ``block_10`` -- notably after a
#: domain-randomization glyph swap, which may now assign it. It is the one glyph whose token is
#: two characters; ``block_10_2`` is still "glyph 10, copy 2".
#: NOTE: :data:`DIGITS` deliberately stays ``1``-``9`` -- the task GENERATORS do not author
#: 10-blocks; this only makes the name parseable when something else produces one.
_GLYPH_BLOCK_PATTERN = re.compile(r"^block_(10|[a-z1-9])(?:_(\d+))?$")


def glyph_usd_subpath(name: str, library: str = DEFAULT_GLYPH_LIBRARY) -> str | None:
    """Return the ``<library>/usd/<GLYPH>.usd`` sub-path for a glyph block name.

    ``library`` selects which glyph asset library to resolve against (one of
    :data:`GLYPH_LIBRARIES`); it defaults to :data:`DEFAULT_GLYPH_LIBRARY`
    (``train_blocks_large``).

    Returns ``None`` for non-glyph names (color cuboids such as ``block_red1``,
    and the legacy ``block_o1`` / ``block_o2`` aliases that carry explicit
    overrides).
    """
    if library not in GLYPH_LIBRARIES:
        raise ValueError(
            f"Unknown glyph library {library!r}; known libraries: {GLYPH_LIBRARIES}"
        )
    match = _GLYPH_BLOCK_PATTERN.match(name)
    if match is None:
        return None
    ext = _GLYPH_LIBRARY_EXT.get(library, "usd")
    return f"{library}/usd/{match.group(1).upper()}.{ext}"


def block_name(glyph: str, copy_idx: int = 1) -> str:
    """PDDL object name for the ``copy_idx``-th occurrence of ``glyph`` (1-based)."""
    base = f"block_{glyph.lower()}"
    return base if copy_idx == 1 else f"{base}_{copy_idx}"


def block_type(glyph: str) -> str:
    """PDDL type for a glyph: :data:`LETTER_TYPE` or :data:`DIGIT_TYPE`."""
    upper = glyph.upper()
    if upper in LETTERS:
        return LETTER_TYPE
    if upper in DIGITS or upper in MULTI_DIGIT_GLYPHS:
        return DIGIT_TYPE
    raise ValueError(f"No block asset for glyph {glyph!r}")


def validate_text(text: str) -> str:
    """Uppercase ``text`` and reject characters without a block asset."""
    upper = text.upper()
    if not upper:
        raise ValueError("text must be non-empty")
    unsupported = sorted({c for c in upper if c not in LETTERS and c not in DIGITS})
    if unsupported:
        hint = " (there is no 0.usd asset)" if "0" in unsupported else ""
        raise ValueError(
            f"No block assets for {unsupported} in {text!r}{hint}. "
            f"Supported: A-Z and 1-9."
        )
    counts = {c: upper.count(c) for c in set(upper)}
    too_many = sorted(c for c, n in counts.items() if n > MAX_GLYPH_COPIES)
    if too_many:
        raise ValueError(
            f"{text!r} repeats {too_many} more than MAX_GLYPH_COPIES={MAX_GLYPH_COPIES} times"
        )
    return upper


def blocks_for_text(text: str) -> list[tuple[str, str, str]]:
    """Return ``(block_name, block_type, glyph)`` per character, in text order.

    Repeated glyphs get distinct object names via the ``_<copy>`` suffix, so
    ``"JAZZ"`` yields ``block_j, block_a, block_z, block_z_2``.
    """
    upper = validate_text(text)
    seen: dict[str, int] = {}
    out: list[tuple[str, str, str]] = []
    for glyph in upper:
        seen[glyph] = seen.get(glyph, 0) + 1
        out.append((block_name(glyph, seen[glyph]), block_type(glyph), glyph))
    return out


def all_glyph_blocks(alphabet: str) -> list[tuple[str, str, str]]:
    """Return ``(block_name, block_type, glyph)`` for one copy of every glyph in ``alphabet``."""
    return [(block_name(g), block_type(g), g) for g in alphabet]
