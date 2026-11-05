"""Shared distractor-block pool for the blocks task generators.

Distractors are blocks present in the scene but absent from the goal. They add
clutter (more collision avoidance, more grounding pressure) without lengthening
the symbolic plan, so they are the cheapest way to make a working task family
harder.

Two kinds are available, and both are goal-irrelevant:

``color``
    The cuboid registry from ``template/codegen.py`` minus whatever the task
    itself uses. Visually similar to the goal cubes, so they stress grounding.
``letter``
    ``block_a`` .. ``block_z``. These never need reorienting when used purely as
    distractors, so they add clutter without pulling in the ``reorient``
    executor, which is the dominant failure mode for glyph tasks.

Pure-Python: returns ``(name, pddl_type)`` pairs and leaves ``Block`` /
``DistractorPool`` construction to the caller, matching :mod:`glyph_blocks` and
keeping this module free of a ``tmp3`` import.
"""

from blocksuite.tamp.blocks.glyph_blocks import LETTERS, all_glyph_blocks

#: Cuboids per color in the codegen registry. Keep in sync with
#: ``template/codegen.py``; ``color_patterns.py`` carries the same table.
COLOR_POOL: dict[str, int] = {
    "red": 5,
    "green": 4,
    "blue": 4,
    "yellow": 3,
    "orange": 2,
    "purple": 2,
    "cyan": 2,
    "gray": 2,
    "pink": 1,
    "brown": 1,
}

DISTRACTOR_KINDS = ("color", "letter", "both")


def cube_name(color: str, index: int) -> str:
    """Nth cube of a color, matching the names registered in ``template/codegen.py``."""
    if color == "red":
        return f"block_red{index}"
    return f"block_{color}" if index == 1 else f"block_{color}{index}"


def color_blocks() -> list[tuple[str, str]]:
    """Every registered color cuboid as ``(name, pddl_type)``."""
    return [
        (cube_name(color, i + 1), f"block_{color}")
        for color, count in COLOR_POOL.items()
        for i in range(count)
    ]


def letter_blocks() -> list[tuple[str, str]]:
    """One block per letter A-Z as ``(name, pddl_type)``."""
    return [(name, btype) for name, btype, _ in all_glyph_blocks(LETTERS)]


def distractor_blocks(
    kind: str = "both",
    exclude: set[str] | None = None,
    exclude_colors: set[str] | None = None,
) -> list[tuple[str, str]]:
    """Return ``(name, pddl_type)`` distractor candidates of ``kind``.

    ``exclude`` should hold every block the task itself declares, so a cube can
    never be both a goal object and a distractor.

    ``exclude_colors`` drops whole color families. Pass the colors the goal
    talks about: leaving a spare cube of a goal color in the scene makes an
    instruction like "stack the red blocks" genuinely ambiguous, even though the
    PDDL goal names concrete cubes and would still check out.
    """
    if kind not in DISTRACTOR_KINDS:
        raise ValueError(f"kind must be one of {DISTRACTOR_KINDS}, got {kind!r}")
    unknown = sorted((exclude_colors or set()) - set(COLOR_POOL))
    if unknown:
        raise ValueError(f"unknown color(s) {unknown}; available: {sorted(COLOR_POOL)}")
    blocks: list[tuple[str, str]] = []
    if kind in ("color", "both"):
        dropped = {f"block_{c}" for c in (exclude_colors or set())}
        blocks += [(n, t) for n, t in color_blocks() if t not in dropped]
    if kind in ("letter", "both"):
        blocks += letter_blocks()
    skip = exclude or set()
    return [(name, btype) for name, btype in blocks if name not in skip]
