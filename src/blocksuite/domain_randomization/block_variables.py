"""One map from a PDDL variable in the rendered goal back to the object it names.

``pddl_goal`` and the LeRobot ``task`` string are rewritten so they describe what the camera sees:
glyph blocks are renamed to the letter they now carry, colour cuboids to the colour they now are.
The POSE columns cannot follow, because a parquet schema is per FILE and the mapping is per
EPISODE -- one file holds eight episodes in which ``block_red1`` became five different colours, so
no single column name is right for all of them.

That leaves a consumer holding ``block_red1`` from the goal with no way to know which
``observation.state.object_pose.*`` it belongs to. This module answers exactly that, for BOTH
block families in one place, keyed by the name the goal uses:

    {"block_8":    {"from": "block_a",      "kind": "glyph",  "glyph": "8", ...},
     "block_red1": {"from": "block_orange", "kind": "cuboid", "color": "red", ...}}

Why keyed by the rewritten name rather than the original: that is the direction every consumer
reads in, and it is the only direction that is unambiguous. A rewritten token can COLLIDE with
some other block's authored name -- observed in a real render, ``block_orange`` recoloured to red
takes the token ``block_red1`` while the block authored ``block_red1`` became blue, so the goal's
``block_red1`` and ``object_pose.block_red1`` are different objects. Keying on the goal token
removes the ambiguity: the key is what the caller is holding.

``glyph_map`` keeps its own forward ``{original: glyph}`` shape; shipped datasets carry it and
nothing is gained by breaking them. This column supersedes it for lookups, it does not replace it.
"""
from __future__ import annotations

import dataclasses
import json
import re
from typing import Any, Dict, Optional, Tuple

POSE_PREFIX = "observation.state.object_pose."

#: Same token shape as glyphs._TOKEN_RE / block_colors._TOKEN_RE, so the inverse rewrite matches
#: exactly the tokens the forward rewrites produced.
_TOKEN_RE = re.compile(r"\bblock_([A-Za-z0-9]+(?:_\d+)?)\b")


def block_variable_map(glyph_mapping: Optional[Dict[str, str]] = None,
                       color_mapping: Optional[Dict[str, str]] = None,
                       palette=None) -> Dict[str, dict]:
    """{rewritten PDDL name: provenance}, over glyph blocks and colour cuboids together.

    ``glyph_mapping`` is {block: GLYPH} and ``color_mapping`` is {block: colour name} -- the two
    forward draws. Both are optional: either randomizer may be off.

    The two families are disjoint by construction (glyphs own ``type=="usd"`` blocks with a
    recoverable authored glyph, colours own ``type=="cuboid"``), so their rewritten names cannot
    collide with each other; this asserts it rather than trusting it, because the whole value of
    this map is that one lookup gives one object.
    """
    from . import glyphs as _g
    from . import block_colors as _bc

    out: Dict[str, dict] = {}
    for name, token in (_g.output_tokens(glyph_mapping or {}) or {}).items():
        out[token] = {"from": name, "pose_column": f"{POSE_PREFIX}{name}",
                      "kind": "glyph", "glyph": (glyph_mapping or {})[name]}

    if color_mapping:
        hsv_by_name = {n: h for n, h in (tuple(palette or ()) or _bc.DATASET_COLORS)}
        for name, token in _bc.output_tokens(color_mapping).items():
            if token in out:
                raise AssertionError(
                    f"rewritten name {token!r} claimed by both a glyph block ({out[token]['from']}) "
                    f"and a cuboid ({name}); the spec-type partition has drifted")
            color = color_mapping[name]
            hsv = hsv_by_name.get(color)
            out[token] = {"from": name, "pose_column": f"{POSE_PREFIX}{name}",
                          "kind": "cuboid", "color": color,
                          "hsv": list(hsv) if hsv is not None else None,
                          "rgb": list(_bc.hsv_to_rgb(hsv)) if hsv is not None else None}
    return out


def block_variable_map_json(glyph_mapping=None, color_mapping=None, palette=None) -> str:
    """Compact, stable serialisation for the per-episode ``block_variable_map`` column."""
    return json.dumps(block_variable_map(glyph_mapping, color_mapping, palette),
                      separators=(",", ":"), sort_keys=True)


def _parse_json_map(raw: Any, column: str) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, dict):
        return dict(raw)
    if not isinstance(raw, str):
        raise ValueError(f"{column} must be a JSON object string, got {type(raw).__name__}")
    if not raw.strip():
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"{column} must be a JSON object, got {parsed!r}")
    return parsed


def parse_block_variable_map(raw: Any) -> Dict[str, dict]:
    """The ``block_variable_map`` column value -> ``{rendered token: entry}``; ``{}`` when empty."""
    parsed = _parse_json_map(raw, "block_variable_map")
    for token, entry in parsed.items():
        if not isinstance(entry, dict) or not entry.get("from"):
            raise ValueError(f"block_variable_map[{token!r}] has no 'from' block: {entry!r}")
    return parsed


def parse_glyph_map(raw: Any) -> Dict[str, str]:
    """The ``glyph_map`` column value -> forward ``{block: glyph}``; ``{}`` when empty."""
    return {str(k): str(v) for k, v in _parse_json_map(raw, "glyph_map").items()}


@dataclasses.dataclass(frozen=True)
class BlockIdentity:
    """One episode's recorded block identities, in both naming directions.

    ``to_source`` maps each rendered goal token to the block (prim) it names. ``glyphs`` and
    ``colors`` are what each block displayed, keyed by block, and ``palette`` holds the
    ``(colour name, hsv)`` pairs the colours came from.
    """

    to_source: Dict[str, str]
    glyphs: Dict[str, str]
    colors: Dict[str, str]
    palette: Tuple[Tuple[str, Tuple[float, float, float]], ...]

    @property
    def to_rendered(self) -> Dict[str, str]:
        return {src: tok for tok, src in self.to_source.items()}

    def __bool__(self) -> bool:
        return bool(self.to_source or self.glyphs or self.colors)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe form, for queue setups that cross the episode-queue wire."""
        return {"to_source": dict(self.to_source), "glyphs": dict(self.glyphs),
                "colors": dict(self.colors),
                "palette": [[name, list(hsv)] for name, hsv in self.palette]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> BlockIdentity:
        return cls(to_source=dict(d.get("to_source") or {}), glyphs=dict(d.get("glyphs") or {}),
                   colors=dict(d.get("colors") or {}),
                   palette=tuple((str(n), (float(h[0]), float(h[1]), float(h[2])))
                                 for n, h in (d.get("palette") or ())))


def identity_assignment(block_variable_map: Optional[Dict[str, dict]],
                        glyph_map: Optional[Dict[str, str]] = None) -> BlockIdentity:
    """Both recorded maps -> a :class:`BlockIdentity`.

    Glyphs come from ``glyph_map`` (it covers every glyph block); without it, from the map's
    glyph entries. Colours and the token map come from ``block_variable_map`` only, so a
    ``glyph_map``-only row has an empty token map and its goal is left as recorded.
    """
    bvm = block_variable_map or {}
    to_source: Dict[str, str] = {}
    glyphs: Dict[str, str] = {}
    colors: Dict[str, str] = {}
    palette: Dict[str, Tuple[float, float, float]] = {}
    for token, entry in bvm.items():
        src = str(entry["from"])
        if src in to_source.values():
            raise ValueError(f"block_variable_map names block {src!r} under two tokens")
        to_source[str(token)] = src
        kind = entry.get("kind")
        if kind == "glyph" and entry.get("glyph"):
            glyphs[src] = str(entry["glyph"])
        elif kind == "cuboid" and entry.get("color"):
            color = str(entry["color"])
            colors[src] = color
            hsv = entry.get("hsv")
            if hsv is None:
                raise ValueError(f"block_variable_map[{token!r}] cuboid entry has no hsv")
            palette.setdefault(color, (float(hsv[0]), float(hsv[1]), float(hsv[2])))
    if glyph_map:
        glyphs = {str(k): str(v) for k, v in glyph_map.items()}
    return BlockIdentity(to_source=to_source, glyphs=glyphs, colors=colors,
                         palette=tuple(palette.items()))


def _rewrite(text: str, mapping: Dict[str, str]) -> str:
    if not text or not mapping:
        return text
    lower = {k.lower(): v for k, v in mapping.items()}
    return _TOKEN_RE.sub(lambda m: lower.get(m.group(0).lower(), m.group(0)), text)


def source_goal(text: str, identity: BlockIdentity) -> str:
    """Rendered goal/task text -> the same text in source block names, in one simultaneous pass.

    Simultaneous because rendered tokens collide with other blocks' source names (``block_1``
    can come from ``block_d`` while block ``block_1`` itself was renamed); unmapped tokens such
    as ``workspace_table`` pass through.
    """
    return _rewrite(text, identity.to_source)


def rendered_text(text: str, identity: BlockIdentity) -> str:
    """Inverse of :func:`source_goal`: source block names -> the rendered tokens."""
    return _rewrite(text, identity.to_rendered)
