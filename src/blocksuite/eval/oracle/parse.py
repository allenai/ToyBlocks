"""Parse oracle high-level action strings into (language, target pixel).

The oracle emits **bare** ``u,v`` pixel coordinates (no angle brackets), e.g.::

    "pick 320,180"          -> ("pick", [320, 180])
    "place 400,120"         -> ("place", [400, 120])
    "rotate 310,195 cw"     -> ("rotate cw", [310, 195])
    "flip 310,195 left"     -> ("flip left", [310, 195])
    "done"                  -> ("done", None)

This is intentionally a separate parser from
``maniflow_serving_utils._parse_target_xy`` / ``_strip_coords``, which only
match the ``<u,v>`` angle-bracket format used by the Molmo HL model.
"""

from __future__ import annotations

import re

import numpy as np

_ORACLE_UV = re.compile(r"(-?\d+)\s*,\s*(-?\d+)")


def parse_oracle_action(action_str: str) -> tuple[str, np.ndarray | None]:
    """Split an oracle action string into a language phrase and a pixel target.

    Returns ``(lang, xy)`` where ``xy`` is a ``float32`` array ``[u, v]`` in the
    front-camera pixel frame, or ``None`` when the string carries no coords
    (e.g. ``"done"``).
    """
    s = (action_str or "").strip()
    m = _ORACLE_UV.search(s)
    xy: np.ndarray | None = None
    if m:
        xy = np.array([float(m.group(1)), float(m.group(2))], dtype=np.float32)
    # Drop the coords and collapse the resulting double space: "rotate  cw" -> "rotate cw".
    lang = _ORACLE_UV.sub("", s)
    lang = re.sub(r"\s+", " ", lang).strip()
    return lang, xy
