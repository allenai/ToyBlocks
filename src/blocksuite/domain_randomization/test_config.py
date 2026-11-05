"""dr_schema 3 flags must reach the interpreter under the internal names it reads."""
import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from blocksuite.domain_randomization.config import canonicalize_flags, load_flags

PRESETS: list[Path] = sorted((Path(__file__).parent / "flags").glob("*.yaml"))


@pytest.mark.parametrize("path", PRESETS, ids=lambda p: p.name)
def test_preset_scenario_light_gate_reaches_interpreter(path: Path) -> None:
    """apply_wave gates per-env lights on mix[].sun (default True), so an unmapped v3
    `local_lights: false` rendered every open_hybrid wave with its local light on."""
    raw: dict[str, Any] = yaml.safe_load(path.read_text())
    flags: dict[str, Any] = load_flags(str(path), verbose=False)
    written: list[dict[str, Any]] = (raw.get("scenarios") or {}).get("mix") or []
    loaded: list[dict[str, Any]] = (flags.get("scenarios") or {}).get("mix") or []
    assert len(loaded) == len(written)
    for w, got in zip(written, loaded):
        assert "local_lights" not in got
        assert got.get("sun", True) is w.get("local_lights", w.get("sun", True))


def test_canonicalize_maps_every_v3_rename() -> None:
    v3: dict[str, Any] = {
        "dr_schema": 3,
        "scene_lights": {
            "global_distant": {"intensity": 500},
            "local_lights": {"random_subset": True,
                             "lights": [{"type": "sphere_light"}, {"type": "distant"}]},
        },
        "scenarios": {"mix": [{"name": "open", "local_lights": False},
                              {"name": "shadow", "local_lights": True}]},
    }
    out: dict[str, Any] = canonicalize_flags(copy.deepcopy(v3))
    sl = out["scene_lights"]
    assert sl["sun"] == {"intensity": 500} and "global_distant" not in sl
    assert "random_subset" not in sl["local_lights"]
    assert sl["local_lights"]["lights"] == [{"type": "distant", "distance": [40, 70]},
                                            {"type": "distant"}]
    assert [m["sun"] for m in out["scenarios"]["mix"]] == [False, True]
    assert all("local_lights" not in m for m in out["scenarios"]["mix"])
    assert canonicalize_flags(copy.deepcopy(out)) == out


def test_canonicalize_leaves_v2_untouched() -> None:
    v2: dict[str, Any] = {
        "scene_lights": {"sun": {"intensity": 1}, "local_lights": {"lights": [{"type": "distant"}]}},
        "scenarios": {"mix": [{"name": "open", "sun": False}]},
    }
    assert canonicalize_flags(copy.deepcopy(v2)) == v2
