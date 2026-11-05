"""episode_queue._apply_block_identity paints recorded glyphs/colours and restores the rest."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from toyblocks.domain_randomization import block_colors, glyphs
from toyblocks.domain_randomization.block_variables import BlockIdentity
from toyblocks.utils import episode_queue
from toyblocks.utils.episode_queue import _apply_block_identity

_GLYPH_USD = f"{glyphs.SUPPORTED_LIBRARY}/block_1.usd"
_SPECS: dict[str, dict[str, Any]] = {
    "block_1": {"type": "usd", "usd_path": _GLYPH_USD},
    "block_5": {"type": "usd", "usd_path": f"{glyphs.SUPPORTED_LIBRARY}/block_5.usd"},
    "block_blue": {"type": "cuboid", "color": (0.0, 0.0, 1.0)},
}
_AUTHORED_BLUE = (0.0, 0.0, 1.0)


class _Attr:
    def __init__(self, value: tuple[float, float, float] | None) -> None:
        self.value = value

    def __bool__(self) -> bool:
        return True

    def Get(self) -> tuple[float, float, float] | None:
        return self.value

    def Set(self, value: tuple[float, float, float]) -> None:
        self.value = value


class _Shader:
    def __init__(self, attr: _Attr) -> None:
        self._attr = attr

    def GetAttribute(self, name: str) -> _Attr:
        assert name == "inputs:diffuseColor"
        return self._attr


def _base_env() -> SimpleNamespace:
    env = SimpleNamespace(scene=SimpleNamespace(rigid_objects=dict.fromkeys(_SPECS)))
    setattr(env, episode_queue._BLOCK_SPECS_ATTR, _SPECS)
    return env


def _setup(glyph_map: dict[str, str], colors: dict[str, str]) -> dict[str, Any]:
    identity = BlockIdentity(
        to_source={}, glyphs=glyph_map, colors=colors,
        palette=(("green", (0.333, 1.0, 1.0)),),
    )
    return {"block_identity": identity.to_dict()}


class _Fakes:
    """Stand-ins for the USD-writing helpers, recording what they were asked to write."""

    def __init__(self) -> None:
        self.glyph_calls: list[dict[int, dict[str, str]]] = []
        self.color_calls: list[dict[int, dict[str, str]]] = []
        self.color_attr = _Attr(_AUTHORED_BLUE)
        self.skip_glyphs: set[str] = set()

    def apply_glyphs(self, assignment: dict[int, dict[str, str]], *, verbose: bool = True):
        self.glyph_calls.append(assignment)
        return {e: {b: g for b, g in m.items() if b not in self.skip_glyphs}
                for e, m in assignment.items()}

    def apply_colors(self, assignment: dict[int, dict[str, str]], palette, *, verbose: bool = True):
        self.color_calls.append(assignment)
        for m in assignment.values():
            for _name in m:
                self.color_attr.Set((0.0, 1.0, 0.0))
        return assignment

    def find_shader(self, env_idx: int, name: str) -> _Shader:
        return _Shader(self.color_attr)


@pytest.fixture
def fakes():
    f = _Fakes()
    with (
        patch.object(glyphs, "apply_wave_glyphs", f.apply_glyphs),
        patch.object(block_colors, "apply_wave_block_colors", f.apply_colors),
        patch.object(block_colors, "find_surface_shader", f.find_shader),
    ):
        yield f


def test_paints_glyphs_and_colours(fakes: _Fakes) -> None:
    env = _base_env()
    setup = _setup({"block_1": "F", "block_5": "L"}, {"block_blue": "green"})
    _apply_block_identity(env, 0, setup, "ep0")
    assert fakes.glyph_calls == [{0: {"block_1": "F", "block_5": "L"}}]
    assert fakes.color_calls == [{0: {"block_blue": "green"}}]
    assert fakes.color_attr.value == (0.0, 1.0, 0.0)
    assert setup["block_identity_status"] == "applied"


def test_next_episode_restores_authored_look(fakes: _Fakes) -> None:
    env = _base_env()
    _apply_block_identity(env, 0, _setup({"block_1": "F"}, {"block_blue": "green"}), "ep0")
    setup = _setup({}, {})
    _apply_block_identity(env, 0, setup, "ep1")
    assert fakes.glyph_calls[-1] == {0: {"block_1": glyphs.authored_glyph("block_1")}}
    assert fakes.color_attr.value == _AUTHORED_BLUE
    assert setup["block_identity_status"] == "none"


def test_setup_without_identity_needs_no_block_specs(fakes: _Fakes) -> None:
    env = SimpleNamespace(scene=SimpleNamespace(rigid_objects={}))
    setup: dict[str, Any] = {}
    _apply_block_identity(env, 0, setup, "ep0")
    assert setup["block_identity_status"] == "none"
    assert fakes.glyph_calls == [] and fakes.color_calls == []


def test_raises_on_block_not_in_scene(fakes: _Fakes) -> None:
    env = _base_env()
    with pytest.raises(ValueError, match="not in the scene"):
        _apply_block_identity(env, 0, _setup({"block_9": "F"}, {}), "ep0")


def test_raises_on_unsupported_glyph_library(fakes: _Fakes) -> None:
    env = _base_env()
    setattr(env, episode_queue._BLOCK_SPECS_ATTR,
            {**_SPECS, "block_1": {"type": "usd", "usd_path": "other_lib/block_1.usd"}})
    with pytest.raises(ValueError, match="can only be painted"):
        _apply_block_identity(env, 0, _setup({"block_1": "F"}, {}), "ep0")


def test_raises_when_glyph_not_applied(fakes: _Fakes) -> None:
    env = _base_env()
    fakes.skip_glyphs = {"block_5"}
    with pytest.raises(RuntimeError, match="block_5"):
        _apply_block_identity(env, 0, _setup({"block_1": "F", "block_5": "L"}, {}), "ep0")


@pytest.mark.parametrize("flag", [False, True])
def test_apply_setup_paints_only_with_flag(flag: bool) -> None:
    base_env = MagicMock()
    base_env.cfg = SimpleNamespace(use_calibrated_cameras=True)
    base_env.replay_block_identity = flag
    base_env.device = "cpu"
    base_env.scene.env_origins = torch.zeros(1, 3)
    base_env.scene.rigid_objects = {}
    base_env.robot.articulation.data.joint_pos = torch.zeros(1, 8)
    with (
        patch(
            "toyblocks.utils.calibrated_camera_intrinsics.apply_calibrated_camera_intrinsics_for_env"
        ),
        patch("toyblocks.utils.episode_queue._park_unused_blocks"),
        patch("toyblocks.utils.episode_queue._apply_table_height"),
        patch("toyblocks.utils.episode_queue._apply_eval_table_z_offset", return_value=0.0),
        patch("toyblocks.utils.episode_queue._apply_recorded_friction"),
        patch("toyblocks.utils.episode_queue._apply_block_identity") as paint,
    ):
        episode_queue.apply_setup(base_env, 0, {})
    assert paint.called is flag
