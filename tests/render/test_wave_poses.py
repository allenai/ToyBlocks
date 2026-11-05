"""WavePoses: one wave's recorded trajectories become the poses render writes every frame.

render.py launches Isaac at import, so the class is pulled out of its AST (as test_wave_sort
does for the pure helpers) and Isaac Lab's ``RigidObject`` / quaternion helpers are stubbed with
plain-torch stand-ins. What is checked is exactly what reaches ``write_root_pose_to_sim``.
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from toyblocks.vecgen.math import quat_apply, quat_mul

RENDER = Path(__file__).resolve().parents[2] / "scripts" / "render" / "render.py"


class _FakeRigidObject:
    """Records every root-pose write as ``(env_ids, poses)``."""

    def __init__(self) -> None:
        self.pose_writes: list[tuple[list[int], torch.Tensor]] = []

    def write_root_pose_to_sim(self, pose: torch.Tensor, env_ids: torch.Tensor) -> None:
        self.pose_writes.append((env_ids.tolist(), pose.clone()))

    def write_root_velocity_to_sim(self, vel: torch.Tensor, env_ids: torch.Tensor) -> None:
        pass


class _FakeScene(dict):
    pass


class _FakeSim:
    def __init__(self, names: list[str], num_envs: int) -> None:
        self.scene = _FakeScene({n: _FakeRigidObject() for n in names})
        self._bases = {
            i: torch.tensor([10.0 * i, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) for i in range(num_envs)
        }

    def _get_env_base_pose(self, env_idx: int) -> torch.Tensor:
        return self._bases[env_idx]


@pytest.fixture
def wave_poses_cls(monkeypatch: pytest.MonkeyPatch) -> type:
    assets = types.ModuleType("isaaclab.assets")
    assets.RigidObject = _FakeRigidObject  # type: ignore[attr-defined]
    math_mod = types.ModuleType("isaaclab.utils.math")
    math_mod.quat_apply = quat_apply  # type: ignore[attr-defined]
    math_mod.quat_mul = quat_mul  # type: ignore[attr-defined]
    for name, mod in (
        ("isaaclab", types.ModuleType("isaaclab")),
        ("isaaclab.assets", assets),
        ("isaaclab.utils", types.ModuleType("isaaclab.utils")),
        ("isaaclab.utils.math", math_mod),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    tree = ast.parse(RENDER.read_text())
    nodes = [
        n
        for n in tree.body
        if (isinstance(n, ast.ClassDef) and n.name == "WavePoses")
        or (isinstance(n, ast.ImportFrom) and n.module == "typing")
    ]
    ns: dict[str, Any] = {"np": np, "torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "render.py", "exec"), ns)
    return ns["WavePoses"]


def _unit(length: int, objects: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "length": length,
        "joint": np.zeros((length, 7), np.float32),
        "gripper": np.zeros(length, np.float32),
        "objects": objects,
    }


def _track(length: int, x0: float) -> np.ndarray:
    poses = np.zeros((length, 7), np.float32)
    poses[:, 0] = x0 + 0.01 * np.arange(length)
    poses[:, 3] = 1.0
    return poses


def test_world_poses_compose_the_env_base_and_hold_the_last_frame(wave_poses_cls: type) -> None:
    sim = _FakeSim(["block_a"], num_envs=2)
    wave = [_unit(3, {"block_a": _track(3, 0.5)}), _unit(5, {"block_a": _track(5, 0.2)})]
    poses = wave_poses_cls(sim, wave, max_t=5, device="cpu")
    for t in range(5):
        poses.write_objects(t, [0, 1], zero_velocity=False)
    writes = sim.scene["block_a"].pose_writes
    assert [ids for ids, _p in writes] == [[0, 1]] * 5
    # env 1 sits 10 m along x; env 0's shorter episode holds its last frame
    assert writes[4][1][0, 0].item() == pytest.approx(0.5 + 0.02)
    assert writes[4][1][1, 0].item() == pytest.approx(10.0 + 0.2 + 0.04)


@pytest.mark.xfail(
    strict=True,
    reason="vecgen multi-problem packages record the parked blocks outside an episode's problem "
    "as all-NaN object poses (IsaacDataRecorderLeRobot._inactive_objects); WavePoses uploads them "
    "verbatim, so every frame writes NaN root poses into PhysX. Eval's _row_to_setup skips the "
    "same NaN poses as absent blocks.",
)
def test_parked_objects_of_a_multi_problem_package_get_no_nan_pose(wave_poses_cls: type) -> None:
    sim = _FakeSim(["block_a", "block_parked"], num_envs=2)
    parked = np.full((4, 7), np.nan, np.float32)
    wave = [
        _unit(4, {"block_a": _track(4, 0.5), "block_parked": parked}),
        _unit(4, {"block_a": _track(4, 0.3), "block_parked": _track(4, 0.1)}),
    ]
    poses = wave_poses_cls(sim, wave, max_t=4, device="cpu")
    for t in range(4):
        poses.write_objects(t, [0, 1], zero_velocity=False)
    for _ids, pose in sim.scene["block_parked"].pose_writes:
        assert not torch.isnan(pose).any()
