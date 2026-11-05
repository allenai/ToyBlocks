"""table_height_delta is the height of the table the reset state placed, not the env slot's.

A reset state runs in any slot whose block sizes and friction match and restores its own table
pose, so the slot's profile height can be centimetres off (slots 4/5 of the oct30 RI group: equal
sizes, tables 6.2 cm apart). Needs cuRobo (vecgen.recorder imports the geometry module).
"""

from __future__ import annotations

import types

import torch

from toyblocks.vecgen.recorder import LeRobotRecorder


class _Scene(dict):
    def __init__(self, items, origins):
        super().__init__(items)
        self.env_origins = origins


def test_placed_table_height_delta_reads_the_placed_table():
    table = types.SimpleNamespace(data=types.SimpleNamespace(
        root_pose_w=torch.tensor([[0.0, 0.0, 0.031, 1, 0, 0, 0], [5.0, 0.0, -0.02, 1, 0, 0, 0]]),
        default_root_state=torch.zeros(2, 13)))
    rec = LeRobotRecorder.__new__(LeRobotRecorder)
    rec.sim = types.SimpleNamespace(base_env=types.SimpleNamespace(
        scene=_Scene({"workspace_table": table}, torch.zeros(2, 3))))
    assert abs(rec._placed_table_height_delta(0) - 0.031) < 1e-6
    assert abs(rec._placed_table_height_delta(1) + 0.02) < 1e-6
    rec.sim = types.SimpleNamespace(base_env=None)
    assert rec._placed_table_height_delta(0) is None
