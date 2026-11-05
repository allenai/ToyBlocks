"""apply_setup uses preset K when use_calibrated_cameras is set."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from blocksuite.utils.episode_queue import apply_setup


def test_apply_setup_uses_calibrated_not_setup_k() -> None:
    base_env = MagicMock()
    base_env.cfg = SimpleNamespace(use_calibrated_cameras=True)
    base_env.device = "cpu"
    base_env.scene.env_origins = torch.zeros(1, 3)
    base_env.scene.rigid_objects = {}
    base_env.robot.articulation.data.joint_pos = torch.zeros(1, 8)
    base_env.robot.articulation.write_joint_state_to_sim = MagicMock()
    setup = {
        "camera_intrinsics": {
            "front_camera": [1.0] * 9,
        },
    }

    with (
        patch(
            "blocksuite.utils.calibrated_camera_intrinsics.apply_calibrated_camera_intrinsics_for_env"
        ) as mock_cal,
        patch(
            "blocksuite.utils.dataset_provenance.apply_camera_intrinsics_for_env"
        ) as mock_rec,
        patch("blocksuite.utils.episode_queue._park_unused_blocks"),
        patch("blocksuite.utils.episode_queue._apply_table_height"),
        patch("blocksuite.utils.episode_queue._apply_eval_table_z_offset", return_value=0.0),
        patch("blocksuite.utils.episode_queue._apply_recorded_friction"),
    ):
        apply_setup(base_env, 0, setup)
        mock_cal.assert_called_once_with(base_env, 0)
        mock_rec.assert_not_called()
