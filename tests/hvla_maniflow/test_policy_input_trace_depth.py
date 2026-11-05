"""Policy-input trace panels with depth tensors (requires maniflow-tb)."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pytest
import torch

from toyblocks.serving.maniflow_serving_utils import summarize_model_inputs
from toyblocks.serving.policy_input_trace import render_ll_input_panel
from toyblocks.utils.wandb_logging import _LL_INPUT_PANEL_RE
from tests.serving.policy_input_fixture import build_obs, build_target_heatmap, normalizer_for
from tests.serving.test_policy_input_trace import HEATMAP_PEAK_XY

from toyblocks.serving.obs_transforms.maniflow_blocks_table import (
    ManiflowBlocksTableObsTransform,
)


def _wire_for_panel(
    state_dim: int, *, depth: bool, heatmap: bool, prev_front: bool
) -> dict[str, Any]:
    transform = ManiflowBlocksTableObsTransform(
        prompt="pick up the red block",
        state_dim=state_dim,
        depth_keys=("front_depth", "gripper_depth") if depth else (),
        include_prev_front_frame=prev_front,
        image_interpolate_device="cpu",
    )
    wire = transform(build_obs())
    wire["agent_pos"] = normalizer_for(state_dim)._normalize_state(
        np.asarray(wire["agent_pos"])
    )
    if heatmap:
        wire["target_heatmap"] = build_target_heatmap(HEATMAP_PEAK_XY)
    return {
        k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
        for k, v in wire.items()
    }


@pytest.mark.parametrize("state_dim", [8, 9, 19])
@pytest.mark.parametrize(
    "depth,heatmap,prev_front,raw_state",
    [
        (True, True, True, True),
        (True, False, True, True),
    ],
    ids=["everything", "depth_prev_no_heatmap"],
)
def test_render_ll_input_panel_over_the_fixture_wire(
    tmp_path, state_dim, depth, heatmap, prev_front, raw_state
) -> None:
    torch_obs = _wire_for_panel(
        state_dim, depth=depth, heatmap=heatmap, prev_front=prev_front
    )
    png = render_ll_input_panel(
        torch_obs,
        out_dir=str(tmp_path / "ll_policy_input_trace"),
        env_idx=1,
        sim_time_ms=12345.6,
        episode_id=7,
        raw_state=np.asarray(torch_obs["agent_pos"]) if raw_state else None,
        dropped=["front_ptmap"] if not depth else [],
        summary=summarize_model_inputs(torch_obs),
    )
    assert png.is_file()
    assert png.stat().st_size > 0


def test_panel_filename_matches_both_consumer_globs(tmp_path) -> None:
    torch_obs = _wire_for_panel(19, depth=True, heatmap=True, prev_front=False)
    trace_dir = tmp_path / "ll_policy_input_trace"
    for episode_id, sim_time_ms in ((3, 100.0), (3, 2000.0), (11, 50.0)):
        render_ll_input_panel(
            torch_obs,
            out_dir=str(trace_dir),
            env_idx=0,
            sim_time_ms=sim_time_ms,
            episode_id=episode_id,
            raw_state=None,
            dropped=[],
            summary="",
        )
    panels = sorted(trace_dir.glob("ll_input_*.png"))
    assert len(panels) == 3
    parsed = []
    for path in panels:
        m = _LL_INPUT_PANEL_RE.match(path.name)
        assert m is not None
        parsed.append((int(m.group(1)), float(m.group(2))))
    assert parsed == [(3, 100.0), (3, 2000.0), (11, 50.0)]


def test_summarize_model_inputs_reports_every_key() -> None:
    torch_obs = _wire_for_panel(19, depth=True, heatmap=True, prev_front=True)
    summary = summarize_model_inputs(torch_obs)
    for key in torch_obs:
        assert re.search(rf"^\s+{re.escape(key)}: ", summary, re.MULTILINE), key
    assert "peaks:" in summary
