"""CPU tests for the policy-input debug/trace surface.

``policy_input_trace.py``, ``eval/policy_input_debug.py`` and the summarizers in
``maniflow_serving_utils`` had no tests, and their failure modes are expensive in
a way ordinary code's are not: these are the tools you reach for *because* a
policy is misbehaving, so a crash takes out the diagnostic step of a live GPU
eval, and -- worse -- a mislabelled state dim sends you chasing a bug that isn't
there.

The sharpest case is :data:`_STATE_BLOCKS_BY_DIM`, which is a **second
hand-maintained copy** of the layout ``_normalize_state`` slices by fixed index.
Nothing checked that the two agree, so this file declares the layout once and
checks both against it: the panel's labels and bands on one side, the
normalizer's actual behaviour on the other. A raw dim drawn with a band, or a
normalized dim drawn without one, is a fabricated failure mode in a diagnostic.

The renderer is exercised over the same seeded wire dict the goldens use, not a
hand-built stub, and the PNG filename is pinned against the globs its two
consumers actually run -- that filename is a cross-process contract carrying the
wandb row keys, because the panel is written inside the policy-server process and
has no way to hand a sidecar back.

Needs matplotlib (in the ``maniflow`` extra and installed explicitly by
``ci-unit.yaml``); a missing one is a lane misconfiguration and fails loudly.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pytest
import torch

from blocksuite.eval.policy_input_debug import (
    _STATE_BLOCKS_BY_DIM,
    _state_rows,
)
from blocksuite.serving.maniflow_serving_utils import (
    _summarize_heatmap_peaks,
    _summarize_value,
    state_norm_metadata,
    summarize_model_inputs,
)
from blocksuite.serving.obs_transforms.maniflow import STATE_DIM_LAYOUTS
from blocksuite.serving.policy_input_trace import (
    _depth_hw,
    _heatmap_hw,
    _image_hwc_uint8,
    _leading_slice,
    _state_vector,
    render_hl_input_panel,
    render_ll_input_panel,
)
from blocksuite.utils.wandb_logging import _HL_INPUT_PANEL_RE, _LL_INPUT_PANEL_RE
from tests.serving.policy_input_fixture import (
    FIXTURE_BATCH,
    build_obs,
    build_target_heatmap,
    load_stats,
    normalizer_for,
)

HEATMAP_PEAK_XY = (61, 148)

# The ``agent_pos`` layout, declared once here as ``(label, width, normalized)``.
# Both hand-maintained copies are checked against this: the panel's
# ``_STATE_BLOCKS_BY_DIM`` and, behaviourally, ``_normalize_state``. Writing it a
# third time is the point -- a test that imported either copy would agree with
# itself.
EXPECTED_STATE_LAYOUT: dict[int, tuple[tuple[str, int, bool], ...]] = {
    8: (("arm", 7, True), ("gripper", 1, False)),
    9: (("arm", 7, True), ("gripper", 1, False), ("pd_mode", 1, False)),
    19: (
        ("arm", 7, True),
        ("jvel", 7, True),
        ("ee", 3, True),
        ("gripper", 1, False),
        ("pd_mode", 1, False),
    ),
}

# Values placed in the raw (never-normalized) dims. Neither may be -1.0, which is
# what a normalized dim fed its band's q01 comes out as -- that contrast is the
# whole discriminator in ``test_normalize_state_matches_declared_layout``.
RAW_SENTINELS = {"gripper": 0.021, "pd_mode": 1.0}


def _expected_labels(state_dim: int) -> list[str]:
    """Row labels the declared layout implies, in order.

    Single-width blocks are unsuffixed (``gripper``, not ``gripper[0]``), which
    is also the key ``_dataset_state`` writes its recorded frame-0 values under.
    """
    labels: list[str] = []
    for label, width, _ in EXPECTED_STATE_LAYOUT[state_dim]:
        if width > 1:
            labels.extend(f"{label}[{j}]" for j in range(width))
        else:
            labels.append(label)
    return labels


# --------------------------------------------------------------------------
# The strict slicers
# --------------------------------------------------------------------------


def _per_env_frames(batch: int = FIXTURE_BATCH, seq: int = 3) -> np.ndarray:
    """``(B, S, 3, 4, 5)`` where every (env, frame, channel) is a distinct value.

    Distinct per env *and* per frame, so slicing env 1 cannot pass by reading env
    0 and taking the last frame cannot pass by reading the first.
    """
    arr = np.zeros((batch, seq, 3, 4, 5), dtype=np.float32)
    for b in range(batch):
        for s in range(seq):
            for c in range(3):
                arr[b, s, c] = (b + 1) * 100 + s * 10 + c
    return arr


def test_leading_slice_takes_the_env_then_the_last_frame():
    arr = _per_env_frames()
    out = _leading_slice(arr, env_idx=1, want_ndim=3, key="front_rgb")
    assert out.shape == (3, 4, 5)
    # env 1, last of 3 frames, channel 2: 200 + 20 + 2.
    assert out[2, 0, 0] == pytest.approx(222.0)
    # env 0 differs, so a hardcoded 0 would have failed here.
    other = _leading_slice(arr, env_idx=0, want_ndim=3, key="front_rgb")
    assert other[2, 0, 0] == pytest.approx(122.0)


def test_leading_slice_passes_through_an_already_reduced_array():
    arr = np.zeros((3, 4, 5), dtype=np.float32)
    assert _leading_slice(arr, 0, 3, "front_rgb") is arr


def test_leading_slice_rejects_too_few_dims():
    with pytest.raises(ValueError, match="expected at least 3 dims"):
        _leading_slice(np.zeros((4, 5)), 0, 3, "front_rgb")


def test_leading_slice_rejects_env_idx_out_of_range():
    with pytest.raises(IndexError, match="env_idx=5 out of range"):
        _leading_slice(_per_env_frames(), 5, 3, "front_rgb")


def test_image_hwc_uint8_transposes_and_scales():
    """CHW float in [0, 1] -> HWC uint8, per env.

    The transpose is the assertion: a panel that skipped it would render a
    3-pixel-tall image and still "work", which is exactly the kind of silent
    wrongness a diagnostic must not have.
    """
    chw = np.zeros((FIXTURE_BATCH, 1, 3, 4, 5), dtype=np.float32)
    chw[1, 0, 0] = 1.0  # env 1, red channel saturated
    chw[1, 0, 1] = 0.5
    chw[0, 0, 2] = 1.0  # env 0 is different
    out = _image_hwc_uint8(chw, env_idx=1, key="front_rgb")
    assert out.shape == (4, 5, 3)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out[..., 0], 255)
    np.testing.assert_array_equal(out[..., 1], 128)  # round(0.5 * 255) = 128
    np.testing.assert_array_equal(out[..., 2], 0)


def test_image_hwc_uint8_clips_out_of_range_floats():
    chw = np.full((1, 1, 3, 2, 2), 4.0, dtype=np.float32)
    chw[0, 0, 1] = -2.0
    out = _image_hwc_uint8(chw, 0, "front_rgb")
    assert out.max() == 255
    assert out.min() == 0


def test_image_hwc_uint8_rejects_a_non_three_channel_frame():
    with pytest.raises(ValueError, match="expected a 3-channel CHW frame"):
        _image_hwc_uint8(np.zeros((1, 1, 4, 8, 8), dtype=np.float32), 0, "front_rgb")


def test_heatmap_hw_reads_channel_zero_of_the_selected_env():
    hm = np.zeros((FIXTURE_BATCH, 1, 3, 6, 7), dtype=np.float32)
    hm[1, 0, 0, 2, 3] = 0.75
    hm[1, 0, 1, 4, 5] = 9.0  # a different channel, which must not be read
    out = _heatmap_hw(hm, env_idx=1)
    assert out.shape == (6, 7)
    assert out[2, 3] == pytest.approx(0.75)
    assert out.max() == pytest.approx(0.75)


def test_depth_hw_unwraps_the_singleton_channel_axis():
    depth = np.zeros((FIXTURE_BATCH, 1, 1, 6, 7), dtype=np.float32)
    depth[1, 0, 0, 2, 3] = 1.25
    out = _depth_hw(depth, env_idx=1, key="front_depth")
    assert out.shape == (6, 7)
    assert out[2, 3] == pytest.approx(1.25)


def test_depth_hw_handles_depth_with_no_channel_axis():
    """``(B, H, W)`` depth takes the other branch and must reduce identically."""
    depth = np.zeros((FIXTURE_BATCH, 6, 7), dtype=np.float32)
    depth[1, 2, 3] = 1.25
    out = _depth_hw(depth, env_idx=1, key="front_depth")
    assert out.shape == (6, 7)
    assert out[2, 3] == pytest.approx(1.25)


def test_state_vector_reduces_to_one_dim_per_env():
    state = np.arange(FIXTURE_BATCH * 1 * 4, dtype=np.float32).reshape(FIXTURE_BATCH, 1, 4)
    np.testing.assert_allclose(_state_vector(state, 1, "agent_pos"), [4.0, 5.0, 6.0, 7.0])


def test_slicers_accept_torch_tensors():
    """The panel is fed the server's ``torch_obs``, so every slicer sees tensors."""
    chw = torch.zeros(FIXTURE_BATCH, 1, 3, 4, 5)
    chw[1, 0, 0] = 1.0
    assert _image_hwc_uint8(chw, 1, "front_rgb").shape == (4, 5, 3)
    assert _state_vector(torch.zeros(FIXTURE_BATCH, 1, 9), 1, "agent_pos").shape == (9,)


# --------------------------------------------------------------------------
# The state layout: two hand-maintained copies, one declaration
# --------------------------------------------------------------------------


def test_state_block_widths_cover_every_declared_state_dim():
    """A width added to ``STATE_DIM_LAYOUTS`` must gain panel labels too.

    Without this, a new width builds a valid ``agent_pos`` that the panel then
    refuses to label -- a crash in the diagnostic of the very eval that needs it.
    """
    assert sorted(_STATE_BLOCKS_BY_DIM) == sorted(STATE_DIM_LAYOUTS)
    assert sorted(_STATE_BLOCKS_BY_DIM) == sorted(EXPECTED_STATE_LAYOUT)


@pytest.mark.parametrize("state_dim", sorted(EXPECTED_STATE_LAYOUT))
def test_state_blocks_match_the_declared_layout(state_dim: int):
    """The panel's block table agrees with the declared layout, block by block."""
    observed = [
        (label, width, normalized)
        for _stats_key, label, width, normalized in _STATE_BLOCKS_BY_DIM[state_dim]
    ]
    assert observed == list(EXPECTED_STATE_LAYOUT[state_dim])
    assert sum(width for _, width, _ in observed) == state_dim


@pytest.mark.parametrize("state_dim", sorted(EXPECTED_STATE_LAYOUT))
def test_state_blocks_agree_with_the_include_flags(state_dim: int):
    """Optional blocks are present exactly when ``STATE_DIM_LAYOUTS`` says so."""
    labels = {label for _, label, _, _ in _STATE_BLOCKS_BY_DIM[state_dim]}
    flags = STATE_DIM_LAYOUTS[state_dim]
    assert ("jvel" in labels) is flags["include_joint_vel"]
    assert ("ee" in labels) is flags["include_ee_pos"]
    assert ("pd_mode" in labels) is flags["include_pd_mode"]


def _state_at_band_floor(state_dim: int) -> np.ndarray:
    """A raw state with every normalized block at q01 and raw dims at a sentinel.

    Fed to ``_normalize_state``, normalized dims come out at exactly -1.0 and raw
    dims come out unchanged -- which is what separates the two classes without
    the test having to know how the normalizer is written.
    """
    stats = load_stats(state_dim)
    state = np.zeros((1, state_dim), dtype=np.float32)
    stats_for_label = {
        "arm": "observation.state.joint_position",
        "jvel": "observation.state.joint_velocity",
        "ee": "observation.state.ee_position",
    }
    offset = 0
    for label, width, normalized in EXPECTED_STATE_LAYOUT[state_dim]:
        if normalized:
            q01 = np.atleast_1d(np.asarray(stats[stats_for_label[label]]["q01"], dtype=np.float32))
            state[0, offset : offset + width] = q01[:width]
        else:
            state[0, offset : offset + width] = RAW_SENTINELS[label]
        offset += width
    return state


@pytest.mark.parametrize("state_dim", sorted(EXPECTED_STATE_LAYOUT))
def test_normalize_state_matches_declared_layout(state_dim: int):
    """``_normalize_state`` normalizes exactly the dims the panel says it does.

    This is the other half of the ``_STATE_BLOCKS_BY_DIM`` cross-check: the panel
    draws a ``[q01, q99]`` band around every dim it marks normalized, so a
    normalizer that stopped touching one of them would have the panel reporting a
    constraint that is no longer applied.
    """
    state = _state_at_band_floor(state_dim)
    out = normalizer_for(state_dim)._normalize_state(state.copy())
    assert out.shape == state.shape

    offset = 0
    for label, width, normalized in EXPECTED_STATE_LAYOUT[state_dim]:
        block = out[0, offset : offset + width]
        if normalized:
            np.testing.assert_allclose(
                block, -1.0, atol=1e-6,
                err_msg=f"{label!r} is declared normalized but was passed through raw",
            )
        else:
            np.testing.assert_allclose(
                block, RAW_SENTINELS[label], atol=1e-6,
                err_msg=f"{label!r} is declared raw but was normalized",
            )
        offset += width


@pytest.mark.parametrize("state_dim", sorted(EXPECTED_STATE_LAYOUT))
def test_state_rows_offsets_labels_and_bands(state_dim: int):
    """Row *i* labels and reports dim *i*, with a band iff the dim is normalized.

    Offsets are checked by value rather than by index arithmetic: the state is
    ``arange``, so a row reading the wrong offset reports the wrong number.
    """
    state_norm = state_norm_metadata(load_stats(state_dim))
    state = np.arange(state_dim, dtype=np.float32)
    rows = _state_rows(state, state_norm, {})

    assert [r["label"] for r in rows] == _expected_labels(state_dim)
    assert [r["value"] for r in rows] == [float(i) for i in range(state_dim)]

    expected_banded: list[bool] = []
    for _label, width, normalized in EXPECTED_STATE_LAYOUT[state_dim]:
        expected_banded.extend([normalized] * width)
    assert [r["q01"] is not None for r in rows] == expected_banded


@pytest.mark.parametrize("state_dim", sorted(EXPECTED_STATE_LAYOUT))
def test_state_rows_bands_come_from_server_metadata(state_dim: int):
    """``state_norm_metadata`` -> ``_state_rows`` is the server/panel contract.

    Neither side was tested, and it is the pair that decides what band the panel
    draws: the client has no stats.json, so if the server's advertised bands ever
    stopped lining up with these rows the panel would flag healthy dims.
    """
    stats = load_stats(state_dim)
    rows = _state_rows(np.zeros(state_dim, dtype=np.float32), state_norm_metadata(stats), {})

    stats_for_label = {
        "arm": "observation.state.joint_position",
        "jvel": "observation.state.joint_velocity",
        "ee": "observation.state.ee_position",
    }
    offset = 0
    for label, width, normalized in EXPECTED_STATE_LAYOUT[state_dim]:
        if normalized:
            entry = stats[stats_for_label[label]]
            for j in range(width):
                assert rows[offset + j]["q01"] == pytest.approx(float(entry["q01"][j]))
                assert rows[offset + j]["q99"] == pytest.approx(float(entry["q99"][j]))
        offset += width


def test_state_rows_reports_the_dataset_column_by_label():
    """The purple dataset marker is keyed by row label, not by index."""
    rows = _state_rows(
        np.zeros(8, dtype=np.float32),
        state_norm_metadata(load_stats(8)),
        {"arm[3]": 0.5, "gripper": 0.03},
    )
    by_label = {r["label"]: r["dataset"] for r in rows}
    assert by_label["arm[3]"] == pytest.approx(0.5)
    assert by_label["gripper"] == pytest.approx(0.03)
    assert by_label["arm[0]"] is None


def test_state_rows_rejects_an_unknown_width():
    with pytest.raises(ValueError, match="known ManiFlow state layouts"):
        _state_rows(np.zeros(12, dtype=np.float32), state_norm_metadata(load_stats(19)), {})


def test_state_rows_rejects_a_normalized_block_with_no_advertised_band():
    """A 19-dim state against short-layout metadata has no jvel / ee bands.

    Drawing that panel would mean inventing a band or silently omitting one, both
    of which misreport what the server does -- so it raises, naming the block.
    """
    short_norm = state_norm_metadata(load_stats(9))
    assert "observation.state.joint_velocity" not in short_norm
    with pytest.raises(ValueError, match="'jvel' block"):
        _state_rows(np.zeros(19, dtype=np.float32), short_norm, {})


# --------------------------------------------------------------------------
# render_ll_input_panel
# --------------------------------------------------------------------------


def _wire_for_panel(
    state_dim: int, *, depth: bool, heatmap: bool, prev_front: bool
) -> dict[str, Any]:
    """A server-side ``torch_obs`` built from the seeded fixture.

    The real transform output, not a stub: the panel's whole job is to render the
    tensors the model receives, so feeding it hand-built arrays would test the
    matplotlib calls and nothing else.
    """
    from blocksuite.serving.obs_transforms.maniflow_blocks_table import (
        ManiflowBlocksTableObsTransform,
    )

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
        (False, False, False, False),
        (False, True, False, True),
    ],
    ids=["minimal", "heatmap_only"],
)
def test_render_ll_input_panel_over_the_fixture_wire(
    tmp_path, state_dim, depth, heatmap, prev_front, raw_state
):
    """Every optional-key combination renders a PNG at the contract filename.

    The combinations are the point: each optional key adds panels (and
    ``front_rgb`` + ``prev_front_rgb`` together add a *derived* difference
    panel), so the grid layout arithmetic is re-derived for each one and an
    off-by-one there is a crash inside a running eval.
    """
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
    assert png.name == "ll_input_ep0007_t0012346ms.png"


def test_hl_panel_filename_matches_consumer_glob(tmp_path):
    """HL panels share the LL filename contract so the consolidator can parse them."""
    rgb = np.zeros((32, 48, 3), dtype=np.uint8)
    rgb[..., 0] = 200
    gripper = np.zeros((32, 48, 3), dtype=np.uint8)
    gripper[..., 1] = 200
    trace_dir = tmp_path / "hl_policy_input_trace"
    png = render_hl_input_panel(
        front_rgb=rgb,
        gripper_rgb=gripper,
        question="Previous action: pick <1, 2>. stack 2 red blocks",
        prev_action="pick <1, 2>",
        hl_output="place <3, 4>",
        out_dir=str(trace_dir),
        env_idx=0,
        sim_time_ms=12345.6,
        episode_id=7,
    )
    assert png.name == "hl_input_ep0007_t0012346ms.png"
    m = _HL_INPUT_PANEL_RE.match(png.name)
    assert m is not None
    assert int(m.group(1)) == 7
    assert float(m.group(2)) == 12346.0


def test_hl_panel_filename_matches_consumer_glob(tmp_path):
    """HL panels share the LL filename contract so the consolidator can parse them."""
    rgb = np.zeros((32, 48, 3), dtype=np.uint8)
    rgb[..., 0] = 200
    gripper = np.zeros((32, 48, 3), dtype=np.uint8)
    gripper[..., 1] = 200
    trace_dir = tmp_path / "hl_policy_input_trace"
    png = render_hl_input_panel(
        front_rgb=rgb,
        gripper_rgb=gripper,
        question="Previous action: pick <1, 2>. stack 2 red blocks",
        prev_action="pick <1, 2>",
        hl_output="place <3, 4>",
        out_dir=str(trace_dir),
        env_idx=0,
        sim_time_ms=12345.6,
        episode_id=7,
    )
    assert png.name == "hl_input_ep0007_t0012346ms.png"
    m = _HL_INPUT_PANEL_RE.match(png.name)
    assert m is not None
    assert int(m.group(1)) == 7
    assert float(m.group(2)) == 12346.0


def test_panel_reports_the_selected_env_task_name(tmp_path):
    """``task_name`` is a ``list[str]``, so the footer must index it per env.

    Rendering env 1's tensors under env 0's prompt is the kind of mislabelling
    that sends someone debugging the wrong episode. Asserted through the rendered
    pixels, because the footer is drawn text with nothing to read back: at
    ``env_idx=1``, editing entry 1 has to change the panel and editing entry 0
    must not. ``summary`` is empty so the footer's ``task_name`` line is the only
    place the prompt appears.
    """
    import matplotlib.image as mpimg

    torch_obs = _wire_for_panel(19, depth=False, heatmap=False, prev_front=False)

    def render(task_name: list[str], out_name: str) -> np.ndarray:
        torch_obs["task_name"] = task_name
        png = render_ll_input_panel(
            torch_obs,
            out_dir=str(tmp_path / out_name),
            env_idx=1,
            sim_time_ms=0.0,
            episode_id=0,
            raw_state=None,
            dropped=[],
            summary="",
        )
        assert png.is_file()
        return mpimg.imread(png)

    baseline = render(["prompt for env 0", "prompt for env 1"], "baseline")
    env0_edited = render(["a much longer prompt", "prompt for env 1"], "env0_edited")
    env1_edited = render(["prompt for env 0", "a much longer prompt"], "env1_edited")

    np.testing.assert_array_equal(
        baseline,
        env0_edited,
        err_msg="the panel changed when only env 0's prompt did, so the footer is "
        "not indexing task_name by env_idx",
    )
    assert not np.array_equal(baseline, env1_edited), (
        "the panel did not change when env 1's prompt did, so it is not reporting "
        "the selected env's task_name"
    )


# --------------------------------------------------------------------------
# The summarizers
# --------------------------------------------------------------------------


def test_summarize_heatmap_peaks_finds_the_planted_pixel():
    """``(x, y)``, in that order: a transposed report is a wrong spatial goal."""
    hm = build_target_heatmap(HEATMAP_PEAK_XY)
    out = _summarize_heatmap_peaks(torch.from_numpy(hm))
    x, y = HEATMAP_PEAK_XY
    assert out.count(f"x={x},y={y}") == FIXTURE_BATCH


@pytest.mark.parametrize("ndim_case", ["chw", "bchw", "bschw"])
def test_summarize_heatmap_peaks_accepts_every_declared_rank(ndim_case: str):
    hm = build_target_heatmap(HEATMAP_PEAK_XY, batch=1)  # (1, 1, 3, 224, 224)
    arr = {"chw": hm[0, 0], "bchw": hm[:, 0], "bschw": hm}[ndim_case]
    x, y = HEATMAP_PEAK_XY
    assert f"x={x},y={y}" in _summarize_heatmap_peaks(arr)


def test_summarize_heatmap_peaks_reports_an_unexpected_rank():
    assert "unexpected ndim=2" in _summarize_heatmap_peaks(np.zeros((4, 4)))
    assert _summarize_heatmap_peaks(None) == "None"


def test_summarize_value_prints_lists_verbatim():
    """``task_name`` is the one input whose *content* is the thing to read."""
    assert _summarize_value(["pick", "place"]) == "list[2]=['pick', 'place']"


def test_summarize_value_handles_none_and_scalars():
    assert _summarize_value(None) == "None"
    assert _summarize_value(3.5) == "float=3.5"
    assert _summarize_value("abs") == "str='abs'"


def test_summarize_value_reports_exact_tensor_and_array_stats():
    t = torch.arange(4, dtype=torch.float32)
    summary = _summarize_value(t)
    assert "Tensor(4,)" in summary and "min=0.0000 max=3.0000 mean=1.5000" in summary
    a = np.arange(4, dtype=np.float32)
    assert "ndarray(4,)" in _summarize_value(a)
