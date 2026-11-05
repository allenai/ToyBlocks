"""Depth monitoring summaries for first-batch eval."""

import numpy as np
import pytest

from blocksuite.serving.depth_monitoring import summarize_depth_meters, summarize_wire_depth_batch


def test_summarize_depth_meters_zeros_and_probe_shelf() -> None:
    depth = np.array(
        [[0.0, 0.05, 0.5, 1.0, 1.0, 1.5]],
        dtype=np.float32,
    )
    s = summarize_depth_meters(depth, clamp_probe_m=(1.0,))
    assert s["count"] == 6
    assert s["count_zero"] == 1
    assert s["min_positive_m"] == pytest.approx(0.05)
    assert s["max_positive_m"] == pytest.approx(1.5)
    assert s["count_at_max_positive"] == 1
    assert s["probe_exact_counts"]["1"] == 2


def test_summarize_wire_depth_batch() -> None:
    wire = {
        "front_depth": np.ones((2, 1, 1, 4, 4), dtype=np.float32) * 0.5,
    }
    wire["front_depth"][1, 0, 0] = 0.0
    rep = summarize_wire_depth_batch(wire, ("front_depth",))
    assert rep["front_depth"]["pooled"]["fraction_zero"] > 0
    assert len(rep["front_depth"]["per_env"]) == 2
