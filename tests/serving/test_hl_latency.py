"""Unit tests for :class:`~toyblocks.serving.hl_latency.HLLatencyBuffer`."""

from __future__ import annotations

from toyblocks.serving.hl_latency import HLLatencyBuffer


def _payload(tag: str) -> dict[str, str]:
    return {"tag": tag}


def test_prune_retains_entry_for_m_when_k_below_chunk_period() -> None:
    """K=0, M=500 with a 1000ms chunk: age must reach M after 2nd infer."""
    chunk_ms = 1000.0
    buf = HLLatencyBuffer(
        update_interval_ms=0.0,
        estimated_inference_time_ms=500.0,
        num_envs=1,
        ll_infer_period_ms=chunk_ms,
    )
    env = 0
    assert buf.should_update(env, 0.0)
    buf.commit(env, 0.0, _payload("a"))
    age0 = buf.selected_age_ms(env, 0.0)
    assert age0 == 0.0

    # Second HL commit one chunk later (server infer cadence).
    assert buf.should_update(env, chunk_ms)
    buf.commit(env, chunk_ms, _payload("b"))
    age1 = buf.selected_age_ms(env, chunk_ms)
    assert age1 is not None and age1 >= 500.0


def test_prune_without_chunk_margin_drops_stale_entry() -> None:
    """Without ll_infer_period_ms the first entry is pruned and age collapses to 0."""
    chunk_ms = 1000.0
    buf = HLLatencyBuffer(
        update_interval_ms=0.0,
        estimated_inference_time_ms=500.0,
        num_envs=1,
        ll_infer_period_ms=0.0,
    )
    env = 0
    buf.commit(env, 0.0, _payload("a"))
    buf.commit(env, chunk_ms, _payload("b"))
    age = buf.selected_age_ms(env, chunk_ms)
    assert age == 0.0


def test_k_ge_chunk_gives_expected_age() -> None:
    """K=1000, M=500 with 1000ms chunks → served age 1000ms."""
    chunk_ms = 1000.0
    buf = HLLatencyBuffer(
        update_interval_ms=chunk_ms,
        estimated_inference_time_ms=500.0,
        num_envs=1,
        ll_infer_period_ms=chunk_ms,
    )
    env = 0
    buf.commit(env, 0.0, _payload("a"))
    assert not buf.should_update(env, 500.0)
    assert buf.should_update(env, chunk_ms)
    buf.commit(env, chunk_ms, _payload("b"))
    age = buf.selected_age_ms(env, chunk_ms)
    assert age is not None and abs(age - chunk_ms) < 1e-6
