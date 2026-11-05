"""Parked (inactive) objects and the HDRI luminance memo."""

from __future__ import annotations

import numpy as np

from toyblocks.domain_randomization import env_sphere
from toyblocks.domain_randomization.episodes import split_parked_objects


def test_all_nan_objects_are_parked_not_replayed():
    good = np.tile(np.array([0.4, 0.1, 0.05, 1, 0, 0, 0], np.float32), (3, 1))
    nan = np.full((3, 7), np.nan, np.float32)
    kept, parked = split_parked_objects({"block_a": good, "block_z": nan})
    assert list(kept) == ["block_a"] and parked == ["block_z"]
    assert np.array_equal(kept["block_a"], good)


def test_partial_nan_rows_are_filled_from_neighbours():
    poses = np.arange(4 * 7, dtype=np.float32).reshape(4, 7)
    poses[0] = np.nan
    poses[2] = np.nan
    kept, parked = split_parked_objects({"b": poses.copy()})
    assert parked == []
    assert np.isfinite(kept["b"]).all()
    assert np.array_equal(kept["b"][0], poses[1])   # leading gap: next valid row
    assert np.array_equal(kept["b"][2], poses[1])   # inner gap: previous valid row


def test_luminance_failure_is_not_memoized_and_source_is_tried(monkeypatch):
    env_sphere._LUMINANCE.clear()
    reads: list[str] = []
    monkeypatch.setattr(env_sphere._texcache, "resolve", lambda p: "/cache/" + p.rsplit("/", 1)[-1])

    def read(path):
        reads.append(path)
        return None if path.startswith("/cache/") else 0.5     # the cached copy was evicted

    monkeypatch.setattr(env_sphere, "_read_luminance", read)
    assert env_sphere._hdri_luminance("/src/sky.hdr") == 0.5
    assert reads == ["/cache/sky.hdr", "/src/sky.hdr"]
    monkeypatch.setattr(env_sphere, "_read_luminance", lambda p: None)
    env_sphere._LUMINANCE.clear()
    assert env_sphere._hdri_luminance("/src/b.hdr") is None
    assert "/src/b.hdr" not in env_sphere._LUMINANCE            # failures are retried next time
