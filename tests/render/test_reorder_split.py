"""reorder_split_episodes: a sorted-wave render is put back into source episode order.

Builds a split shaped like the recorder's output (data/, meta/episodes/, one video file per episode
per feature), renumbers it, and checks every place an episode index lives: data rows and their
global index, the episode table's frame ranges and data/video pointers, and the video file names.
"""

from __future__ import annotations

import ast
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

RENDER = Path(__file__).resolve().parents[2] / "scripts" / "render" / "render.py"
FEATS = ("observation.images.camera:front_camera", "observation.depth.camera:front_camera")


def _load() -> Any:
    tree = ast.parse(RENDER.read_text())
    nodes = [
        n
        for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name == "reorder_split_episodes")
        or (isinstance(n, ast.ImportFrom) and n.module == "typing")
    ]
    ns: dict[str, Any] = {"np": np, "shutil": shutil, "Path": Path}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "render.py", "exec"), ns)
    return ns["reorder_split_episodes"]


reorder_split_episodes = _load()


def _make_split(root: Path, lengths: list[int], rows_per_file: int) -> None:
    """Episode e has `lengths[e]` frames whose payload column is 1000 * e + frame."""
    ep = np.concatenate([np.full(n, e) for e, n in enumerate(lengths)])
    fr = np.concatenate([np.arange(n) for n in lengths])
    data = pa.table(
        {
            "payload": (1000 * ep + fr).astype(np.int64),
            "frame_index": fr.astype(np.int64),
            "episode_index": ep.astype(np.int64),
            "index": np.arange(len(ep), dtype=np.int64),
        }
    )
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    file_of, cur, f = [], 0, 0
    for n in lengths:  # whole episodes per file, as the recorder writes them
        if cur and cur + n > rows_per_file:
            f, cur = f + 1, 0
        file_of.append(f)
        cur += n
    for j in range(f + 1):
        sel = [e for e in range(len(lengths)) if file_of[e] == j]
        lo, hi = int(starts[sel[0]]), int(starts[sel[-1]] + lengths[sel[-1]])
        dst = root / "data" / "chunk-000" / f"file-{j:03d}.parquet"
        dst.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(data.slice(lo, hi - lo), dst)
    cols: dict[str, Any] = {
        "episode_index": np.arange(len(lengths), dtype=np.int64),
        "length": np.asarray(lengths, dtype=np.int64),
        "data/chunk_index": np.zeros(len(lengths), dtype=np.int64),
        "data/file_index": np.asarray(file_of, dtype=np.int64),
        "dataset_from_index": starts.astype(np.int64),
        "dataset_to_index": (starts + np.asarray(lengths)).astype(np.int64),
        "stats/payload/max": [[1000.0 * e + n - 1] for e, n in enumerate(lengths)],
    }
    for k in FEATS:
        cols[f"videos/{k}/chunk_index"] = np.zeros(len(lengths), dtype=np.int64)
        cols[f"videos/{k}/file_index"] = np.arange(len(lengths), dtype=np.int64)
        cols[f"videos/{k}/from_timestamp"] = np.zeros(len(lengths))
        for e in range(len(lengths)):
            v = root / "videos" / k / "chunk-000" / f"file-{e:03d}.mp4"
            v.parent.mkdir(parents=True, exist_ok=True)
            v.write_text(f"{k}:{e}")
    m = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    m.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), m)


def test_reorder_moves_every_index(tmp_path: Path) -> None:
    lengths = [5, 9, 3, 7, 4, 8]
    _make_split(tmp_path, lengths, rows_per_file=14)
    new_of_old = {0: 3, 1: 0, 2: 5, 3: 1, 4: 2, 5: 4}
    reorder_split_episodes(tmp_path, new_of_old)
    old_of_new = {v: k for k, v in new_of_old.items()}

    data = pa.concat_tables(
        [pq.read_table(f) for f in sorted((tmp_path / "data").glob("*/*.parquet"))]
    )
    assert data.column("index").to_pylist() == list(range(sum(lengths)))
    ep, payload = data.column("episode_index").to_numpy(), data.column("payload").to_numpy()
    assert np.all(np.diff(ep) >= 0)
    for e_new in range(len(lengths)):
        rows = payload[ep == e_new]
        o = old_of_new[e_new]
        assert rows.tolist() == [1000 * o + f for f in range(lengths[o])]

    meta = pq.read_table(
        tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ).to_pandas()
    assert meta["episode_index"].tolist() == list(range(len(lengths)))
    assert meta["length"].tolist() == [lengths[old_of_new[e]] for e in range(len(lengths))]
    for _, r in meta.iterrows():
        f = (
            tmp_path
            / "data"
            / f"chunk-{r['data/chunk_index']:03d}"
            / f"file-{r['data/file_index']:03d}.parquet"
        )
        idx = pq.read_table(f, columns=["index", "episode_index"]).to_pandas()
        mine = idx[idx.episode_index == r["episode_index"]]["index"]
        assert (mine.min(), mine.max() + 1) == (r["dataset_from_index"], r["dataset_to_index"])
        assert (
            r["stats/payload/max"][0] == 1000.0 * old_of_new[r["episode_index"]] + r["length"] - 1
        )
        for k in FEATS:
            assert r[f"videos/{k}/file_index"] == r["episode_index"]
            v = tmp_path / "videos" / k / "chunk-000" / f"file-{r['episode_index']:03d}.mp4"
            assert v.read_text() == f"{k}:{old_of_new[r['episode_index']]}"
    assert not list(tmp_path.rglob("*.reorder*"))


def test_a_video_hole_moves_with_its_episode(tmp_path: Path) -> None:
    """replay_job renumbers _video_defects.jsonl with the same map, so the missing file must land
    on the episode's new index and must not be filled by (or overwrite) a neighbour's video."""
    lengths = [4, 6, 5, 3]
    _make_split(tmp_path, lengths, rows_per_file=100)
    hole_feat, hole_old = FEATS[0], 1
    (tmp_path / "videos" / hole_feat / "chunk-000" / f"file-{hole_old:03d}.mp4").unlink()
    new_of_old = {0: 2, 1: 0, 2: 3, 3: 1}
    reorder_split_episodes(tmp_path, new_of_old)
    for k in FEATS:
        for old, new in new_of_old.items():
            v = tmp_path / "videos" / k / "chunk-000" / f"file-{new:03d}.mp4"
            if (k, old) == (hole_feat, hole_old):
                assert not v.exists()
            else:
                assert v.read_text() == f"{k}:{old}"
    assert not list(tmp_path.rglob("*.reorder*"))


def test_rejects_non_permutation(tmp_path: Path) -> None:
    _make_split(tmp_path, [3, 4], rows_per_file=10)
    with pytest.raises(ValueError, match="permutation"):
        reorder_split_episodes(tmp_path, {0: 1, 1: 1})


# --------------------------------------------------- the rank <-> render-uid inversion
def _reorder_map(uid_of_out: dict[int, int]) -> dict[int, int]:
    """What finalize() computes: old per-split output index -> new rank (render.py:3140)."""
    order = sorted(uid_of_out, key=lambda o: uid_of_out[o])
    return {o: r for r, o in enumerate(order)}


def _rank_to_uid(uid_of_out: dict[int, int]) -> list[int]:
    """What the patch pass computes to invert it: rank -> render uid."""
    return sorted(uid_of_out.values())


def test_camera_delta_lookup_survives_the_reorder() -> None:
    """cam_delta's per_ep is keyed by render UID; the patched parquet is keyed by post-reorder rank.

    With one split those coincide. With blocks_success AND blocks_failure they do not: a split's
    uids are a sparse subset, so a rank lookup silently returns ANOTHER episode's camera delta and
    observation.camera_pose describes a camera that never rendered that frame.
    """
    # N=2 over 4 source episodes: uids 0..7, interleaved by per-episode success/failure.
    succeeded = [True, False, True, True]          # src episode -> success?
    success_uids, failure_uids = {}, {}
    for it in range(2):
        for src, ok in enumerate(succeeded):
            uid = it * len(succeeded) + src
            tgt = success_uids if ok else failure_uids
            tgt[len(tgt)] = uid                     # per-split render-order index -> uid

    for split, uid_of_out in (("blocks_success", success_uids),
                              ("blocks_failure", failure_uids)):
        new_of_old = _reorder_map(uid_of_out)
        rank_to_uid = _rank_to_uid(uid_of_out)
        for old_oidx, uid in uid_of_out.items():
            rank = new_of_old[old_oidx]
            assert rank_to_uid[rank] == uid, (split, old_oidx, rank, uid)

    # ...and prove the bug was real: on the success split, looking up by rank (the old code)
    # lands on a DIFFERENT episode's uid for most episodes.
    rank_to_uid = _rank_to_uid(success_uids)
    wrong = [r for r in range(len(rank_to_uid)) if rank_to_uid[r] != r]
    assert len(wrong) >= len(rank_to_uid) // 2, (
        f"expected the rank-keyed lookup to be wrong for most episodes, got {wrong}")


def test_the_patch_pass_actually_inverts_it() -> None:
    """Source-text backstop: the fix is a closure, so pin that the inversion is still wired in."""
    src = RENDER.read_text()
    assert "_rank_to_uid = sorted((uid_of_out or {}).get(split, {}).values())" in src
    assert "_patched_cam_pose(src_rows, c, _uid_at_rank(_rank_to_uid, int(oidx))" in src
    assert "_patched_cam_pose(None, c, _uid_at_rank(_rank_to_uid, int(oidx))" in src
    assert "uid_of_out=_uid_of_out" in src, "the patch pass must receive the map"
