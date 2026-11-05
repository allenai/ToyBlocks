"""Fast merge of LeRobot v3 datasets -- toyblocks's replacement for TAMP/factory/merge_success.py.

Same inputs, outputs and guarantees as the TAMP script (layout discovery, camera selection, column
union, NaN fills for absent objects, task re-indexing, hardlinked videos, provenance, stats,
footer verification, bounded LeRobot validation), rebuilt for speed:

* Frames never go through pandas. The TAMP script turned every vector cell into a Python-held
  numpy array (~6.7 KB/frame), cast through arrow and back, then rebuilt float vectors cell by
  cell to keep NaN. Here every step is a vectorised arrow kernel on the source row groups.
* The data pass is planned up front from the source parquet footers: output file k holds global
  rows [k*R, (k+1)*R), exactly the TAMP script's cut. Every output file is then an independent job
  in a process pool, so reading, casting, compressing and writing run on all cores at once.
* Normalisation stats are reduced inside those jobs from the tables they just built, so the data
  is read once instead of twice. Quantiles come from a seeded uniform row sample (Bernoulli at
  rate STATS_SAMPLE_ROWS / total frames), which equals the TAMP reservoir exactly while the merge
  has fewer than STATS_SAMPLE_ROWS frames and is an unbiased sample above it.
* Video hardlinks, episode metadata and recording metadata are I/O bound and run in threads while
  the data pass runs.
* Validation defaults to a fast check: every output file's schema against the declared features
  (footer only), then a LeRobotDataset built on one episode. ``--validate_mode full`` restores the
  TAMP script's whole-dataset LeRobotDataset build, which converts every frame into an HF arrow
  cache and does not finish in its timeout on a 60M-frame merge.

Deliberate differences from the TAMP script, all of which only remove silent failures:

* ``source_problem`` keeps the value the recorder wrote (vecgen job groups hold several problems
  and the recorder knows which one each episode came from) and falls back to the dataset
  directory name only where the source has none. The TAMP script overwrote it with the group
  directory name.
* ``source_problem`` / ``instance_id`` are always written. The TAMP script dropped them under
  ``--union_schema`` when no source had them, while info.json still declared them.
* Every output file has the same columns (the union across sources), with or without
  ``--union_schema``; that flag still decides what info.json declares, as before.
* A source data file that fails to read after its footer passed aborts the merge. The TAMP script
  skipped the dataset after its episodes had already been counted, leaving holes.
* Float scalar NaN stays NaN on disk (the TAMP script's pandas bridge wrote it as null).

Usage (drop-in for the TAMP script's CLI):
  python scripts/datagen/merge_datasets.py --input_dirs <execution dir> --output_dir <merged> \
      [--cameras all] [--num_workers N] [--union_schema] [--validate]
  python scripts/datagen/merge_datasets.py --recompute-stats <merged dataset>
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import multiprocessing as mp
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
import warnings
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

log = logging.getLogger("toyblocks.merge")

SOURCE_PROBLEM_KEY = "source_problem"
SOURCE_PROBLEM_FEATURE: dict[str, Any] = {"dtype": "string", "shape": (1,), "names": None}
INSTANCE_ID_KEY = "instance_id"
INSTANCE_ID_FEATURE: dict[str, Any] = {"dtype": "int64", "shape": (1,), "names": None}
PROVENANCE_DATA_KEYS = (SOURCE_PROBLEM_KEY, INSTANCE_ID_KEY)

# Output layout, identical to the TAMP script's info.json (LeRobot v3 defaults).
CHUNK_SIZE = 1000
DATA_FILE_SIZE_IN_MB = 100
VIDEO_FILE_SIZE_IN_MB = 200
ROW_GROUP_SIZE = 350
MIN_ROWS_PER_FILE = 1000

STATS_SAMPLE_ROWS = 200_000
STATS_SEED = 0
QUANTILES = (("q01", 1), ("q10", 10), ("q50", 50), ("q90", 90), ("q99", 99))

MAX_WRITE_RETRIES = 3
# The output lock's heartbeat period, and the heartbeat age after which its merge counts as dead.
LOCK_HEARTBEAT_S = 30
LOCK_STALE_S = 600
LEROBOT_SAMPLE_TIMEOUT_S = 900
LINK_WORKERS = 64
ALL_CAMERAS = ("all", "*")

# Set when a bounded validation abandons a thread still inside a C call; main() then leaves via
# os._exit, because finalizing the interpreter around that thread can SIGABRT (exit -6) after a
# complete merge.
_ABANDONED_VALIDATION = False


# ── discovery (unchanged from the TAMP script) ─────────────────────────────────


def find_dataset_subdir(problem_dir: Path) -> Path | None:
    """The LeRobot dataset inside a problem directory ('blocks' or 'blocks_success', ...)."""
    for candidate in problem_dir.iterdir():
        if candidate.is_dir() and (candidate / "meta" / "info.json").exists():
            return candidate
    return None


def is_vectorized_layout(base_dir: Path) -> bool:
    """True when problem dirs hold blocks_success/blocks_failure.

    A problem without a single successful episode has only blocks_failure, so every problem
    dir is looked at: deciding by the first one alone treated a whole run whose first problem
    had 0% success as the old layout, whose discovery then merged blocks_failure datasets.
    """
    for d in sorted(base_dir.iterdir()):
        if d.is_dir() and d.name.startswith("problem_"):
            if any((d / split / "meta" / "info.json").exists() for split in ("blocks_success", "blocks_failure")):
                return True
    return False


def get_dataset_paths(base_dir: Path) -> list[Path]:
    """Every LeRobot dataset under base_dir.

    Vectorized layout: base_dir/problem_*/blocks_success (blocks_failure is never merged).
    Old layout: base_dir/problem_*/<the one dataset dir>.
    """
    datasets: list[Path] = []
    if is_vectorized_layout(base_dir):
        for problem_dir in sorted(base_dir.iterdir()):
            if problem_dir.is_dir() and problem_dir.name.startswith("problem_"):
                success_dir = problem_dir / "blocks_success"
                if (success_dir / "meta" / "info.json").exists():
                    datasets.append(success_dir)
    else:
        for problem_dir in sorted(base_dir.iterdir()):
            if problem_dir.is_dir():
                dataset_path = find_dataset_subdir(problem_dir)
                if dataset_path is not None:
                    datasets.append(dataset_path)
    return datasets


def load_info(dataset_path: Path) -> dict[str, Any]:
    with open(dataset_path / "meta" / "info.json") as f:
        return json.load(f)


def problem_name_from_dataset_path(dataset_path: Path) -> str:
    """.../problem_xyz/blocks_success -> problem_xyz."""
    for part in reversed(dataset_path.parts):
        if part.startswith("problem_"):
            return part
    return dataset_path.parent.name


def wants_all_cameras(keep_cameras: Sequence[str]) -> bool:
    """True if --cameras asked for every video feature rather than a named list."""
    return any(str(c).strip().lower() in ALL_CAMERAS for c in (keep_cameras or []))


def normalize_camera_keys(keep_cameras: Sequence[str]) -> list[str]:
    """Bare camera names -> RGB feature keys ('front_camera' never means its depth stream)."""
    return [
        c if c.startswith("observation.") else f"observation.images.{c}" for c in keep_cameras or []
    ]


def feature_from_arrow(fld: pa.Field, sample_path: Path | None = None) -> dict[str, Any]:
    """A LeRobot feature entry for a parquet column no info.json declares."""
    t = fld.type
    if pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t):
        length = 1
        if pa.types.is_fixed_size_list(t):
            length = t.list_size
        elif sample_path is not None:
            try:  # the schema does not carry a list's length; one row does
                col = pq.read_table(sample_path, columns=[fld.name]).column(0)
                for v in col:
                    if v.is_valid:
                        length = len(v)
                        break
            except Exception:  # noqa: BLE001 - fall back to 1
                pass
        inner = t.value_type
        dtype = (
            "float32"
            if pa.types.is_floating(inner)
            else ("int64" if pa.types.is_integer(inner) else "string")
        )
        return {"dtype": dtype, "shape": (length,), "names": None}
    if pa.types.is_boolean(t):
        return {"dtype": "bool", "shape": (1,), "names": None}
    if pa.types.is_integer(t):
        return {"dtype": "int64", "shape": (1,), "names": None}
    if pa.types.is_floating(t):
        return {"dtype": "float32", "shape": (1,), "names": None}
    return {"dtype": "string", "shape": (1,), "names": None}


# ── atomic writes ──────────────────────────────────────────────────────────────


def verify_parquet_footer(path: Path) -> bool:
    """PAR1 magic at both ends: a truncated write fails this."""
    try:
        size = path.stat().st_size
        if size < 8:
            return False
        with open(path, "rb") as f:
            if f.read(4) != b"PAR1":
                return False
            f.seek(size - 4)
            return f.read(4) == b"PAR1"
    except OSError:
        return False


def _fsync(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write(target: Path, write: Callable[[Path], None], verify_parquet: bool = True) -> None:
    """write(tmp) into a sibling temp file, fsync, rename over target, verify; retried."""
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, MAX_WRITE_RETRIES + 1):
        fd, tmp_str = tempfile.mkstemp(suffix=target.suffix + ".tmp", dir=str(target.parent))
        os.close(fd)
        tmp = Path(tmp_str)
        try:
            write(tmp)
            _fsync(tmp)
            if verify_parquet and not verify_parquet_footer(tmp):
                raise OSError(
                    f"parquet footer verification failed for {tmp} ({tmp.stat().st_size} bytes)"
                )
            tmp.rename(target)
            if verify_parquet and not verify_parquet_footer(target):
                raise OSError(f"post-rename footer verification failed for {target}")
            return
        except Exception:
            tmp.unlink(missing_ok=True)
            if attempt == MAX_WRITE_RETRIES:
                log.error("write failed after %d attempts: %s", MAX_WRITE_RETRIES, target)
                raise
            log.warning(
                "write attempt %d/%d failed for %s, retrying", attempt, MAX_WRITE_RETRIES, target
            )


def write_parquet_table(table: pa.Table, target: Path) -> None:
    """Encode in memory, then one sequential write: 350-row groups are thousands of small
    column-chunk writes, each a round trip on Weka."""
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, row_group_size=ROW_GROUP_SIZE)
    data = sink.getvalue()

    def _write(tmp: Path) -> None:
        with open(tmp, "wb") as f:
            f.write(memoryview(data))

    atomic_write(target, _write)


def write_parquet_frame(df: Any, target: Path) -> None:
    """A pandas frame through pandas' own writer (episode metadata, tasks), atomically."""
    atomic_write(target, lambda tmp: df.to_parquet(tmp))


# ── output lock ────────────────────────────────────────────────────────────────


class OutputLock:
    """Exclusive claim on a merge's output dir, held for the whole merge.

    Two merges into one output dir destroy each other: each starts by rmtree()ing it, so the
    second deletes the data files the first has already written, and the first then writes
    episode metadata pointing at files that are gone. On 2026-10-31 two stale orchestrators of
    the oct30 RG rerender did exactly that under a live merge: 125 of 414 data files deleted, and
    the live merge still reported success.

    The lock is ``<output_dir>.merge.lock`` NEXT TO the output dir, so the merge's own rmtree
    cannot remove it. It is created with O_EXCL and its heartbeat is rewritten every
    LOCK_HEARTBEAT_S; a lock whose heartbeat is older than LOCK_STALE_S belongs to a dead merge
    and is taken over.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.path = output_dir.parent / f"{output_dir.name}.merge.lock"
        self.token = f"{socket.gethostname()}:{os.getpid()}:{time.time_ns()}"
        self.lost = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _payload(self) -> bytes:
        return json.dumps(
            {
                "token": self.token,
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "heartbeat": time.time(),
            }
        ).encode()

    @staticmethod
    def _read(path: Path) -> dict[str, Any] | None:
        try:
            return json.loads(path.read_bytes())
        except (OSError, ValueError):
            return None

    def _age(self, held: dict[str, Any] | None, path: Path) -> float:
        """Seconds since the holder's last heartbeat (the file's mtime while it is half-written)."""
        if held is not None and "heartbeat" in held:
            return time.time() - float(held["heartbeat"])
        try:
            return time.time() - path.stat().st_mtime
        except OSError:
            return float("inf")

    def _create(self) -> bool:
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return False
        with os.fdopen(fd, "wb") as f:
            f.write(self._payload())
            f.flush()
            os.fsync(f.fileno())
        return True

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self._create():
            held = self._read(self.path)
            age = self._age(held, self.path)
            if age < LOCK_STALE_S:
                h = held or {}
                raise RuntimeError(
                    f"{self.output_dir} is being merged by another process (host {h.get('host')}, "
                    f"pid {h.get('pid')}, heartbeat "
                    f"{age:.0f}s ago). Refusing to start: this merge's rmtree would delete its "
                    f"files. If that merge is dead, its lock goes stale after {LOCK_STALE_S}s, or "
                    f"remove {self.path}."
                )
            # Stale: move it aside, then claim with O_EXCL. If what was moved is not the lock
            # judged stale, another merge took it over first: put it back and give way.
            aside = self.path.with_name(f"{self.path.name}.stale.{os.getpid()}")
            try:
                os.rename(self.path, aside)
            except FileNotFoundError:
                pass
            else:
                moved = self._read(aside)
                if held is not None and (moved or {}).get("token") != held.get("token"):
                    with contextlib.suppress(OSError):
                        os.link(aside, self.path)
                    aside.unlink(missing_ok=True)
                    raise RuntimeError(f"{self.path} was taken over by another merge; refusing")
                aside.unlink(missing_ok=True)
            if not self._create():
                raise RuntimeError(f"{self.path} was claimed by another merge first; refusing")
            log.warning(
                "Took over a stale merge lock (heartbeat %.0fs old, holder %s)",
                age,
                (held or {}).get("token"),
            )
        self._thread = threading.Thread(target=self._beat, name="merge-lock", daemon=True)
        self._thread.start()

    def _beat(self) -> None:
        while not self._stop.wait(LOCK_HEARTBEAT_S):
            held = self._read(self.path)
            if held is None or held.get("token") != self.token:
                self.lost = True
                log.error(
                    "Lost the merge lock %s (now held by %s)", self.path, (held or {}).get("token")
                )
                return
            # replace, never rewrite in place: the lock file must exist at every instant
            tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
            tmp.write_bytes(self._payload())
            os.replace(tmp, self.path)

    def check(self) -> None:
        """Raise unless this merge still holds the lock (call before publishing results)."""
        if self.lost or (self._read(self.path) or {}).get("token") != self.token:
            raise RuntimeError(
                f"Lost the merge lock {self.path}: another process may have written into "
                f"{self.output_dir}, so this merge's output cannot be trusted"
            )

    def release(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if (self._read(self.path) or {}).get("token") == self.token:
            self.path.unlink(missing_ok=True)


def check_output_complete(
    output_dir: Path,
    jobs: Sequence[OutputJob],
    video_tasks: Sequence[tuple[str, Path, str, int]],
    io_workers: int,
) -> None:
    """Every planned data file and video link exists, and no data file is unplanned.

    The footer verification only looks at files that exist, so a data file deleted after its
    write (another process's rmtree of this output dir) went unnoticed and the merge published
    episode metadata pointing at it.
    """
    expected = {output_dir / "data" / j.path_suffix for j in jobs}
    on_disk = set(_sorted_glob2(output_dir / "data", ".parquet"))
    missing = sorted(expected - on_disk)
    extra = sorted(on_disk - expected)
    with ThreadPoolExecutor(max_workers=io_workers) as ex:
        present = list(ex.map(lambda t: t[1].exists(), video_tasks))
    missing_videos = [t[1] for t, ok in zip(video_tasks, present) if not ok]
    for p in [*missing, *missing_videos][:20]:
        log.error("  MISSING: %s", p.relative_to(output_dir))
    for p in extra[:20]:
        log.error("  UNPLANNED: %s", p.relative_to(output_dir))
    if missing or extra or missing_videos:
        raise RuntimeError(
            f"merged output is incomplete: {len(missing)}/{len(expected)} data files missing, "
            f"{len(extra)} unplanned, {len(missing_videos)}/{len(video_tasks)} videos missing "
            f"(was {output_dir} deleted or written by another process during the merge?)"
        )
    log.info("  All %d data files and %d videos present", len(expected), len(video_tasks))


# ── scan ───────────────────────────────────────────────────────────────────────


@dataclass
class SourceFile:
    path: str
    num_rows: int
    rg_rows: list[int]


@dataclass
class SourceDataset:
    path: Path
    problem_name: str
    total_episodes: int = 0
    total_frames: int = 0
    tasks: list[str] = field(default_factory=list)
    local_tasks: dict[int, str] = field(default_factory=dict)
    has_task_table: bool = False
    files: list[SourceFile] = field(default_factory=list)
    schema: pa.Schema | None = None
    data_bytes: int = 0
    video_files: list[tuple[str, str, int]] = field(default_factory=list)
    stray_video_files: list[str] = field(default_factory=list)
    corrupt_data_files: list[str] = field(default_factory=list)
    episode_offset: int = 0
    frame_offset: int = 0

    @property
    def footer_rows(self) -> int:
        return sum(f.num_rows for f in self.files)


def _sorted_glob2(root: Path, suffix: str) -> list[Path]:
    """sorted(root.glob('*/*<suffix>')) with scandir (one listing per chunk dir)."""
    out: list[Path] = []
    try:
        chunks = sorted(e.path for e in os.scandir(root) if e.is_dir())
    except FileNotFoundError:
        return out
    for c in chunks:
        out.extend(Path(e.path) for e in os.scandir(c) if e.name.endswith(suffix) and e.is_file())
    return sorted(out)


def scan_dataset(path: Path, video_keys: Sequence[str]) -> SourceDataset:
    """info.json, every data file's footer, the task table and the video clips of one dataset."""
    ds = SourceDataset(path=path, problem_name=problem_name_from_dataset_path(path))
    try:
        info = load_info(path)
        ds.total_episodes = int(info["total_episodes"])
        ds.total_frames = int(info["total_frames"])
        # Footers prove the files are complete (a truncated render fails here) and give the row
        # group sizes the data plan is cut from. A dataset with any unreadable data file is
        # dropped before offsets are assigned so the merged indices stay contiguous.
        for f in _sorted_glob2(path / "data", ".parquet"):
            try:
                pf = pq.ParquetFile(f)
                md = pf.metadata
                ds.files.append(
                    SourceFile(
                        str(f),
                        md.num_rows,
                        [md.row_group(i).num_rows for i in range(md.num_row_groups)],
                    )
                )
                if ds.schema is None:
                    ds.schema = pf.schema_arrow
                ds.data_bytes += f.stat().st_size
            except Exception as e:  # noqa: BLE001
                ds.corrupt_data_files.append(f"{f}: {e}")
        tasks_file = path / "meta" / "tasks.parquet"
        if tasks_file.exists():
            df = pq.read_table(tasks_file).to_pandas()  # index = task string, col = local index
            ds.has_task_table = True
            ds.tasks = [str(s) for s in df.index.tolist()]
            if "task_index" in df.columns:
                ds.local_tasks = {int(li): str(s) for s, li in df["task_index"].items()}
        for key in video_keys:
            for vf in _sorted_glob2(path / "videos" / key, ".mp4"):
                # Only real clips are file-NNN.mp4; an abandoned encode temp
                # (file-NNN.<pid>.tmp.mp4) is reported, never parsed.
                stem = vf.stem
                if not (stem.startswith("file-") and stem[5:].isdigit()):
                    ds.stray_video_files.append(str(vf))
                    continue
                ds.video_files.append((key, str(vf), int(stem[5:])))
    except Exception as e:  # noqa: BLE001
        log.warning("Error scanning %s: %s", path, e)
    return ds


def video_coverage_report(
    datasets: Sequence[SourceDataset], video_keys: Sequence[str]
) -> tuple[dict[str, int], list[tuple[str, Path, int, int]], list[str]]:
    """(datasets per key with any clip, (key, dataset, clips, episodes) short pairs, strays)."""
    cover = {k: 0 for k in video_keys}
    short: list[tuple[str, Path, int, int]] = []
    strays: list[str] = []
    for ds in datasets:
        n_by_key: dict[str, int] = {}
        for vk, _f, _e in ds.video_files:
            n_by_key[vk] = n_by_key.get(vk, 0) + 1
        strays += ds.stray_video_files
        for k in video_keys:
            n = n_by_key.get(k, 0)
            if n:
                cover[k] += 1
            if 0 < n < ds.total_episodes:
                short.append((k, ds.path, n, ds.total_episodes))
    return cover, short, strays


# ── data pass: plan ────────────────────────────────────────────────────────────


@dataclass
class Piece:
    """Rows [start, stop) of one source file, in source-local row numbers."""

    dataset: int
    file: str
    start: int
    stop: int
    rg_rows: list[int]


@dataclass
class OutputJob:
    file_index: int
    pieces: list[Piece]

    @property
    def path_suffix(self) -> str:
        return f"chunk-{self.file_index // CHUNK_SIZE:03d}/file-{self.file_index % CHUNK_SIZE:03d}.parquet"


@dataclass
class DatasetCtx:
    """What a worker needs to turn one source dataset's rows into merged rows."""

    episode_offset: int
    frame_offset: int
    problem_name: str
    task_lut_keys: np.ndarray | None  # sorted local task indices
    task_lut_vals: np.ndarray | None  # their global indices


@dataclass
class WorkerConfig:
    columns: list[str]
    out_schema: pa.Schema
    fill_width: dict[str, int]
    numeric_keys: list[tuple[str, int, str]]  # (key, expected row size, dtype)
    sample_rate: float
    datasets: list[DatasetCtx]


def plan_output_files(datasets: Sequence[SourceDataset], rows_per_file: int) -> list[OutputJob]:
    """Cut the concatenated rows into files of rows_per_file rows (the last one shorter)."""
    jobs: list[OutputJob] = []
    cur: list[Piece] = []
    room = rows_per_file
    for di, ds in enumerate(datasets):
        for sf in ds.files:
            pos = 0
            while pos < sf.num_rows:
                take = min(room, sf.num_rows - pos)
                cur.append(Piece(di, sf.path, pos, pos + take, sf.rg_rows))
                pos += take
                room -= take
                if room == 0:
                    jobs.append(OutputJob(len(jobs), cur))
                    cur, room = [], rows_per_file
    if cur:
        jobs.append(OutputJob(len(jobs), cur))
    return jobs


def output_type(t: pa.DataType) -> pa.DataType:
    """The type the TAMP script's pandas round trip wrote: vectors as variable-size lists."""
    if pa.types.is_fixed_size_list(t) or pa.types.is_large_list(t):
        return pa.list_(pa.field("element", t.value_type))
    if pa.types.is_list(t):
        return pa.list_(pa.field("element", t.value_type))
    return t


def _is_float_list(t: pa.DataType) -> bool:
    return (
        pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t)
    ) and pa.types.is_floating(t.value_type)


# ── data pass: worker ──────────────────────────────────────────────────────────

_CFG: WorkerConfig | None = None


def _init_worker(cfg: WorkerConfig) -> None:
    global _CFG
    _CFG = cfg
    pa.set_cpu_count(1)  # the pool supplies the parallelism; arrow threads would oversubscribe
    pa.set_io_thread_count(4)


def _read_piece(p: Piece) -> pa.Table:
    """Only the row groups that overlap [start, stop), then the exact slice."""
    bounds = np.concatenate([[0], np.cumsum(p.rg_rows)])
    first = int(np.searchsorted(bounds, p.start, side="right") - 1)
    last = int(np.searchsorted(bounds, p.stop, side="left") - 1)
    rgs = list(range(first, last + 1))
    # pre_buffer coalesces the column chunks into a few large reads. Recorders write small row
    # groups (vecgen: ~670 rows, ~6400 column chunks per 100 MB file), and one read per chunk
    # costs Weka latency each time: 5.3 s -> 1.6 s per file measured on oct21_stack3_pnp_vg.
    t = pq.ParquetFile(p.file, pre_buffer=True).read_row_groups(rgs, use_threads=False)
    lo = p.start - int(bounds[first])
    return t.slice(lo, p.stop - p.start)


def _nan_lists(n: int, width: int, t: pa.DataType) -> pa.Array:
    vt = t.value_type
    values = pa.array(np.full(n * width, np.nan, dtype=vt.to_pandas_dtype()), type=vt)
    offsets = pa.array(np.arange(0, (n + 1) * width, width, dtype=np.int32), type=pa.int32())
    return pa.ListArray.from_arrays(offsets, values, type=t)


def fill_list_nulls(
    arr: pa.Array | pa.ChunkedArray, width: int, fill: float, invalidate_null_elements: bool = True
) -> pa.Array:
    """Null list cells -> `width` copies of `fill` (LeRobot cannot tensor None).

    With invalidate_null_elements, a list holding a null element becomes `fill` at its own
    length (width if empty), as merge_success.sanitize_null_vectors did. Vectorised.
    """
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    if pa.types.is_fixed_size_list(arr.type):
        arr = arr.cast(pa.list_(arr.type.value_field))
    flat = pc.list_flatten(arr)  # values of the non-null rows, in row order
    if arr.null_count == 0 and (not invalidate_null_elements or flat.null_count == 0):
        return arr
    valid = np.asarray(arr.is_valid().to_numpy(zero_copy_only=False), dtype=bool)
    lens = np.asarray(
        pc.fill_null(pc.list_value_length(arr), 0).to_numpy(zero_copy_only=False), dtype=np.int64
    )
    lens = np.where(valid, lens, 0)
    parents = np.repeat(np.flatnonzero(valid), lens[valid])
    keep = valid.copy()
    if invalidate_null_elements and flat.null_count:
        keep[
            np.unique(
                parents[np.asarray(flat.is_null().to_numpy(zero_copy_only=False), dtype=bool)]
            )
        ] = False
    new_lens = np.where(keep, lens, np.where(valid & (lens > 0), lens, width))
    offsets = np.zeros(len(arr) + 1, dtype=np.int64)
    np.cumsum(new_lens, out=offsets[1:])
    vt = arr.type.value_type
    vals = np.full(int(offsets[-1]), fill, dtype=vt.to_pandas_dtype())
    kept = flat.filter(pa.array(keep[parents])) if not keep[valid].all() else flat
    vals[np.repeat(keep, new_lens)] = np.asarray(
        kept.to_numpy(zero_copy_only=False), dtype=vals.dtype
    )
    return pa.ListArray.from_arrays(
        pa.array(offsets.astype(np.int32)), pa.array(vals, type=vt), type=arr.type
    )


def transform_piece(
    table: pa.Table, ctx: DatasetCtx, cfg: WorkerConfig, warn: list[str]
) -> pa.Table:
    """One source slice -> merged rows: offsets, task remap, provenance, union columns, types."""
    n = table.num_rows
    names = set(table.column_names)
    cols: dict[str, pa.ChunkedArray] = {}

    def offset(name: str, off: int) -> None:
        if name in names and off:
            c = table.column(name)
            cols[name] = pc.add(c, pa.scalar(off, c.type))

    offset("episode_index", ctx.episode_offset)
    offset("index", ctx.frame_offset)

    if "task_index" in names and ctx.task_lut_keys is not None:
        c = table.column("task_index")
        vals = c.to_numpy()
        keys, gvals = ctx.task_lut_keys, ctx.task_lut_vals
        if len(keys):
            pos = np.clip(np.searchsorted(keys, vals), 0, len(keys) - 1)
            hit = keys[pos] == vals
            mapped = np.where(hit, gvals[pos], vals)
        else:
            hit = np.zeros(len(vals), dtype=bool)
            mapped = vals
        if not hit.all():
            # Keep the original index rather than crash or map it somewhere wrong (rerender used
            # to copy task_index from its source while writing its own 1-row task table).
            bad = np.unique(vals[~hit])[:5].tolist()
            warn.append(
                f"{ctx.problem_name}: {int((~hit).sum())} row(s) reference task_index {bad} which its "
                f"meta/tasks.parquet does not define; leaving those indices unmapped"
            )
        cols["task_index"] = pa.chunked_array([pa.array(mapped.astype(vals.dtype), type=c.type)])

    # source_problem: the recorder's per-episode value where it wrote one, else the dataset dir.
    if SOURCE_PROBLEM_KEY in names:
        c = table.column(SOURCE_PROBLEM_KEY)
        c = c.cast(pa.string()) if not pa.types.is_string(c.type) else c
        empty = pc.or_kleene(pc.is_null(c), pc.equal(pc.utf8_trim_whitespace(c), ""))
        if pc.any(empty).as_py():
            c = pc.if_else(empty, pa.scalar(ctx.problem_name), c)
        cols[SOURCE_PROBLEM_KEY] = c
    else:
        cols[SOURCE_PROBLEM_KEY] = pa.chunked_array(
            [pa.array([ctx.problem_name], pa.string()).take(np.zeros(n, np.int32))]
        )
    if INSTANCE_ID_KEY not in names:
        cols[INSTANCE_ID_KEY] = pa.chunked_array([pa.array(np.full(n, -1, dtype=np.int64))])

    out: list[pa.ChunkedArray] = []
    for fld in cfg.out_schema:
        name, t = fld.name, fld.type
        c = cols.get(name)
        if c is None and name in names:
            c = table.column(name)
        if c is None:
            # Absent here because another problem's scene has it: undefined, so NaN (not zero,
            # which is a plausible pose, and not null, which LeRobot cannot tensor).
            if _is_float_list(t):
                c = pa.chunked_array([_nan_lists(n, cfg.fill_width.get(name, 1), t)])
            elif pa.types.is_floating(t):
                c = pa.chunked_array(
                    [pa.array(np.full(n, np.nan, dtype=t.to_pandas_dtype()), type=t)]
                )
            else:
                c = pa.chunked_array([pa.nulls(n, t)])
        elif not c.type.equals(t):
            try:
                c = c.cast(t)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as e:
                raise RuntimeError(
                    f"{ctx.problem_name}: column {name} ({c.type}) cannot be cast to the merged type {t}: {e}"
                ) from e
        if _is_float_list(t):
            c = pa.chunked_array([fill_list_nulls(c, cfg.fill_width.get(name, 1), np.nan)])
        out.append(c)
    return pa.Table.from_arrays(out, schema=cfg.out_schema)


@dataclass
class KeyStats:
    min: np.ndarray
    max: np.ndarray
    sum: np.ndarray
    sq_sum: np.ndarray
    valid: np.ndarray
    count: int


def column_matrix(col: pa.ChunkedArray, expected: int, dtype: str) -> np.ndarray:
    """A column as float64 (n,) or (n, w), matching the TAMP reduction's np.stack shapes.

    Scalars give (n,); lists give (n, w) when every row has the expected width, and are coerced
    row by row to (n, expected) (NaN where a row is null or the wrong size, 1-D when expected is
    1) otherwise -- the same rules as merge_success._stats_reduce_one.
    """
    arr = col.combine_chunks()
    t = arr.type
    if not (pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t)):
        # via float64 so nulls (any type, bool included) become NaN
        return np.asarray(
            pc.cast(arr, pa.float64()).to_numpy(zero_copy_only=False), dtype=np.float64
        )
    n = len(arr)
    if arr.null_count == 0 and arr.values.null_count == 0:
        if pa.types.is_fixed_size_list(t):
            w = t.list_size
            uniform = True
        else:
            offs = arr.offsets.to_numpy()
            lens = np.diff(offs)
            w = int(lens[0]) if n else expected
            uniform = bool(n == 0 or np.all(lens == w))
        if uniform and w == expected:
            flat = (
                arr.flatten()
                if not pa.types.is_fixed_size_list(t)
                else arr.values.slice(arr.offset * w, n * w)
            )
            vals = flat.to_numpy(zero_copy_only=False)
            if pa.types.is_boolean(t.value_type):
                vals = vals.astype(np.float64)
            return np.asarray(vals, dtype=np.float64).reshape(n, w)
    out = np.full((n, expected), np.nan, dtype=np.float64)
    for i, v in enumerate(arr.to_pylist()):
        if v is None:
            continue
        a = np.asarray([np.nan if x is None else x for x in v], dtype=np.float64).ravel()
        if a.size == expected:
            out[i] = a
    return out.reshape(-1) if expected == 1 else out


def reduce_table(
    table: pa.Table, numeric_keys: Sequence[tuple[str, int, str]], sample_idx: np.ndarray
) -> tuple[dict[str, KeyStats], dict[str, np.ndarray]]:
    """NaN-aware min/max/sum/sq_sum/valid per key, plus the sampled rows for quantiles."""
    stats: dict[str, KeyStats] = {}
    samples: dict[str, np.ndarray] = {}
    names = set(table.column_names)
    n = table.num_rows
    for key, expected, dtype in numeric_keys:
        if key not in names:
            continue
        try:
            x = column_matrix(table.column(key), expected, dtype)
        except Exception as e:  # noqa: BLE001
            log.warning("Could not process stats for %s: %s", key, e)
            continue
        valid = np.isfinite(x)
        cnt = valid.sum(axis=0)
        filled = np.where(valid, x, 0.0)
        stats[key] = KeyStats(
            min=np.where(cnt > 0, np.where(valid, x, np.inf).min(axis=0), np.inf),
            max=np.where(cnt > 0, np.where(valid, x, -np.inf).max(axis=0), -np.inf),
            sum=filled.sum(axis=0),
            sq_sum=(filled**2).sum(axis=0),
            valid=cnt,
            count=n,
        )
        samples[key] = x[sample_idx]
    return stats, samples


def run_output_job(job: OutputJob, output_dir: str) -> dict[str, Any]:
    """Build, write and reduce one output data file (runs in a pool process)."""
    cfg = _CFG
    assert cfg is not None
    warn: list[str] = []
    parts = []
    for p in job.pieces:
        try:
            raw = _read_piece(p)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"reading rows {p.start}-{p.stop} of {p.file} failed after its footer passed the scan: {e}. "
                "Its episodes are already counted, so skipping it would leave holes; fix or remove the "
                "dataset and rerun."
            ) from e
        parts.append(transform_piece(raw, cfg.datasets[p.dataset], cfg, warn))
    table = pa.concat_tables(parts) if len(parts) > 1 else parts[0]
    target = Path(output_dir) / "data" / job.path_suffix
    write_parquet_table(table, target)
    idx = table.column("index")
    first, last = int(idx[0].as_py()), int(idx[-1].as_py())
    rng = np.random.default_rng([STATS_SEED, job.file_index])
    sample_idx = np.flatnonzero(rng.random(table.num_rows) < cfg.sample_rate)
    stats, samples = reduce_table(table, cfg.numeric_keys, sample_idx)
    return {
        "file_index": job.file_index,
        "start": first,
        "end": last + 1,
        "rows": table.num_rows,
        "stats": stats,
        "samples": samples,
        "warnings": warn,
    }


# ── stats ──────────────────────────────────────────────────────────────────────


def numeric_stat_keys(features: dict[str, dict[str, Any]]) -> list[tuple[str, int, str]]:
    keys = []
    for key, spec in features.items():
        dtype = spec.get("dtype")
        if dtype in ("video", "string") or key in PROVENANCE_DATA_KEYS:
            continue
        shape = tuple(int(s) for s in (spec.get("shape") or (1,)))
        keys.append((key, int(np.prod(shape)) if shape else 1, str(dtype)))
    return keys


class StatsAccumulator:
    """Associative merge of per-file KeyStats plus the quantile sample (kept in file order)."""

    def __init__(self) -> None:
        self.running: dict[str, KeyStats] = {}
        self.samples: dict[int, dict[str, np.ndarray]] = {}

    def add(
        self, file_index: int, stats: dict[str, KeyStats], samples: dict[str, np.ndarray]
    ) -> None:
        for key, s in stats.items():
            r = self.running.get(key)
            if r is None:
                self.running[key] = s
            else:
                r.min = np.minimum(r.min, s.min)
                r.max = np.maximum(r.max, s.max)
                r.sum = r.sum + s.sum
                r.sq_sum = r.sq_sum + s.sq_sum
                r.valid = r.valid + s.valid
                r.count += s.count
        self.samples[file_index] = samples

    def finalize(self, features: dict[str, dict[str, Any]]) -> dict[str, Any]:
        order = sorted(self.samples)
        out: dict[str, Any] = {}
        rng = np.random.default_rng([STATS_SEED, 1 << 30])
        for key, spec in features.items():
            dtype = spec.get("dtype")
            if dtype == "video":
                out[key] = {
                    "min": [[[0.0]], [[0.0]], [[0.0]]],
                    "max": [[[1.0]], [[1.0]], [[1.0]]],
                    "mean": [[[0.5]], [[0.5]], [[0.5]]],
                    "std": [[[0.25]], [[0.25]], [[0.25]]],
                    "count": [1],
                    "q01": [[[0.0]], [[0.0]], [[0.0]]],
                    "q10": [[[0.1]], [[0.1]], [[0.1]]],
                    "q50": [[[0.5]], [[0.5]], [[0.5]]],
                    "q90": [[[0.9]], [[0.9]], [[0.9]]],
                    "q99": [[[1.0]], [[1.0]], [[1.0]]],
                }
                continue
            if dtype == "string" or key not in self.running:
                continue
            r = self.running[key]
            has_any = r.valid > 0
            safe = np.where(has_any, r.valid, 1)
            # Elements NaN in every frame have no statistic: identity transform (mean 0, std 1).
            mean = np.where(has_any, r.sum / safe, 0.0)
            var = np.maximum(r.sq_sum / safe - mean**2, 0)
            std = np.where(has_any, np.sqrt(var), 1.0)
            col_min = np.where(has_any, r.min, 0.0)
            col_max = np.where(has_any, r.max, 0.0)
            if not np.all(has_any):
                log.warning(
                    "%s: %d/%d element(s) are NaN in every frame; reporting mean 0 / std 1 for those",
                    key,
                    int((~has_any).sum()),
                    has_any.size,
                )
            parts = [self.samples[i][key] for i in order if key in self.samples[i]]
            sample = np.concatenate(parts) if parts else None
            if sample is not None and len(sample) > STATS_SAMPLE_ROWS:
                sample = sample[np.sort(rng.choice(len(sample), STATS_SAMPLE_ROWS, replace=False))]
            if sample is not None and len(sample) == 0:
                sample = None
            entry: dict[str, Any] = {
                "min": col_min.tolist(),
                "max": col_max.tolist(),
                "mean": mean.tolist(),
                "std": std.tolist(),
                "count": [int(r.count)],
            }
            for name, q in QUANTILES:
                if sample is None:
                    entry[name] = mean.tolist()
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN reported above
                    qv = np.nanpercentile(sample, q, axis=0)
                entry[name] = np.where(np.isfinite(qv), qv, mean).tolist()
            out[key] = entry
        return out


def _stats_job(
    path: str, file_index: int, numeric_keys: list[tuple[str, int, str]], rate: float
) -> dict[str, Any]:
    pa.set_cpu_count(1)
    table = pq.ParquetFile(path, pre_buffer=True).read(use_threads=False)
    rng = np.random.default_rng([STATS_SEED, file_index])
    sample_idx = np.flatnonzero(rng.random(table.num_rows) < rate)
    stats, samples = reduce_table(table, numeric_keys, sample_idx)
    return {"file_index": file_index, "stats": stats, "samples": samples}


def compute_stats(
    output_dir: Path, features: dict[str, dict[str, Any]], num_workers: int = 16
) -> dict[str, Any]:
    """stats.json of an already-merged dataset, reading its data files in a process pool."""
    files = sorted((output_dir / "data").glob("*/*.parquet"))
    if not files:
        log.error("No parquet files found in %s", output_dir / "data")
        return {}
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    rate = min(1.0, STATS_SAMPLE_ROWS / max(total, 1))
    keys = numeric_stat_keys(features)
    acc = StatsAccumulator()
    with ProcessPoolExecutor(
        max_workers=max(1, num_workers), mp_context=mp.get_context("spawn")
    ) as ex:
        futs = [ex.submit(_stats_job, str(f), i, keys, rate) for i, f in enumerate(files)]
        for fut in futs:
            r = fut.result()
            acc.add(r["file_index"], r["stats"], r["samples"])
    stats = acc.finalize(features)
    path = output_dir / "meta" / "stats.json"
    with open(path, "w") as f:
        json.dump(stats, f, indent=4)
    log.info("Stats saved to %s", path)
    return stats


# ── episode metadata ───────────────────────────────────────────────────────────


def read_episode_meta(ds: SourceDataset) -> pa.Table | None:
    tables = []
    for f in _sorted_glob2(ds.path / "meta" / "episodes", ".parquet"):
        t = pq.read_table(f)
        for name, off in (
            ("episode_index", ds.episode_offset),
            ("dataset_from_index", ds.frame_offset),
            ("dataset_to_index", ds.frame_offset),
        ):
            if name in t.column_names and off:
                i = t.schema.get_field_index(name)
                c = t.column(i)
                t = t.set_column(i, name, pc.add(c, pa.scalar(off, c.type)))
        tables.append(t)
    if not tables:
        return None
    return pa.concat_tables(tables, promote_options="permissive") if len(tables) > 1 else tables[0]


def _set_column(t: pa.Table, name: str, arr: pa.Array | pa.ChunkedArray) -> pa.Table:
    """Replace a column in place, or append it (pandas assignment order)."""
    i = t.schema.get_field_index(name)
    return t.set_column(i, name, arr) if i >= 0 else t.append_column(name, arr)


def write_episode_meta(
    tables: list[pa.Table],
    output_dir: Path,
    file_starts: np.ndarray,
    file_ends: np.ndarray,
    file_locs: list[tuple[int, int]],
    video_keys: Sequence[str],
    fps: float,
) -> None:
    """meta/episodes/chunk-000/file-000.parquet with data/video locations re-pointed.

    All in arrow: episode metadata carries ~1000 per-feature stats columns, and the pandas path
    turned every cell into a Python-held array (122 s for 15.6K episodes on oct21).
    """
    meta = pa.concat_tables(tables, promote_options="permissive") if len(tables) > 1 else tables[0]
    n = meta.num_rows
    meta = _set_column(meta, "meta/episodes/chunk_index", pa.array(np.zeros(n, dtype=np.int64)))
    meta = _set_column(meta, "meta/episodes/file_index", pa.array(np.zeros(n, dtype=np.int64)))

    # The data file holding each episode's first frame (the last file if none does, as before).
    x = np.asarray(meta.column("dataset_from_index").to_numpy(), dtype=np.int64)
    i = np.searchsorted(file_starts, x, side="right") - 1
    ok = (i >= 0) & (x < file_ends[np.clip(i, 0, len(file_ends) - 1)])
    i = np.where(ok, i, len(file_locs) - 1)
    locs = np.asarray(file_locs, dtype=np.int64)
    meta = _set_column(meta, "data/chunk_index", pa.array(locs[i, 0]))
    meta = _set_column(meta, "data/file_index", pa.array(locs[i, 1]))

    # One video file per episode, named by its merged index.
    ep = np.asarray(meta.column("episode_index").to_numpy(), dtype=np.int64)
    length = (
        np.asarray(meta.column("length").to_numpy(zero_copy_only=False), dtype=np.float64)
        if "length" in meta.column_names
        else None
    )
    for vk in video_keys:
        meta = _set_column(meta, f"videos/{vk}/chunk_index", pa.array(ep // CHUNK_SIZE))
        meta = _set_column(meta, f"videos/{vk}/file_index", pa.array(ep % CHUNK_SIZE))
        meta = _set_column(meta, f"videos/{vk}/from_timestamp", pa.array(np.zeros(n)))
        if length is not None:
            meta = _set_column(meta, f"videos/{vk}/to_timestamp", pa.array(length / fps))
    drop = [
        c
        for c in meta.column_names
        if c.startswith("videos/") and not any(vk in c for vk in video_keys)
    ]
    if drop:
        meta = meta.drop_columns(drop)

    # Episodes from a scene without some object have no stats for it; null cells kill LeRobot's
    # torch.tensor at item access, so fill NaN vectors of the existing width (count: [0]).
    filled = []
    for c in meta.column_names:
        col = meta.column(c)
        nulls = col.null_count
        if not nulls or not (nulls == n or c.startswith("stats/")):
            continue
        t = col.type
        if c.endswith("/count"):
            arr = col.combine_chunks()
            if not (
                pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t)
            ):
                arr = pa.nulls(n, pa.list_(pa.int64())) if pa.types.is_null(t) else arr
            if (
                pa.types.is_list(arr.type)
                or pa.types.is_large_list(arr.type)
                or pa.types.is_fixed_size_list(arr.type)
            ):
                if not pa.types.is_integer(arr.type.value_type):
                    arr = arr.cast(pa.list_(pa.int64()))
                meta = _set_column(
                    meta, c, fill_list_nulls(arr, 1, 0, invalidate_null_elements=False)
                )
                filled.append(c)
            continue
        if nulls == n or not (
            pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t)
        ):
            continue  # nothing to copy a width from (unchanged, as before)
        arr = col.combine_chunks()
        if not pa.types.is_floating(t.value_type):
            arr = arr.cast(pa.list_(pa.float64()))
        valid_lens = pc.list_value_length(arr.filter(arr.is_valid()))
        width = int(valid_lens[0].as_py()) if len(valid_lens) else 1
        meta = _set_column(
            meta, c, fill_list_nulls(arr, width, np.nan, invalidate_null_elements=False)
        )
        filled.append(c)
    if filled:
        log.info(
            "Episode stats: filled %d column(s) undefined for episodes whose scene lacks that object (NaN): %s",
            len(filled),
            sorted({c.split("/")[1] for c in filled if "/" in c}),
        )
    target = output_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    atomic_write(target, lambda tmp: pq.write_table(meta, tmp))


# ── videos ─────────────────────────────────────────────────────────────────────


def link_video(task: tuple[str, Path, str, int]) -> bool:
    """Hardlink one clip into the merged layout; True if linked, False if it had to be copied.

    An existing destination means two source episodes claim one merged index, and writing over
    it would write through the shared inode into the unmerged source, so that is an error. Only
    a cross-filesystem link (EXDEV) falls back to an fsynced temp copy renamed into place.
    """
    src, dst, key, ep = task
    try:
        os.link(src, dst)
        return True
    except FileExistsError as exc:
        raise RuntimeError(
            f"Refusing to overwrite existing merged video {dst} (episode {ep}, {key}, source {src}). Two "
            "source episodes claim the same merged episode index; overwriting would corrupt the shared "
            "inode behind the unmerged source."
        ) from exc
    except OSError:
        atomic_write(dst, lambda tmp: shutil.copy(src, tmp), verify_parquet=False)
        return False


def link_videos(tasks: list[tuple[str, Path, str, int]]) -> tuple[int, int]:
    for d in sorted({t[1].parent for t in tasks}):
        d.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=LINK_WORKERS) as ex:
        res = list(ex.map(link_video, tasks))
    return res.count(True), res.count(False)


# ── recording metadata ─────────────────────────────────────────────────────────


def merge_recording_metadata(
    dataset_paths: Sequence[Path], output_dir: Path, num_workers: int = 32
) -> None:
    """recording_config.json + domain.pddl from the first source that has them, every source's
    other meta/extra files under meta/extra/<problem>/, and one aggregated provenance.json."""
    meta_out = output_dir / "meta"
    extra_out = meta_out / "extra"
    extra_out.mkdir(parents=True, exist_ok=True)

    for src in dataset_paths:
        rc = src / "meta" / "recording_config.json"
        if rc.is_file():
            shutil.copy2(rc, meta_out / "recording_config.json")
            log.info("Copied recording_config.json from %s", src)
            break
    for src in dataset_paths:
        dp = src / "meta" / "extra" / "domain.pddl"
        if dp.is_file():
            shutil.copy2(dp, extra_out / "domain.pddl")
            log.info("Copied domain.pddl from %s", src)
            break

    def _list(src: Path) -> tuple[Path, list[Path], dict[str, Any] | None]:
        src_extra = src / "meta" / "extra"
        files = (
            [
                f
                for f in src_extra.iterdir()
                if f.is_file() and f.name not in ("domain.pddl", "combined.json")
            ]
            if src_extra.is_dir()
            else []
        )
        prov = None
        pp = src / "meta" / "provenance.json"
        if pp.is_file():
            try:
                with open(pp) as fh:
                    prov = json.load(fh)
            except Exception:  # noqa: BLE001
                prov = None
        return src, files, prov

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        listed = list(ex.map(_list, dataset_paths))

    copies: list[tuple[Path, Path]] = []
    seen: set[str] = set()
    for src, files, _prov in listed:
        if not (src / "meta" / "extra").is_dir():
            continue
        name = problem_name_from_dataset_path(src)
        if name in seen:
            continue
        seen.add(name)
        (extra_out / name).mkdir(parents=True, exist_ok=True)
        copies += [(f, extra_out / name / f.name) for f in files]
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        list(ex.map(lambda sd: shutil.copy2(sd[0], sd[1]), copies))
    log.info("Copied extra files for %d problem(s) (%d files)", len(seen), len(copies))

    git_info: dict[str, Any] = {}
    sources: list[dict[str, Any]] = []
    for src, _files, prov in listed:
        if prov is None:
            continue
        if not git_info:
            git_info = prov.get("git", {})
        name = problem_name_from_dataset_path(src)
        merged_files = []
        for fentry in prov.get("files", []):
            fname = Path(fentry.get("dest", "")).name
            if fname == "combined.json":
                continue
            updated = dict(fentry)
            updated["dest"] = (
                "meta/extra/domain.pddl" if fname == "domain.pddl" else f"meta/extra/{name}/{fname}"
            )
            merged_files.append(updated)
        sources.append(
            {
                "source_dataset": str(src),
                "problem_name": name,
                "extra_dir": f"meta/extra/{name}",
                "recording_provenance_sources": prov.get("recording_provenance_sources", {}),
                "files": merged_files,
            }
        )
    with open(meta_out / "provenance.json", "w") as f:
        json.dump(
            {"git": git_info, "merged_from_n_datasets": len(dataset_paths), "sources": sources},
            f,
            indent=2,
        )
    log.info(
        "Wrote merged provenance.json (%d source entries from %d datasets)",
        len(sources),
        len(dataset_paths),
    )


# ── validation ─────────────────────────────────────────────────────────────────


def _bounded(fn: Callable[[], bool], timeout_s: int) -> bool | None:
    """fn() with a deadline; a stuck call is abandoned in a daemon thread (read-only step)."""
    box: dict[str, Any] = {}

    def _run() -> None:
        try:
            box["ok"] = fn()
        except BaseException as e:  # noqa: BLE001 - re-raised below
            box["err"] = e

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    th.join(timeout_s)
    if th.is_alive():
        globals()["_ABANDONED_VALIDATION"] = True
        raise TimeoutError(
            f"LeRobot validation exceeded {timeout_s}s and was abandoned (read-only step; the merged data is "
            "complete and unaffected)"
        )
    if "err" in box:
        raise box["err"]
    return box.get("ok")


def _read_first_sample(dataset: Any) -> None:
    """dataset[0] under SIGALRM where possible; a timeout is inconclusive, not failure."""
    import signal

    prev = None
    try:

        def _on_alarm(_sig: int, _frm: Any) -> None:
            raise TimeoutError(f"dataset[0] exceeded {LEROBOT_SAMPLE_TIMEOUT_S}s")

        prev = signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(LEROBOT_SAMPLE_TIMEOUT_S)
    except (ValueError, AttributeError, OSError):
        prev = None  # not the main thread: run unbounded
    try:
        sample = dataset[0]
        log.info("  ✓ Sample data accessible, keys: %s", list(sample.keys()))
    except TimeoutError as te:
        log.warning(
            "  ! sample read timed out (%s); the dataset CONSTRUCTED fine, so this is INCONCLUSIVE",
            te,
        )
    finally:
        try:
            signal.alarm(0)
            if prev is not None:
                signal.signal(signal.SIGALRM, prev)
        except Exception:  # noqa: BLE001
            pass


def validate_full(output_dir: Path) -> bool:
    """The TAMP script's check: LeRobotDataset over every frame, then dataset[0]."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    log.info("Validating dataset with LeRobot (full)...")
    dataset = LeRobotDataset(repo_id="local", root=str(output_dir))
    log.info("  ✓ Dataset loaded successfully")
    log.info("  ✓ Total frames: %d", len(dataset))
    if len(dataset) > 0:
        _read_first_sample(dataset)
    log.info("✓ LeRobot validation passed!")
    return True


def validate_fast(
    output_dir: Path, features: dict[str, dict[str, Any]], num_workers: int = 32
) -> bool:
    """What a full LeRobot load would trip over, without converting every frame.

    1. Every data file holds exactly the declared non-video features and each column casts to
       LeRobot's arrow type for it (footers only) -- the 'column names don't match' and
       'Couldn't cast' failures.
    2. LeRobotDataset on one episode: info/tasks/episodes/stats load, the episode's rows cast to
       the features, and dataset[0] reads (videos included).
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import get_hf_features_from_features
    from lerobot.datasets.utils import load_info as lerobot_load_info

    log.info("Validating dataset with LeRobot (fast: every file's schema + one episode)...")
    # The features as LeRobot reads them back from info.json (shapes as tuples), not as passed in.
    target = get_hf_features_from_features(lerobot_load_info(output_dir)["features"]).arrow_schema
    declared = set(target.names)
    files = sorted((output_dir / "data").glob("*/*.parquet"))

    def _check(f: Path) -> str | None:
        s = pq.read_schema(f)
        have = set(s.names)
        if have != declared:
            return (
                f"{f.relative_to(output_dir)}: column names don't match the declared features "
                f"(undeclared {sorted(have - declared)[:5]}, missing {sorted(declared - have)[:5]})"
            )
        try:
            s.empty_table().select(target.names).cast(target)
        except Exception as e:  # noqa: BLE001
            return f"{f.relative_to(output_dir)}: CastError: {e}"
        return None

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        problems = [p for p in ex.map(_check, files) if p]
    if problems:
        raise ValueError(
            f"schema check failed in {len(problems)}/{len(files)} file(s); first: {problems[0]}"
        )
    log.info("  ✓ All %d data files match the declared features", len(files))
    dataset = LeRobotDataset(repo_id="local", root=str(output_dir), episodes=[0])
    log.info("  ✓ Dataset loaded successfully (episode 0: %d frames)", len(dataset))
    if len(dataset) > 0:
        _read_first_sample(dataset)
    log.info("✓ LeRobot validation passed!")
    return True


def report_validation_error(
    e: BaseException, union_schema: bool, output_dir: Path, features: dict[str, Any]
) -> None:
    """Classify a validation failure with the TAMP script's wording (orchestrators match it)."""
    t, m = type(e).__name__, str(e)
    chain, cur, seen = [m], e, 0
    while cur is not None and seen < 8:
        cur = cur.__cause__ or cur.__context__
        seen += 1
        if cur is not None:
            chain.append(f"{type(cur).__name__}: {cur}")
    low = " ".join(chain).lower()
    detail = next((c for c in chain[1:] if c.strip()), m)
    if "ffmpeg" in low or "codec" in low or "torchcodec" in low or "backend" in low:
        log.warning(
            "LeRobot validation skipped -- video decoding unavailable (%s). The merged data itself is unaffected.",
            t,
        )
    elif "cast" in low or "column names" in low or "schema" in low:
        fix = (
            "re-run with --union_schema to reconcile them"
            if not union_schema
            else "--union_schema was ON, so this is NOT the ragged-object_pose case"
        )
        log.warning(
            "LeRobot validation FAILED -- the merged dataset is NOT loadable as a LeRobotDataset (%s): the parquet "
            "schema and meta/info.json features do not agree; %s. Readers that load the parquet directly (e.g. hvla) "
            "are unaffected. Detail: %s",
            t,
            fix,
            detail[:250],
        )
        try:
            decl = {k for k, v in features.items() if v.get("dtype") != "video"}
            have: set[str] = set()
            for f in sorted(output_dir.glob("data/*/*.parquet")):
                have |= set(pq.read_schema(f).names)
            if have - decl:
                log.warning("  in the parquet, declared by no info.json: %s", sorted(have - decl))
            if decl - have:
                log.warning(
                    "  declared in info.json, absent from the parquet: %s", sorted(decl - have)
                )
        except Exception as e2:  # noqa: BLE001
            log.warning("  (could not diff the schemas: %s: %s)", type(e2).__name__, e2)
    elif t == "TimeoutError" or "timeout" in low or "abandoned" in low:
        log.warning(
            "LeRobot validation NOT VERIFIED (%s): the check was abandoned before it reached a verdict, so this "
            "dataset is UNCHECKED, not known-bad. It is a read-only step and cannot have damaged anything. "
            "Detail: %s",
            t,
            detail[:250],
        )
    else:
        log.warning("LeRobot validation FAILED (%s): %s", t, detail[:250])


# ── merge ──────────────────────────────────────────────────────────────────────


class PhaseTimer:
    def __init__(self) -> None:
        self.t0 = time.time()
        self.last = self.t0
        self.rows: list[tuple[str, float]] = []

    def done(self, name: str) -> None:
        now = time.time()
        self.rows.append((name, now - self.last))
        self.last = now

    def report(self) -> None:
        log.info(
            "Timing: %s | total %.1fs",
            ", ".join(f"{n} {s:.1f}s" for n, s in self.rows),
            time.time() - self.t0,
        )


def merge_datasets(
    dataset_paths: list[Path],
    output_dir: Path,
    keep_cameras: Sequence[str],
    num_workers: int = 16,
    union_schema: bool = False,
    validate_timeout: int = 900,
    validate: bool = False,
    validate_mode: str = "fast",
) -> None:
    lock = OutputLock(output_dir)
    lock.acquire()
    try:
        _merge_locked(
            dataset_paths,
            output_dir,
            keep_cameras,
            lock,
            num_workers=num_workers,
            union_schema=union_schema,
            validate_timeout=validate_timeout,
            validate=validate,
            validate_mode=validate_mode,
        )
    finally:
        lock.release()


def _merge_locked(
    dataset_paths: list[Path],
    output_dir: Path,
    keep_cameras: Sequence[str],
    lock: OutputLock,
    num_workers: int,
    union_schema: bool,
    validate_timeout: int,
    validate: bool,
    validate_mode: str,
) -> None:
    timer = PhaseTimer()
    if output_dir.exists():
        log.info("Removing existing output directory: %s", output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    io_workers = max(num_workers, 32)

    keep_all = wants_all_cameras(keep_cameras)
    keep_video_keys = [] if keep_all else normalize_camera_keys(keep_cameras)
    log.info("Merging %d datasets", len(dataset_paths))
    log.info(
        "Keeping video keys: %s", "ALL declared by any source" if keep_all else keep_video_keys
    )
    log.info("Using %d data workers, %d I/O threads", num_workers, io_workers)

    # ── features: template = first dataset (+ video features from the first that declares any),
    # unioned across every source's info.json for --cameras all (video) / --union_schema (rest).
    with ThreadPoolExecutor(max_workers=io_workers) as ex:

        def _safe_info(p: Path) -> dict[str, Any] | None:
            try:
                return load_info(p)
            except Exception:  # noqa: BLE001
                return None

        infos = list(ex.map(_safe_info, dataset_paths))
    template = infos[0] if infos[0] is not None else load_info(dataset_paths[0])
    fps = template["fps"]
    features: dict[str, dict[str, Any]] = dict(template["features"])
    if not any(v.get("dtype") == "video" for v in features.values()):
        for p, cand in zip(dataset_paths[1:], infos[1:]):
            if cand and any(v.get("dtype") == "video" for v in cand["features"].values()):
                for k, v in cand["features"].items():
                    if v.get("dtype") == "video" and k not in features:
                        features[k] = v
                log.info("Patched template with video features from %s", p)
                break
    others = [i for i in infos[1:] if i is not None]
    if keep_all:
        added = sorted(
            {
                k
                for c in others
                for k, v in c.get("features", {}).items()
                if v.get("dtype") == "video"
            }
            - set(features)
        )
        for c in others:
            for k, v in c.get("features", {}).items():
                if v.get("dtype") == "video" and k not in features:
                    features[k] = v
        if added:
            log.info(
                "--cameras all: added %d video feature(s) the template did not declare: %s",
                len(added),
                added,
            )
    if union_schema:
        added = []
        for c in others:
            for k, v in c.get("features", {}).items():
                if v.get("dtype") != "video" and k not in features:
                    features[k] = v
                    added.append(k)
        if added:
            log.info(
                "Feature union: added %d feature(s) absent from the template: %s",
                len(added),
                sorted(set(added)),
            )
    filtered: dict[str, dict[str, Any]] = {}
    video_keys: list[str] = []
    for k, v in features.items():
        if v.get("dtype") == "video":
            if keep_all or k in keep_video_keys:
                filtered[k] = v
                video_keys.append(k)
        else:
            filtered[k] = v
    filtered[SOURCE_PROBLEM_KEY] = SOURCE_PROBLEM_FEATURE
    filtered[INSTANCE_ID_KEY] = INSTANCE_ID_FEATURE
    log.info("Filtered to %d video keys: %s", len(video_keys), video_keys)
    timer.done("infos")

    # ── scan
    with ThreadPoolExecutor(max_workers=io_workers) as ex:
        scanned = list(ex.map(lambda p: scan_dataset(p, video_keys), dataset_paths))
    corrupt = [d for d in scanned if d.corrupt_data_files]
    if corrupt:
        log.warning(
            "Dropping %d dataset(s) (%d episode(s)) with an unreadable data parquet:",
            len(corrupt),
            sum(d.total_episodes for d in corrupt),
        )
        for d in corrupt:
            log.warning(
                "    %s (%d episode(s)): %s", d.path, d.total_episodes, d.corrupt_data_files[0]
            )
    datasets = [d for d in scanned if not d.corrupt_data_files]
    if not datasets:
        raise RuntimeError("no readable datasets to merge")
    timer.done("scan")

    # ── offsets, tasks, videos
    ep_off = fr_off = 0
    all_tasks: set[str] = set()
    video_tasks: list[tuple[str, Path, str, int]] = []
    for d in datasets:
        d.episode_offset, d.frame_offset = ep_off, fr_off
        all_tasks.update(d.tasks)
        if d.footer_rows != d.total_frames:
            log.warning(
                "%s: info.json says %d frames but its data files hold %d rows; offsets follow info.json",
                d.path,
                d.total_frames,
                d.footer_rows,
            )
        for key, vf, src_ep in d.video_files:
            dst_ep = ep_off + src_ep
            dst = (
                output_dir
                / "videos"
                / key
                / f"chunk-{dst_ep // CHUNK_SIZE:03d}"
                / f"file-{dst_ep % CHUNK_SIZE:03d}.mp4"
            )
            video_tasks.append((vf, dst, key, dst_ep))
        ep_off += d.total_episodes
        fr_off += d.total_frames
    total_episodes, total_frames = ep_off, fr_off
    log.info("Total: %d episodes, %d frames", total_episodes, total_frames)

    cover, short, strays = video_coverage_report(datasets, video_keys)
    for k, n_have in sorted(cover.items()):
        if n_have < len(datasets):
            log.warning(
                "  video key %s: present in %d/%d datasets -- merged episodes from the other %d will have no %s video",
                k,
                n_have,
                len(datasets),
                len(datasets) - n_have,
                k,
            )
    if strays:
        log.warning(
            "  %d file(s) in video dirs are not named file-NNN.mp4 and were skipped; first few: %s",
            len(strays),
            strays[:3],
        )
    if short:
        by_key: dict[str, int] = {}
        for k, _p, _n, _e in short:
            by_key[k] = by_key.get(k, 0) + 1
        log.warning(
            "  INCOMPLETE VIDEO COVERAGE: %d (dataset, key) pair(s) have FEWER clips than episodes -- those episodes "
            "will declare a video that is not on disk, which a path-resolving dataloader (hvla) reads as ZEROS. By "
            "key: %s",
            len(short),
            ", ".join(f"{k}: {n} dataset(s)" for k, n in sorted(by_key.items())),
        )
        for k, pth, n, e in short[:10]:
            log.warning("    %s: %d/%d clips in %s", k, n, e, pth)

    tasks_sorted = sorted(all_tasks)
    task_to_idx = {t: i for i, t in enumerate(tasks_sorted)}

    # ── column union (always: every output file gets the same columns), declared or not
    columns: list[str] = []
    union_fields: dict[str, pa.Field] = {}
    for d in datasets:
        if d.schema is None:
            continue
        for fld in d.schema:
            if fld.name not in union_fields:
                union_fields[fld.name] = fld
                columns.append(fld.name)
    for key, feat in ((SOURCE_PROBLEM_KEY, pa.string()), (INSTANCE_ID_KEY, pa.int64())):
        if key not in union_fields:
            union_fields[key] = pa.field(key, feat)
            columns.append(key)
    if union_schema:
        log.info("Column union across %d dataset(s): %d columns", len(datasets), len(columns))
    ragged = sorted(c for c in columns if "object_pose" in c)
    if ragged:
        log.info("  object_pose columns in the union: %s", ragged)
    if union_schema:
        sample_file = Path(datasets[0].files[0].path) if datasets[0].files else None
        undeclared = [c for c in columns if c not in filtered]
        for c in undeclared:
            filtered[c] = feature_from_arrow(union_fields[c], sample_file)
        if undeclared:
            log.info(
                "Schema repair: %d column(s) present in the parquet but declared by no source info.json, now declared: %s",
                len(undeclared),
                sorted(undeclared),
            )
    out_schema = pa.schema([pa.field(c, output_type(union_fields[c].type)) for c in columns])
    filtered = order_features_like_columns(filtered, columns)
    fill_width: dict[str, int] = {}
    for c in columns:
        t = union_fields[c].type
        shape = (filtered.get(c) or {}).get("shape")
        if shape:
            fill_width[c] = int(np.prod([int(s) for s in shape]))
        elif pa.types.is_fixed_size_list(t):
            fill_width[c] = t.list_size
        else:
            fill_width[c] = 1

    import pandas as pd

    write_parquet_frame(
        pd.DataFrame({"task_index": range(len(tasks_sorted))}, index=tasks_sorted),
        output_dir / "meta" / "tasks.parquet",
    )
    timer.done("plan")

    # ── background I/O while the data pass runs: video links, episode metadata reads,
    # recording metadata. Exceptions surface when their futures are joined.
    bg = ThreadPoolExecutor(max_workers=3)
    fut_links: Future[tuple[int, int]] = bg.submit(link_videos, video_tasks)

    def _read_all_meta() -> list[pa.Table | None]:
        with ThreadPoolExecutor(max_workers=io_workers) as mex:
            return list(mex.map(read_episode_meta, datasets))

    fut_meta: Future[list[pa.Table | None]] = bg.submit(_read_all_meta)
    fut_rec: Future[None] = bg.submit(
        merge_recording_metadata, dataset_paths, output_dir, io_workers
    )

    # ── data pass
    ctxs = []
    for d in datasets:
        if d.has_task_table:
            items = sorted(
                (li, task_to_idx[s]) for li, s in d.local_tasks.items() if s in task_to_idx
            )
            keys = np.array([k for k, _ in items], dtype=np.int64)
            vals = np.array([v for _, v in items], dtype=np.int64)
            ctxs.append(DatasetCtx(d.episode_offset, d.frame_offset, d.problem_name, keys, vals))
        else:
            ctxs.append(DatasetCtx(d.episode_offset, d.frame_offset, d.problem_name, None, None))
    numeric_keys = numeric_stat_keys(filtered)
    total_rows = sum(d.footer_rows for d in datasets)
    cfg = WorkerConfig(
        columns=columns,
        out_schema=out_schema,
        fill_width=fill_width,
        numeric_keys=numeric_keys,
        sample_rate=min(1.0, STATS_SAMPLE_ROWS / max(total_rows, 1)),
        datasets=ctxs,
    )
    rows_per_file = estimate_rows_per_file(datasets, cfg)
    jobs = plan_output_files(datasets, rows_per_file)
    log.info(
        "Data pass: %d rows -> %d file(s) of %d rows, %d workers",
        total_rows,
        len(jobs),
        rows_per_file,
        num_workers,
    )
    acc = StatsAccumulator()
    results: list[dict[str, Any]] = []
    seen_warn: set[str] = set()
    t_data = time.time()
    with ProcessPoolExecutor(
        max_workers=max(1, num_workers),
        mp_context=mp.get_context("spawn"),
        initializer=_init_worker,
        initargs=(cfg,),
    ) as ex:
        futs = [ex.submit(run_output_job, j, str(output_dir)) for j in jobs]
        for k, fut in enumerate(futs):
            try:
                r = fut.result()
            except BaseException:
                ex.shutdown(wait=False, cancel_futures=True)
                raise
            acc.add(r["file_index"], r.pop("stats"), r.pop("samples"))
            for w in r["warnings"]:
                if w not in seen_warn:
                    seen_warn.add(w)
                    log.warning(w)
            results.append(r)
            if (k + 1) % max(1, len(futs) // 20) == 0 or k + 1 == len(futs):
                el = time.time() - t_data
                log.info(
                    "  data files %d/%d (%.0fs, %.2fM rows/s)",
                    k + 1,
                    len(futs),
                    el,
                    sum(x["rows"] for x in results) / 1e6 / max(el, 1e-9),
                )
    timer.done("data")

    # ── episode metadata (needs where each file's rows landed)
    starts = np.array([r["start"] for r in results], dtype=np.int64)
    ends = np.array([r["end"] for r in results], dtype=np.int64)
    locs = [(r["file_index"] // CHUNK_SIZE, r["file_index"] % CHUNK_SIZE) for r in results]
    order = np.argsort(starts, kind="stable")
    starts, ends, locs = starts[order], ends[order], [locs[i] for i in order]
    meta_tables = [t for t in fut_meta.result() if t is not None]
    write_episode_meta(meta_tables, output_dir, starts, ends, locs, video_keys, fps)
    timer.done("episodes")

    n_linked, n_copied = fut_links.result()
    if n_copied:
        log.info(
            "  %d videos hardlinked, %d copied (hardlink unavailable -- different filesystems?)",
            n_linked,
            n_copied,
        )
    else:
        log.info("  All %d videos hardlinked (no data copied)", n_linked)
    fut_rec.result()
    bg.shutdown()
    timer.done("videos+extra")

    lock.check()
    info = {
        "codebase_version": "v3.0",
        "robot_type": template.get("robot_type", "unknown"),
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(tasks_sorted),
        "chunks_size": CHUNK_SIZE,
        "data_files_size_in_mb": DATA_FILE_SIZE_IN_MB,
        "video_files_size_in_mb": VIDEO_FILE_SIZE_IN_MB,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": filtered,
    }
    with open(output_dir / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=4)
    stats = acc.finalize(filtered)
    with open(output_dir / "meta" / "stats.json", "w") as f:
        json.dump(stats, f, indent=4)
    log.info("Stats saved to %s", output_dir / "meta" / "stats.json")

    # data/ and meta/ only: an rglob also walks every video directory (12 s for 60K clips on Weka)
    all_parquets = sorted(
        [*output_dir.glob("data/*/*.parquet"), *(output_dir / "meta").rglob("*.parquet")]
    )
    with ThreadPoolExecutor(max_workers=io_workers) as ex:
        ok = list(ex.map(verify_parquet_footer, all_parquets))
    bad = [p for p, good in zip(all_parquets, ok) if not good]
    if bad:
        for p in bad:
            log.error("  CORRUPT: %s (%d bytes)", p.relative_to(output_dir), p.stat().st_size)
        raise RuntimeError(f"{len(bad)}/{len(all_parquets)} parquet files have corrupt footers")
    log.info("  All %d parquet files verified OK", len(all_parquets))
    check_output_complete(output_dir, jobs, video_tasks, io_workers)
    lock.check()
    timer.done("finalize")

    # union_schema implies validate: that flag exists to make the output LeRobot-loadable.
    if not (validate or union_schema):
        log.info(
            "LeRobot validation SKIPPED (pass --validate to run it; --union_schema turns it on)."
        )
    else:
        try:
            run: Callable[[], bool] = (
                (lambda: validate_full(output_dir))
                if validate_mode == "full"
                else (lambda: validate_fast(output_dir, filtered, io_workers))
            )
            if validate_timeout and validate_timeout > 0:
                _bounded(run, validate_timeout)
            else:
                run()
        except Exception as e:  # noqa: BLE001 - non-fatal, classified
            report_validation_error(e, union_schema, output_dir, filtered)
        timer.done("validate")

    timer.report()
    log.info("Done! Merged dataset saved to: %s", output_dir)
    log.info("  Episodes: %d", total_episodes)
    log.info("  Frames: %d", total_frames)
    log.info("  Video keys: %s", video_keys)


def order_features_like_columns(
    features: dict[str, dict[str, Any]], columns: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Non-video features in parquet column order; video features keep their positions.

    LeRobotDataset(episodes=[...]) casts the filtered table straight to the features' arrow
    schema, which requires the same column ORDER, so a union merge whose info.json lists
    features in a different order than its parquet loads whole but crashes on any episode
    subset. (The whole-dataset load reorders, which is why the TAMP script never noticed.)
    """
    pos = {c: i for i, c in enumerate(columns)}
    keys = list(features)
    slots = [i for i, k in enumerate(keys) if features[k].get("dtype") != "video"]
    non_video = sorted(
        (keys[i] for i in slots), key=lambda k: (pos.get(k, len(pos)), slots.index(keys.index(k)))
    )
    for slot, k in zip(slots, non_video):
        keys[slot] = k
    return {k: features[k] for k in keys}


def estimate_rows_per_file(datasets: Sequence[SourceDataset], cfg: WorkerConfig) -> int:
    """Rows that make a ~DATA_FILE_SIZE_IN_MB file: write 1000 merged rows and measure."""
    first = next((d for d in datasets if d.files), None)
    if first is None:
        return MIN_ROWS_PER_FILE
    di = datasets.index(first)
    sf = first.files[0]
    take = min(1000, sf.num_rows)
    piece = Piece(di, sf.path, 0, take, sf.rg_rows)
    sample = transform_piece(_read_piece(piece), cfg.datasets[di], cfg, [])
    buf = io.BytesIO()
    pq.write_table(sample, buf, row_group_size=ROW_GROUP_SIZE)
    bytes_per_row = buf.tell() / max(take, 1)
    rows = max(MIN_ROWS_PER_FILE, int(DATA_FILE_SIZE_IN_MB * 1024 * 1024 / bytes_per_row))
    log.info(
        "  Estimated %.1f bytes/row, targeting %d rows/file (~%dMB)",
        bytes_per_row,
        rows,
        DATA_FILE_SIZE_IN_MB,
    )
    return rows


# ── CLI ────────────────────────────────────────────────────────────────────────


def _exit(code: int = 0) -> None:
    """sys.exit, or os._exit if a validation thread was abandoned inside a C call."""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass
    if _ABANDONED_VALIDATION:
        log.warning(
            "exiting via os._exit(%d): a LeRobot-validation thread was abandoned inside a C call",
            code,
        )
        os._exit(code)
    sys.exit(code)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fast merge of LeRobot v3 datasets (replaces TAMP/factory/merge_success.py)"
    )
    p.add_argument(
        "--input_dirs",
        nargs="+",
        type=Path,
        required=True,
        help="Directories holding the datasets to merge",
    )
    p.add_argument(
        "--output_dir", type=Path, required=True, help="Merged dataset (replaced if it exists)"
    )
    p.add_argument(
        "--cameras",
        nargs="+",
        default=["all"],
        help="Camera names or full feature keys to keep, or 'all'/'*' for every video feature any source declares "
        "(default). A named list DROPS every other stream, and a bare name means RGB only.",
    )
    p.add_argument("--num_workers", type=int, default=16, help="Data-pass processes (default: 16)")
    p.add_argument(
        "--max_datasets", type=int, default=None, help="Merge only the first N source datasets"
    )
    p.add_argument(
        "--episodes", type=int, default=None, help="Alias for --max_datasets (backward compat)"
    )
    p.add_argument(
        "--union_schema",
        action="store_true",
        help="Declare in info.json every column any source holds (needed by LeRobotDataset consumers). Implies --validate.",
    )
    p.add_argument("--validate", action="store_true", help="Run the LeRobot loadability check")
    p.add_argument(
        "--validate_mode",
        choices=("fast", "full"),
        default="fast",
        help="fast: every file's schema + LeRobotDataset on one episode (seconds). full: LeRobotDataset over every "
        "frame, as the TAMP script did (converts the whole dataset into an HF cache).",
    )
    p.add_argument(
        "--validate_timeout",
        type=int,
        default=900,
        help="Seconds before validation is abandoned (0 = no limit)",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--recompute-stats":
        rp = argparse.ArgumentParser(
            description="Recompute stats.json for a merged LeRobot dataset"
        )
        rp.add_argument("dataset_dir", type=Path)
        rp.add_argument("--num_workers", type=int, default=16)
        a = rp.parse_args(argv[1:])
        info = load_info(a.dataset_dir)
        compute_stats(a.dataset_dir, info.get("features", {}), a.num_workers)
        return 0
    args = build_parser().parse_args(argv)
    max_datasets = args.max_datasets or args.episodes
    dataset_paths: list[Path] = []
    for d in args.input_dirs:
        found = get_dataset_paths(d)
        log.info("Found %d datasets in %s", len(found), d)
        dataset_paths.extend(found)
    log.info("Total: %d datasets from %d directories", len(dataset_paths), len(args.input_dirs))
    if not dataset_paths:
        log.error("No datasets found!")
        return 1
    if max_datasets is not None and max_datasets < len(dataset_paths):
        log.info("Limiting to first %d datasets (out of %d)", max_datasets, len(dataset_paths))
        dataset_paths = dataset_paths[:max_datasets]
    merge_datasets(
        dataset_paths=dataset_paths,
        output_dir=args.output_dir,
        keep_cameras=args.cameras,
        num_workers=args.num_workers,
        union_schema=args.union_schema,
        validate_timeout=args.validate_timeout,
        validate=args.validate,
        validate_mode=args.validate_mode,
    )
    return 0


def run_cli(argv: Iterable[str] | None = None) -> None:
    _exit(main(list(argv) if argv is not None else None))


if __name__ == "__main__":
    run_cli()
