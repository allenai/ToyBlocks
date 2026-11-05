"""Read and reconcile blocksuite scene provenance from a LeRobot dataset.

Provenance is the set of scene knobs the eval must match for the dataset's
recorded block/joint poses to make sense: ``robot_id`` and ``scene_preset_id``,
plus the recorded camera calibration. Datagen writes the first two as top-level
per-step parquet columns rather than into ``meta/info.json``, which only stores a
generic ``robot_type``.

See ``docs/source/user_guide/workflows/evals/launch_eval.md`` ("Dataset scene provenance").
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from blocksuite.utils.episode_dataset import _discover_lerobot_roots, _load_lerobot_info

logger = logging.getLogger(__name__)

# info.json keys per canonical field, first match wins. ``scene_id`` is datagen's
# name for what eval calls ``scene_preset_id``.
_PROVENANCE_ALIASES: dict[str, tuple[str, ...]] = {
    "robot_id": ("robot_id",),
    "scene_preset_id": ("scene_preset_id", "scene_id"),
}

# Fallback source: canonical field -> per-step parquet column name.
_PROVENANCE_COLUMNS: dict[str, str] = {
    "robot_id": "robot_id",
    "scene_preset_id": "scene_id",
}

# Per-camera calibration columns, one pair per camera:
#   observation.camera_pose.camera:<name>       [x, y, z, qw, qx, qy, qz], robot
#       base frame, scalar-FIRST quaternion, written per frame. A camera cfg with
#       update_latest_camera_pose=True therefore gives a real trajectory, while
#       the IsaacLab default repeats the init-time pose and reads as world-fixed.
#   observation.camera_intrinsics.camera:<name> row-major 3x3 pixel matrix.
_COL_CAMERA_POSE_PREFIX = "observation.camera_pose."
_COL_CAMERA_INTRINSICS_PREFIX = "observation.camera_intrinsics."
# Spread below this counts as constant: generous against float-repr noise, far
# tighter than any real fixed-vs-arm-mounted gap.
_CALIB_WITHIN_EPISODE_ATOL = 1e-4
# Cross-episode spread of first-frame poses above this marks a camera as
# arm-mounted, since it starts each episode wherever the initial joints put it.
_CALIB_WORLD_FIXED_ATOL = 1e-3


class CameraCalibColumnsMissing(LookupError):
    """Raised when a dataset has no ``camera_pose``/``camera_intrinsics`` columns.

    Distinct from a malformed-calib error so callers that only optionally want
    calibration can branch on it while genuine corruption still propagates.
    """


def read_dataset_provenance(dataset_path: str | Path) -> dict[str, Optional[str]]:
    """Return ``{robot_id, scene_preset_id, source}`` for the dataset.

    Reads ``meta/info.json`` first, then falls back to the per-step parquet
    columns for anything still unset. Values are ``None`` when recorded nowhere.
    """
    roots = _discover_lerobot_roots(Path(dataset_path))
    if not roots:
        raise FileNotFoundError(
            f"no LeRobot dataset found at or under {dataset_path}; cannot read "
            "scene provenance."
        )
    info = _load_lerobot_info(roots[0])
    nested = info.get("blocksuite")
    block = nested if isinstance(nested, dict) else {}
    out: dict[str, Optional[str]] = {"source": str(roots[0])}
    for canonical, aliases in _PROVENANCE_ALIASES.items():
        value = next(
            (block.get(a, info.get(a)) for a in aliases if block.get(a, info.get(a)) is not None),
            None,
        )
        out[canonical] = str(value) if value is not None else None

    missing = [k for k in _PROVENANCE_ALIASES if out[k] is None]
    if missing:
        from_columns = _read_provenance_columns(roots[0])
        for k in missing:
            if from_columns.get(k) is not None:
                out[k] = from_columns[k]
    return out


def _read_provenance_columns(root: Path) -> dict[str, Optional[str]]:
    """Read scene identity from the first data shard's per-step columns."""
    from blocksuite.utils.episode_dataset import _resolve_parquet_files

    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    files = _resolve_parquet_files(root)
    names = set(pq.read_schema(str(files[0])).names)
    present = {k: c for k, c in _PROVENANCE_COLUMNS.items() if c in names}
    if not present:
        return {}
    table = pq.read_table(str(files[0]), columns=list(present.values()))
    out: dict[str, Optional[str]] = {}
    for canonical, col in present.items():
        distinct = [
            str(v).strip()
            for v in pc.unique(table.column(col)).to_pylist()
            if v is not None and str(v).strip()
        ]
        if len(distinct) > 1:
            raise ValueError(
                f"{root}: provenance column {col!r} has multiple distinct values "
                f"{sorted(distinct)}; scene identity must be constant per dataset."
            )
        out[canonical] = distinct[0] if distinct else None
    return out


def camera_short_name(column: str, prefix: str) -> str:
    """Reduce a calib column name to its short camera name.

    The short name is how recorded calibration, expert videos and the sim's
    ``obs["images"]`` are matched up, so this is the sole owner of the
    convention.
    """
    rest = column[len(prefix):] if column.startswith(prefix) else column
    return rest.rpartition(":")[-1]


def read_dataset_camera_calib(dataset_path: str | Path) -> dict[str, dict[str, Any]]:
    """Return recorded per-camera calibration, keyed by short camera name.

    Each entry holds ``K`` (3x3 pixel matrix, first episode's first frame),
    ``pos`` and ``quat_wxyz`` (scalar-first, robot base frame, taken at the
    first episode's first frame), ``pose_world_fixed`` and
    ``pose_moves_within_episode``. ``K`` is required to be constant *within*
    each episode; it may vary across episodes (randomized-camera rerenders).
    Per-episode ``K`` for eval ingest lives on the queue setup
    (``setup["camera_intrinsics"]``).
    """
    import numpy as np
    import pyarrow.parquet as pq

    from blocksuite.utils.episode_dataset import (
        _COL_EPISODE_INDEX,
        _resolve_parquet_files,
    )

    roots = _discover_lerobot_roots(Path(dataset_path))
    if not roots:
        raise FileNotFoundError(
            f"no LeRobot dataset found at or under {dataset_path}; cannot read "
            "camera calibration."
        )
    root = roots[0]
    shard = _resolve_parquet_files(root)[0]
    names = pq.read_schema(str(shard)).names

    pose_cols = {
        camera_short_name(c, _COL_CAMERA_POSE_PREFIX): c
        for c in names if c.startswith(_COL_CAMERA_POSE_PREFIX)
    }
    intr_cols = {
        camera_short_name(c, _COL_CAMERA_INTRINSICS_PREFIX): c
        for c in names if c.startswith(_COL_CAMERA_INTRINSICS_PREFIX)
    }
    if not pose_cols and not intr_cols:
        raise CameraCalibColumnsMissing(
            f"{root}: no {_COL_CAMERA_POSE_PREFIX}* or "
            f"{_COL_CAMERA_INTRINSICS_PREFIX}* columns; this dataset predates "
            "camera-calibration recording."
        )
    cams = sorted(pose_cols.keys() | intr_cols.keys())
    incomplete = [c for c in cams if c not in pose_cols or c not in intr_cols]
    if incomplete:
        raise ValueError(
            f"{root}: camera(s) {incomplete} are missing a pose or intrinsics "
            f"column (pose={sorted(pose_cols)}, intrinsics={sorted(intr_cols)}); "
            "expected both per camera."
        )

    read_cols = [_COL_EPISODE_INDEX, *pose_cols.values(), *intr_cols.values()]
    table = pq.read_table(str(shard), columns=read_cols)
    if table.num_rows == 0:
        raise ValueError(f"{shard}: 0 rows; cannot read camera calibration.")
    episodes = np.asarray(table.column(_COL_EPISODE_INDEX).to_pylist())

    out: dict[str, dict[str, Any]] = {}
    for cam in cams:
        K = _read_episode_intrinsics(table, intr_cols[cam], episodes, shard, cam)
        pos, quat_wxyz, world_fixed, moves = _read_pose_per_episode(
            table, pose_cols[cam], episodes, shard
        )
        out[cam] = {
            "K": K,
            "pos": pos,
            "quat_wxyz": quat_wxyz,
            "pose_world_fixed": world_fixed,
            "pose_moves_within_episode": moves,
        }
    return out


def _read_episode_intrinsics(
    table: Any, col: str, episodes: Any, shard: Path, cam: str
) -> list[list[float]]:
    """Return first-episode first-frame K; K must be constant within each episode."""
    import numpy as np

    arr = np.asarray(table.column(col).to_pylist(), dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 9:
        raise ValueError(
            f"{shard}: intrinsics column {col!r} has shape {arr.shape}; "
            "expected (num_steps, 9) row-major 3x3."
        )
    for ep in np.unique(episodes):
        ep_rows = arr[episodes == ep]
        spread = float(np.max(ep_rows.max(axis=0) - ep_rows.min(axis=0)))
        if spread > _CALIB_WITHIN_EPISODE_ATOL:
            raise ValueError(
                f"{shard}: camera {cam!r} intrinsics vary within episode {ep} "
                f"(spread={spread:.6g} > {_CALIB_WITHIN_EPISODE_ATOL:g}); "
                "K must be constant for a given episode."
            )
    first_ep = np.unique(episodes)[0]
    return arr[episodes == first_ep][0].reshape(3, 3).tolist()


def _read_pose_per_episode(
    table: Any, col: str, episodes: Any, shard: Path
) -> tuple[list[float], list[float], bool, bool]:
    """Return ``(pos, quat_wxyz, world_fixed, moves_within_episode)`` for a camera.

    The pose returned is the first episode's first frame, which is the only one
    eval can match after resetting to the episode's recorded initial state.
    """
    import numpy as np

    arr = np.asarray(table.column(col).to_pylist(), dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 7:
        raise ValueError(
            f"{shard}: pose column {col!r} has shape {arr.shape}; expected "
            "(num_steps, 7) = xyz + wxyz."
        )
    reps: list[Any] = []
    within = 0.0
    for ep in np.unique(episodes):
        ep_rows = arr[episodes == ep]
        within = max(within, float(np.max(ep_rows.max(axis=0) - ep_rows.min(axis=0))))
        reps.append(ep_rows[0])
    reps_arr = np.stack(reps)
    cross = float(np.max(reps_arr.max(axis=0) - reps_arr.min(axis=0)))
    moves_within_episode = within > _CALIB_WITHIN_EPISODE_ATOL
    world_fixed = not moves_within_episode and cross <= _CALIB_WORLD_FIXED_ATOL
    rep = reps_arr[0]
    return rep[:3].tolist(), rep[3:7].tolist(), world_fixed, moves_within_episode


def apply_camera_intrinsics_for_env(
    base_env: Any, env_idx: int, camera_intrinsics: dict[str, Sequence[float]]
) -> list[str]:
    """Write one env's recorded 9-vector K into the live cameras.

    Patches that env's row of each camera's ``(num_envs, 3, 3)`` tensor so
    other envs keep the K they already have. Returns the short camera names
    that were written. A recorded camera the scene does not have is skipped
    with a warning (same as dataset-level ingest), unless the env dropped it on
    purpose (``BaseEnvCfg.drop_cameras``).
    """
    import torch

    cameras = getattr(base_env, "cameras", {}) or {}
    dropped = set(getattr(getattr(base_env, "cfg", None), "drop_cameras", ()))
    set_names: list[str] = []
    for name, k9 in camera_intrinsics.items():
        cam = cameras.get(name)
        if cam is None and name in dropped:
            continue
        if cam is None:
            logger.warning(
                "setup records intrinsics for camera %r but the live scene has "
                "no such camera (%s); skipping it.",
                name, sorted(cameras.keys()),
            )
            continue
        if len(k9) != 9:
            raise ValueError(
                f"camera {name!r} intrinsics have {len(k9)} entries, expected 9"
            )
        mats = cam.data.intrinsic_matrices.clone()
        if env_idx < 0 or env_idx >= mats.shape[0]:
            raise ValueError(
                f"env_idx {env_idx} is outside camera {name!r} batch "
                f"{tuple(mats.shape)}"
            )
        mats[env_idx] = torch.tensor(list(k9), dtype=mats.dtype, device=mats.device).reshape(3, 3)
        cam.set_intrinsic_matrices(mats)
        set_names.append(name)
    return set_names


def ingest_dataset_camera_intrinsics(
    base_env: Any,
    dataset_path: str | Path,
    setups: Sequence[dict | None] | None = None,
) -> list[str]:
    """Set each live sim camera's intrinsics to the recorded ``K``.

    Call after the first ``env.reset()``, once the sensors have initialized.
    When ``setups`` is the current per-env queue state, each env gets that
    episode's first-frame K (required for randomized-camera datasets). When
    ``setups`` is omitted, one first-episode K is broadcast to every env.
    Returns the names that were set.
    """
    import torch

    if setups is not None:
        return _ingest_intrinsics_from_setups(base_env, dataset_path, setups)

    try:
        calib = read_dataset_camera_calib(dataset_path)
    except CameraCalibColumnsMissing as e:
        raise ValueError(
            f"camera-intrinsics ingestion was requested but {e}. Eval would then "
            "render at the preset camera_cfgs calibration, which differs from the "
            "recorded one, while reporting a matched scene. Regenerate the dataset "
            "with camera calib columns, or pass --no-ingest-camera-intrinsics to "
            "accept the preset calibration deliberately."
        ) from e

    cameras = getattr(base_env, "cameras", {}) or {}
    set_names: list[str] = []
    for name, cam_calib in calib.items():
        cam = cameras.get(name)
        if cam is None:
            logger.warning(
                "dataset %s recorded intrinsics for camera %r but the live scene "
                "has no such camera (%s); skipping it.",
                dataset_path, name, sorted(cameras.keys()),
            )
            continue
        n = int(cam.data.intrinsic_matrices.shape[0])
        K = torch.tensor(cam_calib["K"], dtype=torch.float32)
        cam.set_intrinsic_matrices(K.unsqueeze(0).repeat(n, 1, 1))
        set_names.append(name)
        logger.info(
            "ingested recorded intrinsics for camera %r "
            "(fx=%.3f fy=%.3f cx=%.3f cy=%.3f) from %s",
            name, cam_calib["K"][0][0], cam_calib["K"][1][1],
            cam_calib["K"][0][2], cam_calib["K"][1][2], dataset_path,
        )
    if not set_names:
        raise ValueError(
            f"camera-intrinsics ingestion was requested for {dataset_path} but none of "
            f"the recorded cameras {sorted(calib)} matched a live scene camera "
            f"{sorted(cameras)}. Every camera would render at preset calibration while "
            "the run reported ingested provenance. Check the scene preset, or pass "
            "--no-ingest-camera-intrinsics."
        )
    return set_names


def _ingest_intrinsics_from_setups(
    base_env: Any,
    dataset_path: str | Path,
    setups: Sequence[dict | None],
) -> list[str]:
    """Apply each non-None setup's ``camera_intrinsics`` to that env."""
    n_envs = int(base_env.num_envs)
    if len(setups) != n_envs:
        raise ValueError(
            f"camera-intrinsics ingestion got {len(setups)} setups for "
            f"{n_envs} envs"
        )
    set_names: set[str] = set()
    applied = 0
    missing: list[int] = []
    for env_idx, setup in enumerate(setups):
        if setup is None:
            continue
        cam_k = setup.get("camera_intrinsics")
        if not cam_k:
            missing.append(env_idx)
            continue
        written = apply_camera_intrinsics_for_env(base_env, env_idx, cam_k)
        set_names.update(written)
        applied += 1
    if missing:
        raise ValueError(
            f"camera-intrinsics ingestion was requested for {dataset_path} but "
            f"env(s) {missing} have no setup['camera_intrinsics']. Randomized-"
            "camera datasets must record observation.camera_intrinsics on every "
            "episode; older datasets without the column should pass "
            "--no-ingest-camera-intrinsics."
        )
    if not set_names:
        raise ValueError(
            f"camera-intrinsics ingestion was requested for {dataset_path} but "
            "no live scene camera matched a recorded name. Check the scene "
            "preset, or pass --no-ingest-camera-intrinsics."
        )
    logger.info(
        "ingested per-env camera intrinsics for %d env(s), cameras %s, from %s",
        applied, sorted(set_names), dataset_path,
    )
    return sorted(set_names)


def reconcile_env_cfg_with_dataset(
    env_cfg: Any,
    dataset_path: str | Path,
    *,
    argv: Optional[list[str]] = None,
) -> None:
    """Default ``env_cfg`` robot/scene to the dataset's values.

    An explicit ``--env-cfg.*`` override always wins but is warned about when it
    disagrees. ``argv`` defaults to ``sys.argv`` and is only used to detect those
    overrides.
    """
    argv = list(sys.argv) if argv is None else argv
    provenance = read_dataset_provenance(dataset_path)
    _reconcile_field(
        env_cfg, "robot_id", provenance["robot_id"], dataset_path,
        ("--env-cfg.robot-id", "--env-cfg.robot_id"), argv,
    )
    _reconcile_field(
        env_cfg, "scene_preset_id", provenance["scene_preset_id"], dataset_path,
        ("--env-cfg.scene-preset-id", "--env-cfg.scene_preset_id"), argv,
    )


def _explicitly_set(argv: list[str], flags: tuple[str, ...]) -> bool:
    """True if any of ``flags`` appears in ``argv``."""
    for arg in argv:
        for flag in flags:
            if arg == flag or arg.startswith(flag + "="):
                return True
    return False


def _reconcile_field(
    env_cfg: Any,
    attr: str,
    dataset_value: Optional[str],
    dataset_path: str | Path,
    flags: tuple[str, ...],
    argv: list[str],
) -> None:
    current = getattr(env_cfg, attr, None)
    if dataset_value is None:
        if _explicitly_set(argv, flags):
            # Honored, but unverifiable against provenance never recorded.
            logger.warning(
                "dataset %s records no %s; cannot verify the eval %s=%r matches "
                "datagen. Honoring your explicit %s. See docs/source/user_guide/workflows/evals/launch_eval.md "
                "'Dataset scene provenance'.",
                dataset_path, attr, attr, current, flags[0],
            )
            return
        # Neither provenance nor an override: fail rather than run a
        # possibly-wrong scene.
        raise SystemExit(
            f"dataset {dataset_path} records no {attr} (checked meta/info.json "
            f"and the per-step provenance columns) and no {flags[0]} override "
            f"was given, so the eval scene cannot be matched to datagen. Pass "
            f"{flags[0]} explicitly, or record {attr} in the dataset. See "
            "docs/source/user_guide/workflows/evals/launch_eval.md 'Dataset scene provenance'."
        )
    if _explicitly_set(argv, flags):
        if current != dataset_value:
            logger.warning(
                "OVERRIDE MISMATCH: eval %s=%r but dataset %s recorded %s=%r. "
                "Honoring your %s override; the eval scene will NOT match the "
                "dataset.",
                attr, current, dataset_path, attr, dataset_value, flags[0],
            )
        return
    if current != dataset_value:
        # The expected, healthy path, so INFO rather than a warning.
        logger.info(
            "defaulting eval %s to dataset value %r (env default was %r); pass "
            "%s to override.",
            attr, dataset_value, current, flags[0],
        )
        setattr(env_cfg, attr, dataset_value)
