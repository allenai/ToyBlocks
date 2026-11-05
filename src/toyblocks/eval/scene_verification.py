"""Scene-verification utilities for queue eval.

Detects mismatches between a LeRobot dataset and the eval scene it initializes:
table height, blocks present, block/table poses, camera calibration and
first-frame images. Shared by ``scripts/eval/verify_dataset_loading.py`` and by
``run_eval`` under ``--verify-scene`` so the two stay in lock-step.
``compare_array``, ``compare_key_set`` and ``malformed_golden`` are also used by
``tests/env_randomization/test_rollout_determinism.py``.

A hard mismatch is one the data can prove wrong, and a check that cannot run
counts as one: an unverifiable scene is not a verified one. Callers decide
whether that aborts.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from toyblocks.utils.dataset_provenance import camera_short_name
from toyblocks.utils.episode_dataset import SCENE_FURNITURE_PREFIXES, load_setups
from toyblocks.utils.episode_queue import EXPECTED_TABLE_Z_ATTR

logger = logging.getLogger(__name__)

# LeRobot RGB video feature prefix (depth lives under observation.depth.*).
_IMAGE_KEY_PREFIX = "observation.images."

# Travels with the metrics JSON so the numbers aren't read as pure scene error.
_FIRST_FRAME_CALIB_CAVEAT = (
    "recorded camera intrinsics are ingested (set_intrinsic_matrices at reset) "
    "and both intrinsics and world-fixed-camera pose are checked by scene "
    "verification (see docs/source/user_guide/workflows/evals/launch_eval.md 'Dataset scene provenance'). "
    "Residual first-frame differences therefore reflect scene state plus any "
    "remaining extrinsics gap (arm-mounted camera world pose is verify-only, not "
    "overridden), not an un-ingested FOV. ssim_edge and ssim_photonorm still "
    "separate lighting from combined geometry+camera error."
)

# Default tolerances (meters / quaternion chord distance).
DEFAULT_TABLE_Z_TOL = 5e-3
DEFAULT_BLOCK_POS_TOL = 1e-2
DEFAULT_BLOCK_QUAT_TOL = 1e-2
# A larger gap means the env's spawn slot doesn't match the episode.
DEFAULT_BLOCK_SIZE_TOL = 2e-3
# Intrinsics in pixels (recorded fx ~ hundreds, so 1px is tight), position in
# meters, rotation as a geodesic angle in radians (~2.9deg).
DEFAULT_CAMERA_INTRINSICS_TOL = 1.0
DEFAULT_CAMERA_POS_TOL = 1e-2
DEFAULT_CAMERA_ROT_TOL_RAD = 0.05


@dataclass
class DatasetSceneSpec:
    """The dataset's ground-truth view of one episode's initial scene."""

    dataset_path: str
    episode_index: int
    episode_id: Optional[str] = None
    robot_joint_positions: Optional[list[float]] = None
    robot_gripper_position: Optional[float] = None
    blocks: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    has_table_column: bool = False
    table_pose: Optional[dict[str, list[float]]] = None
    table_height_delta: Optional[float] = None
    pddl_goal: Optional[str] = None
    # Recorded isotropic block side length (meters); eval must spawn at this size.
    dataset_cube_scale: Optional[float] = None
    expert_videos: dict[str, dict[str, Any]] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)
    # None until datagen writes it (see docs/source/user_guide/workflows/evals/launch_eval.md).
    dataset_robot_id: Optional[str] = None
    dataset_scene_preset_id: Optional[str] = None
    # Keyed by short camera name: {"K": 3x3, "pos": [x,y,z], "quat_wxyz": [..],
    # "pose_world_fixed": bool, "pose_moves_within_episode": bool}, pose taken at
    # the episode's first frame. Empty for datagen predating the calib columns.
    recorded_camera_calib: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def block_names(self) -> set[str]:
        return set(self.blocks.keys())


@dataclass
class SimSceneState:
    """The actual spawned + post-reset state read off the live env."""

    env_idx: int
    robot_id: Optional[str] = None
    scene_preset_id: Optional[str] = None
    robot_usd_path: Optional[str] = None
    table_local_pos: Optional[list[float]] = None
    table_local_quat: Optional[list[float]] = None
    # World-frame table z ``apply_setup`` targeted, recorded at reset. The preset
    # baseline a ``table_height_delta`` is relative to is gone once the pose is
    # written, so this is the only numeric check of the delta path.
    expected_table_z_w: Optional[float] = None
    table_world_z: Optional[float] = None
    blocks: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    block_sizes: dict[str, list[float]] = field(default_factory=dict)
    cameras: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def block_names(self) -> set[str]:
        return set(self.blocks.keys())


@dataclass
class FieldDiff:
    """One compared field: dataset vs sim, with pass/fail against a tolerance."""

    name: str
    passed: bool
    hard: bool
    dataset_value: Any = None
    sim_value: Any = None
    diff: Optional[float] = None
    tol: Optional[float] = None
    note: Optional[str] = None
    # Fraction of the allowance consumed, so a field about to flake is visible
    # while it still passes. Defaults to diff/tol; golden comparisons set it
    # explicitly because their allowance is asymmetric.
    margin: Optional[float] = None

    def __post_init__(self) -> None:
        if self.margin is None and self.diff is not None and self.tol:
            self.margin = float(self.diff) / float(self.tol)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "hard": self.hard,
            "dataset_value": self.dataset_value,
            "sim_value": self.sim_value,
            "diff": self.diff,
            "tol": self.tol,
            "margin": self.margin,
            "note": self.note,
        }


@dataclass
class SceneMismatchReport:
    """Structured comparison result; ``ok`` is False on any hard mismatch."""

    dataset_path: str
    episode_index: int
    diffs: list[FieldDiff] = field(default_factory=list)
    sim_context: dict[str, Any] = field(default_factory=dict)

    @property
    def hard_failures(self) -> list[FieldDiff]:
        return [d for d in self.diffs if d.hard and not d.passed]

    @property
    def soft_failures(self) -> list[FieldDiff]:
        return [d for d in self.diffs if not d.hard and not d.passed]

    @property
    def ok(self) -> bool:
        return not self.hard_failures

    def to_dict(self) -> dict:
        return {
            "dataset_path": self.dataset_path,
            "episode_index": self.episode_index,
            "ok": self.ok,
            "num_hard_failures": len(self.hard_failures),
            "num_soft_failures": len(self.soft_failures),
            "sim_context": self.sim_context,
            "diffs": [d.to_dict() for d in self.diffs],
        }

    def format_console(self) -> str:
        lines = [
            f"Scene verification for {self.dataset_path} episode {self.episode_index}: "
            f"{'OK' if self.ok else 'MISMATCH'} "
            f"({len(self.hard_failures)} hard, {len(self.soft_failures)} soft)",
        ]
        lines.extend(format_diff_line(d) for d in self.diffs)
        return "\n".join(lines)


def format_diff_line(d: FieldDiff) -> str:
    """Format one diff as a console line."""
    status = "PASS" if d.passed else ("FAIL" if d.hard else "warn")
    extra = ""
    if d.diff is not None and d.tol is not None:
        extra = f" diff={d.diff:.5f} tol={d.tol:.5f}"
    if d.margin is not None:
        extra += f" margin={d.margin:.0%}"
    note = f"  ({d.note})" if d.note else ""
    return (
        f"  [{status}] {d.name}{extra}"
        f"  dataset={d.dataset_value} sim={d.sim_value}{note}"
    )


def compare_array(
    name: str,
    observed: Any,
    golden: Any,
    tol: float,
    report: SceneMismatchReport,
) -> None:
    """Compare two numeric arrays, reporting the worst element as one diff."""
    obs = np.asarray(observed, dtype=float)
    gold = np.asarray(golden, dtype=float)
    if obs.shape != gold.shape:
        report.diffs.append(
            FieldDiff(
                name=name,
                passed=False,
                hard=True,
                dataset_value=list(gold.shape),
                sim_value=list(obs.shape),
                margin=float("inf"),
                note="shape changed, so there is nothing comparable element-wise",
            )
        )
        return
    if not obs.size:
        report.diffs.append(
            FieldDiff(
                name=name,
                passed=True,
                hard=True,
                dataset_value=0.0,
                sim_value=0.0,
                diff=0.0,
                tol=tol,
                note="both sides are empty, so there is nothing to compare",
            )
        )
        return
    abs_diff = np.abs(obs - gold)
    worst = np.unravel_index(int(np.argmax(abs_diff)), abs_diff.shape)
    diff = float(abs_diff[worst])
    report.diffs.append(
        FieldDiff(
            name=name,
            passed=diff <= tol,
            hard=True,
            dataset_value=float(gold[worst]),
            sim_value=float(obs[worst]),
            diff=diff,
            tol=tol,
            note=f"worst element at index {[int(i) for i in np.atleast_1d(worst)]}",
        )
    )


def compare_key_set(
    name: str, observed: Any, golden: Any, report: SceneMismatchReport
) -> None:
    """Hard-fail when two collections do not cover the same keys."""
    if sorted(observed) == sorted(golden):
        return
    report.diffs.append(
        FieldDiff(
            name=name,
            passed=False,
            hard=True,
            dataset_value=sorted(golden),
            sim_value=sorted(observed),
            margin=float("inf"),
        )
    )


def malformed_golden(
    section: str, report: SceneMismatchReport, *, remedy: str
) -> None:
    """Report a golden that is missing a whole payload section as a hard failure."""
    report.diffs.append(
        FieldDiff(
            name=f"golden_malformed[{section}]",
            passed=False,
            hard=True,
            dataset_value=None,
            sim_value=section,
            margin=float("inf"),
            note=(
                f"the golden has no '{section}' section, so there is nothing to "
                f"compare this run against. {remedy}"
            ),
        )
    )


def extract_dataset_scene_spec(
    dataset_path: str | Path,
    episode_index: int,
    *,
    init_after_hl_action: str | None = None,
) -> DatasetSceneSpec:
    """Build the dataset's scene spec for one episode.

    ``init_after_hl_action`` uses the same cut as the eval queue so a
    mid-episode start is verified against the post-cut poses, not frame 0.
    """
    setups = load_setups(
        dataset_path,
        episode_indices=[episode_index],
        init_after_hl_action=init_after_hl_action,
    )
    if not setups:
        raise ValueError(
            f"load_setups returned no setup for episode_index={episode_index} "
            f"at {dataset_path}"
        )
    if len(setups) > 1:
        raise ValueError(
            f"episode_index={episode_index} matched {len(setups)} setups at "
            f"{dataset_path}; pass a single LeRobot root, not a collection."
        )
    setup = setups[0]
    info = _read_dataset_info(Path(dataset_path), setup)
    return dataset_scene_spec_from_setup(
        setup, dataset_path=str(dataset_path), info=info
    )


def dataset_scene_spec_from_setup(
    setup: dict,
    *,
    dataset_path: Optional[str] = None,
    info: Optional[dict] = None,
) -> DatasetSceneSpec:
    """Build a scene spec from an already-loaded setup dict."""
    ep_idx = setup.get("episode_index")
    if ep_idx is None:
        raise ValueError(
            f"setup {setup.get('episode_id', '<unknown>')!r} has no "
            "episode_index; cannot build a scene spec."
        )
    # Prefer the setup's own scene-identity columns; re-read provenance from disk
    # only for what the setup didn't carry.
    robot_id = setup.get("robot_id")
    scene_preset_id = setup.get("scene_id")
    if robot_id is None or scene_preset_id is None:
        provenance = _read_provenance(dataset_path or setup.get("source_dataset_path"))
        robot_id = robot_id if robot_id is not None else provenance.get("robot_id")
        scene_preset_id = (
            scene_preset_id if scene_preset_id is not None else provenance.get("scene_preset_id")
        )
    resolved_path = str(dataset_path or setup.get("source_dataset_path", ""))
    return DatasetSceneSpec(
        dataset_path=resolved_path,
        episode_index=int(ep_idx),
        episode_id=setup.get("episode_id"),
        robot_joint_positions=setup.get("robot_joint_positions"),
        robot_gripper_position=setup.get("robot_gripper_position"),
        blocks=setup.get("objects", {}) or {},
        has_table_column=bool(setup.get("has_table_column")),
        table_pose=setup.get("table_pose"),
        table_height_delta=setup.get("table_height_delta"),
        pddl_goal=setup.get("pddl_goal"),
        dataset_cube_scale=setup.get("cube_scale"),
        expert_videos=setup.get("expert_videos", {}) or {},
        info=info or {},
        dataset_robot_id=robot_id,
        dataset_scene_preset_id=scene_preset_id,
        recorded_camera_calib=_overlay_setup_intrinsics(
            _read_camera_calib(resolved_path), setup
        ),
    )


def _read_provenance(dataset_path: Optional[str | Path]) -> dict[str, Optional[str]]:
    """Best-effort read of recorded robot/scene provenance (empty if no path)."""
    if not dataset_path:
        return {}
    from toyblocks.utils.dataset_provenance import read_dataset_provenance

    return read_dataset_provenance(dataset_path)


def _read_camera_calib(dataset_path: Optional[str | Path]) -> dict[str, dict[str, Any]]:
    """Read recorded camera calib, or ``{}`` when the dataset predates the columns."""
    if not dataset_path:
        return {}
    from toyblocks.utils.dataset_provenance import (
        CameraCalibColumnsMissing,
        read_dataset_camera_calib,
    )

    try:
        return read_dataset_camera_calib(dataset_path)
    except CameraCalibColumnsMissing:
        return {}


def _overlay_setup_intrinsics(
    calib: dict[str, dict[str, Any]], setup: dict
) -> dict[str, dict[str, Any]]:
    """Replace dataset-level K with this episode's first-frame K when present."""
    setup_k = setup.get("camera_intrinsics")
    if not setup_k:
        return calib
    out = {cam: dict(entry) for cam, entry in calib.items()}
    for cam, k9 in setup_k.items():
        k33 = np.asarray(k9, dtype=float).reshape(3, 3).tolist()
        if cam in out:
            out[cam]["K"] = k33
        else:
            out[cam] = {"K": k33}
    return out


def _read_dataset_info(dataset_path: Path, setup: dict) -> dict:
    """Read the source root's ``meta/info.json`` (best-effort, informational)."""
    root = Path(setup.get("source_dataset_path", dataset_path))
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        return {}
    with open(info_path) as f:
        return json.load(f)


def read_sim_scene_state(base_env: Any, env_idx: int) -> SimSceneState:
    """Read the live env's spawned + post-reset state for one sub-env."""
    state = SimSceneState(env_idx=int(env_idx))

    cfg = getattr(base_env, "cfg", None)
    if cfg is not None:
        state.robot_id = getattr(cfg, "robot_id", None)
        state.scene_preset_id = getattr(cfg, "scene_preset_id", None)

    state.robot_usd_path = _read_robot_usd_path(base_env)

    origin = base_env.scene.env_origins[env_idx]
    rigid_objects = base_env.scene.rigid_objects

    table_name = "workspace_table"
    if table_name in rigid_objects:
        table = rigid_objects[table_name]
        table_local = (table.data.root_pos_w[env_idx] - origin).detach().cpu().tolist()
        state.table_local_pos = [float(x) for x in table_local]
        state.table_local_quat = [
            float(x) for x in table.data.root_quat_w[env_idx].detach().cpu().tolist()
        ]
        state.table_world_z = float(table.data.root_pos_w[env_idx][2].item())
        expected = getattr(base_env, EXPECTED_TABLE_Z_ATTR, None)
        if expected is not None and env_idx < len(expected):
            state.expected_table_z_w = expected[env_idx]

    block_sizes = _read_block_sizes(base_env, env_idx)
    for name, obj in rigid_objects.items():
        # SCENE_FURNITURE_PREFIXES is the single furniture source of truth.
        if name.startswith(SCENE_FURNITURE_PREFIXES):
            continue
        local_pos = (obj.data.root_pos_w[env_idx] - origin).detach().cpu().tolist()
        quat = obj.data.root_quat_w[env_idx].detach().cpu().tolist()
        state.blocks[name] = {
            "position": [float(x) for x in local_pos],
            "quaternion": [float(x) for x in quat],
        }
        if name in block_sizes:
            state.block_sizes[name] = block_sizes[name]

    state.cameras = _read_camera_specs(base_env, env_idx, origin)
    return state


def _read_robot_usd_path(base_env: Any) -> Optional[str]:
    """Resolve the spawned robot USD path from the articulation cfg."""
    robot = getattr(base_env, "robot", None)
    articulation = getattr(robot, "articulation", None)
    cfg = getattr(articulation, "cfg", None)
    spawn = getattr(cfg, "spawn", None)
    usd_path = getattr(spawn, "usd_path", None)
    return str(usd_path) if usd_path is not None else None


def _read_block_sizes(base_env: Any, env_idx: int) -> dict[str, list[float]]:
    """Read per-block spawn sizes from the codegen ``randomizations`` table."""
    env_scene_profile = getattr(base_env, "env_scene_profile", None)
    if callable(env_scene_profile):
        return dict(env_scene_profile(env_idx)["block_sizes"])
    randomizations = getattr(base_env, "randomizations", None)
    if not isinstance(randomizations, dict):
        return {}
    block_sizes = randomizations.get("block_sizes")
    if not isinstance(block_sizes, dict):
        return {}
    out: dict[str, list[float]] = {}
    for name, sizes in block_sizes.items():
        if isinstance(sizes, torch.Tensor):
            sizes = sizes.detach().cpu().tolist()
        if not isinstance(sizes, (list, tuple)) or not sizes:
            continue
        # One row per cube-scale slot; packages without env_scene_profile spawn
        # env i at slot i % #slots.
        row = sizes[env_idx % len(sizes)]
        if isinstance(row, (list, tuple)):
            out[name] = [float(x) for x in row]
    return out


def _read_camera_specs(
    base_env: Any, env_idx: int, origin: Any
) -> dict[str, dict[str, Any]]:
    """Read live per-camera world-frame calibration off the spawned sensors."""
    cameras = getattr(base_env, "cameras", None)
    if not cameras:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for name, cam in cameras.items():
        data = getattr(cam, "data", None)
        if data is None:
            continue
        entry: dict[str, Any] = {
            "prim_path": getattr(getattr(cam, "cfg", None), "prim_path", None),
        }
        K = getattr(data, "intrinsic_matrices", None)
        if K is not None:
            entry["K"] = K[env_idx].detach().cpu().tolist()
        pos_w = getattr(data, "pos_w", None)
        if pos_w is not None:
            local = (pos_w[env_idx] - origin).detach().cpu().tolist()
            entry["pos"] = [float(x) for x in local]
        for conv in ("world", "ros", "opengl"):
            quat = getattr(data, f"quat_w_{conv}", None)
            if quat is not None:
                entry[f"quat_w_{conv}"] = [
                    float(x) for x in quat[env_idx].detach().cpu().tolist()
                ]
        shape = getattr(data, "image_shape", None)
        if shape is not None:
            entry["height"], entry["width"] = int(shape[0]), int(shape[1])
        else:
            cfg = getattr(cam, "cfg", None)
            entry["width"] = getattr(cfg, "width", None)
            entry["height"] = getattr(cfg, "height", None)
        out[str(name)] = entry
    return out


def compare_scene(
    spec: DatasetSceneSpec,
    sim_state: SimSceneState,
    *,
    table_z_tol: float = DEFAULT_TABLE_Z_TOL,
    block_pos_tol: float = DEFAULT_BLOCK_POS_TOL,
    block_quat_tol: float = DEFAULT_BLOCK_QUAT_TOL,
    camera_intrinsics_tol: float = DEFAULT_CAMERA_INTRINSICS_TOL,
    camera_pos_tol: float = DEFAULT_CAMERA_POS_TOL,
    camera_rot_tol_rad: float = DEFAULT_CAMERA_ROT_TOL_RAD,
) -> SceneMismatchReport:
    """Diff the dataset spec against the live sim state into a structured report."""
    report = SceneMismatchReport(
        dataset_path=spec.dataset_path,
        episode_index=spec.episode_index,
        sim_context={
            "robot_id": sim_state.robot_id,
            "scene_preset_id": sim_state.scene_preset_id,
            "robot_usd_path": sim_state.robot_usd_path,
            "block_sizes": sim_state.block_sizes,
            "cameras": sim_state.cameras,
        },
    )

    _compare_provenance(spec, sim_state, report)
    _compare_block_presence(spec, sim_state, report)
    _compare_table(spec, sim_state, report, table_z_tol)
    _compare_block_poses(spec, sim_state, report, block_pos_tol, block_quat_tol)
    _compare_block_scale(spec, sim_state, report)
    _compare_cameras(
        spec, sim_state, report,
        camera_intrinsics_tol, camera_pos_tol, camera_rot_tol_rad,
    )
    return report


def _compare_block_scale(
    spec: DatasetSceneSpec,
    sim_state: SimSceneState,
    report: SceneMismatchReport,
    size_tol: float = DEFAULT_BLOCK_SIZE_TOL,
) -> None:
    """Check the spawned block side length against the recorded cube_scale."""
    if spec.dataset_cube_scale is None:
        return
    cs = float(spec.dataset_cube_scale)
    if not sim_state.block_sizes:
        # Hard, not soft: the recorded block poses are only valid at the recorded
        # size, which is exactly what a size-routing miss breaks.
        report.diffs.append(
            FieldDiff(
                name="cube_scale",
                passed=False,
                hard=True,
                dataset_value=cs,
                sim_value=None,
                note=(
                    "dataset records cube_scale but the sim exposed no block sizes, so "
                    "the spawned block size cannot be checked against it. Block scale is "
                    "non-sim-state, so an unverifiable size means unverifiable results."
                ),
            )
        )
        return
    for name in sorted(sim_state.block_sizes):
        size = sim_state.block_sizes[name]
        diff = max(abs(float(s) - cs) for s in size)
        report.diffs.append(
            FieldDiff(
                name=f"cube_scale[{name}]",
                passed=diff <= size_tol,
                hard=True,
                dataset_value=cs,
                sim_value=size,
                diff=diff,
                tol=size_tol,
            )
        )


def _compare_cameras(
    spec: DatasetSceneSpec,
    sim_state: SimSceneState,
    report: SceneMismatchReport,
    intrinsics_tol: float,
    pos_tol: float,
    rot_tol_rad: float,
) -> None:
    """Check recorded camera intrinsics and pose against the live readback.

    Camera-set differences are soft, since a preset may legitimately use a
    different camera set than datagen.
    """
    calib = spec.recorded_camera_calib or {}
    if not calib:
        return
    sim_cams = sim_state.cameras or {}
    for name in sorted(calib):
        rec = calib[name]
        sim = sim_cams.get(name)
        if sim is None:
            report.diffs.append(
                FieldDiff(
                    name=f"camera[{name}]",
                    passed=True,
                    hard=False,
                    dataset_value=name,
                    sim_value=sorted(sim_cams.keys()),
                    note="recorded in dataset but absent from the live scene; not verified",
                )
            )
            continue
        _compare_camera_intrinsics(name, rec, sim, report, intrinsics_tol)
        _compare_camera_pose(name, rec, sim, report, pos_tol, rot_tol_rad)

    for name in sorted(set(sim_cams) - set(calib)):
        report.diffs.append(
            FieldDiff(
                name=f"camera[{name}]",
                passed=True,
                hard=False,
                dataset_value=sorted(calib.keys()),
                sim_value=name,
                note="present in the live scene but no recorded calib to verify against",
            )
        )


def _compare_camera_intrinsics(
    name: str,
    rec: dict,
    sim: dict,
    report: SceneMismatchReport,
    intrinsics_tol: float,
) -> None:
    sim_K = sim.get("K")
    if sim_K is None:
        report.diffs.append(
            FieldDiff(
                name=f"camera_intrinsics[{name}]",
                passed=True,
                hard=False,
                dataset_value=rec["K"],
                sim_value=None,
                note="live camera exposed no intrinsic_matrices; cannot verify",
            )
        )
        return
    diff = float(np.max(np.abs(np.array(rec["K"]) - np.array(sim_K))))
    report.diffs.append(
        FieldDiff(
            name=f"camera_intrinsics[{name}]",
            passed=diff <= intrinsics_tol,
            hard=True,
            dataset_value=rec["K"],
            sim_value=sim_K,
            diff=diff,
            tol=intrinsics_tol,
        )
    )


def _compare_camera_pose(
    name: str,
    rec: dict,
    sim: dict,
    report: SceneMismatchReport,
    pos_tol: float,
    rot_tol_rad: float,
) -> None:
    # World-fixed (hard-checked) requires both signals: the data says constant
    # across episodes AND the prim isn't under the robot. Either alone is
    # ambiguous (single-episode shards, unusual prim layouts).
    prim_path = sim.get("prim_path") or ""
    mounted = (not rec.get("pose_world_fixed", True)) or ("/Robot/" in prim_path)
    hard = not mounted
    note = None if hard else (
        "arm-mounted camera: world pose moves with the robot (recorded live per "
        "frame; compared at the episode's first frame) and is intentionally not "
        "overridden, so pose is reported soft"
    )

    # Frames rendered from a transform the eval scene cannot reproduce. Reported
    # here rather than in the reader, which has no prim path to tell the two
    # cases apart.
    if rec.get("pose_moves_within_episode") and "/Robot/" not in prim_path:
        report.diffs.append(
            FieldDiff(
                name=f"camera_pose[{name}].static",
                passed=False,
                hard=True,
                dataset_value="pose varies within an episode",
                sim_value=prim_path or None,
                note=(
                    "recorded pose moves during the episode but the camera is not "
                    "mounted on the robot prim, so its recorded extrinsics are not "
                    "a scene constant eval can match"
                ),
            )
        )

    sim_pos = sim.get("pos")
    if sim_pos is not None:
        pos_diff = float(np.linalg.norm(np.array(rec["pos"]) - np.array(sim_pos)))
        report.diffs.append(
            FieldDiff(
                name=f"camera_pose[{name}].position",
                passed=pos_diff <= pos_tol,
                hard=hard,
                dataset_value=rec["pos"],
                sim_value=sim_pos,
                diff=pos_diff,
                tol=pos_tol,
                note=note,
            )
        )

    # The recorder writes IsaacLab's scalar-first quat_w_world through its
    # world->local transform (see utils/dataset_provenance.py), so a missing key
    # is a broken readback, not a cue to try another convention.
    rec_wxyz = np.asarray(rec["quat_wxyz"], dtype=float)
    sim_q = sim["quat_w_world"]
    angle = _quat_geodesic_angle(rec_wxyz, sim_q)
    report.diffs.append(
        FieldDiff(
            name=f"camera_pose[{name}].rotation",
            passed=angle <= rot_tol_rad,
            hard=hard,
            dataset_value=rec["quat_wxyz"],
            sim_value=sim_q,
            diff=angle,
            tol=rot_tol_rad,
            note=note,
        )
    )


def _quat_geodesic_angle(a_wxyz: np.ndarray, b_wxyz: list[float]) -> float:
    """Geodesic angle in radians between two unit quaternions."""
    dot = abs(float(np.dot(a_wxyz, np.array(b_wxyz, dtype=float))))
    return float(2.0 * np.arccos(min(1.0, dot)))


def _compare_provenance(
    spec: DatasetSceneSpec, sim_state: SimSceneState, report: SceneMismatchReport
) -> None:
    """Compare recorded robot/scene provenance against the live env.

    A mismatch is hard because the recorded block Z and ``table_height_delta``
    are relative to the datagen preset's table baseline.
    """
    for name, dataset_value, sim_value in (
        ("robot_id", spec.dataset_robot_id, sim_state.robot_id),
        ("scene_preset_id", spec.dataset_scene_preset_id, sim_state.scene_preset_id),
    ):
        if dataset_value is None:
            report.diffs.append(
                FieldDiff(
                    name=name,
                    passed=True,
                    hard=False,
                    dataset_value=None,
                    sim_value=sim_value,
                    note=(
                        "dataset has no recorded "
                        f"{name}; cannot verify the eval matches datagen "
                        "(see docs/source/user_guide/workflows/evals/launch_eval.md 'Dataset scene provenance')"
                    ),
                )
            )
            continue
        matched = str(dataset_value) == str(sim_value)
        report.diffs.append(
            FieldDiff(
                name=name,
                passed=matched,
                hard=True,
                dataset_value=dataset_value,
                sim_value=sim_value,
                note=None if matched else (
                    f"the eval built a different {name} than datagen recorded, so the "
                    "recorded block/table geometry describes another scene. Regenerate "
                    "the task package from this dataset (generate_task_from_dataset.py "
                    f"bakes the recorded {name}) or drop the conflicting "
                    "--env-cfg override."
                ),
            )
        )


def _compare_block_presence(
    spec: DatasetSceneSpec, sim_state: SimSceneState, report: SceneMismatchReport
) -> None:
    dataset_names = spec.block_names
    sim_names = sim_state.block_names
    if not dataset_names:
        report.diffs.append(
            FieldDiff(
                name="blocks_present",
                passed=True,
                hard=False,
                dataset_value=sorted(dataset_names),
                sim_value=sorted(sim_names),
                note="dataset has no object_pose.* block columns to compare",
            )
        )
        return
    missing = sorted(dataset_names - sim_names)
    extra = sorted(sim_names - dataset_names)
    passed = not missing and not extra
    report.diffs.append(
        FieldDiff(
            name="blocks_present",
            passed=passed,
            hard=True,
            dataset_value=sorted(dataset_names),
            sim_value=sorted(sim_names),
            note=None if passed else f"missing_in_sim={missing} extra_in_sim={extra}",
        )
    )


def _compare_table(
    spec: DatasetSceneSpec,
    sim_state: SimSceneState,
    report: SceneMismatchReport,
    table_z_tol: float,
) -> None:
    sim_table_z = sim_state.table_local_pos[2] if sim_state.table_local_pos else None

    if not spec.has_table_column or spec.table_pose is None:
        # No absolute table pose, so the only anchor is table_height_delta, which
        # apply_setup adds to the preset baseline. With neither there is no
        # ground truth and no fallback (apply_setup raises), so hard-fail rather
        # than guess the surface from block positions.
        if spec.table_height_delta is not None:
            # The baseline the delta is relative to is gone once apply_setup has
            # written the pose, so check against the world z it targeted. Catches
            # the delta not landing, e.g. a later randomization overwriting it.
            if sim_state.expected_table_z_w is None:
                report.diffs.append(
                    FieldDiff(
                        name="table_height",
                        passed=False,
                        hard=True,
                        dataset_value=f"delta={spec.table_height_delta:+.4f}",
                        sim_value=sim_table_z,
                        note=(
                            "dataset records table_height_delta="
                            f"{spec.table_height_delta:+.4f} but this env has no "
                            "recorded expected table z, so apply_setup's table write "
                            "never ran for it. The recorded block Z is relative to the "
                            "datagen table, so the scene cannot be trusted. See "
                            "docs/source/user_guide/workflows/evals/launch_eval.md 'Dataset scene provenance'."
                        ),
                    )
                )
                return
            diff = abs(float(sim_state.table_world_z) - float(sim_state.expected_table_z_w))
            report.diffs.append(
                FieldDiff(
                    name="table_height",
                    passed=diff <= table_z_tol,
                    hard=True,
                    dataset_value=(
                        f"delta={spec.table_height_delta:+.4f} "
                        f"-> expected_world_z={sim_state.expected_table_z_w:.4f}"
                    ),
                    sim_value=sim_state.table_world_z,
                    diff=diff,
                    tol=table_z_tol,
                    note=(
                        "dataset has NO absolute table pose, so the height is anchored "
                        "by table_height_delta added to the preset baseline (the preset "
                        "itself anchored by the recorded scene_id). This checks the "
                        "reconstruction took; that the baseline is the right one is "
                        "confirmed by the first-frame image overlay. See "
                        "docs/source/user_guide/workflows/evals/launch_eval.md 'Dataset scene provenance'."
                    ),
                )
            )
            return
        report.diffs.append(
            FieldDiff(
                name="table_height",
                passed=False,
                hard=True,
                dataset_value=None,
                sim_value=sim_table_z,
                note=(
                    "dataset records NO absolute table pose AND NO table_height_delta, "
                    "so the eval table height cannot be anchored to datagen. Record "
                    "table_height_delta (or object_pose.table) at datagen, or pin "
                    "scene_preset_id. See docs/source/user_guide/workflows/evals/launch_eval.md 'Dataset scene provenance'."
                ),
            )
        )
        return
    if sim_state.table_local_pos is None:
        report.diffs.append(
            FieldDiff(
                name="table_height",
                passed=False,
                hard=True,
                dataset_value=spec.table_pose["position"][2],
                sim_value=None,
                note="no workspace_table in scene to read table height from",
            )
        )
        return
    dataset_z = float(spec.table_pose["position"][2])
    sim_z = float(sim_state.table_local_pos[2])
    diff = abs(dataset_z - sim_z)
    report.diffs.append(
        FieldDiff(
            name="table_height",
            passed=diff <= table_z_tol,
            hard=True,
            dataset_value=dataset_z,
            sim_value=sim_z,
            diff=diff,
            tol=table_z_tol,
        )
    )


def _compare_block_poses(
    spec: DatasetSceneSpec,
    sim_state: SimSceneState,
    report: SceneMismatchReport,
    block_pos_tol: float,
    block_quat_tol: float,
) -> None:
    for name in sorted(spec.block_names & sim_state.block_names):
        ds = spec.blocks[name]
        sim = sim_state.blocks[name]
        pos_diff = float(
            np.max(np.abs(np.array(ds["position"]) - np.array(sim["position"])))
        )
        quat_diff = _quat_distance(ds["quaternion"], sim["quaternion"])
        report.diffs.append(
            FieldDiff(
                name=f"block_pose[{name}].position",
                passed=pos_diff <= block_pos_tol,
                hard=True,
                dataset_value=ds["position"],
                sim_value=sim["position"],
                diff=pos_diff,
                tol=block_pos_tol,
            )
        )
        report.diffs.append(
            FieldDiff(
                name=f"block_pose[{name}].quaternion",
                passed=quat_diff <= block_quat_tol,
                hard=True,
                dataset_value=ds["quaternion"],
                sim_value=sim["quaternion"],
                diff=quat_diff,
                tol=block_quat_tol,
            )
        )


def _quat_distance(a: list[float], b: list[float]) -> float:
    """Sign-invariant chord distance between two unit quaternions."""
    dot = abs(float(np.dot(np.array(a), np.array(b))))
    return 1.0 - min(dot, 1.0)


# --------------------------- first-frame image comparison ---------------------------


def compare_first_frames(
    spec: DatasetSceneSpec,
    obs: dict,
    env_idx: int,
    output_dir: str | Path,
    *,
    compute_ssim: bool = True,
    cameras: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compare the dataset's expert first frames against the post-reset sim obs.

    Writes MAE / MSE / SSIM metrics and a side-by-side PNG per camera.
    ``cameras`` limits the comparison to an allowlist of short camera names.
    """
    from toyblocks.utils.wandb_logging import _extract_expert_first_frame

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_images = obs.get("images") if isinstance(obs, dict) else None
    if not sim_images:
        raise ValueError(
            "obs has no 'images' key; first-frame comparison needs RGB "
            "observations (run eval with an rgb obs_mode)."
        )
    if not spec.expert_videos:
        raise ValueError(
            f"episode {spec.episode_index} has no expert_videos; cannot "
            "compare first frames."
        )

    allowed = None if cameras is None else set(cameras)
    per_camera: dict[str, dict[str, Any]] = {}
    for ds_key, video in spec.expert_videos.items():
        # RGB only; skip depth (observation.depth.*) and other modalities.
        if not ds_key.startswith(_IMAGE_KEY_PREFIX):
            continue
        cam = camera_short_name(ds_key, _IMAGE_KEY_PREFIX)
        if allowed is not None and cam not in allowed:
            continue
        if cam not in sim_images or "rgb" not in sim_images[cam]:
            logger.warning(
                "expert camera %r (sim key %r) absent from obs['images'] %s; skipping",
                ds_key, cam, sorted(sim_images.keys()),
            )
            continue
        dataset_frame = _extract_expert_first_frame(
            video["path"], video.get("from_timestamp")
        )
        if dataset_frame is None:
            raise RuntimeError(
                f"failed to decode expert first frame for camera {cam} "
                f"from {video['path']}"
            )
        sim_frame = _sim_frame_for_camera(sim_images, cam, env_idx)
        metrics, png_path = _compare_one_camera(
            cam, dataset_frame, sim_frame, out_dir, compute_ssim
        )
        per_camera[cam] = {**metrics, "png": str(png_path)}

    if not per_camera:
        dataset_rgb_cams = sorted(
            camera_short_name(k, _IMAGE_KEY_PREFIX)
            for k in spec.expert_videos
            if k.startswith(_IMAGE_KEY_PREFIX)
        )
        raise ValueError(
            "no overlapping RGB cameras between dataset expert_videos "
            f"(rgb cameras: {dataset_rgb_cams}; all keys: "
            f"{sorted(spec.expert_videos.keys())}), obs['images'] "
            f"({sorted(sim_images.keys())}) and the requested allowlist "
            f"({sorted(allowed) if allowed is not None else 'all'}); cannot "
            "compare first frames."
        )

    result = {
        "cameras": per_camera,
        "caveat": _FIRST_FRAME_CALIB_CAVEAT,
    }
    metrics_json = out_dir / f"first_frame_metrics_ep{spec.episode_index}.json"
    with open(metrics_json, "w") as f:
        json.dump(result, f, indent=2)
    result["metrics_json"] = str(metrics_json)
    return result


def _sim_frame_for_camera(
    sim_images: dict, cam: str, env_idx: int
) -> np.ndarray:
    """Extract the sim RGB frame for one camera and env as HxWx3 uint8.

    Goes through the recorder's own ``_extract_frame`` so verification compares
    the frame the recorder would have saved.
    """
    from toyblocks.utils.dataset_recording_utils import _extract_frame

    env_slice = {"images": {cam: {"rgb": sim_images[cam]["rgb"][env_idx]}}}
    frame = _extract_frame(env_slice, cam)
    if frame is None:
        raise ValueError(
            f"sim obs camera {cam!r} produced no frame for env {env_idx}"
        )
    return frame


def _compare_one_camera(
    cam: str,
    dataset_frame: np.ndarray,
    sim_frame: np.ndarray,
    out_dir: Path,
    compute_ssim: bool,
) -> tuple[dict[str, Any], Path]:
    """Compute metrics and write the side-by-side PNG for one camera."""
    if dataset_frame.shape[-1] != 3 or sim_frame.shape[-1] != 3:
        raise ValueError(
            f"camera {cam}: expected 3-channel RGB frames, got dataset "
            f"{dataset_frame.shape} sim {sim_frame.shape}"
        )
    sim_shape = sim_frame.shape
    if dataset_frame.shape != sim_shape:
        # Downsize the dataset frame rather than upscaling sim, which would blur
        # sim and depress SSIM.
        dataset_frame = _resize(dataset_frame, sim_shape[:2])
    logger.info(
        "first-frame compare cam=%s: dataset %s -> sim %s (RGB assumed)",
        cam, dataset_frame.shape, sim_shape,
    )

    diff = np.abs(dataset_frame.astype(np.float32) - sim_frame.astype(np.float32))
    metrics: dict[str, Any] = {
        "mae": float(diff.mean()),
        "mse": float((diff ** 2).mean()),
        "shape": list(sim_shape),
    }

    ssim_map = None
    if compute_ssim:
        ssim_metrics, ssim_map = _ssim_metrics(dataset_frame, sim_frame)
        metrics.update(ssim_metrics)

    png_path = out_dir / f"first_frame_{cam}.png"
    _write_comparison_png(
        cam, dataset_frame, sim_frame, diff, metrics, png_path, ssim_map
    )
    return metrics, png_path


def _ssim_metrics(
    dataset_frame: np.ndarray, sim_frame: np.ndarray
) -> tuple[dict[str, float], np.ndarray]:
    """Compute an SSIM family that separates lighting from geometry differences.

    ``ssim_gray`` rising above ``ssim`` indicates a color difference,
    ``ssim_edge`` is near lighting-invariant, and the ``*_photonorm`` variants
    histogram-match sim to the dataset frame to factor exposure out.
    """
    from skimage.color import rgb2gray
    from skimage.exposure import match_histograms
    from skimage.filters import sobel
    from skimage.metrics import structural_similarity

    # Canonical Wang et al. (2004) params, not skimage's noisier 7x7 default.
    ssim_kwargs = dict(
        gaussian_weights=True, sigma=1.5, use_sample_covariance=False, win_size=11,
    )

    ssim_color, ssim_map = structural_similarity(
        dataset_frame, sim_frame,
        channel_axis=-1, data_range=255, full=True, **ssim_kwargs,
    )
    ds_gray = rgb2gray(dataset_frame)
    sm_gray = rgb2gray(sim_frame)
    ssim_gray = structural_similarity(ds_gray, sm_gray, data_range=1.0, **ssim_kwargs)
    ssim_edge = structural_similarity(
        sobel(ds_gray), sobel(sm_gray), data_range=1.0, **ssim_kwargs,
    )

    sim_matched = match_histograms(sim_frame, dataset_frame, channel_axis=-1)
    mae_photonorm = float(
        np.abs(dataset_frame.astype(np.float32) - sim_matched.astype(np.float32)).mean()
    )
    ssim_photonorm = structural_similarity(
        dataset_frame, sim_matched.astype(np.uint8),
        channel_axis=-1, data_range=255, **ssim_kwargs,
    )

    metrics = {
        "ssim": float(ssim_color),
        "ssim_gray": float(ssim_gray),
        "ssim_edge": float(ssim_edge),
        "ssim_photonorm": float(ssim_photonorm),
        "mae_photonorm": mae_photonorm,
    }
    color_map = ssim_map.mean(axis=-1) if ssim_map.ndim == 3 else ssim_map
    return metrics, color_map


def _resize(img: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Resize an HxWx3 uint8 image to ``(h, w)`` using PIL (no extra deps)."""
    from PIL import Image

    return np.asarray(Image.fromarray(img).resize((hw[1], hw[0]), Image.BILINEAR))


def verify_setups_against_sim(
    base_env: Any,
    obs: dict,
    setups: list[Optional[dict]],
    output_dir: str | Path,
    *,
    compute_first_frames: bool = True,
    strict: bool = False,
) -> list[SceneMismatchReport]:
    """Run scene and first-frame verification on a batch of live-eval setups.

    Writes a combined ``scene_verification.json`` to ``output_dir``. ``strict``
    raises on a hard mismatch instead of only logging it.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reports: list[SceneMismatchReport] = []
    combined: list[dict] = []
    has_images = isinstance(obs, dict) and bool(obs.get("images"))
    for env_idx, setup in enumerate(setups):
        if setup is None:
            continue
        if setup.get("episode_index") is None:
            logger.warning(
                "env %d setup has no episode_index; skipping scene verification",
                env_idx,
            )
            continue
        spec = dataset_scene_spec_from_setup(setup)
        sim_state = read_sim_scene_state(base_env, env_idx)
        report = compare_scene(spec, sim_state)
        record = report.to_dict()
        record["env_idx"] = env_idx

        if compute_first_frames and has_images and spec.expert_videos:
            record["first_frames"] = compare_first_frames(spec, obs, env_idx, out_dir)

        combined.append(record)
        reports.append(report)
        if report.ok:
            logger.info(
                "scene verification env=%d episode=%s OK", env_idx, spec.episode_index
            )
        else:
            logger.error(
                "scene verification env=%d episode=%s MISMATCH:\n%s",
                env_idx, spec.episode_index, report.format_console(),
            )

    report_path = out_dir / "scene_verification.json"
    with open(report_path, "w") as f:
        json.dump(combined, f, indent=2)
    logger.info(
        "wrote scene verification for %d env(s) to %s", len(combined), report_path
    )

    if strict:
        failed = [r for r in reports if not r.ok]
        if failed:
            raise RuntimeError(
                f"scene verification failed for {len(failed)}/{len(reports)} env(s) "
                f"(episodes {[r.episode_index for r in failed]}); see {report_path}. "
                "The eval scene does not match the dataset it replays, so any metric "
                "measured here describes a different scene. Drop "
                "--verify_scene_strict to treat this as advisory."
            )
    return reports


def _write_comparison_png(
    cam: str,
    dataset_frame: np.ndarray,
    sim_frame: np.ndarray,
    diff: np.ndarray,
    metrics: dict[str, Any],
    png_path: Path,
    ssim_map: Optional[np.ndarray] = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    diff_gray = diff.mean(axis=-1)
    n_panels = 4 if ssim_map is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    axes[0].imshow(dataset_frame)
    axes[0].set_title(f"dataset ({cam})")
    axes[1].imshow(sim_frame)
    axes[1].set_title("sim (post-reset)")
    im = axes[2].imshow(diff_gray, cmap="inferno")
    ssim_txt = f" ssim={metrics['ssim']:.3f}" if "ssim" in metrics else ""
    axes[2].set_title(f"|diff| mae={metrics['mae']:.1f} mse={metrics['mse']:.1f}{ssim_txt}")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    if ssim_map is not None:
        # Localizes disagreement (a moved block vs global lighting); dark regions
        # are where the frames diverge most.
        im2 = axes[3].imshow(ssim_map, cmap="viridis", vmin=0, vmax=1)
        edge_txt = f" edge={metrics['ssim_edge']:.3f}" if "ssim_edge" in metrics else ""
        axes[3].set_title(f"SSIM map{edge_txt}")
        fig.colorbar(im2, ax=axes[3], fraction=0.046, pad=0.04)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(png_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
