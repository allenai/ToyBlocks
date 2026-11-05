"""Replay a rendered dataset's recorded camera pose and appearance in eval.

The render stage gives every episode its own camera offsets and look (HDRI, table
material, local lights, exposure, hidden backdrop) and records the applied values per
episode in ``meta/render_params.json``. A merged dataset keeps one such file per source
problem, reachable through ``meta/provenance.json`` ``sources[].source_dataset``.

:class:`RecordedRenderReplay` maps each queue setup to its record and re-applies it with
the same DR tools the render used (``interpreter.setup_prims`` / ``apply_record``,
``camera._apply_camera_local_offsets``):

* the merged episode's local index is its rank among the merged episodes that share its
  ``source_problem``; a per-problem count check and a per-episode field check (front
  camera position, ``table_height_delta``) reject records that do not describe the
  episode. A rejected episode gets nothing applied, never a run failure, and its
  ``render_replay_status`` says why.
* scene-global parts (sky, dome, fill, sun, exposure) are one value for the whole stage,
  so they are applied separately from the per-env parts (materials, local lights,
  background sphere, cameras); see ``apply_staged``.

:class:`WaveGate` is the bookkeeping for ``--wave_resets``: envs that finish are parked
until every active env is parked, then released together so a wave shares its globals.

Module-level imports are stdlib + numpy only: ``scripts/evaluate.py`` reads the kit args
from here before ``launch_app()``.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from toyblocks.utils.episode_dataset import _discover_lerobot_roots, _resolve_parquet_files

logger = logging.getLogger(__name__)

RENDER_PARAMS_RELPATH = Path("meta") / "render_params.json"
PROVENANCE_RELPATH = Path("meta") / "provenance.json"

STATUS_APPLIED = "applied"
STATUS_COUNT_MISMATCH = "count_mismatch"
STATUS_FIELD_MISMATCH = "field_mismatch"
STATUS_NO_RECORD = "no_record"

# Record parts that are one value for the whole stage (dome/sky, sphere fill, shared
# sun, tonemap exposure): a batch can only show one episode's values at a time.
GLOBAL_RECORD_KEYS: tuple[str, ...] = (
    "skybox", "lighting_intensity", "fill_intensity", "sun", "exposure",
)
CAMERA_POSITION_TOL_M = 1e-4
TABLE_HEIGHT_DELTA_TOL_M = 1e-5
# Flags whose effect (block glyphs / colours, rewritten goals) render replay leaves alone:
# the dataset rows record them (glyph_map / block_variable_map), and
# --replay_block_identity paints them per episode (episode_queue.apply_setup).
_UNREPLAYED_FLAGS: tuple[str, ...] = ("randomize_task_letters", "randomize_block_colors")
# Keys IsaacLab's balanced.kit re-applies after launch, discarding a launch-time value
# (see domain_randomization/YAML_REFERENCE.md, `carb_settings`).
_BALANCED_KIT_KEYS: frozenset[str] = frozenset({
    "/rtx/translucency/enabled",
    "/rtx/reflections/enabled",
    "/rtx/reflections/denoiser/enabled",
    "/rtx/directLighting/sampledLighting/denoisingTechnique",
    "/rtx/directLighting/sampledLighting/enabled",
    "/rtx/sceneDb/ambientLightIntensity",
    "/rtx/shadows/enabled",
    "/rtx/indirectDiffuse/enabled",
    "/rtx/indirectDiffuse/denoiser/enabled",
    "/rtx/domeLight/upperLowerStrategy",
    "/rtx/ambientOcclusion/enabled",
    "/rtx/ambientOcclusion/denoiserMode",
    "/rtx/raytracing/subpixel/mode",
    "/rtx/raytracing/cached/enabled",
    "/rtx/post/dlss/execMode",
    "/rtx/post/aa/op",
})
# Tiled cameras spawned directly under the env Xform; their USD local pose is env-local.
_ENV_PARENT_RE = re.compile(r"(.*/env_(\.\*|\d+)|\{ENV_REGEX_NS\})$")
_CAMERA_SENSOR_PREFIX = "camera:"


def kit_args_from_flags(flags: dict[str, Any] | None) -> str:
    """``DRSession.kit_args(flags)`` without importing torch, so it can run before launch.

    Kept identical to ``DRSession.kit_args`` (pinned by a unit test).
    """
    cs = (flags or {}).get("carb_settings")
    if not isinstance(cs, dict) or not cs:
        return ""

    def kv(k: Any, v: Any) -> str:
        v = ("true" if v else "false") if isinstance(v, bool) else v
        return f"--/{str(k).lstrip('/')}={v}"

    return " ".join(kv(k, v) for k, v in cs.items())


def balanced_kit_collisions(flags: dict[str, Any] | None) -> list[str]:
    """``carb_settings`` keys that balanced.kit overrides after launch (silently lost)."""
    cs = (flags or {}).get("carb_settings") or {}
    return sorted(k for k in cs if "/" + str(k).lstrip("/") in _BALANCED_KIT_KEYS)


def merge_kit_args(existing: str, extra: str) -> str:
    """Append ``extra`` to a ``--kit_args`` value, skipping settings already present."""
    have = set(existing.split())
    added = [tok for tok in extra.split() if tok not in have]
    return " ".join([existing.strip(), *added]).strip()


def _short_camera_name(key: str) -> str:
    return key.rpartition(":")[-1]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        logger.warning("render replay: could not read %s (%s)", path, e)
        return None


def _scan_episode_problems(root: Path) -> dict[int, str | None]:
    """``episode_index -> source_problem`` from the frame-0 rows of every parquet shard."""
    import pyarrow.parquet as pq

    out: dict[int, str | None] = {}
    for pq_path in _resolve_parquet_files(root):
        names = pq.read_schema(str(pq_path)).names
        cols = [c for c in ("episode_index", "source_problem") if c in names]
        if "episode_index" not in cols:
            continue
        if "frame_index" in names:
            filters: list[tuple[str, str, Any]] | None = [("frame_index", "==", 0)]
        elif "is_first" in names:
            filters = [("is_first", "==", True)]
        else:
            filters = None
        table = pq.read_table(str(pq_path), columns=cols, filters=filters).to_pydict()
        problems = table.get("source_problem") or [None] * len(table["episode_index"])
        for ep, sp in zip(table["episode_index"], problems):
            if isinstance(sp, (list, tuple)):
                sp = sp[0] if sp else None
            if isinstance(sp, bytes):
                sp = sp.decode("utf-8")
            out.setdefault(int(ep), sp.strip() if isinstance(sp, str) and sp.strip() else None)
    return out


@dataclass
class _RootRecords:
    """Lazily loaded render_params records for one LeRobot root.

    ``render_params`` maps problem name to its file; the single key ``None`` means the
    root itself is one render output (no merge, every episode in one file).
    """

    root: Path
    render_params: dict[str | None, Path]
    flags_sha256: str | None
    episode_problems: dict[int, str | None] | None = None
    _ranks: dict[int, tuple[str | None, int]] | None = None
    _counts: dict[str | None, int] | None = None
    _records: dict[str | None, dict[int, dict[str, Any]] | None] = field(default_factory=dict)
    _record_counts: dict[str | None, int] = field(default_factory=dict)
    _count_warned: set[str | None] = field(default_factory=set)

    @property
    def merged(self) -> bool:
        return None not in self.render_params

    def _build_index(self) -> None:
        if self.episode_problems is None:
            self.episode_problems = _scan_episode_problems(self.root)
        by_problem: dict[str | None, list[int]] = {}
        for ep, problem in self.episode_problems.items():
            by_problem.setdefault(problem if self.merged else None, []).append(ep)
        self._ranks = {}
        self._counts = {}
        for problem, eps in by_problem.items():
            eps.sort()
            self._counts[problem] = len(eps)
            for rank, ep in enumerate(eps):
                self._ranks[ep] = (problem, rank)

    def locate(self, episode_index: int) -> tuple[str | None, int, int] | None:
        """``(problem, local_index, merged_count)`` of a merged episode, or None if unknown."""
        if self._ranks is None:
            self._build_index()
        assert self._ranks is not None and self._counts is not None
        hit = self._ranks.get(int(episode_index))
        if hit is None:
            return None
        problem, rank = hit
        return problem, rank, self._counts[problem]

    def records(self, problem: str | None) -> tuple[dict[int, dict[str, Any]] | None, int]:
        """The problem's records keyed by local episode index, plus how many the file lists."""
        if problem not in self._records:
            path = self.render_params.get(problem)
            payload = _read_json(path) if path is not None and path.is_file() else None
            if payload is None:
                self._records[problem] = None
                self._record_counts[problem] = 0
            else:
                sha = payload.get("flags_sha256")
                if self.flags_sha256 is not None and sha != self.flags_sha256:
                    raise ValueError(
                        f"render replay: {path} was rendered with flags_sha256={sha!r} but the "
                        f"scene was built from flags_sha256={self.flags_sha256!r}; one eval "
                        "process can only replay a dataset rendered with a single flags yaml"
                    )
                episodes = payload.get("episodes") or []
                self._records[problem] = {
                    int(e.get("episode_index", k)): e for k, e in enumerate(episodes)
                }
                self._record_counts[problem] = len(episodes)
        return self._records[problem], self._record_counts[problem]

    def count_mismatch_warned(self, problem: str | None) -> bool:
        if problem in self._count_warned:
            return True
        self._count_warned.add(problem)
        return False


def _root_records(root: Path) -> tuple[_RootRecords, dict[str, Any]] | None:
    """The root's render_params layout plus the flags payload of its first readable file."""
    direct = root / RENDER_PARAMS_RELPATH
    if direct.is_file():
        payload = _read_json(direct)
        if payload is None:
            return None
        return _RootRecords(root, {None: direct}, payload.get("flags_sha256")), payload
    provenance = _read_json(root / PROVENANCE_RELPATH) if (root / PROVENANCE_RELPATH).is_file() else None
    sources = (provenance or {}).get("sources") or []
    paths: dict[str | None, Path] = {}
    for src in sources:
        name, src_dir = src.get("problem_name"), src.get("source_dataset")
        if name and src_dir:
            paths[str(name)] = Path(src_dir) / RENDER_PARAMS_RELPATH
    for path in paths.values():
        if path.is_file():
            payload = _read_json(path)
            if payload is not None:
                return _RootRecords(root, paths, payload.get("flags_sha256")), payload
            break
    return None


class RecordedRenderReplay:
    """Per-episode replay of a dataset's recorded render parameters into the eval scene.

    Lifecycle: ``from_dataset`` (pure) -> ``kit_args`` (pre-launch) -> ``configure_build``
    (before ``gym.make``) -> ``bind`` (sim exists) -> per reset ``stage`` + ``apply_staged``.
    """

    def __init__(
        self,
        roots: dict[str, _RootRecords],
        flags: dict[str, Any],
        render_args: dict[str, Any] | None = None,
        *,
        assets_paths: Sequence[str] = (),
    ) -> None:
        self._roots = roots
        self.flags: dict[str, Any] = flags
        self._render_args: dict[str, Any] = dict(render_args or {})
        # HDRI roots searched before the repo's assets/environments (render's --assets_paths).
        self._assets_paths: tuple[str, ...] = tuple(str(p) for p in assets_paths)
        self._warned_textures: set[str] = set()
        self._base_env: Any = None
        self._prims: Any = None
        self._static_cams: dict[str, Any] = {}
        self._link_cams: dict[str, Any] = {}
        self._staged: dict[int, tuple[dict[str, Any] | None, dict[str, Any] | None]] = {}
        # Record each env's scene currently shows (None: nothing applied / rejected).
        self._current: dict[int, dict[str, Any] | None] = {}
        self._status_counts: dict[str, int] = {}

    # ---- pure: lookup + checks ------------------------------------------------------
    @classmethod
    def from_dataset(
        cls, dataset_path: str | Path, *, assets_paths: Sequence[str] = ()
    ) -> RecordedRenderReplay | None:
        """Replay for ``dataset_path``, or None (info log) when it has no render_params.json."""
        path = Path(dataset_path)
        roots: dict[str, _RootRecords] = {}
        flags_payload: dict[str, Any] | None = None
        for root in _discover_lerobot_roots(path) if path.is_dir() else []:
            found = _root_records(root)
            if found is None:
                continue
            rr, payload = found
            if flags_payload is None:
                flags_payload = payload
            elif rr.flags_sha256 != flags_payload.get("flags_sha256"):
                raise ValueError(
                    f"render replay: {root} was rendered with flags_sha256={rr.flags_sha256!r}, "
                    f"other roots under {path} with {flags_payload.get('flags_sha256')!r}"
                )
            roots[str(root)] = rr
        if flags_payload is None:
            logger.info("render replay off: no %s under %s", RENDER_PARAMS_RELPATH, path)
            return None
        return cls(
            roots,
            flags_payload.get("flags") or {},
            flags_payload.get("args"),
            assets_paths=assets_paths,
        )

    def kit_args(self) -> str:
        return kit_args_from_flags(self.flags)

    def _root_for(self, setup: dict[str, Any]) -> _RootRecords | None:
        src = setup.get("source_dataset_path")
        if src is not None and str(src) in self._roots:
            return self._roots[str(src)]
        if src is None and len(self._roots) == 1:
            return next(iter(self._roots.values()))
        return None

    def record_for(
        self, setup: dict[str, Any] | None
    ) -> tuple[dict[str, Any] | None, str]:
        """``(record, status)`` for a queue setup; record is None unless status is applied."""
        if not setup or setup.get("episode_index") is None:
            return None, STATUS_NO_RECORD
        ep = int(setup["episode_index"])
        root = self._root_for(setup)
        loc = root.locate(ep) if root is not None else None
        if root is None or loc is None:
            logger.warning("render replay: episode %d has no render_params record", ep)
            return None, STATUS_NO_RECORD
        problem, local, merged_count = loc
        records, record_count = root.records(problem)
        if records is None:
            logger.warning(
                "render replay: episode %d (problem %s) has no readable render_params.json",
                ep, problem,
            )
            return None, STATUS_NO_RECORD
        if record_count != merged_count:
            if not root.count_mismatch_warned(problem):
                logger.warning(
                    "render replay off for problem %s: the merged dataset has %d episode(s) "
                    "but its render_params.json lists %d, so the position-based lookup is "
                    "not trustworthy", problem, merged_count, record_count,
                )
            return None, STATUS_COUNT_MISMATCH
        rec = records.get(local)
        if rec is None:
            logger.warning(
                "render replay: problem %s has no record for local episode %d (merged %d)",
                problem, local, ep,
            )
            return None, STATUS_NO_RECORD
        mismatch = self._field_mismatch(setup, rec)
        if mismatch is not None:
            logger.warning(
                "render replay: merged episode %d -> problem %s local episode %d rejected: %s",
                ep, problem, local, mismatch,
            )
            return None, STATUS_FIELD_MISMATCH
        return rec, STATUS_APPLIED

    def _field_mismatch(self, setup: dict[str, Any], rec: dict[str, Any]) -> str | None:
        poses: dict[str, list[float]] = setup.get("camera_poses") or {}
        rec_cams: dict[str, dict[str, Any]] = rec.get("cameras") or {}
        for key, cam in rec_cams.items():
            short = _short_camera_name(key)
            if self._link_cams and key in self._link_cams:
                continue  # the recorded pose of an arm camera is not a scene constant
            pose = poses.get(short)
            if pose is None or "position" not in cam:
                continue
            d = float(np.linalg.norm(np.asarray(pose[:3]) - np.asarray(cam["position"][:3])))
            if d > CAMERA_POSITION_TOL_M:
                return f"{short} position differs by {d:.2e} m"
        rec_delta = rec.get("table_height_delta")
        setup_delta = setup.get("table_height_delta")
        if rec_delta is not None and setup_delta is not None:
            d = abs(float(rec_delta) - float(setup_delta))
            if d > TABLE_HEIGHT_DELTA_TOL_M:
                return f"table_height_delta differs by {d:.2e} m"
        return None

    # ---- build time -------------------------------------------------------------------
    def configure_build(self, env_cfg: Any) -> None:
        """Render's build-time scene config (env spacing, GI, render mode, AA). Before gym.make."""
        from toyblocks.domain_randomization.build import apply_build_flags

        for name in _UNREPLAYED_FLAGS:
            if self.flags.get(name):
                logger.warning(
                    "render replay: flags use %s, which render replay does not apply; "
                    "pass --replay_block_identity to paint each episode's recorded block "
                    "glyphs/colours, otherwise they stay as the eval task spawns them", name,
                )
        lost = balanced_kit_collisions(self.flags)
        if lost:
            logger.warning(
                "render replay: carb_settings %s are re-applied by balanced.kit after launch, "
                "so the render's values are lost", lost,
            )
        apply_build_flags(
            env_cfg,
            self.flags,
            antialiasing=self._render_args.get("antialiasing"),
            disable_dlss=bool(self._render_args.get("disable_dlss", False)),
            enable_gi=bool(self._render_args.get("enable_gi", False)),
        )

    def bind(self, base_env: Any) -> None:
        """Create the DR prims and run render's post-build stage mutators. Sim must exist."""
        from toyblocks.domain_randomization import build as dr_build
        from toyblocks.domain_randomization.assets import AssetPaths
        from toyblocks.domain_randomization.camera import _is_link_mounted
        from toyblocks.domain_randomization.interpreter import setup_prims
        from toyblocks.domain_randomization.resolve import build_basename_index

        assets = AssetPaths.with_search_paths(list(self._assets_paths))
        num_hdris = len(build_basename_index(assets.hdri_dirs))
        if num_hdris == 0 and (self.flags.get("skybox") or self.flags.get("env_sphere")):
            logger.warning(
                "render replay: no HDRI found under %s, so the recorded skybox and backdrop "
                "cannot be shown; pass --render_replay_assets_paths <dir>/assets/environments",
                list(assets.hdri_dirs),
            )
        self._base_env = base_env
        self._prims = setup_prims(base_env, self.flags, base_env.num_envs, assets=assets)
        dr_build.apply_lighting_flags(self.flags, base_env)
        dr_build.hide_debug_visuals()
        dr_build.hide_scene_objects(self.flags)
        for key, sensor in base_env.scene.sensors.items():
            if not key.startswith(_CAMERA_SENSOR_PREFIX) or getattr(sensor, "_view", None) is None:
                continue
            (self._link_cams if _is_link_mounted(sensor) else self._static_cams)[key] = sensor
        logger.info(
            "render replay bound: %d env(s), %d HDRI(s) under %s, static cameras %s, "
            "arm cameras %s", base_env.num_envs, num_hdris, list(assets.hdri_dirs),
            sorted(self._static_cams), sorted(self._link_cams),
        )

    @property
    def bound(self) -> bool:
        return self._prims is not None

    # ---- per reset ---------------------------------------------------------------------
    def stage(self, env_idx: int, setup: dict[str, Any] | None) -> str:
        """Look up the env's next record; writes ``render_replay_status`` into ``setup``."""
        rec, status = self.record_for(setup)
        self._staged[int(env_idx)] = (rec, setup)
        self._status_counts[status] = self._status_counts.get(status, 0) + 1
        if setup is not None:
            setup["render_replay_status"] = status
        return status

    @property
    def status_counts(self) -> dict[str, int]:
        return dict(self._status_counts)

    def apply_staged(self, env_ids: Iterable[int], *, apply_globals: bool) -> None:
        """Write the staged records for ``env_ids`` to USD (no render).

        Per-env parts for every staged env. With ``apply_globals`` the scene-global parts
        come from the lowest env id in ``env_ids`` whose scene shows a record (staged now
        or earlier). Envs without a record get their cameras back on ``cfg.offset`` so a
        previous episode's pose does not carry over.
        """
        if not self.bound:
            return
        from toyblocks.domain_randomization.interpreter import apply_record

        ids = sorted(int(e) for e in env_ids)
        for i in ids:
            if i not in self._staged:
                continue
            rec, setup = self._staged.pop(i)
            self._current[i] = rec
            if rec is None:
                self._apply_default_cameras(i)
                continue
            self._warn_unresolved_textures(rec)
            per_env = {k: v for k, v in rec.items() if k not in GLOBAL_RECORD_KEYS}
            apply_record(self._prims, self._base_env, per_env, active=(i,))
            if "local_lights" in rec and not rec["local_lights"]:
                self._zero_local_lights(i, rec.get("episode_index", 0))
            self._apply_cameras(i, rec, setup or {})
        if not apply_globals:
            return
        leader_idx = next((i for i in ids if self._current.get(i) is not None), None)
        if leader_idx is None:
            return
        leader = self._current[leader_idx]
        assert leader is not None
        globals_only = {k: leader[k] for k in GLOBAL_RECORD_KEYS if k in leader}
        globals_only["episode_index"] = leader.get("episode_index", 0)
        # apply_record turns off the local lights of a record that lists none, and the
        # per-env lights were already set from each env's own record above.
        global_prims = copy.copy(self._prims)
        global_prims.local_spec = None
        apply_record(global_prims, self._base_env, globals_only, active=(leader_idx,))

    def _warn_unresolved_textures(self, rec: dict[str, Any]) -> None:
        """apply_record hands an unresolvable HDRI basename through as-is (no sky shown)."""
        assets = getattr(self._prims, "assets", None)
        if assets is None:
            return
        from toyblocks.domain_randomization.resolve import resolve_basename

        for part in ("skybox", "env_sphere"):
            tex = (rec.get(part) or {}).get("texture")
            if not tex or tex in self._warned_textures:
                continue
            if not Path(resolve_basename(tex, assets.hdri_dirs)).is_file():
                self._warned_textures.add(tex)
                logger.warning(
                    "render replay: %s texture %r not found under %s",
                    part, tex, list(assets.hdri_dirs),
                )

    def _zero_local_lights(self, env_idx: int, episode_key: int) -> None:
        """A record with ``local_lights: []`` had its lights off; apply_record skips it."""
        if not getattr(self._prims, "local_prims", None):
            return
        from toyblocks.domain_randomization.lights import _apply_local_lights

        wave = [{"episode_index": episode_key} for _ in range(env_idx + 1)]
        origins = self._base_env.scene.env_origins.cpu().numpy()
        _apply_local_lights(self._prims.local_prims, wave, [env_idx], {}, origins, enabled=False)

    def _apply_cameras(self, env_idx: int, rec: dict[str, Any], setup: dict[str, Any]) -> None:
        """Static cameras from the recorded parquet pose; arm cameras from the recorded offset."""
        from toyblocks.domain_randomization.camera import (
            _apply_camera_local_offsets,
            offsets_from_absolute,
        )

        key = int(rec.get("episode_index", 0))
        offsets = offsets_from_absolute({}, [rec]).get(key, {})
        poses: dict[str, list[float]] = setup.get("camera_poses") or {}
        by_offset: dict[str, Any] = {}
        for cam_key, cam in self._static_cams.items():
            pose = poses.get(_short_camera_name(cam_key))
            if pose is not None and self._set_static_camera_pose(env_idx, cam_key, cam, pose):
                continue
            if cam_key in offsets:
                by_offset[cam_key] = cam
        for cam_key, cam in self._link_cams.items():
            if cam_key in offsets:
                by_offset[cam_key] = cam
        if by_offset:
            wave = [{"episode_index": key} for _ in range(env_idx + 1)]
            _apply_camera_local_offsets(
                by_offset, wave, [env_idx], {key: offsets}, self._base_env.device
            )

    def _set_static_camera_pose(
        self, env_idx: int, cam_key: str, cam: Any, pose: Sequence[float]
    ) -> bool:
        """USD local pose of an env-mounted camera from a robot-base-frame recorded pose."""
        import torch
        from isaaclab.utils.math import (
            convert_camera_frame_orientation_convention,
            quat_apply,
            quat_mul,
        )

        parent = str(getattr(cam.cfg, "prim_path", "")).rpartition("/")[0]
        arr = np.asarray(pose, dtype=np.float32).reshape(-1)
        if (
            not _ENV_PARENT_RE.match(parent)
            or arr.size != 7
            or not np.isfinite(arr).all()
            or float(np.abs(arr[3:]).max()) == 0.0
        ):
            return False
        env = self._base_env
        device = env.device
        root = env.robot.articulation.data.root_pose_w[env_idx : env_idx + 1]
        p_root = root[:, :3] - env.scene.env_origins[env_idx : env_idx + 1]
        q_root = root[:, 3:7]
        p_rec = torch.tensor(arr[:3], device=device, dtype=torch.float32).unsqueeze(0)
        q_rec = torch.tensor(arr[3:], device=device, dtype=torch.float32).unsqueeze(0)
        pos = p_root + quat_apply(q_root, p_rec)
        quat_gl = convert_camera_frame_orientation_convention(
            quat_mul(q_root, q_rec), origin="world", target="opengl"
        )
        cam._view.set_local_poses(pos, quat_gl, indices=[env_idx])
        return True

    def _apply_default_cameras(self, env_idx: int) -> None:
        from toyblocks.domain_randomization.camera import _apply_camera_local_offsets

        cams = {**self._static_cams, **self._link_cams}
        if not cams:
            return
        zero = {"position_offset_cm": [0.0, 0.0, 0.0], "quat_delta": [1.0, 0.0, 0.0, 0.0]}
        wave = [{"episode_index": -1} for _ in range(env_idx + 1)]
        _apply_camera_local_offsets(
            cams, wave, [env_idx], {-1: {k: dict(zero) for k in cams}}, self._base_env.device
        )


class WaveGate:
    """Which envs are parked waiting for the rest of their wave.

    An env is parked when its episode ends (it has already been reset into its next
    setup). The wave is ready once every *active* env -- one with a live setup -- is
    parked; envs without a setup (queue drained) never block it.
    """

    def __init__(self, num_envs: int) -> None:
        self._num_envs = int(num_envs)
        self._parked: set[int] = set()

    def park(self, env_ids: Iterable[int]) -> None:
        self._parked.update(int(i) for i in env_ids)

    @property
    def parked(self) -> list[int]:
        return sorted(self._parked)

    def ready(self, active_env_ids: Iterable[int]) -> bool:
        active = {int(i) for i in active_env_ids}
        if not self._parked:
            return False
        return active.issubset(self._parked)

    def release(self) -> list[int]:
        wave = sorted(self._parked)
        self._parked.clear()
        return wave
