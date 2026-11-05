"""Live-sim verification of what the policy actually receives.

Every CPU test of the policy-input pipeline feeds a hand-built observation, so
the *input side* of the contract is an assumption: if ``_get_observations``
renames a key, changes depth units, changes the ``ee_pos`` frame, or a scene
preset changes camera resolution, the whole CPU suite stays green because its
goldens are computed over a fixture rather than over sim. Closing that is the one
gap that cannot be closed on CPU, and this module is the leg that closes it.

It is called from ``scripts/eval/verify_dataset_loading.py``'s ``verify_episode``
rather than from a script of its own, because that function already has, at once:
a live env reset into a recorded episode's setup through the **real**
``apply_setup`` path with cameras rendered, the dataset's ground-truth spec,
the ``target_env`` matching the episode's ``cube_scale``, recorded camera
intrinsics already ingested, and a ``SceneMismatchReport`` whose ``FieldDiff``
list drives the console output, the margin ranking, the report JSON and the exit
code. Appending diffs to that report means a policy-input failure is reported,
uploaded and exit-coded with no new CI step and no second Isaac cold start.

Two tiers, deliberately split:

* **Against the dataset, no golden.** The arm block of the raw ``agent_pos``
  equals the episode's recorded joints, the emitted ``camera_intrinsics`` equal
  the recorded per-camera K rescaled to the model grid, depth is metric,
  ``ee_pos`` is env-local, RGB is non-degenerate. This mirrors the script's
  stated philosophy -- there is nothing to re-baseline -- which matters here
  specifically because which episodes a datagen run produces depends on which
  policies happened to succeed, so a numeric golden keyed to an episode is
  exactly the staleness the script already rejects for dynamics.
* **One golden: the observation contract.** Per scene preset, since the preset
  decides camera names and native resolution. Structure only -- keys, dtypes,
  ranks, the native resolution, and the emitted wire layout -- with no
  episode-specific numbers, which is what makes it stable across datagen runs.
  It is also the seam that lets the CPU lane check its synthetic fixture's layout
  against the real observation (``tests/serving/test_fixture_matches_sim.py``).

The transform is built as ``maniflow_blocks_table`` at ``state_dim=19`` with both
depth keys -- the widest input, so one construction covers every branch -- and
left at its **default** interpolate device, which makes this the only place the
CUDA resize path is exercised at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from toyblocks.eval.scene_verification import (
    DatasetSceneSpec,
    FieldDiff,
    SceneMismatchReport,
    SimSceneState,
    compare_array,
    compare_key_set,
    malformed_golden,
)
from toyblocks.serving.obs_transforms import BaseObsTransform, _depth_as_nhw

if TYPE_CHECKING:
    # Deferred so this module stays importable (and the CPU parity test cheap)
    # without pulling torch in: the transform is constructed inside
    # :func:`build_policy_input_transform`, not at import time.
    from toyblocks.serving.obs_transforms.maniflow import ManiflowObsTransform
    from toyblocks.serving.obs_transforms.maniflow_blocks_table import (
        ManiflowBlocksTableObsTransform,
    )
#: The widest ManiFlow state layout, so one transform covers every state block.
POLICY_INPUT_STATE_DIM = 19

#: Both depth cameras, so the depth + intrinsics branch and the two-row
#: ``camera_intrinsics`` stacking order are both live-covered.
POLICY_INPUT_DEPTH_KEYS = ("front_depth", "gripper_depth")

#: ``apply_setup`` writes joints directly with no settle step, so the state the
#: policy sees should equal the recorded state to well under a milliradian. Kept
#: at 1e-3 rather than tighter only to absorb the float32 round-trip through the
#: dataset parquet and the sim's articulation buffers.
JOINT_TOL_RAD = 1e-3
GRIPPER_TOL_M = 1e-3

#: ``ingest_dataset_camera_intrinsics`` ran in the warm-up reset, so live K *is*
#: recorded K; this only absorbs the float32 rescale.
INTRINSICS_TOL_PX = 1e-2

#: A metres/millimetres flip pushes every sample past the transform's depth
#: cutoff and
#: ``process_depth`` zeroes it, so the surviving fraction is the unit check. The
#: floor is loose on purpose: the real bound to catch is "almost nothing
#: survived", and the legitimate fraction varies a lot between the wrist camera
#: (near field, mostly valid) and a front camera clipping at 2.4 m.
MIN_VALID_DEPTH_FRACTION = 0.05

#: ``agent.ee_pos`` is the ``ee_body_name`` link minus ``env_origins``, so it is
#: env-local. With ``env_spacing=5.0`` on ``sim2real`` a lost origin subtraction
#: is metres out, which this box catches without pretending the value is
#: comparable to the recorder's world-frame ``observation.state.ee_position``.
EE_LOCAL_BOX_XY_M = 2.0
EE_LOCAL_BOX_Z_M = (-1.0, 2.0)

#: Per-channel standard deviation floor, in 0-255 units, so a black or constant
#: frame fails rather than passing every shape and range check.
MIN_RGB_CHANNEL_STD = 1.0

OBS_CONTRACT_GOLDEN_PREFIX = "policy_input_obs_contract_"

REGEN_HINT = (
    "Re-record it on a GPU box with the datasets present:\n"
    "    ./tests/tamp/datagen.sh --mode all --task flip_pnp --record-data --no-ui\n"
    "    ./tests/eval/scene_load.sh --write-policy-input-golden\n"
    "then review `git diff tests/eval/goldens/` and commit it."
)


def contract_golden_path(golden_dir: str | Path, scene_preset_id: str) -> Path:
    """Where a scene preset's observation-contract golden lives."""
    return Path(golden_dir) / f"{OBS_CONTRACT_GOLDEN_PREFIX}{scene_preset_id}.json"


def build_policy_input_transform() -> ManiflowBlocksTableObsTransform:
    """The transform this leg drives, at the widest layout.

    Left at the default ``image_interpolate_device`` on purpose: everything in
    ``tests/serving`` pins ``"cpu"`` so its goldens do not depend on the runner
    having a GPU, which leaves the CUDA resize kernel -- the one every real eval
    actually uses -- covered nowhere else.
    """
    from toyblocks.serving.obs_transforms.maniflow_blocks_table import (
        ManiflowBlocksTableObsTransform,
    )

    return ManiflowBlocksTableObsTransform(
        state_dim=POLICY_INPUT_STATE_DIM,
        depth_keys=POLICY_INPUT_DEPTH_KEYS,
    )


# ---------------------------------------------------------------------------
# The structural signature of the live observation
# ---------------------------------------------------------------------------


def _leaf(obs: dict[str, Any], path: str) -> np.ndarray:
    """The array at a ``/``-separated observation path.

    Raises rather than returning ``None``: every path asked for here is one the
    transform itself reads, so a missing one is the renamed key this leg exists
    to catch, and it should name the path.
    """
    node: Any = obs
    for part in path.split("/"):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(
                f"the live observation has no {path!r} (missing {part!r}; available: "
                f"{sorted(node) if isinstance(node, dict) else type(node).__name__}). "
                "The ManiFlow transform reads this key, so the observation "
                "contract changed."
            )
        node = node[part]
    return np.asarray(node)


def _spatial_descriptor(arr: np.ndarray) -> dict[str, Any]:
    """Rank and channel layout of a camera leaf, with the resolution elided.

    Resolution is recorded separately because it is preset-specific while the
    layout is not -- which is what lets the CPU fixture (180x320) be checked
    against live sim (640x360) for the thing that actually matters.
    """
    return {
        "dtype": str(arr.dtype),
        "ndim": int(arr.ndim),
        # 4 dims means a trailing channel axis: 3 for RGB, 1 for the depth
        # layouts that keep one. ``_depth_as_nhw`` accepts both, so which one it
        # is belongs in the contract rather than in an assumption.
        "trailing_channels": int(arr.shape[-1]) if arr.ndim == 4 else None,
    }


def _fixed_descriptor(arr: np.ndarray) -> dict[str, Any]:
    """dtype and batch-elided shape of a leaf whose shape is resolution-free."""
    return {"dtype": str(arr.dtype), "shape": [int(n) for n in arr.shape[1:]]}


def observed_obs_contract(
    obs: dict[str, Any],
    transform: ManiflowObsTransform,
    *,
    scene_preset_id: str,
    robot_id: str,
    obs_mode: str,
) -> dict[str, Any]:
    """The structural signature of an observation and the wire it produces.

    ``obs`` is a **numpy, batched** observation dict. Split into three sections
    so each can be compared where it is meaningful:

    * ``structure`` -- resolution-independent, so both the live observation and
      the CPU fixture can be held to it;
    * ``resolution`` -- native camera ``[H, W]``, which the scene preset decides;
    * ``wire`` -- what the transform emits, on the model's fixed 224 grid, so it
      is resolution-independent too.

    ``bands`` (min/max/mean per numeric leaf) is returned alongside but is
    episode-specific and never compared; it rides into the report's context so a
    failure is readable without a second run.
    """
    external, wrist = transform.external_cam, transform.wrist_cam

    fixed_paths = ["agent/joint_pos"]
    if transform.include_joint_vel:
        fixed_paths.append("agent/joint_vel")
    if transform.include_ee_pos:
        fixed_paths.append("agent/ee_pos")

    spatial_paths = [f"images/{external}/rgb", f"images/{wrist}/rgb"]
    for depth_key in transform.depth_keys:
        cam = getattr(transform, transform.DEPTH_KEY_CAMERAS[depth_key])
        spatial_paths.append(f"images/{cam}/depth")
        fixed_paths.append(f"images/{cam}/intrinsics")

    structure: dict[str, Any] = {}
    bands: dict[str, Any] = {}
    for path in sorted(fixed_paths):
        arr = _leaf(obs, path)
        structure[path] = _fixed_descriptor(arr)
        bands[path] = _numeric_band(arr)
    for path in sorted(spatial_paths):
        arr = _leaf(obs, path)
        structure[path] = _spatial_descriptor(arr)
        bands[path] = _numeric_band(arr)

    resolution: dict[str, list[int]] = {}
    for cam in (external, wrist):
        rgb = _leaf(obs, f"images/{cam}/rgb")
        resolution[cam] = [int(rgb.shape[-3]), int(rgb.shape[-2])]

    wire = transform(obs)
    wire_desc: dict[str, Any] = {}
    for key in sorted(wire):
        value = wire[key]
        if isinstance(value, np.ndarray):
            wire_desc[key] = {
                "dtype": str(value.dtype),
                "shape": [int(n) for n in value.shape[1:]],
            }
        elif isinstance(value, (list, tuple)):
            wire_desc[key] = {"dtype": "list", "shape": []}
        else:
            wire_desc[key] = {"dtype": type(value).__name__, "shape": []}

    return {
        "identity": {
            "scene_preset_id": scene_preset_id,
            "robot_id": robot_id,
            "obs_mode": obs_mode,
            "transform_class": type(transform).__name__,
            "state_dim": int(POLICY_INPUT_STATE_DIM),
            "depth_keys": list(transform.depth_keys),
            "cameras": [external, wrist],
        },
        "structure": structure,
        "resolution": resolution,
        "wire": wire_desc,
        "bands": bands,
    }


def _numeric_band(arr: np.ndarray) -> dict[str, float]:
    """min / max / mean over the finite samples of a leaf, for the report only."""
    flat = arr.astype(np.float64).reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan"),
                "finite_fraction": 0.0}
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "finite_fraction": float(finite.size) / float(flat.size),
    }


def _compare_descriptor_map(
    name: str,
    observed: dict[str, Any],
    golden: dict[str, Any],
    report: SceneMismatchReport,
) -> None:
    """Diff two ``{path: descriptor}`` maps: key set first, then field by field."""
    compare_key_set(f"{name}[keys]", observed, golden, report)
    for path in sorted(set(observed) & set(golden)):
        obs_desc, gold_desc = observed[path], golden[path]
        for field in sorted(set(obs_desc) | set(gold_desc)):
            got, want = obs_desc.get(field), gold_desc.get(field)
            report.diffs.append(
                FieldDiff(
                    name=f"{name}[{path}].{field}",
                    passed=got == want,
                    hard=True,
                    dataset_value=want,
                    sim_value=got,
                    margin=None if got == want else float("inf"),
                )
            )


def compare_obs_contract(
    observed: dict[str, Any],
    golden_dir: str | Path,
    report: SceneMismatchReport,
) -> None:
    """Diff the observation contract against its committed per-preset golden.

    A missing golden is a hard failure carrying the recapture command, not a
    skip: an unverified contract is the state this leg exists to remove, and the
    CPU-lane parity test reads the same file.
    """
    preset = observed["identity"]["scene_preset_id"]
    path = contract_golden_path(golden_dir, preset)
    if not path.is_file():
        report.diffs.append(
            FieldDiff(
                name=f"obs_contract_golden_missing[{preset}]",
                passed=False,
                hard=True,
                dataset_value=str(path),
                sim_value=None,
                margin=float("inf"),
                note=(
                    "no committed observation contract for this scene preset, so "
                    "there is nothing to check the live observation's layout "
                    f"against. {REGEN_HINT}"
                ),
            )
        )
        return

    with open(path) as f:
        golden = json.load(f)

    for section in ("identity", "structure", "resolution", "wire"):
        if section not in golden:
            malformed_golden(f"obs_contract.{section}", report, remedy=REGEN_HINT)
            return

    _compare_descriptor_map("obs_contract.structure", observed["structure"],
                            golden["structure"], report)
    _compare_descriptor_map("obs_contract.wire", observed["wire"], golden["wire"], report)

    compare_key_set(
        "obs_contract.identity[keys]", observed["identity"], golden["identity"], report
    )
    for key in sorted(set(observed["identity"]) & set(golden["identity"])):
        got, want = observed["identity"][key], golden["identity"][key]
        report.diffs.append(
            FieldDiff(
                name=f"obs_contract.identity[{key}]",
                passed=got == want,
                hard=True,
                dataset_value=want,
                sim_value=got,
                margin=None if got == want else float("inf"),
            )
        )

    compare_key_set(
        "obs_contract.resolution[keys]", observed["resolution"], golden["resolution"], report
    )
    for cam in sorted(set(observed["resolution"]) & set(golden["resolution"])):
        report.diffs.append(
            FieldDiff(
                name=f"obs_contract.resolution[{cam}]",
                passed=observed["resolution"][cam] == golden["resolution"][cam],
                hard=True,
                dataset_value=golden["resolution"][cam],
                sim_value=observed["resolution"][cam],
                margin=None
                if observed["resolution"][cam] == golden["resolution"][cam]
                else float("inf"),
                note="the scene preset's native camera resolution decides the "
                "intrinsics rescale, so a change here silently rescales K wrongly",
            )
        )


def write_obs_contract(
    observed: dict[str, Any], golden_dir: str | Path, report: SceneMismatchReport
) -> Path:
    """Re-record the contract golden, refusing to launder a failing run.

    ``bands`` is stripped: it is episode-specific, and a golden carrying it would
    go stale the moment a datagen run's success pattern shifted -- exactly the
    staleness the surrounding script rejects for dynamics.

    A ``raise`` rather than an ``assert``: this is the only gate stopping a broken
    run from re-baselining the contract every other check is read against, and
    ``python -O`` would strip an assert out of it.
    """
    if not report.ok:
        raise ValueError(
            "refusing to write the observation-contract golden from a run whose "
            f"dataset-derived checks already failed:\n{report.format_console()}"
        )
    payload = {k: v for k, v in observed.items() if k != "bands"}
    path = contract_golden_path(golden_dir, observed["identity"]["scene_preset_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


# ---------------------------------------------------------------------------
# Dataset-derived checks (no golden)
# ---------------------------------------------------------------------------


def _state_block_offsets(transform: ManiflowObsTransform) -> dict[str, int]:
    """Where each ``agent_pos`` block starts, from the transform's own flags.

    Derived rather than hardcoded to 19-dim indices: the flags are what assemble
    the state, so reading them is what makes this check fail on a reordering
    instead of silently reading the wrong column.
    """
    offset = transform.arm_joints
    offsets = {"arm": 0}
    if transform.include_joint_vel:
        offsets["jvel"] = offset
        offset += transform.arm_joints
    if transform.include_ee_pos:
        offsets["ee"] = offset
        offset += 3
    offsets["gripper"] = offset
    offset += 1
    if transform.include_pd_mode:
        offsets["pd_mode"] = offset
    return offsets


def _check_state_matches_recorded_setup(
    raw_state: np.ndarray,
    transform: ManiflowObsTransform,
    spec: DatasetSceneSpec,
    report: SceneMismatchReport,
) -> None:
    """The state the policy sees is the state the episode was recorded at.

    The strongest single assertion available here, and it is available only
    because ``apply_setup`` writes joints directly with no settle step. It
    catches an ``arm_joints`` / ``gripper_joint_index`` mistake, a joint
    reordering and a robot-config change in one comparison -- and unlike the
    scene tier's block poses, it is about the *robot*, which nothing else checks
    against the dataset.
    """
    offsets = _state_block_offsets(transform)
    if spec.robot_joint_positions is None:
        report.diffs.append(
            FieldDiff(
                name="policy_input.agent_pos_arm",
                passed=False,
                hard=True,
                dataset_value=None,
                sim_value="unverifiable",
                margin=float("inf"),
                note="the episode's setup carries no robot_joint_positions, so the "
                "state the policy sees cannot be checked against what was recorded",
            )
        )
    else:
        n = transform.arm_joints
        compare_array(
            "policy_input.agent_pos_arm",
            raw_state[offsets["arm"] : offsets["arm"] + n],
            np.asarray(spec.robot_joint_positions, dtype=np.float64)[:n],
            JOINT_TOL_RAD,
            report,
        )

    if spec.robot_gripper_position is None:
        report.diffs.append(
            FieldDiff(
                name="policy_input.agent_pos_gripper",
                passed=False,
                hard=True,
                dataset_value=None,
                sim_value="unverifiable",
                margin=float("inf"),
                note="the episode's setup carries no robot_gripper_position",
            )
        )
    else:
        compare_array(
            "policy_input.agent_pos_gripper",
            [float(raw_state[offsets["gripper"]])],
            [float(spec.robot_gripper_position)],
            GRIPPER_TOL_M,
            report,
        )


def _check_intrinsics_match_recorded_k(
    wire: dict[str, Any],
    obs: dict[str, Any],
    transform: ManiflowObsTransform,
    target_env: int,
    spec: DatasetSceneSpec,
    report: SceneMismatchReport,
) -> None:
    """Emitted ``camera_intrinsics`` equal recorded K rescaled to the model grid.

    Exact, because ``ingest_dataset_camera_intrinsics`` ran in the warm-up reset,
    so live K *is* recorded K. What this catches is the rescale itself: the two
    axes have different factors (a 640x360 frame squashed into 224x224 gives
    ``sx = 0.35`` and ``sy ~ 0.622``), so a single-factor bug, a swapped pair or a
    wrong row order all fail here and are invisible everywhere else.

    The native size is read from the **depth** leaf, because that is what
    ``_depth_inputs`` divides by -- so a depth stream at a different resolution
    than the RGB the recorded K describes is caught rather than absorbed.
    """
    rows = np.asarray(wire["camera_intrinsics"])[target_env, 0]  # (N_cams, 4)
    for row, depth_key in enumerate(transform.depth_keys):
        cam = getattr(transform, transform.DEPTH_KEY_CAMERAS[depth_key])
        calib = (spec.recorded_camera_calib or {}).get(cam)
        if not calib or calib.get("K") is None:
            report.diffs.append(
                FieldDiff(
                    name=f"policy_input.camera_intrinsics[{cam}]",
                    passed=False,
                    hard=True,
                    dataset_value=None,
                    sim_value="unverifiable",
                    margin=float("inf"),
                    note="the dataset records no camera calibration for this camera, "
                    "so the rescale the model's back-projection depends on cannot "
                    "be checked (see dataset_provenance.read_dataset_camera_calib)",
                )
            )
            continue

        depth_hw = _native_depth_hw(_leaf(obs, f"images/{cam}/depth"))
        rgb = _leaf(obs, f"images/{cam}/rgb")
        rgb_hw = (int(rgb.shape[-3]), int(rgb.shape[-2]))
        report.diffs.append(
            FieldDiff(
                name=f"policy_input.depth_matches_rgb_resolution[{cam}]",
                passed=depth_hw == rgb_hw,
                hard=True,
                dataset_value=list(rgb_hw),
                sim_value=list(depth_hw),
                margin=None if depth_hw == rgb_hw else float("inf"),
                note="the intrinsics rescale divides by the depth resolution while "
                "the recorded K describes the colour frame, so the two diverging "
                "rescales K wrongly rather than raising",
            )
        )
        native_h, native_w = depth_hw

        K = np.asarray(calib["K"], dtype=np.float64).reshape(3, 3)
        target_h, target_w = transform.image_size
        sx = target_w / float(native_w)
        sy = target_h / float(native_h)
        expected = np.array(
            [K[0, 0] * sx, K[1, 1] * sy, K[0, 2] * sx, K[1, 2] * sy], dtype=np.float64
        )
        compare_array(
            f"policy_input.camera_intrinsics[{cam}]",
            rows[row],
            expected,
            INTRINSICS_TOL_PX,
            report,
        )


def _native_depth_hw(depth: np.ndarray) -> tuple[int, int]:
    """``(H, W)`` of a depth leaf, whichever of the accepted layouts it uses."""
    if depth.ndim >= 3 and depth.shape[-1] == 1:
        return int(depth.shape[-3]), int(depth.shape[-2])
    return int(depth.shape[-2]), int(depth.shape[-1])


def _check_depth_is_metric(
    obs: dict[str, Any],
    transform: ManiflowObsTransform,
    target_env: int,
    report: SceneMismatchReport,
) -> None:
    """Enough depth survives ``process_depth`` for the units to be metres.

    This is a unit check dressed as a fraction. ``process_depth`` zeroes every
    sample past ``depth_max_m``, so a metres/millimetres flip multiplies
    everything by 1000 and zeroes essentially the whole frame -- a collapse no
    shape or dtype check sees.

    Uses the live sensor depth **before** the ptmap Z invalidation mask applied
    in :meth:`~toyblocks.serving.obs_transforms.maniflow.ManiflowObsTransform._depth_inputs`.
    That mask enforces training crop bounds (e.g. front depths above 1.0 m are
    dropped for ``sim2real``-style near-field front cameras) and can zero an
    entire ``droid_with_box_table`` front frame even when depth is correctly in
    metres -- which would false-fail this check.
    """
    cutoff = transform.depth_max_m
    for depth_key in transform.depth_keys:
        cam = getattr(transform, transform.DEPTH_KEY_CAMERAS[depth_key])
        raw = _depth_as_nhw(transform.get_depth(obs, cam))
        resized = BaseObsTransform.process_depth(
            raw,
            transform.image_size,
            max_depth_m=cutoff,
            interpolate_device=transform._image_interpolate_device,
        )
        depth = np.asarray(resized)[target_env]
        valid = depth > 0.0
        fraction = float(valid.mean())
        report.diffs.append(
            FieldDiff(
                name=f"policy_input.depth_valid_fraction[{depth_key}]",
                passed=fraction >= MIN_VALID_DEPTH_FRACTION,
                hard=True,
                dataset_value=f">= {MIN_VALID_DEPTH_FRACTION}",
                sim_value=fraction,
                diff=fraction,
                tol=MIN_VALID_DEPTH_FRACTION,
                margin=(MIN_VALID_DEPTH_FRACTION - fraction) / MIN_VALID_DEPTH_FRACTION,
                note=(
                    f"valid range [{float(depth[valid].min()):.3f}, "
                    f"{float(depth[valid].max()):.3f}] m against a "
                    f"{cutoff} m cutoff"
                    if valid.any()
                    else "no valid samples at all, which is what a millimetre-unit "
                    "depth stream looks like after the metre cutoff"
                ),
            )
        )


def _check_ee_pos_is_env_local(
    raw_state: np.ndarray, transform: ManiflowObsTransform, report: SceneMismatchReport
) -> None:
    """``ee_pos`` is finite and inside a box around the env's own origin.

    Not compared to the dataset, and that is the honest choice rather than a weak
    one: under the default ``legacy_ee_pos=False`` the recorder's
    ``observation.state.ee_position`` is the last body's world pose while
    ``agent.ee_pos`` is the ``ee_body_name`` link minus ``env_origins``, so an
    equality would fail by construction. The box pins the divergence that matters
    -- with ``env_spacing=5.0`` a lost origin subtraction is metres out -- rather
    than papering over the frame difference.
    """
    if not transform.include_ee_pos:
        return
    offsets = _state_block_offsets(transform)
    # Python floats, not numpy scalars: the comparisons below become the report's
    # ``passed`` field, and a numpy bool_ there is unserializable by json.dump.
    ee = [float(v) for v in raw_state[offsets["ee"] : offsets["ee"] + 3]]
    in_box = (
        bool(np.all(np.isfinite(ee)))
        and abs(ee[0]) <= EE_LOCAL_BOX_XY_M
        and abs(ee[1]) <= EE_LOCAL_BOX_XY_M
        and EE_LOCAL_BOX_Z_M[0] <= ee[2] <= EE_LOCAL_BOX_Z_M[1]
    )
    report.diffs.append(
        FieldDiff(
            name="policy_input.ee_pos_env_local",
            passed=in_box,
            hard=True,
            dataset_value=f"|xy| <= {EE_LOCAL_BOX_XY_M}, z in {list(EE_LOCAL_BOX_Z_M)}",
            sim_value=[round(v, 4) for v in ee],
            margin=None if in_box else float("inf"),
            note="env-local by subtraction of env_origins; a world-frame value "
            "lands metres outside this box on any env but the first",
        )
    )


def _check_rgb_is_a_real_frame(
    wire: dict[str, Any], target_env: int, report: SceneMismatchReport
) -> None:
    """RGB is exactly ``uint8/255`` and carries real per-channel variation.

    The ``uint8/255`` identity is what the CPU goldens' stored uint8 digests
    depend on being true of the live path too, and the spread floor is what makes
    a black frame -- a camera that never rendered, the single most common way an
    eval silently produces garbage -- fail rather than pass every other check.
    """
    for key in sorted(k for k in wire if k.endswith("_rgb")):
        frame = np.asarray(wire[key])[target_env, 0]  # (3, H, W)
        scaled = frame.astype(np.float64) * 255.0
        max_frac = float(np.max(np.abs(scaled - np.round(scaled))))
        report.diffs.append(
            FieldDiff(
                name=f"policy_input.rgb_is_uint8_over_255[{key}]",
                passed=max_frac <= 1e-4
                and float(frame.min()) >= 0.0
                and float(frame.max()) <= 1.0,
                hard=True,
                dataset_value="uint8 / 255 in [0, 1]",
                sim_value=f"max fractional part {max_frac:.3e}, range "
                f"[{float(frame.min()):.4f}, {float(frame.max()):.4f}]",
                diff=max_frac,
                tol=1e-4,
            )
        )
        stds = [float(scaled[c].std()) for c in range(scaled.shape[0])]
        report.diffs.append(
            FieldDiff(
                name=f"policy_input.rgb_channel_spread[{key}]",
                passed=min(stds) >= MIN_RGB_CHANNEL_STD,
                hard=True,
                dataset_value=f">= {MIN_RGB_CHANNEL_STD}",
                sim_value=[round(s, 3) for s in stds],
                diff=min(stds),
                tol=MIN_RGB_CHANNEL_STD,
                margin=(MIN_RGB_CHANNEL_STD - min(stds)) / MIN_RGB_CHANNEL_STD,
                note="per-channel std in 0-255 units; a constant or black frame "
                "passes every shape and range check and fails only here",
            )
        )


def check_policy_input_against_dataset(
    obs: dict[str, Any],
    wire: dict[str, Any],
    *,
    transform: ManiflowObsTransform,
    spec: DatasetSceneSpec,
    target_env: int,
    report: SceneMismatchReport,
) -> None:
    """Append every dataset-derived policy-input diff for one episode."""
    raw_state = np.asarray(wire["agent_pos"])[target_env, 0]
    _check_state_matches_recorded_setup(raw_state, transform, spec, report)
    _check_ee_pos_is_env_local(raw_state, transform, report)
    if transform.depth_keys:
        _check_intrinsics_match_recorded_k(wire, obs, transform, target_env, spec, report)
        _check_depth_is_metric(obs, transform, target_env, report)
    _check_rgb_is_a_real_frame(wire, target_env, report)


# ---------------------------------------------------------------------------
# The one entry point the script calls
# ---------------------------------------------------------------------------


def append_policy_input_diffs(
    obs: dict[str, Any],
    report: SceneMismatchReport,
    *,
    spec: DatasetSceneSpec,
    sim_state: SimSceneState,
    target_env: int,
    obs_mode: str,
    golden_dir: str | Path,
    write_golden: bool = False,
) -> None:
    """Run the policy-input leg over one episode's reset observation.

    Appends to ``report``, so the caller's existing console output, margin
    ranking, report JSON, artifact upload and exit code all carry the result
    unchanged. ``obs`` is the live (torch, batched) observation from
    ``env.reset()``.

    The dataset-derived checks run first and unconditionally, because they are
    what gates ``write_golden``: a run that cannot reproduce its own episode's
    recorded state has no business re-recording the contract everything else is
    checked against.
    """
    from toyblocks.utils.common_utils import to_numpy

    obs_np = to_numpy(obs)
    transform = build_policy_input_transform()
    wire = transform(obs_np)

    check_policy_input_against_dataset(
        obs_np, wire, transform=transform, spec=spec, target_env=target_env, report=report
    )

    if sim_state.scene_preset_id is None:
        # The contract golden is keyed by preset because the preset decides camera
        # names and native resolution, so without one there is no file to compare
        # against and writing "None.json" would compare presets to each other.
        report.diffs.append(
            FieldDiff(
                name="policy_input.scene_preset_id",
                passed=False,
                hard=True,
                dataset_value=spec.dataset_scene_preset_id,
                sim_value=None,
                margin=float("inf"),
                note="the live env exposes no scene_preset_id, so the observation "
                "contract cannot be keyed to a preset",
            )
        )
        return

    observed = observed_obs_contract(
        obs_np,
        transform,
        scene_preset_id=str(sim_state.scene_preset_id),
        robot_id=str(sim_state.robot_id),
        obs_mode=obs_mode,
    )
    report.sim_context["policy_input_bands"] = observed["bands"]

    if write_golden:
        path = write_obs_contract(observed, golden_dir, report)
        print(f"Wrote observation contract golden: {path}", flush=True)
        return
    compare_obs_contract(observed, golden_dir, report)
