"""The seeded synthetic observation the policy-input goldens are computed over.

The path under test (``ManiflowObsTransform.transform`` -> ``_normalize_state``
-> ``select_and_validate_obs``) is a pure function, so its input only has to be
**fixed**, not realistic -- which is what lets these goldens be regenerated with
no GPU, no Isaac and no committed binary fixture. Realism does not disappear, it
moves: ``toyblocks.eval.policy_input_verification`` records the *layout* of the
live observation from the Isaac lane, and ``test_fixture_matches_sim.py`` checks
:func:`build_obs` against it, so the structural assumptions made here are
verified rather than assumed.

Two properties matter and neither is free:

* **RGB is real noise over a per-channel ramp, not zeros.** Against a zero frame
  a channel swap, a CHW/HWC transpose, a missing ``/255`` and a changed resize
  mode are all invisible. Against this frame each one moves the block means.
* **Every state block sits strictly inside its training band**, read off the
  checked-in stats fixtures rather than hardcoded, so the normalized output lands
  in ``(-1, 1)`` and a band regression shows up as a moved number instead of
  being hidden by the clip at the edge.

:func:`obs_digest` is pinned into every golden's identity block, so a change to
this generator reports itself **by name** instead of as an unattributable array
diff -- the trick ``tests/env_randomization/test_rollout_determinism.py`` uses
with ``actions_sha256``.

Runs without Isaac Lab or hvla.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from toyblocks.serving.maniflow_serving_utils import (
    ManiFlowNormalizationMixin,
    TRAIN_MAX_DEPTH_M,
)

FIXTURES = Path(__file__).parent / "fixtures"

# Bumping this invalidates every golden by design: the identity block carries
# obs_digest(), so the failure names the generator rather than the arrays.
SEED = 20260814

#: Two envs, so a per-env slicing bug cannot pass by symmetry.
FIXTURE_BATCH = 2

#: Camera resolution: deliberately neither 224 nor square, so both resize axes
#: run and the intrinsics rescale has ``sx != sy``.
FIXTURE_CAM_HW = (180, 320)

#: Native pinhole intrinsics per camera, distinct so a row-order bug in
#: ``camera_intrinsics`` is a wrong number rather than a wrong shape.
FIXTURE_INTRINSICS: dict[str, list[list[float]]] = {
    "front_camera": [[210.0, 0.0, 160.5], [0.0, 185.0, 90.5], [0.0, 0.0, 1.0]],
    "gripper_camera": [[305.0, 0.0, 149.0], [0.0, 412.0, 101.0], [0.0, 0.0, 1.0]],
}

#: Cameras :class:`ManiflowBlocksTableObsTransform` reads, in its own order.
FIXTURE_CAMERAS = ("front_camera", "gripper_camera")


# --------------------------------------------------------------------------
# Checkpoint config / stats fixtures
# --------------------------------------------------------------------------


def load_fixture(state_dim: int) -> tuple[DictConfig, DictConfig]:
    """``(cfg, fw)`` for a trimmed checkpoint config.

    ``fw`` is ``cfg.framework`` directly rather than merged with hvla's
    ``_MANIFLOW_MM_DEFAULTS``, which is what keeps the callers importable on a
    CPU box without the maniflow extra -- and also why
    ``derive_model_input_spec`` takes ``fw`` as a parameter. The fixtures
    therefore carry only keys a real config sets itself; anything the defaults
    would have supplied raises rather than reading back a fixture-only
    invention.
    """
    cfg = OmegaConf.load(FIXTURES / f"config_state{state_dim}.yaml")
    return cfg, cfg.framework


def load_depth_fixture() -> tuple[DictConfig, DictConfig]:
    """``(cfg, fw)`` for the depth-conditioned 9-dim checkpoint fixture."""
    cfg = OmegaConf.load(FIXTURES / "config_state9_depth.yaml")
    return cfg, cfg.framework


def stats_name(state_dim: int) -> str:
    """Which stats fixture a state width normalizes with.

    The 8- and 9-dim layouts share one file: both normalize the arm only and
    pass the tail (gripper, and pd_mode when present) through raw, so neither
    carries ``joint_velocity`` / ``ee_position`` bands -- which is what keeps
    ``_has_extended_state_stats`` False for them.
    """
    return "stats_state19.json" if state_dim == 19 else "stats_state_short.json"


def load_stats(state_dim: int) -> dict:
    """Normalization bands for a state width."""
    with open(FIXTURES / stats_name(state_dim)) as f:
        return json.load(f)


def stats_action_mode(state_dim: int) -> str:
    """The action mode whose stats key the matching stats fixture carries.

    ``_init_stats`` raises on a missing ``action.*`` key, and the two fixtures
    differ: the 19-dim one holds ``action.relative_joint_target``, the short one
    ``action.joint_target``. Pairing them here keeps that coupling in the same
    place as :func:`stats_name` instead of in each caller.
    """
    return "relative" if state_dim == 19 else "absolute"


class _Normalizer(ManiFlowNormalizationMixin):
    """Bare mixin host: normalization has no dependency on the policy itself."""


def normalizer_for(state_dim: int) -> _Normalizer:
    """A normalizer bound to the stats fixture for ``state_dim``."""
    n = _Normalizer()
    n._init_stats(load_stats(state_dim), stats_action_mode(state_dim))
    return n


def to_torch(obs: dict) -> dict:
    """Mirror the server's numpy -> tensor step; non-arrays pass through."""
    return {
        k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
        for k, v in obs.items()
    }


# --------------------------------------------------------------------------
# Training bands the fixture state is placed inside
# --------------------------------------------------------------------------


def _band(key: str, width: int) -> tuple[np.ndarray, np.ndarray]:
    """The ``[q01, q99]`` band for ``key``, intersected over both stats fixtures.

    Intersected rather than read from one file so a single state vector is
    in-band for every width: an 8-dim golden and a 19-dim golden normalize the
    same raw arm values with different stats files, and a value inside only one
    of them would clip in the other and stop reporting band changes at all.
    """
    los: list[np.ndarray] = []
    his: list[np.ndarray] = []
    for state_dim in (8, 19):
        entry = load_stats(state_dim).get(key)
        if entry is None:
            continue
        los.append(np.broadcast_to(np.asarray(entry["q01"], dtype=np.float64), (width,)))
        his.append(np.broadcast_to(np.asarray(entry["q99"], dtype=np.float64), (width,)))
    if not los:
        raise KeyError(
            f"neither stats fixture carries {key!r}, so the fixture cannot place "
            "this block inside its training band; add the band to "
            f"{stats_name(19)} before widening the state layout."
        )
    lo = np.max(np.stack(los), axis=0)
    hi = np.min(np.stack(his), axis=0)
    if not np.all(hi > lo):
        raise ValueError(
            f"the {key!r} bands in the two stats fixtures do not overlap "
            f"(lo={lo.tolist()}, hi={hi.tolist()}); no single raw value is "
            "in-band for both widths."
        )
    return lo, hi


def _inside(rng: np.random.Generator, key: str, width: int, batch: int) -> np.ndarray:
    """``(batch, width)`` values strictly inside the ``key`` band.

    Sampled over the middle 70% of each band, so no dim lands close enough to an
    edge that a float32 round-trip could push its normalized value past +/-1 and
    turn an exact golden into a boundary flake.
    """
    lo, hi = _band(key, width)
    frac = rng.uniform(0.15, 0.85, size=(batch, width))
    return (lo + frac * (hi - lo)).astype(np.float32)


# --------------------------------------------------------------------------
# The observation
# --------------------------------------------------------------------------


def _rgb_frame(rng: np.random.Generator, batch: int, bias: float) -> np.ndarray:
    """``(batch, H, W, 3)`` uint8: per-channel ramps plus noise.

    The ramps are what make a failure *readable*: R rises left to right, G top to
    bottom, B right to left, so a channel swap, a transpose or a changed resize
    mode each move the stored block means in a recognisable direction. The noise
    is what makes the sha256 sensitive to anything the ramps alone would smooth
    over.
    """
    h, w = FIXTURE_CAM_HW
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ramps = np.stack(
        [
            255.0 * xx / (w - 1),
            255.0 * yy / (h - 1),
            255.0 * (1.0 - xx / (w - 1)),
        ],
        axis=-1,
    )
    # Per-env offset so env 0 and env 1 are never byte-identical: the panels and
    # the adapter both slice by env, and identical frames would let a wrong
    # index pass.
    per_env = np.arange(batch, dtype=np.float32).reshape(batch, 1, 1, 1) * 9.0
    noise = rng.normal(0.0, 12.0, size=(batch, h, w, 3))
    return np.clip(ramps[None, ...] + per_env + bias + noise, 0.0, 255.0).astype(np.uint8)


def _front_depth(rng: np.random.Generator, batch: int) -> np.ndarray:
    """Metric depth carrying every validity class ``process_depth`` decides on.

    Row bands, not scattered pixels, so the nearest-neighbour resize maps them to
    predictable regions of the 224x224 grid and the zeroed fraction is a stable
    number rather than a sampling artifact.

    Returned as ``(batch, H, W, 1)``: Isaac's ``distance_to_image_plane`` keeps a
    trailing singleton channel, which is what the recorded observation contract
    carries and therefore what ``test_fixture_matches_sim.py`` compares against.
    ``_depth_as_nhw`` strips the axis before any arithmetic, so no stored golden
    number depends on it.
    """
    h, w = FIXTURE_CAM_HW
    depth = rng.uniform(0.35, 2.2, size=(batch, h, w)).astype(np.float32)
    depth[:, :20] = np.inf  # background under depth_clipping_behavior="none"
    depth[:, 20:40] = -1.0  # negative: invalid
    depth[:, 40:60] = TRAIN_MAX_DEPTH_M + 0.75  # beyond the training cutoff
    return depth[..., None]


def _gripper_depth(rng: np.random.Generator, batch: int) -> np.ndarray:
    """Wrist depth: all-valid except one non-finite patch, ``(batch, H, W, 1)``.

    A second camera whose invalid *fraction* differs from the front one, so a
    bug that reported one camera's mask for both fails.
    """
    h, w = FIXTURE_CAM_HW
    depth = rng.uniform(0.05, 1.8, size=(batch, h, w)).astype(np.float32)
    depth[:, -25:, -40:] = np.inf
    return depth[..., None]


def build_obs(batch: int = FIXTURE_BATCH) -> dict[str, Any]:
    """The seeded toyblocks-shaped observation the goldens are computed over.

    Carries every key :class:`ManiflowBlocksTableObsTransform` can read, so one
    observation covers the depth branch, the ptmap branch and the
    ``prev_front_rgb`` branch; which of them is exercised is decided by the
    transform's flags, not by rebuilding the input.
    """
    rng = np.random.default_rng(SEED)

    arm = _inside(rng, "observation.state.joint_position", 7, batch)
    gripper = _inside(rng, "observation.state.gripper_position", 1, batch)
    joint_vel = _inside(rng, "observation.state.joint_velocity", 7, batch)
    ee_pos = _inside(rng, "observation.state.ee_position", 3, batch)

    obs: dict[str, Any] = {
        "agent": {
            "joint_pos": np.concatenate([arm, gripper], axis=-1),
            "joint_vel": joint_vel,
            "ee_pos": ee_pos,
        },
        "images": {
            "front_camera": {
                "rgb": _rgb_frame(rng, batch, bias=0.0),
                "depth": _front_depth(rng, batch),
                "intrinsics": np.tile(
                    np.asarray(FIXTURE_INTRINSICS["front_camera"], dtype=np.float32),
                    (batch, 1, 1),
                ),
            },
            "gripper_camera": {
                "rgb": _rgb_frame(rng, batch, bias=-31.0),
                "depth": _gripper_depth(rng, batch),
                "intrinsics": np.tile(
                    np.asarray(FIXTURE_INTRINSICS["gripper_camera"], dtype=np.float32),
                    (batch, 1, 1),
                ),
            },
        },
        # The aged HL conditioning frame a ``use_previous_front_frame``
        # checkpoint reads as prev_front_rgb. Present unconditionally and only
        # read when the transform sets include_prev_front_frame, mirroring the
        # eval harness, which publishes it for the HL-latency video regardless.
        "hl_conditioning_image": _rgb_frame(rng, batch, bias=44.0),
    }
    return obs


def build_target_heatmap(
    peak_xy: tuple[int, int] = (61, 148), batch: int = FIXTURE_BATCH
) -> np.ndarray:
    """``(batch, 1, 3, 224, 224)`` heatmap with a planted single-pixel peak.

    The peak location is the assertion: a summarizer or panel that reports the
    argmax of the wrong axis order returns ``(148, 61)`` instead, which a
    Gaussian blob centred in frame would hide.
    """
    hm = np.zeros((batch, 1, 3, 224, 224), dtype=np.float32)
    x, y = peak_xy
    hm[:, :, :, y, x] = 1.0
    hm[:, :, :, y, x - 1] = 0.5
    hm[:, :, :, y - 1, x] = 0.5
    return hm


def obs_digest(obs: dict[str, Any]) -> str:
    """sha256 over a canonical serialization of an observation dict.

    Pinned into every golden's identity block so a change to :func:`build_obs`
    fails as ``golden_identity[obs_digest]`` -- naming the generator -- instead
    of as an array diff with no attribution. Array leaves contribute dtype,
    shape and bytes, so a dtype change is caught even when the values are equal.
    """
    h = hashlib.sha256()

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key in sorted(value):
                walk(f"{prefix}/{key}", value[key])
        elif isinstance(value, np.ndarray):
            h.update(f"{prefix}|{value.dtype.str}|{tuple(value.shape)}|".encode())
            h.update(np.ascontiguousarray(value).tobytes())
        else:
            h.update(f"{prefix}|{type(value).__name__}|{value!r}".encode())

    walk("", obs)
    return h.hexdigest()


def array_sha256(arr: np.ndarray) -> str:
    """sha256 of an array's bytes, tagged with dtype and shape.

    Tagged so a reshape or a dtype change is a digest change: the bytes of a
    ``(2, 3, 224, 224)`` uint8 array and its transpose can coincide.
    """
    h = hashlib.sha256()
    h.update(f"{arr.dtype.str}|{tuple(arr.shape)}|".encode())
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()