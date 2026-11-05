"""Golden-array regression test for the whole policy-input pipeline.

``test_model_config_contract.py`` pins the *structure* -- shapes, dtypes, key
sets, block order, error messages -- over an observation whose RGB is
``np.zeros``. That leaves a wide class of change silently green: image scaling,
resize mode, channel order, the per-axis intrinsics rescale, depth validity, a
normalization band. Each one produces a correctly-shaped tensor full of different
numbers, which is exactly the failure that otherwise shows up only as a policy
behaving worse.

So this file runs the real chain over the seeded fixture and diffs the numbers:

    derive_model_input_spec -> ManiflowBlocksTableObsTransform.transform
      -> ManiFlowNormalizationMixin._normalize_state
      -> select_and_validate_obs

One golden per config fixture, spanning both state widths, the
depth+intrinsics branch, the kept-ptmap branch and the dropped-ptmap branch.
``config_state9.yaml`` gets none: its RGB is byte-identical to the others and its
state is ``state8`` plus a raw ``pd_mode`` column, so a fourth near-identical
blob would be churn without matching risk -- the contract test already covers
that width's concat order.

Everything reached by float32 arithmetic or integer index math is compared
bit-exactly. The bilinear RGB resize is not: its last bit depends on the CPU
kernel, so an arm64 golden and an x86_64 CI run disagree on a handful of pixels
by one level (see :data:`RGB_BLOCK_MEAN_TOL`). Its three stored statistics are
therefore graded rather than exact -- the byte digest reports as a warn line,
and the block means and neighbour differences carry tolerances chosen well below
any real regression and well above that flip. The two numeric ones are
complementary on purpose: block means catch a channel swap, a transpose or a
changed scale; neighbour differences catch a changed resize mode, which averaging
784 pixels per block would hide.

What is asserted in code instead of stored, because storing it is either
tautological or megabytes of zeros: that the wire float is exactly ``uint8/255``,
that the stand-in ptmaps are all zero, and that the normalized state lands inside
``[-1, 1]``. Those run *before* any golden is read, so they also gate
``--write-policy-input-golden``.

Regenerate with:

    pytest tests/serving/test_policy_input_goldens.py --write-policy-input-golden
    git diff tests/serving/goldens/

No Isaac, no GPU, no checkpoint.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from blocksuite.serving.maniflow_serving_utils import (
    derive_model_input_spec,
    select_and_validate_obs,
)
from blocksuite.serving.obs_transforms.maniflow import STATE_DIM_LAYOUTS
from blocksuite.serving.obs_transforms.maniflow_blocks_table import (
    ManiflowBlocksTableObsTransform,
)
from tests.serving import golden_harness as goldens
from tests.serving.policy_input_fixture import (
    FIXTURE_BATCH,
    array_sha256,
    build_obs,
    load_depth_fixture,
    load_fixture,
    normalizer_for,
    obs_digest,
    stats_action_mode,
    stats_name,
    to_torch,
)

# Golden name -> the config fixture it is computed against. ``state9_depth``
# exercises the depth + intrinsics branch; ``state19`` keeps its declared
# ptmaps; ``state8`` has them stripped, so it is the one that covers
# select_and_validate_obs actually dropping a key the client sent.
CASES = {
    "policy_input_state19": 19,
    "policy_input_state8": 8,
}

# Exercised in tests/hvla_maniflow (needs maniflow-tb for ptmap depth masking).
HVLA_ONLY_CASES = {
    "policy_input_state9_depth": 9,
}


def _state_dim_for_golden(golden_name: str) -> int:
    if golden_name in CASES:
        return CASES[golden_name]
    return HVLA_ONLY_CASES[golden_name]

#: Resize device pinned for every golden: the CUDA and CPU interpolate kernels
#: do not agree bit-for-bit, so a golden that depended on the runner having a GPU
#: would fail on the CPU lane for a reason unrelated to the code under test.
INTERPOLATE_DEVICE = "cpu"

PROMPT = "pick up the red block"

#: Arm joints a Franka's ``agent_pos`` leads with, and the widths of the two
#: optional blocks ``_normalize_state`` also scales, keyed by the
#: ``STATE_DIM_LAYOUTS`` flag that adds each. Structural rather than
#: per-checkpoint: joint_vel carries one value per arm joint and ee_pos is xyz.
ARM_JOINTS = 7
NORMALIZED_OPTIONAL_WIDTHS = {"include_joint_vel": ARM_JOINTS, "include_ee_pos": 3}

#: 8x8 grid of 28x28 blocks over the 224x224 model grid. A channel swap, a
#: transpose, a missing /255 or a changed image scale each move these means in a
#: direction a reviewer can read off the diff.
RGB_BLOCKS = 8

#: Allowance on a stored block mean, in levels. The bilinear resize is the only
#: step in this pipeline whose last bit is not portable: the frame round-trips
#: ``uint8 -> x1/255 -> bilinear -> x255 -> round``, and where the scaled value
#: lands on a ``.5`` tie a one-ULP kernel difference flips the byte. Those ties
#: are structural rather than unlucky -- 320 -> 224 under ``align_corners=False``
#: maps output column ``d`` to ``(d + 0.5) * 10/7 - 0.5``, exactly ``*.5`` for
#: every ``d = 3 (mod 7)`` -- so an arm64 golden and an x86_64 run disagree on a
#: handful of pixels by one level. One flipped level moves a 28x28 block mean by
#: ``1/784 = 0.0013``; this allows ~39 flips per block, while a channel swap or a
#: transpose moves a block mean by tens.
RGB_BLOCK_MEAN_TOL = 0.05

#: Allowance on a stored neighbour-difference mean, in levels. A flipped pixel
#: perturbs at most 4 of the ~50k differences along an axis by one level, so
#: platform noise is bounded near ``1e-4`` per flip; a resize that started
#: smoothing moves these by whole levels.
RGB_NEIGHBOR_DIFF_TOL = 0.02


def _spec_and_transform(golden_name: str) -> tuple[dict[str, Any], Any, str]:
    """The model input spec, a matching transform, and the config fixture name."""
    state_dim = _state_dim_for_golden(golden_name)
    if golden_name.endswith("_depth"):
        cfg, fw = load_depth_fixture()
        config_name = "config_state9_depth.yaml"
    else:
        cfg, fw = load_fixture(state_dim)
        config_name = f"config_state{state_dim}.yaml"
    spec = derive_model_input_spec(cfg, fw)

    # ``depth_keys`` is negotiated off the checkpoint the way
    # ``_make_policy_client`` does, because a checkpoint that declares no depth
    # cannot be sent any. ``include_ptmap`` deliberately stays at its default
    # True: that is what a client sends when a server advertises no obs_keys, and
    # it is what makes the state8 case exercise the drop path at all.
    transform = ManiflowBlocksTableObsTransform(
        prompt=PROMPT,
        state_dim=spec["state_dim"],
        depth_keys=sorted(k for k in spec["obs_shapes"] if k.endswith("_depth")),
        image_interpolate_device=INTERPOLATE_DEVICE,
    )
    return spec, transform, config_name


def _rgb_uint8(wire_value: np.ndarray) -> np.ndarray:
    """The uint8 frame a wire RGB float was built from.

    Exact rather than a rounding guess: the transform's only float step is
    ``uint8 * (1/255)``, every one of whose 256 outputs is uniquely recoverable.
    :func:`_check_wire_is_uint8_over_255` is what proves that premise still holds
    before this is used.
    """
    return np.round(np.asarray(wire_value) * 255.0).astype(np.uint8)


def _block_means(uint8_chw: np.ndarray) -> list[list[float]]:
    """Per-channel ``RGB_BLOCKS x RGB_BLOCKS`` mean grid for one env's frame."""
    c, h, w = uint8_chw.shape
    bh, bw = h // RGB_BLOCKS, w // RGB_BLOCKS
    if bh * RGB_BLOCKS != h or bw * RGB_BLOCKS != w:
        raise ValueError(
            f"a {h}x{w} frame does not tile into {RGB_BLOCKS}x{RGB_BLOCKS} blocks; "
            "the block-mean digest assumes the model's square 224 grid."
        )
    grid = uint8_chw.astype(np.float64).reshape(c, RGB_BLOCKS, bh, RGB_BLOCKS, bw)
    return np.round(grid.mean(axis=(2, 4)), 4).tolist()


def _neighbor_diffs(uint8_chw: np.ndarray) -> dict[str, list[float]]:
    """Per-channel mean ``|neighbour difference|`` along each axis, in levels.

    This is what carries the sensitivity the byte digest used to. A block mean
    averages 784 pixels, so a resize that started smoothing -- bilinear to
    nearest, an ``antialias`` flag, a changed ``align_corners`` -- barely moves
    it while changing the local high-frequency content of the fixture's noise by
    whole levels. Split by axis rather than pooled because the fixture's ramps
    run along different axes per channel, so a transpose swaps the two lists
    instead of leaving them both intact.

    Deliberately blind to a red/blue swap, which negates a ramp's gradient and
    leaves ``|difference|`` alone: that is the block means' job, and two
    statistics that fail together cover less than two that fail apart.
    """
    f = uint8_chw.astype(np.float64)
    return {
        "x": np.round(np.abs(np.diff(f, axis=2)).mean(axis=(1, 2)), 6).tolist(),
        "y": np.round(np.abs(np.diff(f, axis=1)).mean(axis=(1, 2)), 6).tolist(),
    }


# --------------------------------------------------------------------------
# Golden-independent invariants (also the gate on --write-policy-input-golden)
# --------------------------------------------------------------------------


def _check_wire_is_uint8_over_255(
    wire: dict[str, Any], report: goldens.SceneMismatchReport
) -> None:
    """Every RGB wire value is exactly some uint8 divided by 255.

    Asserted rather than stored because a 224x224x3 float array is megabytes of
    unreadable JSON, and because this is the invariant that makes the stored
    uint8 sha256 a faithful description of the float actually sent. A resize that
    started emitting continuous floats, or a normalization slipped in after the
    resize, breaks here and nowhere else.
    """
    for key in sorted(k for k in wire if k.endswith("_rgb")):
        value = np.asarray(wire[key])
        scaled = value * 255.0
        max_frac = float(np.max(np.abs(scaled - np.round(scaled))))
        report.diffs.append(
            goldens.FieldDiff(
                name=f"wire_is_uint8_over_255[{key}]",
                passed=max_frac <= 1e-4 and float(value.min()) >= 0.0
                and float(value.max()) <= 1.0,
                hard=True,
                dataset_value="uint8 / 255 in [0, 1]",
                sim_value=f"max fractional part {max_frac:.3e}, "
                f"range [{float(value.min()):.4f}, {float(value.max()):.4f}]",
                diff=max_frac,
                tol=1e-4,
            )
        )


def _check_ptmaps_are_zero(
    wire: dict[str, Any], report: goldens.SceneMismatchReport
) -> None:
    """The stand-in ptmaps are all zero (1.2 MB of zeros is not worth storing)."""
    for key in sorted(k for k in wire if k.endswith("_ptmap")):
        value = np.asarray(wire[key])
        peak = float(np.max(np.abs(value)))
        report.diffs.append(
            goldens.FieldDiff(
                name=f"ptmap_is_zero[{key}]",
                passed=peak == 0.0,
                hard=True,
                dataset_value=0.0,
                sim_value=peak,
                diff=peak,
                tol=0.0,
                note="the transform emits a zero-filled stand-in; a non-zero "
                "value means it started deriving something from depth",
            )
        )


def _normalized_state_width(state_dim: int) -> int:
    """How many leading ``agent_pos`` dims ``_normalize_state`` scales.

    Derived from :data:`STATE_DIM_LAYOUTS` rather than tabulated per width: those
    flags are what decide which blocks ``agent_pos`` carries, and the normalized
    ones (arm, joint_vel, ee_xyz) are contiguous at its front, so a width added
    to the layouts cannot leave this checking a slice that is partly raw.
    """
    flags = STATE_DIM_LAYOUTS[state_dim]
    return ARM_JOINTS + sum(
        width for flag, width in NORMALIZED_OPTIONAL_WIDTHS.items() if flags[flag]
    )


def _check_normalized_state_in_band(
    normalized: np.ndarray, state_dim: int, report: goldens.SceneMismatchReport
) -> None:
    """Normalized dims land inside ``[-1, 1]``; raw tail dims are exempt.

    Not a restatement of the golden: it says the *fixture* still places every
    normalized block strictly inside its training band, so the stored numbers are
    reporting band changes rather than sitting clipped at an edge where a band
    change is invisible.
    """
    block = normalized[..., : _normalized_state_width(state_dim)]
    peak = float(np.max(np.abs(block)))
    report.diffs.append(
        goldens.FieldDiff(
            name="normalized_state_inside_unit_band",
            passed=peak < 1.0,
            hard=True,
            dataset_value="< 1.0",
            sim_value=peak,
            diff=peak,
            tol=1.0,
            note="the fixture places every normalized block inside its band on "
            "purpose; at the edge a band regression stops being observable",
        )
    )


# --------------------------------------------------------------------------
# The observed payload
# --------------------------------------------------------------------------


def _observe(golden_name: str) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    """``(payload, emitted_wire, normalized_state)`` for one config fixture.

    The wire dict comes back so the invariant checks run over what the transform
    *emitted*, not over what survived validation: the state8 case drops its
    ptmaps, and checking the filtered dict would silently stop covering them.
    """
    state_dim = _state_dim_for_golden(golden_name)
    spec, transform, config_name = _spec_and_transform(golden_name)

    obs = build_obs()
    wire = transform(obs)

    # The server normalizes; the transform forwards raw. Both are stored: block
    # order, units, frame and the bands are all only visible in the pair.
    raw_state = np.asarray(wire["agent_pos"])
    normalized = normalizer_for(state_dim)._normalize_state(raw_state.copy())

    # The server replaces agent_pos with its normalized form before validating,
    # so the validated view stored below is the model's actual input, not the
    # client's.
    torch_wire = to_torch(wire)
    torch_wire["agent_pos"] = torch.from_numpy(normalized)
    filtered, dropped = select_and_validate_obs(
        torch_wire, spec["obs_shapes"], spec["conditioning_keys"]
    )

    depth: dict[str, dict[str, Any]] = {}
    for key in sorted(k for k in wire if k.endswith("_depth")):
        value = np.asarray(wire[key])
        zero_mask = value == 0.0
        depth[key] = {
            # process_depth's validity logic is what decides model input and it
            # is bit-exact, so the mask gets a digest rather than a fraction that
            # a moved region could preserve.
            "min": float(value.min()),
            "max": float(value.max()),
            "zeroed_fraction": float(zero_mask.mean()),
            "zero_mask_sha256": array_sha256(zero_mask),
        }

    rgb: dict[str, dict[str, Any]] = {}
    for key in sorted(k for k in wire if k.endswith("_rgb")):
        uint8 = _rgb_uint8(wire[key])
        rgb[key] = {
            # Soft: see RGB_BLOCK_MEAN_TOL. Kept because "the bytes moved" is the
            # first thing worth knowing when the tolerant checks start drifting.
            "uint8_sha256": array_sha256(uint8),
            # env 0 only for both: the other envs' frames come off the same
            # resize with the same weights, so a second copy would widen the JSON
            # without widening what is covered.
            "block_means_env0": _block_means(uint8[0, 0]),
            "neighbor_diffs_env0": _neighbor_diffs(uint8[0, 0]),
        }

    intrinsics = wire.get("camera_intrinsics")
    payload = {
        goldens.IDENTITY: goldens.identity(
            obs_digest=obs_digest(obs),
            config=config_name,
            stats=stats_name(state_dim),
            transform_class=type(transform).__name__,
            state_dim=state_dim,
            depth_keys=list(transform.depth_keys),
            include_ptmap=transform.include_ptmap,
            image_size=list(transform.image_size),
            image_interpolate_device=INTERPOLATE_DEVICE,
            action_mode=stats_action_mode(state_dim),
        ),
        goldens.BODY: {
            "emitted": goldens.describe_wire(wire),
            "validated": goldens.describe_wire(
                {k: v.numpy() if hasattr(v, "numpy") else v for k, v in filtered.items()}
            ),
            "dropped": dropped,
            # Full, both envs: 2x19 is 38 numbers, and this is where the real
            # regression risk lives (block order, units, frame, bands).
            "agent_pos_raw": raw_state[:, 0, :].astype(float).tolist(),
            "agent_pos_normalized": normalized[:, 0, :].astype(float).tolist(),
            # Rows follow sorted(depth_keys), which is what the encoder reshapes
            # positionally; null when the checkpoint reads no depth.
            "camera_intrinsics": (
                np.asarray(intrinsics)[:, 0].astype(float).tolist()
                if intrinsics is not None
                else None
            ),
            "depth": depth,
            "rgb": rgb,
        },
    }
    return payload, wire, normalized


# --------------------------------------------------------------------------
# The test
# --------------------------------------------------------------------------


@pytest.mark.parametrize("golden_name", sorted(CASES))
def test_policy_input_matches_golden(golden_name: str, request) -> None:
    write = request.config.getoption("--write-policy-input-golden")
    state_dim = _state_dim_for_golden(golden_name)
    report = goldens.new_report(golden_name)

    payload, wire, normalized = _observe(golden_name)

    _check_wire_is_uint8_over_255(wire, report)
    _check_ptmaps_are_zero(wire, report)
    _check_normalized_state_in_band(normalized, state_dim, report)

    if write:
        path = goldens.write_golden(golden_name, payload, report)
        pytest.skip(f"wrote {path}; review the diff and commit it")

    golden = goldens.load_golden(golden_name)
    if not goldens.compare_identity(payload[goldens.IDENTITY], golden, report):
        # Short-circuit: with a different input or a different transform the
        # array diff below would be noise, and reading it as a regression is the
        # mistake this gate exists to prevent.
        assert report.ok, (
            "the policy-input golden's identity does not match this run, so the "
            "numbers are not comparable:\n" + report.format_console()
        )

    body = payload[goldens.BODY]
    for section in ("emitted", "validated"):
        gold = goldens.compare_section(section, body, golden, report)
        if gold is not None:
            goldens.compare_described(section, body[section], gold, report)

    dropped_gold = goldens.compare_section("dropped", body, golden, report)
    if dropped_gold is not None:
        goldens.compare_key_set("dropped", body["dropped"], dropped_gold, report)

    for section in ("agent_pos_raw", "agent_pos_normalized"):
        gold = goldens.compare_section(section, body, golden, report)
        if gold is not None:
            goldens.compare_exact_array(section, body[section], gold, report)

    intrinsics_gold = goldens.compare_section("camera_intrinsics", body, golden, report)
    if (body["camera_intrinsics"] is None) != (intrinsics_gold is None):
        goldens.compare_exact(
            "camera_intrinsics[present]",
            body["camera_intrinsics"] is not None,
            intrinsics_gold is not None,
            report,
        )
    elif intrinsics_gold is not None:
        goldens.compare_exact_array(
            "camera_intrinsics", body["camera_intrinsics"], intrinsics_gold, report
        )

    depth_gold = goldens.compare_section("depth", body, golden, report)
    if depth_gold is not None:
        goldens.compare_key_set("depth[keys]", body["depth"], depth_gold, report)
        for key in sorted(set(body["depth"]) & set(depth_gold)):
            for field in ("min", "max", "zeroed_fraction"):
                goldens.compare_exact_array(
                    f"depth[{key}].{field}",
                    body["depth"][key][field],
                    depth_gold[key][field],
                    report,
                )
            goldens.compare_exact(
                f"depth[{key}].zero_mask_sha256",
                body["depth"][key]["zero_mask_sha256"],
                depth_gold[key]["zero_mask_sha256"],
                report,
            )

    rgb_gold = goldens.compare_section("rgb", body, golden, report)
    if rgb_gold is not None:
        goldens.compare_key_set("rgb[keys]", body["rgb"], rgb_gold, report)
        for key in sorted(set(body["rgb"]) & set(rgb_gold)):
            goldens.compare_exact(
                f"rgb[{key}].uint8_sha256",
                body["rgb"][key]["uint8_sha256"],
                rgb_gold[key]["uint8_sha256"],
                report,
                hard=False,
            )
            goldens.compare_array(
                f"rgb[{key}].block_means_env0",
                body["rgb"][key]["block_means_env0"],
                rgb_gold[key]["block_means_env0"],
                RGB_BLOCK_MEAN_TOL,
                report,
            )
            for axis in ("x", "y"):
                goldens.compare_array(
                    f"rgb[{key}].neighbor_diffs_env0[{axis}]",
                    body["rgb"][key]["neighbor_diffs_env0"][axis],
                    rgb_gold[key]["neighbor_diffs_env0"][axis],
                    RGB_NEIGHBOR_DIFF_TOL,
                    report,
                )

    assert report.ok, report.format_console()


def test_fixture_state_spans_both_envs() -> None:
    """The two envs carry different state, so an env-slicing bug cannot pass.

    Cheap, and it protects every per-env assertion in this directory: a fixture
    that tiled one env's values across the batch (as the older ``dummy_obs``
    does, deliberately) would let ``arr[0]`` stand in for ``arr[1]`` everywhere.
    """
    obs = build_obs()
    joint_pos = obs["agent"]["joint_pos"]
    assert joint_pos.shape[0] == FIXTURE_BATCH
    assert not np.allclose(joint_pos[0], joint_pos[1])
    for cam in ("front_camera", "gripper_camera"):
        rgb = obs["images"][cam]["rgb"]
        assert not np.array_equal(rgb[0], rgb[1])


def test_fixture_is_deterministic() -> None:
    """Two builds agree, so ``obs_digest`` describes the generator not the run."""
    assert obs_digest(build_obs()) == obs_digest(build_obs())
