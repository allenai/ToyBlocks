"""The one test that stops the CPU goldens from verifying a fiction.

Every other test in this directory feeds :func:`policy_input_fixture.build_obs`,
a hand-built observation. That makes the goldens regenerable with no GPU and no
Isaac, but it also means the *input side* of the contract is an assumption: if
``_get_observations`` renamed ``agent.ee_pos``, started emitting depth as
``(B, H, W, 1)``, or handed over float32 RGB, the whole CPU suite would stay green
because its goldens are computed over the fixture rather than over sim.

This test closes that by comparing the fixture against the observation contract
the Isaac lane records
(``blocksuite.eval.policy_input_verification.observed_obs_contract``, captured by
``scripts/eval/verify_dataset_loading.py`` and committed under
``tests/eval/goldens/``). Both sides are described by the *same* descriptor
functions, so what is compared is exactly what those functions consider
structural -- there is no second notion of "layout" to drift.

Deliberately **not** compared, because these differ by design rather than by
regression:

* **Native camera resolution.** The fixture is 180x320 so both resize axes run
  with ``sx != sy``; sim is whatever the scene preset renders. The transform
  reads the resolution at runtime, so the difference is the point.
* **``agent/joint_pos`` and ``agent/joint_vel`` width.** A Franka reports its
  finger joints too, so sim is 13 wide against the fixture's 8 and 7. The
  transform slices both blocks down -- ``joint_pos`` to the arm columns plus the
  gripper column, ``joint_vel`` to the arm columns -- so only that each side is
  wide enough for what is indexed matters, and that is what is checked.
* **``identity``'s ``scene_preset_id`` / ``robot_id`` / ``obs_mode``.** Sim-side
  facts with no fixture counterpart. The parts of the identity that describe the
  *transform* are compared, since a fixture built for a different camera set or
  state width would compare nothing meaningful.

Runs on CPU. It needs no GPU because it reads the golden rather than sim -- which
is the whole reason the Isaac lane writes one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blocksuite.eval.policy_input_verification import (
    OBS_CONTRACT_GOLDEN_PREFIX,
    POLICY_INPUT_DEPTH_KEYS,
    POLICY_INPUT_STATE_DIM,
    build_policy_input_transform,
    observed_obs_contract,
)
from tests.serving.policy_input_fixture import FIXTURE_CAMERAS, build_obs

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "eval" / "goldens"

JOINT_POS_PATH = "agent/joint_pos"
JOINT_VEL_PATH = "agent/joint_vel"

RECAPTURE_HINT = (
    "Record it on a GPU box with the datagen datasets present:\n"
    "    ./tests/tamp/datagen.sh --mode all --record-data --no-ui\n"
    "    ./tests/eval/scene_load.sh --write-policy-input-golden\n"
    "then review `git diff tests/eval/goldens/` and commit it."
)

#: Identity fields that describe the transform rather than the sim run, so the
#: fixture must agree on them or the two sides are not describing one layout.
SHARED_IDENTITY = ("transform_class", "state_dim", "depth_keys", "cameras")


def _contract_goldens() -> list[Path]:
    return sorted(GOLDEN_DIR.glob(f"{OBS_CONTRACT_GOLDEN_PREFIX}*.json"))


def _fixture_contract() -> dict:
    """The fixture's contract, through the same transform the Isaac lane drives.

    ``build_policy_input_transform`` rather than a locally-constructed transform:
    if the two ever configured different camera names, state widths or depth keys,
    this test would compare two layouts neither side actually uses.
    """
    return observed_obs_contract(
        build_obs(),
        build_policy_input_transform(),
        scene_preset_id="fixture",
        robot_id="fixture",
        obs_mode="fixture",
    )


def test_a_scene_preset_contract_is_committed():
    """No golden means the fixture's layout is checked against nothing.

    A hard failure rather than a skip: an unverified fixture is precisely the
    state this file exists to remove, and a skip here reads as a pass on the one
    test that makes every other golden in this directory trustworthy.
    """
    assert _contract_goldens(), (
        f"no observation-contract golden under {GOLDEN_DIR}, so the synthetic "
        "fixture every policy-input golden is computed over is not checked "
        f"against any real observation. {RECAPTURE_HINT}"
    )


def test_the_fixture_covers_the_transform_the_isaac_lane_drives():
    """The fixture supplies every leaf the widest transform reads.

    Not a golden comparison -- it fails on the fixture alone, before any golden
    exists -- but it is what makes the comparisons below meaningful: a fixture
    missing a camera or the ``ee_pos`` block would build a contract with fewer
    keys and report the absence as a sim-side change.
    """
    contract = _fixture_contract()
    assert contract["identity"]["state_dim"] == POLICY_INPUT_STATE_DIM
    assert contract["identity"]["depth_keys"] == sorted(POLICY_INPUT_DEPTH_KEYS)
    assert contract["identity"]["cameras"] == list(FIXTURE_CAMERAS), (
        "the fixture's cameras and the transform's cameras disagree, so its RGB "
        "and depth leaves are read from names the transform never looks at"
    )
    for cam in FIXTURE_CAMERAS:
        for leaf in ("rgb", "depth", "intrinsics"):
            assert f"images/{cam}/{leaf}" in contract["structure"]
    for path in (JOINT_POS_PATH, "agent/joint_vel", "agent/ee_pos"):
        assert path in contract["structure"]


@pytest.mark.parametrize(
    "golden_path",
    _contract_goldens(),
    ids=lambda p: p.stem[len(OBS_CONTRACT_GOLDEN_PREFIX) :],
)
def test_fixture_layout_matches_the_live_observation(golden_path: Path):
    """Every structural field of the fixture equals the live observation's.

    The exclusions are enumerated at the top of this file; everything else is
    exact, including the emitted wire, which is resolution-independent because the
    transform resizes to a fixed 224x224 grid.
    """
    golden = json.loads(golden_path.read_text())
    observed = _fixture_contract()

    for key in SHARED_IDENTITY:
        assert observed["identity"][key] == golden["identity"][key], (
            f"the fixture and {golden_path.name} describe different transform "
            f"configurations ({key}), so their layouts are not comparable"
        )

    assert sorted(observed["structure"]) == sorted(golden["structure"]), (
        "the fixture and the live observation do not carry the same set of leaves; "
        "an observation key the transform reads was renamed, added or dropped. "
        f"{RECAPTURE_HINT}"
    )

    transform = build_policy_input_transform()
    # How many columns the transform indexes out of each sliced state block.
    required_width = {
        JOINT_POS_PATH: max(transform.arm_joints, transform.gripper_joint_index + 1),
        JOINT_VEL_PATH: transform.arm_joints,
    }

    for path in sorted(golden["structure"]):
        want, got = golden["structure"][path], observed["structure"][path]
        if path in required_width:
            _assert_state_block_is_wide_enough(path, want, got, required_width[path])
            continue
        assert got == want, (
            f"the fixture's {path} does not match the live observation's: sim "
            f"{want}, fixture {got}. The fixture is what every golden in this "
            "directory is computed over, so fix the fixture -- or, if sim really "
            f"changed, re-record the goldens. {RECAPTURE_HINT}"
        )

    assert observed["wire"] == golden["wire"], (
        "the transform emits a different wire layout over the fixture than over "
        f"the live observation: sim {golden['wire']}, fixture {observed['wire']}"
    )


def _assert_state_block_is_wide_enough(
    path: str, want: dict, got: dict, required: int
) -> None:
    """Both sides carry at least the ``required`` columns the transform indexes.

    Width is not compared because a Franka reports its two finger joints while the
    fixture stops at the arm block plus the single gripper scalar the transform
    reads. What has to hold is that the columns the transform indexes exist on
    both sides, and that the dtype matches -- a float64 state would change every
    stored number.
    """
    assert got["dtype"] == want["dtype"], (
        f"{path} dtype differs: sim {want['dtype']}, fixture {got['dtype']}"
    )
    assert len(got["shape"]) == len(want["shape"]) == 1, (
        f"{path} rank differs: sim {want['shape']}, fixture {got['shape']}; the "
        "transform only accepts (n,) or (B, n)"
    )
    for label, desc in (("live observation", want), ("fixture", got)):
        assert desc["shape"][0] >= required, (
            f"the {label}'s {path} has {desc['shape'][0]} joints but the transform "
            f"reads {required} of them"
        )
