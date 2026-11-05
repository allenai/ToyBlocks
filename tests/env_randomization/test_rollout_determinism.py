"""Golden for the sim's dynamics: seeded free-space rollouts' joint trajectories.

This is the one thing the eval-side scene checks cannot see. ``scene_load.sh``
and ``toyblocks.eval.scene_verification`` diff a *static* scene against a
dataset's recorded ground truth, so a drift in how the arm actually moves under
known actions -- a controller gain, an actuator profile, a solver setting, an
Isaac upgrade -- passes every one of them.

Deliberately independent of datagen: no dataset, no episode, no randomization
profile, no recorded provenance. Everything that decides what the numbers mean
lives in :func:`identity` -- the task, the seed, the action schedule and its
hash, the robot / control mode / scene preset, the episode horizon, the gains
actually applied, and which frames were persisted. A change to any of them
reports itself by name and short-circuits the trajectory comparison, so a
failure is never ambiguous about which of the two happened.

Three properties make the golden stable enough to be worth committing:

- Actions are drawn on a **CPU** ``torch.Generator``, so the sequence is
  bit-identical across GPUs and driver versions. Otherwise the golden would
  encode the machine's RNG rather than the scene's physics.
- The rollout stays in **free space**, asserted rather than assumed
  (:func:`check_free_space`). Contact-rich PhysX is not reproducible across
  driver or Isaac versions, so a golden that included contact would demand a
  re-baseline on every environment bump -- which is the failure mode that gets
  goldens deleted instead of read. The scene is settled before the first
  snapshot (``SETTLE_STEPS``) so that check sees only what the arm did.
- The **arm** is a numeric golden; the **gripper** is not. The binary gripper
  command is phased (closed for a run of steps, then open) and checked by the
  command-to-response relation in :func:`check_gripper_phases`, because travel
  speed is the gripper actuator's effort limit rather than anything about the
  scene. Its keyframes are still recorded, reported soft, so a magnitude shift
  stays readable in the diff.

Parametrized over three named gain cases (:data:`ROLLOUT_CASES`), one golden
file each. Needs a GPU box with Isaac Lab
(``tests/env_randomization/conftest.py`` owns the AppLauncher):

    pytest tests/env_randomization/test_rollout_determinism.py -v

Re-baseline only for a deliberate Isaac / driver / physics change, then review
the diff -- each golden is nine keyframes of the arm's joints, small enough to
read:

    pytest tests/env_randomization/test_rollout_determinism.py \
        --write-rollout-golden [--rollout-case softer_pd_stiff]
    git diff tests/env_randomization/
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.helpers import teardown_sim
from toyblocks.eval.pd_mode_controller import PDModeController
from toyblocks.eval.scene_verification import (
    FieldDiff,
    SceneMismatchReport,
    compare_array,
    compare_key_set,
    malformed_golden,
)
from toyblocks.tasks.lift.lift_env import LiftEnv
from toyblocks.tasks.lift.lift_env_cfg import LiftEnvCfg
from toyblocks.utils.episode_dataset import SCENE_FURNITURE_PREFIXES

GOLDEN_DIR = Path(__file__).parent

REBASELINE_HINT = (
    "Re-record it with 'pytest "
    "tests/env_randomization/test_rollout_determinism.py --write-rollout-golden "
    "--rollout-case <case>' and commit the result, but only for a deliberate "
    "Isaac / driver / physics change."
)

# ``PDModeController`` takes its gains as explicit dicts rather than baking any in,
# so these mirror the two arm configs it can be asked to switch to/from. Built here
# instead of imported: an edit to either source has to be echoed here to keep
# passing, so it can't drift silently.
# Mirrors ``actuator_profiles.TAMP_ACTUATORS`` (880/80; effort 87 on every arm joint).
PD_STIFF_GAINS: dict[str, list[float]] = {
    "stiffness": [880.0] * 7,
    "damping": [80.0] * 7,
    "effort": [87.0] * 7,
}
# Mirrors ``fr3_softer.FRANKA_DROID_ARTICULATION_CFG.actuators`` (40/13; effort 87 on
# panda_joint1-4, 12 on panda_joint5-7) -- exactly ``fr3_robotiq_gripper_softer``'s own
# config, per the ROLLOUT_CASES comment below.
PD_COMPLIANT_GAINS: dict[str, list[float]] = {
    "stiffness": [40.0] * 7,
    "damping": [13.0] * 7,
    "effort": [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0],
}

# Named gain setups, not a robot x mode cross product. ``PDModeController._commit``
# overwrites stiffness/damping/effort on all 7 arm joints, so under a pinned PD
# mode ``_soft`` and ``_softer`` are identical on the arm and differ only in the
# gripper actuator. And ``PD_COMPLIANT_GAINS`` (40/13) is exactly
# ``fr3_robotiq_gripper_softer``'s own config, so compliant mode on that robot is
# a no-op. These three are the distinct arm setups that exist.
ROLLOUT_CASES = {
    # The datagen/eval robot at its config defaults (40/13), which is also what
    # PD compliant mode asks for: what runs whenever no policy drives pd_mode.
    "softer_config": {"robot_id": "fr3_robotiq_gripper_softer", "pd_mode": None},
    # Same robot under PD_STIFF_GAINS (880/80), 22x stiffer on the arm -- the
    # other mode a pd_mode policy selects, and otherwise tested nowhere.
    "softer_pd_stiff": {"robot_id": "fr3_robotiq_gripper_softer", "pd_mode": 1},
    # The only genuinely third arm gain set (80/18), and the robot
    # ``tests.helpers.SCENE_ROBOT_MAP`` names for both scene presets.
    "soft_config": {"robot_id": "fr3_robotiq_gripper_soft", "pd_mode": None},
}

# A checked-in task rather than an autogen one: autogen packages are gitignored
# and only exist after a datagen run, which would make the golden depend on the
# very pipeline it is meant to be independent of. Lift is the smallest task with
# a block in it.
TASK = "toyblocks.tasks.lift.LiftEnv"
SEED = 0

# All pinned in the fixture rather than inherited from cfg defaults, and all
# recorded in identity(). BaseEnvCfg.robot_id is the legacy
# ``franka_robotiq_gripper``, which neither datagen nor eval ever runs, so a gain
# or actuator change in the configs they *do* run used to be invisible here.
CONTROL_MODE = "arm_relative_joint_gripper_binary"
SCENE_PRESET_ID = "droid_with_box_table"
EPISODE_LENGTH_S = 2.56

# Uniform in +/- this, per arm action dimension. Small enough that the arm never
# reaches the block from the default reset pose; check_free_space hard-fails if
# that stops being true.
ACTION_SCALE = 0.05

# The gripper command is phased instead of noisy. At +/-ACTION_SCALE its sign
# flipped roughly every step, so the binary command chattered and finger_joint
# was captured mid-flight, never settling and demonstrating nothing about whether
# the command took. Long enough that the slowest case (soft, gripper effort limit
# 0.2) travels a readable distance.
GRIPPER_CLOSE_STEPS = 16
GRIPPER_OPEN_STEPS = 16
NUM_STEPS = GRIPPER_CLOSE_STEPS + GRIPPER_OPEN_STEPS

# Raw physics substeps run with the arm held after reset. ``LiftEnv._reset_idx``
# spawns the block 5e-4 m above the table (to avoid initial penetration) and
# ``env.reset()`` only calls ``sim.forward()``, so without this the first
# snapshot catches the block mid-air: it then free-falls that gap and overshoots
# into the table before the solver resolves it, which check_free_space reads as
# ~1 mm of block motion during the rollout. These are raw ``sim.step`` calls, so
# they do not consume the episode's step budget.
SETTLE_STEPS = 30

# Persist every Nth frame plus the phase boundaries and the last frame; every
# step is still recorded in memory for the free-space check.
KEYFRAME_STRIDE = 4

# Radians. Wide enough to absorb PhysX solver nondeterminism on a fixed Isaac
# build, far tighter than any real dynamics change. Provisional until the
# same-machine repeat spread has been measured on the runner (re-run twice
# unflagged and read the max arm diff); set to roughly 5-10x it.
JOINT_POS_TOL = 1e-2
# Soft, and deliberately loose: see check_gripper_phases for what actually gates
# the gripper.
GRIPPER_POS_TOL = 5e-2
# The commanded reset pose is written verbatim by ``BaseEnv._reset_robots`` with
# no physics in between, so this is a float round-trip tolerance, not a physical
# one.
DEFAULT_JOINT_POS_TOL = 1e-6
# Any block moving more than this over the whole rollout means the actions
# reached it, and the golden would then encode contact physics.
FREE_SPACE_BLOCK_TOL_M = 1e-3

# ``ROBOTIQ_GRIPPER_BINARY_ACTION`` closes finger_joint to this and opens to 0.0.
GRIPPER_CLOSE_TARGET_RAD = 0.72
# Floors on the command-to-response relation, sized for the slowest case rather
# than per case. The achieved travel is reported as the diff's sim_value, so a
# failure prints the number needed to re-sit them.
GRIPPER_CLOSE_TRAVEL_MIN_RAD = 0.05
GRIPPER_OPEN_RETURN_MIN_RAD = 0.02


def golden_path(case: str) -> Path:
    """One golden per case, so each diff stays readable and re-recording one
    case cannot touch the others."""
    return GOLDEN_DIR / f"rollout_determinism_golden_{case}.json"


def keyframe_indices() -> list[int]:
    """Frame indices persisted to the golden.

    Storage granularity and check granularity are independent: every step is
    recorded in memory (:func:`check_free_space` needs it -- a block nudged and
    returned between keyframes would otherwise slip through) and only these are
    written out. Comparing keyframes *reduces* flakiness rather than trading
    safety for size, because a transient that re-converges is PhysX noise rather
    than a dynamics change. Both phase boundaries and the last frame are always
    included, and the list goes in :func:`identity` -- otherwise a stride change
    silently compares a different set of frames.
    """
    strided = set(range(0, NUM_STEPS + 1, KEYFRAME_STRIDE))
    return sorted(strided | {0, GRIPPER_CLOSE_STEPS, NUM_STEPS})


@pytest.fixture
def rollout_env(rollout_case):
    """One Lift env for the named case: reset, gains applied, then settled."""
    import omni.usd

    omni.usd.get_context().new_stage()

    case = ROLLOUT_CASES[rollout_case]

    cfg = LiftEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = "cuda:0"
    cfg.robot_id = case["robot_id"]
    cfg.control_mode = CONTROL_MODE
    cfg.scene_preset_id = SCENE_PRESET_ID
    # ``env.step()`` auto-resets a truncated env, so a rollout that overran the
    # horizon would silently rebaseline the golden onto a fresh episode. Pinned
    # here (rather than relying on the cfg default) so the hard step-budget
    # assert in run_free_space_rollout is checking a known number.
    cfg.episode_length_s = EPISODE_LENGTH_S

    # The block spawn is randomized per reset. It is not compared (blocks must
    # not move at all), but seeding keeps the whole scene reproducible so a
    # failure cannot be blamed on where the block happened to land.
    torch.manual_seed(SEED)
    env = LiftEnv(cfg=cfg)
    try:
        env.reset()
        if case["pd_mode"] is not None:
            # Applied through the controller rather than as hand-built gain
            # tensors, so the golden covers the path eval takes (including the
            # first-time-init bypass).
            PDModeController(
                env,
                torch.device(env.device),
                num_envs=1,
                stiff_gains=PD_STIFF_GAINS,
                compliant_gains=PD_COMPLIANT_GAINS,
            ).apply(
                torch.tensor([float(case["pd_mode"])], device=env.device)
            )
        # Let the block come to rest on the table before the rollout starts, so
        # the motion check measures what the arm did and not the spawn offset
        # falling. Passing no gains makes this purely a settle: it holds the
        # arm's joint positions and steps physics SETTLE_STEPS times, under
        # whatever gains the case just established.
        env.set_actuator_gains(settle_steps=SETTLE_STEPS)
        yield env
    finally:
        env.close()
        teardown_sim()


def joint_groups(env) -> tuple[list[int], list[int]]:
    """Arm and gripper column indices into ``data.joint_pos``, resolved by name.

    By name rather than by position because the two are compared differently:
    the arm is the numeric golden and the gripper is not. The names themselves go
    in :func:`identity`, so a joint reordering reports itself instead of shifting
    every number.
    """
    names = list(env.robot.articulation.joint_names)
    arm = [i for i, name in enumerate(names) if name.startswith("panda_joint")]
    gripper = [i for i in range(len(names)) if i not in set(arm)]
    assert arm, f"no panda_joint* columns among {names}"
    assert "finger_joint" in [names[i] for i in gripper], (
        f"no finger_joint among the non-arm joints {[names[i] for i in gripper]}; "
        "check_gripper_phases has nothing to assert on"
    )
    return arm, gripper


def rollout_snapshot(env, arm: list[int], gripper: list[int]) -> dict:
    """Joint positions + env-local block positions for sub-env 0, as plain lists."""
    origin = env.scene.env_origins[0]
    joints = env.robot.articulation.data.joint_pos[0].detach().cpu().tolist()
    blocks: dict[str, list[float]] = {}
    for name, obj in env.scene.rigid_objects.items():
        if name.startswith(SCENE_FURNITURE_PREFIXES):
            continue
        local = (obj.data.root_pos_w[0] - origin).detach().cpu().tolist()
        blocks[name] = [float(x) for x in local]
    return {
        "arm_joint_position": [float(joints[i]) for i in arm],
        "gripper_joint_position": [float(joints[i]) for i in gripper],
        "block_position": blocks,
    }


def build_actions(env) -> torch.Tensor:
    """Seeded arm noise throughout, with the gripper command phased close->open."""
    generator = torch.Generator().manual_seed(SEED)
    shape = (NUM_STEPS, *tuple(env.action_space.shape))
    actions = (
        2.0 * torch.rand(shape, generator=generator, dtype=torch.float32) - 1.0
    ) * ACTION_SCALE
    # The gripper is the last action dim (``BaseEnv._apply_action`` walks the
    # controller dict in order, arm then gripper). BinaryJointAction convention:
    # negative closes, positive opens.
    actions[:GRIPPER_CLOSE_STEPS, :, -1] = -1.0
    actions[GRIPPER_CLOSE_STEPS:, :, -1] = 1.0
    return actions


def applied_arm_gains(env, arm: list[int]) -> dict[str, list[float]]:
    """The gains the arm is actually running under, read off the live articulation.

    Read back rather than taken from the case's constants, so the robot configs'
    own defaults are covered too: a gain edit in ``fr3_softer.py`` /
    ``fr3_soft.py`` then fails the identity of its case by name instead of
    surfacing as an opaque radian number in the trajectory.
    """
    data = env.robot.articulation.data
    return {
        field: [round(float(getattr(data, attr)[0, i].item()), 6) for i in arm]
        for field, attr in (
            ("arm_stiffness", "joint_stiffness"),
            ("arm_damping", "joint_damping"),
            ("arm_effort_limits", "joint_effort_limits"),
        )
    }


def run_free_space_rollout(env, case: str) -> tuple[dict, list[dict]]:
    """Step the env with the seeded schedule; return (golden payload, all frames).

    Every step is kept in memory for the behavioral checks; the payload carries
    only :func:`keyframe_indices`.
    """
    horizon = int(round(EPISODE_LENGTH_S / env.step_dt))
    assert NUM_STEPS < horizon, (
        f"the rollout is {NUM_STEPS} steps but the episode truncates at "
        f"{horizon} (episode_length_s={EPISODE_LENGTH_S}, step_dt={env.step_dt}). "
        "env.step() auto-resets a truncated env, so the tail of this rollout "
        "would silently be a fresh episode."
    )

    names = list(env.robot.articulation.joint_names)
    arm, gripper = joint_groups(env)
    actions = build_actions(env)

    frames = [rollout_snapshot(env, arm, gripper)]
    for step in range(NUM_STEPS):
        env.step(actions[step].to(env.device))
        frames.append(rollout_snapshot(env, arm, gripper))

    keyframes = keyframe_indices()
    kept = [frames[i] for i in keyframes]
    payload = {
        "case": case,
        "task": TASK,
        "seed": SEED,
        "robot_id": env.cfg.robot_id,
        "control_mode": env.cfg.control_mode,
        "scene_preset_id": env.cfg.scene_preset_id,
        "episode_length_s": env.cfg.episode_length_s,
        "settle_steps": SETTLE_STEPS,
        "num_steps": NUM_STEPS,
        "gripper_close_steps": GRIPPER_CLOSE_STEPS,
        "gripper_open_steps": GRIPPER_OPEN_STEPS,
        "action_scale": ACTION_SCALE,
        "actions_sha256": hashlib.sha256(actions.numpy().tobytes()).hexdigest(),
        "arm_joint_names": [names[i] for i in arm],
        "gripper_joint_names": [names[i] for i in gripper],
        "keyframe_stride": KEYFRAME_STRIDE,
        "keyframe_indices": keyframes,
        # The commanded reset pose: an input to the rollout and, after the
        # settle, the thing frame 0 is measured against. Compared on its own so
        # a reset-pose or robot-config change fails on *it* rather than showing
        # up as an unattributable trajectory diff.
        "default_joint_pos": [
            float(x)
            for x in env.robot.articulation.data.default_joint_pos[0]
            .detach()
            .cpu()
            .tolist()
        ],
        "arm_joint_position": [f["arm_joint_position"] for f in kept],
        "gripper_joint_position": [f["gripper_joint_position"] for f in kept],
        "block_position": {
            name: [f["block_position"][name] for f in kept]
            for name in sorted(kept[0]["block_position"])
        },
        **applied_arm_gains(env, arm),
    }
    return payload, frames


def check_free_space(frames: list[dict], report: SceneMismatchReport) -> None:
    """Hard-fail if any block moved during the rollout.

    Reads every in-memory frame rather than the persisted keyframes: a block
    nudged and returned in between would otherwise slip through.
    """
    names = sorted(frames[0]["block_position"])
    if not names:
        # Otherwise this check emits no diffs at all and reports a pass having
        # compared nothing -- a scene with no blocks is not a free-space scene,
        # it is a scene the snapshot failed to read.
        report.diffs.append(
            FieldDiff(
                name="rollout_free_space[block_set]",
                passed=False,
                hard=True,
                dataset_value="at least one non-furniture block",
                sim_value=[],
                margin=float("inf"),
                note=(
                    "the rollout snapshot found no non-furniture rigid objects, so "
                    "free space was asserted over nothing. Either the task spawned no "
                    "blocks or they are all named with a SCENE_FURNITURE_PREFIXES "
                    "prefix."
                ),
            )
        )
        return
    for name in names:
        track = np.array([f["block_position"][name] for f in frames], dtype=float)
        moved = float(np.max(np.abs(track - track[0])))
        report.diffs.append(
            FieldDiff(
                name=f"rollout_free_space[{name}]",
                passed=moved <= FREE_SPACE_BLOCK_TOL_M,
                hard=True,
                dataset_value=0.0,
                sim_value=moved,
                diff=moved,
                tol=FREE_SPACE_BLOCK_TOL_M,
                note=(
                    None
                    if moved <= FREE_SPACE_BLOCK_TOL_M
                    else (
                        "this block moved, so the golden would encode contact-rich "
                        "PhysX and need re-baselining on every driver/Isaac bump. "
                        "Centimetre-scale motion means the arm reached it: lower "
                        "ACTION_SCALE or NUM_STEPS. Sub-millimetre motion means the "
                        "scene was still settling at frame 0: raise SETTLE_STEPS."
                    )
                ),
            )
        )


def check_gripper_phases(
    frames: list[dict], payload: dict, report: SceneMismatchReport
) -> None:
    """Assert the binary gripper command took, behaviorally rather than numerically.

    Travel *speed* is the gripper actuator's effort limit, which differs per case
    (soft 0.2, softer 1.0), so a numeric bound on it would be a per-case number
    that is really just the actuator's. What must hold for every case is the
    relation between command and response: the close phase drives finger_joint
    toward :data:`GRIPPER_CLOSE_TARGET_RAD`, the open phase brings it back.

    ``margin`` is set explicitly because the allowance here is a floor, not a
    ceiling: 1.0 still means "exactly on the bound" and >1.0 the breach, but the
    ratio is inverted relative to a tolerance.
    """
    column = payload["gripper_joint_names"].index("finger_joint")
    track = [f["gripper_joint_position"][column] for f in frames]
    start, closed, opened = track[0], track[GRIPPER_CLOSE_STEPS], track[NUM_STEPS]

    for name, travel, floor, note in (
        (
            "close_travel",
            closed - start,
            GRIPPER_CLOSE_TRAVEL_MIN_RAD,
            f"the close phase must drive finger_joint toward "
            f"{GRIPPER_CLOSE_TARGET_RAD} rad. It travelled "
            f"{closed - start:+.4f} rad over {GRIPPER_CLOSE_STEPS} steps "
            f"({start:.4f} -> {closed:.4f}), so the command did not take: check "
            "the BinaryJointAction sign convention (negative closes) and the "
            "gripper actuator's effort limit.",
        ),
        (
            "open_return",
            closed - opened,
            GRIPPER_OPEN_RETURN_MIN_RAD,
            f"the open phase must bring finger_joint back toward 0. It returned "
            f"{closed - opened:+.4f} rad over {GRIPPER_OPEN_STEPS} steps "
            f"({closed:.4f} -> {opened:.4f}), so the gripper is not tracking the "
            "command in both directions.",
        ),
    ):
        passed = travel >= floor
        report.diffs.append(
            FieldDiff(
                name=f"rollout_gripper[{name}]",
                passed=passed,
                hard=True,
                dataset_value=f">= {floor} rad",
                sim_value=travel,
                diff=travel,
                tol=floor,
                margin=(floor / travel) if travel > 0 else float("inf"),
                note=None if passed else note,
            )
        )


def identity(payload: dict) -> dict:
    """The fields that must match before the trajectory numbers mean anything.

    Read leniently so a golden written before a field existed surfaces as a
    diff on that field rather than a ``KeyError`` with no context; the observed
    payload always has every key.
    """
    return {
        key: payload.get(key)
        for key in (
            "case",
            "task",
            "seed",
            "robot_id",
            "control_mode",
            "scene_preset_id",
            "episode_length_s",
            "settle_steps",
            "num_steps",
            "gripper_close_steps",
            "gripper_open_steps",
            "action_scale",
            "actions_sha256",
            "arm_joint_names",
            "gripper_joint_names",
            "arm_stiffness",
            "arm_damping",
            "arm_effort_limits",
            "keyframe_stride",
            "keyframe_indices",
        )
    }


def compare_identity(
    observed: dict, golden: dict, report: SceneMismatchReport
) -> bool:
    """Diff the identity fields; returns whether they all matched."""
    obs_id, gold_id = identity(observed), identity(golden)
    all_matched = True
    for key, value in obs_id.items():
        matched = value == gold_id[key]
        all_matched = all_matched and matched
        report.diffs.append(
            FieldDiff(
                name=f"golden_identity[{key}]",
                passed=matched,
                hard=True,
                dataset_value=gold_id[key],
                sim_value=value,
                margin=0.0 if matched else float("inf"),
                note=None if matched else (
                    "this is not a dynamics regression: the run itself differs from "
                    "the one the golden was recorded on, so its numbers describe a "
                    f"different rollout. {REBASELINE_HINT}"
                ),
            )
        )
    return all_matched


def compare_trajectory(
    observed: dict, golden: dict, report: SceneMismatchReport
) -> None:
    """Diff the recorded keyframes, once the identity is known to match."""
    for section in ("arm_joint_position", "default_joint_pos"):
        if section not in golden:
            malformed_golden(section, report, remedy=REBASELINE_HINT)
            return
    compare_key_set(
        "golden_rollout[block_set]",
        observed["block_position"],
        golden.get("block_position", {}),
        report,
    )
    compare_array(
        "golden_rollout[default_joint_pos]",
        observed["default_joint_pos"],
        golden["default_joint_pos"],
        DEFAULT_JOINT_POS_TOL,
        report,
    )
    compare_array(
        "golden_rollout[arm_joint_position]",
        observed["arm_joint_position"],
        golden["arm_joint_position"],
        JOINT_POS_TOL,
        report,
    )
    locate_max_diff(
        "golden_rollout[arm_joint_position]",
        observed["arm_joint_position"],
        golden["arm_joint_position"],
        observed,
        report,
    )
    compare_gripper_keyframes(observed, golden, report)


def locate_max_diff(
    name: str,
    observed: list,
    golden: list,
    payload: dict,
    report: SceneMismatchReport,
) -> None:
    """Name the keyframe and joint carrying the max diff, on the diff that failed.

    ``compare_array`` reports one max-abs number for the whole trajectory, which
    on its own needs a manual reconstruction to interpret. Attached to the
    existing diff's ``note`` rather than added as a second diff, so the report
    still has one entry per compared field.
    """
    diff = next((d for d in report.diffs if d.name == name), None)
    if diff is None or diff.passed or diff.diff is None:
        return
    obs = np.asarray(observed, dtype=float)
    gold = np.asarray(golden, dtype=float)
    if obs.shape != gold.shape:
        return
    row, col = np.unravel_index(int(np.argmax(np.abs(obs - gold))), obs.shape)
    frame = payload["keyframe_indices"][row]
    joint = payload["arm_joint_names"][col]
    where = (
        f"max diff is at step {frame} (keyframe {row}) on {joint}: "
        f"golden {gold[row, col]:.6f} vs sim {obs[row, col]:.6f}."
    )
    diff.note = where if diff.note is None else f"{diff.note} {where}"


def compare_gripper_keyframes(
    observed: dict, golden: dict, report: SceneMismatchReport
) -> None:
    """Report the gripper keyframes as a visible magnitude, not a pass/fail bound.

    Kept in the golden so a magnitude shift (a gripper effort/stiffness change, a
    different close target) is readable in the diff, but reported soft --
    :func:`check_gripper_phases` is what gates the gripper. A *shape* change is
    still hard, since there is then nothing comparable element-wise.
    """
    obs = np.asarray(observed["gripper_joint_position"], dtype=float)
    gold = np.asarray(golden.get("gripper_joint_position", []), dtype=float)
    if obs.shape != gold.shape:
        compare_array(
            "golden_rollout[gripper_joint_position]", obs, gold, GRIPPER_POS_TOL, report
        )
        return
    diff = float(np.max(np.abs(obs - gold))) if obs.size else 0.0
    report.diffs.append(
        FieldDiff(
            name="golden_rollout[gripper_joint_position]",
            passed=diff <= GRIPPER_POS_TOL,
            hard=False,
            dataset_value=float(np.max(np.abs(gold))) if gold.size else 0.0,
            sim_value=float(np.max(np.abs(obs))) if obs.size else 0.0,
            diff=diff,
            tol=GRIPPER_POS_TOL,
            note=(
                "soft by design: gripper travel speed follows the actuator's "
                "effort limit, so this is a magnitude to read rather than a bound "
                "to hold. rollout_gripper[*] carries the pass/fail."
            ),
        )
    )


def test_free_space_rollout_matches_golden(rollout_env, rollout_case, request):
    write_golden = request.config.getoption("--write-rollout-golden")
    path = golden_path(rollout_case)

    report = SceneMismatchReport(dataset_path=str(path), episode_index=-1)
    observed, frames = run_free_space_rollout(rollout_env, rollout_case)
    check_free_space(frames, report)
    check_gripper_phases(frames, observed, report)

    if write_golden:
        # Only a clean run may become the baseline: a rollout that touched a
        # block, or whose gripper never tracked its command, bakes the very
        # failure in as the new expectation.
        assert report.ok, (
            "refusing to write the golden from a run that already failed:\n"
            + report.format_console()
        )
        path.write_text(json.dumps(observed, indent=2) + "\n")
        pytest.skip(f"wrote {path}; review the diff and commit it")

    if not path.is_file():
        pytest.fail(
            f"{path} does not exist, so this run compared nothing. " + REBASELINE_HINT
        )
    golden = json.loads(path.read_text())

    # The trajectory only means something once the identity matches: with a
    # different action sequence, robot or gain set its numbers describe a
    # different rollout, and reporting them as breaches would bury the one diff
    # that explains it.
    if compare_identity(observed, golden, report):
        compare_trajectory(observed, golden, report)

    assert report.ok, report.format_console()
