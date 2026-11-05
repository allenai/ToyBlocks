"""CPU tests for PDModeController's switching logic.

The GPU rollout golden
(``tests/env_randomization/test_rollout_determinism.py``, case
``softer_pd_stiff``) pins what the gains *do* to the arm. This pins the logic
that decides *when* they change, which no rollout can reach: the dead zone
between ``PD_COMPLIANT_THRESH`` and ``PD_STIFF_THRESH`` leaving gains untouched,
debounce requiring N consecutive opposite predictions, the ``_prev_pd_mode == -1``
first-time bypass, ``reset_envs`` restoring that bypass, and per-env
independence under a vectorized ``apply``. Together they are the whole contract
of the thing eval runs whenever a policy emits the ``pd_mode`` channel.

Pure torch on CPU, so it belongs in the CPU tier and needs no Isaac Lab.
``_commit`` reads ``robot.data.joint_stiffness`` / ``joint_damping`` /
``joint_effort_limits`` and calls ``base_env.set_actuator_gains``, so the stub
below exposes those three tensors and records every call -- including the
``settle_env_mask``, which is how the tests assert that a switch touched only the
envs that qualified.

Run:
    pytest tests/eval/test_pd_mode_controller.py -v
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from blocksuite.eval.pd_mode_controller import (
    DeferredSettle,
    PDModeController,
    settle_hold_action,
)
from blocksuite.serving.action_chunk_broker import ActionChunkBroker
from blocksuite.serving.obs_transforms import ObsTransformPolicy
from blocksuite.serving.obs_transforms.maniflow import ManiflowObsTransform

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
# config, which is what real eval runs when no policy drives pd_mode.
PD_COMPLIANT_GAINS: dict[str, list[float]] = {
    "stiffness": [40.0] * 7,
    "damping": [13.0] * 7,
    "effort": [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0],
}

N_ARM = 7
# One more than the arm, standing in for the gripper's finger_joint: PD mode must
# leave it alone.
N_JOINTS = N_ARM + 1
# Distinct from every PD gain value, so "untouched" is provable rather than
# coincidental.
BASE_STIFFNESS = 5.0
BASE_DAMPING = 6.0
BASE_EFFORT = 7.0


class StubArticulationData:
    def __init__(self, num_envs: int):
        self.joint_pos = torch.zeros(num_envs, N_JOINTS)
        self.joint_pos_target = torch.zeros(num_envs, N_JOINTS)
        self.joint_stiffness = torch.full((num_envs, N_JOINTS), BASE_STIFFNESS)
        self.joint_damping = torch.full((num_envs, N_JOINTS), BASE_DAMPING)
        self.joint_effort_limits = torch.full((num_envs, N_JOINTS), BASE_EFFORT)


class StubArticulation:
    def __init__(self, num_envs: int):
        self.data = StubArticulationData(num_envs)
        self.joint_names = [f"panda_joint{i + 1}" for i in range(N_ARM)] + ["finger_joint"]


#: ``ROBOTIQ_GRIPPER_BINARY_ACTION``'s finger_joint commands.
GRIPPER_OPEN = 0.0
GRIPPER_CLOSE = 0.72


class StubArmAction:
    """The attributes ``settle_hold_action`` reads off IsaacLab's ``JointPositionAction``."""

    def __init__(self):
        self._joint_ids = list(range(N_ARM))


class StubGripperAction:
    """The attributes ``settle_hold_action`` reads off IsaacLab's ``BinaryJointPositionAction``."""

    def __init__(self):
        self._joint_ids = [N_ARM]
        self._open_command = torch.tensor([GRIPPER_OPEN])
        self._close_command = torch.tensor([GRIPPER_CLOSE])


class StubRobot:
    def __init__(self, num_envs: int):
        self.articulation = StubArticulation(num_envs)
        self.controller = {"arm": StubArmAction(), "gripper": StubGripperAction()}


class StubEnvCfg:
    decimation: int = 4


class StubEnv:
    """Records the gains handed to ``set_actuator_gains`` instead of applying them.

    Deliberately does not write them back into ``robot.data``: the controller
    reads the live gains as the base for each commit, and leaving them at the
    BASE_* constants means every recorded call shows exactly which columns that
    commit wrote.
    """

    def __init__(self, num_envs: int = 1):
        self.robot = StubRobot(num_envs)
        self.cfg = StubEnvCfg()
        self.calls: list[dict] = []

    def set_actuator_gains(
        self, stiffness=None, damping=None, effort_limits=None,
        settle_steps=0, settle_env_mask=None,
        settle_joint_ids=None, settle_zero_velocity=True,
    ):
        self.calls.append(
            {
                "stiffness": stiffness.clone(),
                "damping": damping.clone(),
                "effort_limits": effort_limits.clone(),
                "settle_steps": settle_steps,
                "settle_env_mask": settle_env_mask.clone(),
            }
        )


def controller(env: StubEnv, *, num_envs: int = 1, debounce: int = 0) -> PDModeController:
    return PDModeController(
        env,
        torch.device("cpu"),
        num_envs=num_envs,
        debounce=debounce,
        stiff_gains=PD_STIFF_GAINS,
        compliant_gains=PD_COMPLIANT_GAINS,
    )


def apply(ctrl: PDModeController, *values: float) -> None:
    for value in values:
        ctrl.apply(torch.tensor(list(_as_tuple(value)), dtype=torch.float32))


def _as_tuple(value):
    return value if isinstance(value, (tuple, list)) else (value,)


# --------------------------------------------------------------------------
# gain values
# --------------------------------------------------------------------------


def test_stiff_mode_writes_pd_stiff_gains_to_the_arm_only():
    """The constants themselves, on the columns they are supposed to reach.

    A stub rather than a rollout because this is the only place the exact
    numbers are checkable: in sim they are one input to a trajectory.
    """
    env = StubEnv()
    apply(controller(env), 1.0)

    (call,) = env.calls
    assert call["stiffness"][0, :N_ARM].tolist() == PD_STIFF_GAINS["stiffness"]
    assert call["damping"][0, :N_ARM].tolist() == PD_STIFF_GAINS["damping"]
    assert call["effort_limits"][0, :N_ARM].tolist() == PD_STIFF_GAINS["effort"]
    # The gripper column must survive a PD-mode switch untouched: it is the one
    # actuator the fr3 soft/softer configs differ on.
    assert call["stiffness"][0, N_ARM].item() == BASE_STIFFNESS
    assert call["damping"][0, N_ARM].item() == BASE_DAMPING
    assert call["effort_limits"][0, N_ARM].item() == BASE_EFFORT


def test_compliant_mode_writes_pd_compliant_gains():
    env = StubEnv()
    apply(controller(env), 0.0)

    (call,) = env.calls
    assert call["stiffness"][0, :N_ARM].tolist() == PD_COMPLIANT_GAINS["stiffness"]
    assert call["damping"][0, :N_ARM].tolist() == PD_COMPLIANT_GAINS["damping"]
    # Not a flat vector: the wrist joints are effort-limited to 12.
    assert call["effort_limits"][0, :N_ARM].tolist() == PD_COMPLIANT_GAINS["effort"]


def test_gains_are_written_without_settle_steps():
    """The normal ``env.step()`` substeps absorb the transition.

    Settle steps here would freeze the arm mid-episode, which is a visible
    hitch in a policy rollout rather than a smooth gain change.
    """
    env = StubEnv()
    apply(controller(env), 1.0)
    assert env.calls[0]["settle_steps"] == 0


# --------------------------------------------------------------------------
# dead zone
# --------------------------------------------------------------------------


def test_dead_zone_prediction_leaves_gains_untouched():
    """Between the two thresholds, nothing is written at all.

    Not even on the first call: an ambiguous prediction is not evidence for
    either mode, so initializing from one would be a coin flip baked into the
    episode.
    """
    env = StubEnv()
    ctrl = controller(env)
    apply(ctrl, 0.5)
    assert env.calls == []
    # And it did not quietly record a mode either, so a later clear prediction
    # still takes the first-time path.
    apply(ctrl, 0.5, 0.31, 0.69)
    assert env.calls == []


def test_dead_zone_does_not_disturb_an_established_mode():
    env = StubEnv()
    ctrl = controller(env)
    apply(ctrl, 1.0)
    apply(ctrl, 0.5, 0.6, 0.4)
    assert len(env.calls) == 1


def test_thresholds_are_inclusive_at_both_bounds():
    """Exactly on a threshold counts as that mode, not as the dead zone."""
    stiff_env, compliant_env = StubEnv(), StubEnv()
    apply(controller(stiff_env), 0.7)
    apply(controller(compliant_env), 0.3)
    assert len(stiff_env.calls) == 1
    assert len(compliant_env.calls) == 1


# --------------------------------------------------------------------------
# first-time bypass / debounce
# --------------------------------------------------------------------------


def test_first_prediction_bypasses_debounce():
    """The robot must start the episode at the mode the policy asked for.

    Waiting out the debounce would run the first N steps under whatever gains
    the robot config happened to ship.
    """
    env = StubEnv()
    apply(controller(env, debounce=5), 1.0)
    assert len(env.calls) == 1


def test_debounce_requires_consecutive_opposite_predictions():
    env = StubEnv()
    ctrl = controller(env, debounce=2)
    apply(ctrl, 1.0)
    assert len(env.calls) == 1

    apply(ctrl, 0.0)
    assert len(env.calls) == 1, "one opposite prediction must not switch at debounce=2"

    apply(ctrl, 0.0)
    assert len(env.calls) == 2
    assert env.calls[1]["stiffness"][0, :N_ARM].tolist() == PD_COMPLIANT_GAINS["stiffness"]


def test_debounce_counter_resets_on_a_same_as_current_prediction():
    """Non-consecutive opposite predictions must never accumulate into a switch.

    This is the thrashing case debounce exists for: a policy alternating around
    the boundary would otherwise switch gains every other step.
    """
    env = StubEnv()
    ctrl = controller(env, debounce=2)
    apply(ctrl, 1.0, 0.0, 1.0, 0.0)
    assert len(env.calls) == 1


def test_zero_debounce_switches_immediately():
    """The shipped default (PD_MODE_DEBOUNCE = 0) is hysteresis-only."""
    env = StubEnv()
    ctrl = controller(env)
    apply(ctrl, 1.0, 0.0, 1.0)
    assert len(env.calls) == 3


def test_repeating_the_current_mode_does_not_rewrite_gains():
    env = StubEnv()
    ctrl = controller(env)
    apply(ctrl, 1.0, 1.0, 0.9)
    assert len(env.calls) == 1


# --------------------------------------------------------------------------
# reset_envs
# --------------------------------------------------------------------------


def test_reset_envs_restores_the_first_time_bypass():
    """A new episode starts at the policy's requested mode, not the last one's."""
    env = StubEnv()
    ctrl = controller(env, debounce=5)
    apply(ctrl, 1.0)
    ctrl.reset_envs(torch.tensor([True]))

    apply(ctrl, 0.0)
    assert len(env.calls) == 2, "post-reset the first prediction must bypass debounce"
    assert env.calls[1]["stiffness"][0, :N_ARM].tolist() == PD_COMPLIANT_GAINS["stiffness"]


def test_reset_envs_clears_a_pending_switch():
    """Debounce progress must not carry across an episode boundary."""
    env = StubEnv()
    ctrl = controller(env, debounce=2)
    apply(ctrl, 1.0, 0.0)
    ctrl.reset_envs(torch.tensor([True]))
    # Re-establishes stiff through the first-time path, so the earlier single
    # compliant prediction is gone rather than one step from committing.
    apply(ctrl, 1.0)
    assert len(env.calls) == 2
    assert env.calls[1]["stiffness"][0, :N_ARM].tolist() == PD_STIFF_GAINS["stiffness"]


def test_reset_envs_only_resets_the_masked_envs():
    env = StubEnv(num_envs=2)
    ctrl = controller(env, num_envs=2, debounce=5)
    apply(ctrl, (1.0, 1.0))
    ctrl.reset_envs(torch.tensor([True, False]))

    apply(ctrl, (0.0, 0.0))
    # env 0 was reset so it re-initializes; env 1 keeps its mode and is 5
    # predictions short of switching.
    assert env.calls[-1]["settle_env_mask"].tolist() == [True, False]


# --------------------------------------------------------------------------
# per-env independence
# --------------------------------------------------------------------------


def test_each_env_switches_on_its_own_prediction():
    env = StubEnv(num_envs=3)
    ctrl = controller(env, num_envs=3)
    # stiff / dead zone / compliant in one vectorized call.
    apply(ctrl, (1.0, 0.5, 0.0))

    (call,) = env.calls
    assert call["settle_env_mask"].tolist() == [True, False, True]
    assert call["stiffness"][0, :N_ARM].tolist() == PD_STIFF_GAINS["stiffness"]
    assert call["stiffness"][2, :N_ARM].tolist() == PD_COMPLIANT_GAINS["stiffness"]
    # The dead-zone env keeps the gains it came in with. It is included in the
    # written tensor (set_actuator_gains takes a full (num_envs, num_joints)
    # array), so the mask is what has to protect it.
    assert call["stiffness"][1].tolist() == [BASE_STIFFNESS] * N_JOINTS
    assert call["damping"][1].tolist() == [BASE_DAMPING] * N_JOINTS
    assert call["effort_limits"][1].tolist() == [BASE_EFFORT] * N_JOINTS


def test_one_env_s_debounce_does_not_gate_another_s():
    env = StubEnv(num_envs=2)
    ctrl = controller(env, num_envs=2, debounce=2)
    apply(ctrl, (1.0, 1.0))
    # env 0 asks to switch twice in a row and gets it; env 1 never asks.
    apply(ctrl, (0.0, 1.0), (0.0, 1.0))

    assert len(env.calls) == 2
    assert env.calls[1]["settle_env_mask"].tolist() == [True, False]
    assert env.calls[1]["stiffness"][0, :N_ARM].tolist() == PD_COMPLIANT_GAINS["stiffness"]


def test_a_scalar_prediction_is_accepted_for_a_single_env():
    """``run_eval`` passes ``action[8].unsqueeze(0)`` for unbatched actions, but a
    0-dim tensor reaches ``apply`` from callers that index a single env."""
    env = StubEnv()
    controller(env).apply(torch.tensor(1.0))
    assert len(env.calls) == 1


# --------------------------------------------------------------------------
# agent_pos pd_mode sync (controller -> ManiflowObsTransform)
# --------------------------------------------------------------------------
#
# Training's agent_pos pd_mode is the current row's ``action.pd_mode``, which the
# recorder derives from the stiffness in sim at capture time. The obs transform
# only runs at chunk boundaries, so ``run_eval`` resyncs it from the controller's
# committed modes after every ``apply``. These drive the real client chain
# (ActionChunkBroker -> ObsTransformPolicy -> ManiflowObsTransform) through the
# same per-step sequence ``run_eval`` uses and check what the server receives.

ACTION_HORIZON = 25


class StubServer:
    """Records each request's agent_pos pd_mode and answers with a scripted chunk.

    ``chunk_pd_modes(call_idx)`` returns a ``(T, B)`` pd_mode schedule; the rest
    of each 9-dim action is zero.
    """

    def __init__(self, chunk_pd_modes: Callable[[int], np.ndarray]):
        self._chunk_pd_modes = chunk_pd_modes
        self.seen_pd_modes: list[np.ndarray] = []

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        self.seen_pd_modes.append(obs["agent_pos"][:, 0, -1].copy())
        pd = np.asarray(self._chunk_pd_modes(len(self.seen_pd_modes) - 1), dtype=np.float32)
        actions = np.zeros((*pd.shape, 9), dtype=np.float32)
        actions[..., 8] = pd
        return {"actions": actions}


def _obs(num_envs: int) -> dict[str, Any]:
    image = np.zeros((num_envs, 4, 4, 3), dtype=np.uint8)
    return {
        "agent": {"joint_pos": np.zeros((num_envs, 8), dtype=np.float32)},
        "images": {
            "external_camera": {"rgb": image},
            "wrist_camera": {"rgb": image},
        },
    }


def _client(server: StubServer) -> tuple[ActionChunkBroker, ManiflowObsTransform]:
    transform = ManiflowObsTransform(
        state_dim=9,
        include_pd_mode=True,
        include_ptmap=False,
        image_size=(4, 4),
        image_interpolate_device="cpu",
    )
    return ActionChunkBroker(ObsTransformPolicy(server, transform), ACTION_HORIZON), transform


def _step(broker: ActionChunkBroker, transform: ManiflowObsTransform, ctrl: PDModeController,
          num_envs: int) -> None:
    """One ``run_eval`` control step: act, apply pd_mode, resync the transform."""
    action = broker.infer(_obs(num_envs))["actions"]
    ctrl.apply(torch.from_numpy(action[:, 8]))
    transform.set_pd_mode(ctrl.current_pd_mode())


def test_agent_pos_pd_mode_reports_a_switch_made_mid_chunk():
    """A switch inside a chunk reaches the next inference's agent_pos.

    Without the resync the transform keeps the chunk's first-step mode, so the
    server would see stiff (1.0) while the compliant gains are in sim.
    """
    schedule = np.ones((ACTION_HORIZON, 1), dtype=np.float32)
    schedule[10:] = 0.0  # stiff for 10 steps, then compliant
    server = StubServer(lambda _call: schedule)
    broker, transform = _client(server)
    ctrl = controller(StubEnv())

    for _ in range(ACTION_HORIZON + 1):
        _step(broker, transform, ctrl, num_envs=1)

    assert len(server.seen_pd_modes) == 2
    assert server.seen_pd_modes[1].tolist() == [0.0]


def test_agent_pos_pd_mode_ignores_chunks_an_env_did_not_take():
    """A mid-chunk env keeps its own mode when another env re-plans.

    After env 1's episode ends, the next batched inference is taken only by
    env 1, but ``action_postprocess`` stores that chunk's pd_mode for every env.
    Env 0 is still executing its compliant chunk, so its next request must say
    compliant rather than the stiff row it never executed.
    """
    compliant_then_stiff = [
        np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (ACTION_HORIZON, 1)),
        np.ones((ACTION_HORIZON, 2), dtype=np.float32),
    ]
    server = StubServer(lambda call: compliant_then_stiff[min(call, 1)])
    broker, transform = _client(server)
    ctrl = controller(StubEnv(num_envs=2), num_envs=2)

    for _ in range(5):
        _step(broker, transform, ctrl, num_envs=2)
    broker.reset_envs([1])
    # Env 1 re-plans now; env 0 needs a fresh chunk at its own step 25.
    for _ in range(ACTION_HORIZON - 5 + 1):
        _step(broker, transform, ctrl, num_envs=2)

    assert len(server.seen_pd_modes) == 3
    assert server.seen_pd_modes[2][0] == 0.0


# --------------------------------------------------------------------------
# deferred settle
# --------------------------------------------------------------------------
#
# The in-place settle runs raw sim substeps, which advance every env in the
# scene. At every env count, run_eval instead writes the gains at once and has
# each switching env spend its settle as ordinary hold steps (DeferredSettle),
# sized by settle_ticks().

#: ``DATASET_SWITCH_SETTLE_TICKS``: (compliant, stiff).
SETTLE_TICKS = (5, 10)
CPU = torch.device("cpu")


def settling_controller(
    env: StubEnv, *, num_envs: int, defer_settle: bool = True, debounce: int = 0
) -> PDModeController:
    """Built the way run_eval builds it for a dataset-driven eval."""
    return PDModeController(
        env,
        CPU,
        num_envs=num_envs,
        debounce=debounce,
        initial_mode=1,
        stiff_gains=PD_STIFF_GAINS,
        compliant_gains=PD_COMPLIANT_GAINS,
        switch_settle_ticks=SETTLE_TICKS,
        defer_settle=defer_settle,
    )


def test_legacy_settle_still_runs_its_substeps_in_place():
    """``defer_settle=False`` still runs the raw-substep settle: ``ticks * decimation``."""
    env = StubEnv()
    ctrl = settling_controller(env, num_envs=1, defer_settle=False)
    ctrl.apply(torch.tensor([0.0]))
    assert env.calls[-1]["settle_steps"] == SETTLE_TICKS[0] * StubEnvCfg.decimation


def test_deferred_settle_writes_the_gains_with_no_settle_steps():
    env = StubEnv(num_envs=2)
    ctrl = settling_controller(env, num_envs=2)
    ctrl.apply(torch.tensor([0.0, 1.0]))
    ctrl.apply(torch.tensor([1.0, 0.0]))
    # initial_mode write + one per switching apply
    assert len(env.calls) == 3
    assert [c["settle_steps"] for c in env.calls] == [0, 0, 0]
    assert env.calls[1]["stiffness"][0, :N_ARM].tolist() == PD_COMPLIANT_GAINS["stiffness"]


def test_apply_returns_the_envs_it_switched():
    env = StubEnv(num_envs=3)
    ctrl = settling_controller(env, num_envs=3)
    assert ctrl.apply(torch.tensor([0.0, 1.0, 0.0])).tolist() == [True, False, True]
    # Same-as-current and dead-zone predictions switch nothing.
    assert ctrl.apply(torch.tensor([0.0, 0.5, 0.0])).tolist() == [False, False, False]


def test_apply_skips_masked_out_envs():
    env = StubEnv(num_envs=2)
    ctrl = settling_controller(env, num_envs=2)
    switched = ctrl.apply(torch.tensor([0.0, 0.0]), env_mask=torch.tensor([True, False]))
    assert switched.tolist() == [True, False]
    assert ctrl.applied_modes.tolist() == [0.0, 1.0]


def test_a_masked_out_env_keeps_its_debounce_state():
    """A busy env's predictions are placeholders, so they must not count either way.

    Unmasked, env 1's same-as-current 1.0 would reset its pending count; masked,
    the count survives and the next opposite prediction completes the switch.
    """
    env = StubEnv(num_envs=2)
    ctrl = settling_controller(env, num_envs=2, debounce=2)
    assert ctrl.apply(torch.tensor([0.0, 0.0])).tolist() == [False, False]
    masked = torch.tensor([True, False])
    assert ctrl.apply(torch.tensor([1.0, 1.0]), env_mask=masked).tolist() == [False, False]
    assert ctrl.apply(torch.tensor([1.0, 0.0])).tolist() == [False, True]


def test_settle_ticks_follow_the_mode_each_env_is_now_in():
    env = StubEnv(num_envs=2)
    ctrl = settling_controller(env, num_envs=2)
    ctrl.apply(torch.tensor([0.0, 1.0]))
    compliant_ticks, stiff_ticks = SETTLE_TICKS
    assert ctrl.settle_ticks().tolist() == [compliant_ticks, stiff_ticks]


def test_settle_ticks_are_zero_without_a_dataset_settle():
    env = StubEnv(num_envs=2)
    assert controller(env, num_envs=2).settle_ticks().tolist() == [0, 0]


ACTION_DIM = N_ARM + 1
HOLD = -100.0


def _policy_action(step: int, num_envs: int) -> torch.Tensor:
    return torch.full((num_envs, ACTION_DIM), float(step))


def _drive(
    settle: DeferredSettle, num_steps: int, starts: dict[int, tuple[list[bool], int]]
) -> tuple[list[list[float]], list[list[bool]], list[list[bool]]]:
    """Run run_eval's per-step sequence: read busy, maybe start, compose.

    ``starts`` maps a step to ``(switched_mask, ticks)``. The policy action at
    step ``t`` is all ``t``, the hold action all ``HOLD``, so the stepped value
    says exactly which action each env ran.
    """
    stepped, busy, holding = [], [], []
    for t in range(num_steps):
        num_envs = settle.busy.shape[0]
        busy.append(settle.busy.tolist())
        action = _policy_action(t, num_envs)
        if t in starts:
            mask, ticks = starts[t]
            settle.start(
                torch.tensor(mask),
                torch.full((num_envs,), ticks, dtype=torch.long),
                torch.full((num_envs, ACTION_DIM), HOLD),
                action,
            )
        out, held = settle.compose(action)
        stepped.append(out[:, 0].tolist())
        holding.append(held.tolist())
    return stepped, busy, holding


def test_deferred_settle_timeline():
    """Switch at step t with k ticks: hold t..t+k-1, run a_t at t+k, busy t+1..t+k."""
    settle = DeferredSettle(2, CPU)
    stepped, busy, holding = _drive(settle, 8, {2: ([True, False], 3)})

    assert [s[0] for s in stepped] == [0, 1, HOLD, HOLD, HOLD, 2, 6, 7]
    assert [b[0] for b in busy] == [False, False, False, True, True, True, False, False]
    assert [h[0] for h in holding] == [False, False, True, True, True, False, False, False]
    # The other env never notices.
    assert [s[1] for s in stepped] == list(range(8))
    assert not any(b[1] for b in busy) and not any(h[1] for h in holding)


def test_a_single_env_settles_as_hold_steps_and_cannot_reswitch_while_busy():
    """``num_envs == 1`` runs the same deferred settle, driven the way run_eval drives it.

    A compliant switch at step 1 holds steps 1-5 and runs a_1 at step 6. The
    stiff predictions on the busy steps 2-6 would switch straight back if they
    were not masked out.
    """
    env = StubEnv()
    ctrl = settling_controller(env, num_envs=1)
    settle = DeferredSettle(1, CPU)
    preds = [1.0, 0.0] + [1.0] * 5 + [0.0] * 2
    stepped: list[float] = []
    busy: list[bool] = []
    holding: list[bool] = []
    switched: list[bool] = []
    for t, pred in enumerate(preds):
        settle_busy = settle.busy
        busy.append(bool(settle_busy[0]))
        action = _policy_action(t, 1)
        switch_mask = ctrl.apply(torch.tensor([pred]), env_mask=~settle_busy)
        switched.append(bool(switch_mask[0]))
        if bool(switch_mask.any()):
            settle.start(
                switch_mask, ctrl.settle_ticks(), torch.full((1, ACTION_DIM), HOLD), action
            )
        out, held = settle.compose(action)
        stepped.append(out[0, 0].item())
        holding.append(bool(held[0]))

    assert SETTLE_TICKS[0] == 5
    assert stepped == [0.0] + [HOLD] * 5 + [1.0, 7.0, 8.0]
    assert busy == [False, False] + [True] * 5 + [False, False]
    assert holding == [False] + [True] * 5 + [False] * 3
    assert switched == [False, True] + [False] * 7
    assert ctrl.applied_modes.tolist() == [0.0]
    assert [c["settle_steps"] for c in env.calls] == [0, 0]


def test_overlapping_settles_are_independent():
    settle = DeferredSettle(2, CPU)
    stepped, _, _ = _drive(
        settle, 8, {1: ([True, False], 3), 2: ([False, True], 2)}
    )
    assert [s[0] for s in stepped] == [0, HOLD, HOLD, HOLD, 1, 5, 6, 7]
    assert [s[1] for s in stepped] == [0, 1, HOLD, HOLD, 2, 5, 6, 7]


def test_clear_drops_a_settle_mid_hold():
    """An episode ending mid-hold must not leak its deferred action into the next one."""
    settle = DeferredSettle(1, CPU)
    settle.start(
        torch.tensor([True]),
        torch.tensor([3]),
        torch.full((1, ACTION_DIM), HOLD),
        _policy_action(0, 1),
    )
    settle.compose(_policy_action(0, 1))
    settle.clear(torch.tensor([True]))

    assert settle.busy.tolist() == [False]
    out, held = settle.compose(_policy_action(1, 1))
    assert held.tolist() == [False]
    assert out[0, 0].item() == 1.0


def test_zero_ticks_runs_the_deferred_action_on_the_switch_step():
    settle = DeferredSettle(1, CPU)
    stepped, busy, holding = _drive(settle, 3, {1: ([True], 0)})
    assert [s[0] for s in stepped] == [0, 1, 2]
    assert busy == [[False]] * 3
    assert holding == [[False]] * 3


def test_compose_is_a_pass_through_when_nothing_settles():
    settle = DeferredSettle(2, CPU)
    action = _policy_action(4, 2)
    out, held = settle.compose(action)
    assert out is action
    assert held.tolist() == [False, False]


# --------------------------------------------------------------------------
# the hold action
# --------------------------------------------------------------------------


def test_hold_action_is_the_arm_pose_and_the_gripper_command_of_the_finger_target():
    env = StubEnv(num_envs=3)
    data = env.robot.articulation.data
    data.joint_pos[:, :N_ARM] = torch.arange(3 * N_ARM, dtype=torch.float32).reshape(3, N_ARM)
    # The finger's own position is mid-travel in every env: only its target counts.
    data.joint_pos[:, N_ARM] = 0.4
    data.joint_pos_target[:, N_ARM] = torch.tensor([GRIPPER_CLOSE, GRIPPER_OPEN, 0.3])

    hold = settle_hold_action(env)

    assert hold.shape == (3, ACTION_DIM)
    assert torch.equal(hold[:, :N_ARM], data.joint_pos[:, :N_ARM])
    # BinaryJointPositionAction closes on a negative command.
    assert hold[:, N_ARM].tolist() == [-1.0, 1.0, 1.0]


def test_hold_action_is_a_snapshot():
    """The arm keeps moving after the switch; the hold must stay where it was."""
    env = StubEnv(num_envs=1)
    hold = settle_hold_action(env)
    env.robot.articulation.data.joint_pos += 1.0
    assert hold[0, :N_ARM].tolist() == [0.0] * N_ARM


def test_a_switch_on_an_episode_s_first_step_holds_the_reset_pose_gripper():
    """``_reset_idx`` clears ``prev_actions`` but not ``actions``.

    So the previous episode's closing command is still on ``base_env.actions``
    while the articulation's finger target is already the reset pose (open).
    Holding with the former would close the gripper on a fresh scene.
    """
    env = StubEnv(num_envs=2)
    env.actions = torch.zeros(2, ACTION_DIM)
    env.actions[:, N_ARM] = -1.0
    env.robot.articulation.data.joint_pos_target[:, N_ARM] = GRIPPER_OPEN

    assert settle_hold_action(env)[:, N_ARM].tolist() == [1.0, 1.0]


def test_hold_action_handles_a_gripper_that_closes_toward_lower_targets():
    env = StubEnv(num_envs=2)
    gripper = env.robot.controller["gripper"]
    gripper._open_command = torch.tensor([0.8])
    gripper._close_command = torch.tensor([0.0])
    env.robot.articulation.data.joint_pos_target[:, N_ARM] = torch.tensor([0.0, 0.8])
    assert settle_hold_action(env)[:, N_ARM].tolist() == [-1.0, 1.0]


def test_hold_action_needs_a_binary_gripper_term():
    env = StubEnv(num_envs=1)
    env.robot.controller = {"arm": StubArmAction(), "gripper": object()}
    try:
        settle_hold_action(env)
    except ValueError as err:
        assert "binary" in str(err)
    else:
        raise AssertionError("expected a ValueError for a non-binary gripper term")
