"""Play data on the vecgen engine: goal policy -> FF -> batched skills, no reset between plans.

The TAMP-free successor of the former TAMP play loop (``playdata/orchestrator.py``). Every env plays a
*chain* of plans from one reset state:

1. **settle** - after a reset (fixed ``pre_episode_ticks``) or after the previous plan
   (a short minimum, then until every block and the gripper are at rest, capped);
   nothing is recorded meanwhile.
2. **goal** - the symbolic state is read off the sim with the same batched predicates
   vecgen verifies with (``on`` / ``on_table`` / ``clear`` / ``ori_of``, plus the derived
   geometric facts), and the goal policy (balanced skills or random goals, from
   :mod:`blocksuite.playdata`) picks a goal for it.
3. **plan** - one FF call per goal on a worker thread; the plan becomes a skill program.
4. **execute** - the engine runs the program as one recorded episode, verifies the
   goal, and the env starts over from the state the plan left behind.

A chain ends when its budget is met (``max_plans_per_env`` plans,
``min_successful_plans_per_env`` goals reached, or ``max_plan_failures_per_env``
consecutive failures to get a plan); with ``one_chain_per_reset_state`` the env then
resets to the next unplayed reset state of its scene profile, otherwise it retires.

With ``abandon_plan_prob`` a pick-and-place plan is cut off partway through its
transport: the episode is kept as a success, the block stays in the gripper, and the
next goal moves a different block, so that episode opens by dropping it.
"""

from __future__ import annotations

import logging
import random
import time
from collections import Counter, deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from blocksuite.playdata.abandon import blocks_below, followup_actions_enabled, followup_pnp_goal
from blocksuite.playdata.goal_policy import Decision, GoalPolicy
from blocksuite.playdata.goal_proposer import ProposedGoal
from blocksuite.playdata.skill_balance import skill_class
from blocksuite.playdata.symbolic_state import SymbolicState
from blocksuite.vecgen.engine import Engine, EngineConfig, Recorder
from blocksuite.vecgen.geometry import Geometry
from blocksuite.vecgen.program import Phase, Program
from blocksuite.vecgen.sim import VecSim
from blocksuite.vecgen.skills.base import Skill
from blocksuite.vecgen.symbolic.ff import FFResult, run_ff
from blocksuite.vecgen.symbolic.pddl import Fact, Problem, parse_facts, render_problem
from blocksuite.vecgen.symbolic.predicates import (
    classify_orientation,
    goal_facts_satisfied,
    on,
    on_table,
    ori_of,
)
from blocksuite.vecgen.symbolic.programs import UnsupportedAction, program_from_plan
from blocksuite.vecgen.symbolic.state import derive_init_facts

log = logging.getLogger("vecgen")

#: Facts the live state determines; every other fact of the problem's init is static.
DYNAMIC_PREDICATES: frozenset[str] = frozenset(
    {"on", "on_table", "clear", "ori_of", "handempty", "holding", "gripper_at_init"}
)
#: Goal predicates vecgen can evaluate; ``holds`` reports False for any other one, so a
#: random goal naming it is never discarded as already satisfied.
EVALUATED_PREDICATES: frozenset[str] = frozenset({"on", "on_table", "clear", "ori_of", "next_to"})
#: Segment label of a place's transport (the carry an abandon cuts off).
TRANSPORT_LABEL = "place_traj"
PLACE_SKILLS: frozenset[str] = frozenset({"place_on", "place_table", "place_side"})

DerivedFn = Callable[[Problem, Tensor], list[list[Fact]]]


@dataclass
class PlayConfig:
    """Chain budgets, settle timing (in engine ticks) and the abandon knobs."""

    max_plans_per_env: int | None = None
    max_plans_total: int | None = None
    min_successful_plans_per_env: int | None = None
    one_chain_per_reset_state: bool = False
    max_plan_failures_per_env: int = 8
    settle_ticks: int = 15
    settle_min_ticks: int = 2
    settle_lin_vel: float = 0.01
    settle_ang_vel: float = 0.1
    settle_gripper_vel: float = 0.05
    poll_ticks: int = 2
    pre_episode_ticks: int = 12
    missed_goals_to_failure: bool = False
    abandon_plan_prob: float = 0.0
    abandon_fraction_range: tuple[float, float] = (0.2, 0.8)
    allowed_actions: tuple[str, ...] | None = None
    max_stack_height: int = 3
    ff_timeout_s: float = 60.0
    ff_workers: int = 16


def program_skills(program: Program) -> list[str]:
    """Skill classes of a program's high-level actions (each action counted once)."""
    skills: list[str] = []
    previous: tuple[str, tuple[str, ...]] | None = None
    for call in program.skills:
        key = (call.hl_name, tuple(call.hl_args))
        if not call.hl_name or key == previous:
            continue
        previous = key
        cls = skill_class(call.hl_name, list(call.hl_args))
        if cls is not None:
            skills.append(cls)
    return skills


def read_symbolic_state(
    problem: Problem,
    block_index: dict[str, int],
    poses: Tensor,
    sizes: Tensor,
    table_pose: Tensor,
) -> tuple[SymbolicState, list[Fact]]:
    """One env's symbolic state and the planner init facts for it.

    ``poses`` ``[K, 7]`` / ``sizes`` ``[K, 3]`` cover every scene block (``block_index``
    order), so scene blocks the problem does not declare (distractors, a superset
    package's parked blocks) still count as support and as cover.
    """
    table = next(iter(problem.objects_of_type("table")), "workspace_table")
    gripper = next(iter(problem.objects_of_type("gripper")), "robot")
    blocks = [b for b in problem.blocks if b in block_index]
    facts = derive_init_facts(
        problem, block_index, poses, sizes, table_pose, table_name=table, robot_name=gripper
    )
    state = SymbolicState(blocks=blocks, table=table, gripper=gripper)
    state.missing_from_scene = [b for b in problem.blocks if b not in block_index]
    for f in facts:
        if f.pred == "on":
            state.on[f.args[0]] = f.args[1]
        elif f.pred == "on_table":
            state.on_table.append(f.args[0])
        elif f.pred == "ori_of":
            state.ori[f.args[0]] = f.args[1]
    declared = set(blocks)
    scene = list(block_index)
    idx = torch.tensor([block_index[b] for b in scene], device=poses.device)
    p = poses[idx]
    k = len(scene)
    on_mat = on(p[:, None, :].expand(-1, k, -1), p[None, :, :].expand(k, -1, -1))
    on_mat[torch.arange(k), torch.arange(k)] = False
    row = {b: i for i, b in enumerate(scene)}
    covered: set[str] = set()
    for b in blocks:
        i = row[b]
        below = [scene[j] for j in on_mat[i].nonzero().flatten().tolist()]
        if any(x not in declared for x in below) or (
            not below and not bool(on_table(p[i], sizes[block_index[b]], table_pose))
        ):
            # on an undeclared block, or on nothing at all (held, leaning, mid-fall)
            state.unsupported.append(b)
        above = [scene[j] for j in on_mat[:, i].nonzero().flatten().tolist()]
        if any(x not in declared for x in above):
            state.covered_by_untracked.append(b)
            covered.add(b)
    cls = classify_orientation(poses[[block_index[b] for b in blocks]][:, 3:7])
    exact = ori_of(poses[[block_index[b] for b in blocks]][:, 3:7], cls)
    state.tilted = [b for b, ok in zip(blocks, exact.tolist(), strict=True) if not ok]
    facts = [f for f in facts if not (f.pred == "clear" and f.args[0] in covered)]
    return state, facts


@dataclass
class _Pending:
    """An env's FF call in flight."""

    future: Future
    problem: Problem
    goal: ProposedGoal
    reason: str


class PlayEngine(Engine):
    """The vecgen engine driving chains of policy-chosen goals instead of fixed problems."""

    def __init__(
        self,
        sim: VecSim,
        geo: Geometry,
        skills: dict[str, Skill],
        cfg: EngineConfig,
        play: PlayConfig,
        goal_policy: GoalPolicy,
        problem_for: Callable[[int], Problem],
        ff_domain: Path,
        derived_fn: DerivedFn,
        recorder: Recorder | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(sim, geo, skills, lambda _e, _r: None, cfg, recorder=recorder, seed=seed)
        self.replan_fn = self._replan_live
        self.play = play
        self.goal_policy = goal_policy
        self.problem_for = problem_for
        self.ff_domain = Path(ff_domain)
        self.derived_fn = derived_fn
        self.rng = random.Random(seed + 1)
        self._ff_threads = ThreadPoolExecutor(
            max_workers=max(1, play.ff_workers), thread_name_prefix="vecgen-play-ff"
        )
        n = self.n
        # per env: "reset" (needs a chain), "settle", "plan" (FF in flight), "episode", "done"
        self.mode: list[str] = ["reset"] * n
        self._wait: list[int] = [0] * n
        self._settle_budget: list[int] = [0] * n
        self._pending: dict[int, _Pending] = {}
        self._episode_problem: dict[int, str] = {}
        self._episode_goal: dict[int, str] = {}
        self._abandon_at: list[float | None] = [None] * n
        self._held_after_abandon: list[str | None] = [None] * n
        self._chain_rid: list[int | None] = [None] * n
        self._chain_plans: list[int] = [0] * n
        self._chain_successes: list[int] = [0] * n
        self._plan_failures: list[int] = [0] * n
        self._no_goal: list[int] = [0] * n
        self._plans_executed: list[int] = [0] * n
        self._skipped_goals: list[int] = [0] * n
        self._problems: dict[str, Problem] = {}
        # one chain per reset state: each slot's compatible states, in order; a state that
        # matches several slots (equal profile values) is played once in total
        self._unplayed: dict[int, deque[int]] = {
            slot: deque(states) for slot, states in self._rs_by_slot.items()
        }
        self._played: set[int] = set()
        self._can_stack, self._can_unstack = followup_actions_enabled(play.allowed_actions)
        self.chains: list[dict[str, Any]] = []
        self.executed_skills: Counter[str] = Counter()
        self.plans_submitted = 0
        self.plans_installed = 0
        self.plans_failed = 0
        self.goals_reached = 0
        self.goals_missed = 0
        self.plans_abandoned = 0
        self.abandon_followup_goals = 0
        # goal-policy TERMINATE (a DFA strategy accepted or rejected): env -> its final state
        self._terminated: dict[int, str] = {}
        self.policy_accepts = 0
        self.policy_rejects = 0
        self.policy_restarts = 0
        if hasattr(goal_policy, "set_scene_accessors"):
            goal_policy.set_scene_accessors(pose_of=self._pose_of, table="workspace_table")

    # ------------------------------------------------------------------ run
    def run(self, episodes: int = 0, max_ticks: int | None = None) -> list[Any]:  # noqa: ARG002
        """Play until every env retired (or ``max_ticks``); in-flight episodes are then cut."""
        self.sim.apply_profile(self.cfg.profile)
        t_start = time.time()
        while any(m != "done" for m in self.mode):
            if max_ticks is not None and self.ticks >= max_ticks:
                break
            self.tick()
        cut = [e for e in range(self.n) if self.mode[e] == "episode"]
        for e in cut:
            self.mode[e] = "done"
            self.ctx.params[e]["_outcome"] = (False, "cut: tick limit")
            self._close_episode(e)
        if cut:
            log.info("play: cut %d in-flight episodes at the tick limit", len(cut))
        self._ff_threads.shutdown(wait=False, cancel_futures=True)
        wall = time.time() - t_start
        log.info(
            "play: %d episodes, %d chains in %d ticks, %.1f s (%.1f ms/tick), timing %s",
            len(self.records),
            len(self.chains),
            self.ticks,
            wall,
            1000 * wall / max(1, self.ticks),
            {k: round(v, 1) for k, v in self.timing.items()},
        )
        return self.records

    # ------------------------------------------------------------ scheduling
    def _assign_idle(self) -> None:
        """Advance every env that is not executing: reset -> settle -> goal -> FF -> episode."""
        self._check_abandon()
        self._start_chains([e for e in range(self.n) if self.mode[e] == "reset"])
        settling = [e for e in range(self.n) if self.mode[e] == "settle"]
        if settling:
            at_rest = self._at_rest()
            ready: list[int] = []
            for e in settling:
                if self._wait[e] > 0:
                    self._wait[e] -= 1
                elif self._settle_budget[e] > 0 and not bool(at_rest[e]):
                    self._settle_budget[e] -= 1
                else:
                    ready.append(e)
            if ready:
                self._request_goals(ready)
        if self._pending:
            self._poll_ff()

    def _start_chains(self, envs: list[int]) -> None:
        """Reset ``envs`` to the next reset state of their profile; retire those with none."""
        if not envs:
            return
        starts: list[tuple[int, int]] = []
        for e in envs:
            rid = self._chain_reset_state(e)
            if rid is None:
                self.mode[e] = "done"
                continue
            starts.append((e, rid))
        if not starts:
            return
        ids = torch.tensor([e for e, _ in starts], dtype=torch.long, device=self.device)
        inst = torch.tensor([r for _, r in starts], dtype=torch.long, device=self.device)
        self.sim.reset_to_instances(ids, inst)
        self.sim.command_gripper(
            torch.zeros(len(starts), dtype=torch.bool, device=self.device), ids
        )
        self.sim.flush()
        self.sim.forward()
        self.geo.detach(ids)
        self.ctx.held[ids] = -1
        for e, rid in starts:
            self._chain_rid[e] = rid
            self._chain_plans[e] = 0
            self._chain_successes[e] = 0
            self._plan_failures[e] = 0
            self._no_goal[e] = 0
            self._held_after_abandon[e] = None
            self.goal_policy.reset(e)
            # fresh reset: blocks were teleported with zero velocity, so "at rest" says
            # nothing about their contacts yet; settle the full fixed window
            self.mode[e] = "settle"
            self._wait[e] = self.play.pre_episode_ticks
            self._settle_budget[e] = 0

    def _chain_reset_state(self, e: int) -> int | None:
        """The reset state env ``e``'s next chain starts from (``None``: retire the env)."""
        slot = self._env_slot[e]
        if self.play.one_chain_per_reset_state:
            queue = self._unplayed.get(slot) or deque()
            while queue:
                rid = queue.popleft()
                if rid not in self._played:
                    self._played.add(rid)
                    return rid
            return None
        if self._chain_rid[e] is not None or self.mode[e] == "done":
            return None  # one chain per env for the whole run
        states = self._rs_by_slot.get(slot) or []
        if not states:
            log.warning("env %d: no reset state was sampled under its scene profile", e)
            return None
        return self.rng.choice(states)

    def _park(self, e: int, *, fresh_goal_wait: bool = False) -> None:
        """Between plans: release the gripper (unless a block is held after an abandon) and
        settle adaptively before the next goal."""
        if self._held_after_abandon[e] is None:
            ids = torch.tensor([e], device=self.device)
            self.sim.command_gripper(torch.zeros(1, dtype=torch.bool, device=self.device), ids)
            self.sim.flush()
        self.mode[e] = "settle"
        if fresh_goal_wait:
            self._wait[e] = self.play.poll_ticks
            self._settle_budget[e] = 0
            return
        min_ticks = min(self.play.settle_min_ticks, self.play.settle_ticks)
        self._wait[e] = min_ticks
        self._settle_budget[e] = self.play.settle_ticks - min_ticks

    def _at_rest(self) -> Tensor:
        """``[N]`` True where every block and the gripper are (nearly) still."""
        vel = torch.stack([self.sim.object_velocity(b) for b in self.sim.block_names], dim=1)
        lin = vel[..., :3].norm(dim=-1).amax(dim=1)
        ang = vel[..., 3:6].norm(dim=-1).amax(dim=1)
        grip = self.sim.art.data.joint_vel[:, self.sim.gripper_id].abs()
        return (
            (lin <= self.play.settle_lin_vel)
            & (ang <= self.play.settle_ang_vel)
            & (grip <= self.play.settle_gripper_vel)
        )

    def _budget_reached(self, e: int) -> bool:
        play = self.play
        if play.max_plans_per_env is not None and self._chain_plans[e] >= play.max_plans_per_env:
            return True
        if (
            play.min_successful_plans_per_env is not None
            and self._chain_successes[e] >= play.min_successful_plans_per_env
        ):
            return True
        if play.max_plans_total is not None and self.plans_installed >= play.max_plans_total:
            return True
        return self._plan_failures[e] >= play.max_plan_failures_per_env

    def _end_chain(self, e: int) -> None:
        """Record env ``e``'s chain; start the next one (one chain per reset state) or retire."""
        rid = self._chain_rid[e]
        if rid is not None:
            need = self.play.min_successful_plans_per_env
            self.chains.append(
                {
                    "reset_state": rid,
                    "env_idx": e,
                    "plans": self._chain_plans[e],
                    "successes": self._chain_successes[e],
                    "reached_min_success": need is None or self._chain_successes[e] >= need,
                }
            )
        total = self.play.max_plans_total
        more = self.play.one_chain_per_reset_state and (
            total is None or self.plans_installed < total
        )
        self.mode[e] = "reset" if more else "done"

    # ------------------------------------------------------------ goals / FF
    def _pose_of(self, e: int, block: str) -> list[float] | None:
        if block not in self.ctx.block_index:
            return None
        return [float(v) for v in self.sim.object_pose(block)[e].tolist()]

    def _request_goals(self, envs: list[int]) -> None:
        """Read the state, ask the goal policy (or the abandon follow-up), queue FF."""
        poses = self.ctx.block_poses()
        table = self.ctx.table_pose()
        for e in envs:
            if (
                self.play.max_plans_total is not None
                and self.plans_installed >= self.play.max_plans_total
            ):
                self._end_chain(e)
                self.mode[e] = "done"
                continue
            rid = self._chain_rid[e]
            assert rid is not None
            prob = self.problem_for(rid)
            state, facts = read_symbolic_state(
                prob, self.ctx.block_index, poses[e], self.ctx.sizes[e], table[e]
            )
            goal, reason = self._next_goal(e, state, poses, table)
            if e in self._terminated:
                self._on_policy_terminate(e, self._terminated.pop(e))
                continue
            if goal is None:
                self._skipped_goals[e] += 1
                self._no_goal[e] += 1
                if self._no_goal[e] >= self.play.max_plan_failures_per_env:
                    log.info(
                        "env %d: no goal for %d tries (%s); ending its chain",
                        e,
                        self._no_goal[e],
                        reason,
                    )
                    self._end_chain(e)
                else:
                    self._park(e, fresh_goal_wait=True)
                continue
            self._no_goal[e] = 0
            static = [f for f in prob.init if f.pred not in DYNAMIC_PREDICATES]
            ids = torch.tensor([e], device=self.device)
            derived = self.derived_fn(prob, ids)[0]
            pid = f"play_env{e}_plan{self._plans_executed[e]}"
            goal_facts = parse_facts(goal.render())
            episode_problem = Problem(
                name=pid,
                domain=prob.domain,
                objects=dict(prob.objects),
                init=facts + static + derived,
                goal=goal_facts,
            )
            text = render_problem(episode_problem)
            fut = self._ff_threads.submit(
                run_ff, self.ff_domain, text, None, self.play.ff_timeout_s
            )
            self._pending[e] = _Pending(fut, episode_problem, goal, reason)
            self.plans_submitted += 1
            self.mode[e] = "plan"
            log.info("env %d: goal %s (%s) | %s", e, goal.text, reason, state.describe())

    def _next_goal(
        self, e: int, state: SymbolicState, poses: Tensor, table: Tensor
    ) -> tuple[ProposedGoal | None, str]:
        held = self._held_after_abandon[e]
        if held is not None:
            # consumed: a failed follow-up parks normally (gripper opened) next time
            self._held_after_abandon[e] = None
            hj = self.ctx.block_index[held]
            block_xy = {
                b: (
                    float(poses[e, self.ctx.block_index[b], 0]),
                    float(poses[e, self.ctx.block_index[b], 1]),
                )
                for b in state.blocks
                if b != held
            }
            avoid = blocks_below((float(poses[e, hj, 0]), float(poses[e, hj, 1])), block_xy)
            goal = followup_pnp_goal(
                state,
                held,
                self.rng,
                can_stack=self._can_stack,
                can_unstack=self._can_unstack,
                max_stack_height=self.play.max_stack_height,
                avoid=avoid,
            )
            if goal is not None:
                self.abandon_followup_goals += 1
                return goal, "abandon-followup"
            log.info("env %d: no pick-and-place goal without %s after the abandon", e, held)

        def holds(name: str, args: list[str]) -> bool:
            if name not in EVALUATED_PREDICATES or any(
                a not in self.ctx.block_index and a != state.table for a in args
            ):
                return False
            ok = goal_facts_satisfied(
                [Fact(name, tuple(args))],
                self.ctx.block_index,
                poses[e : e + 1],
                self.ctx.sizes[e : e + 1],
                table[e : e + 1],
            )
            return bool(ok[0][0])

        decision = self.goal_policy.next_goal(e, state, holds)
        if decision.kind is Decision.GOAL and decision.goal is not None:
            return decision.goal, decision.reason
        if decision.kind is Decision.TERMINATE:
            self._terminated[e] = decision.reason or "ACCEPT"
        return None, decision.reason or decision.kind.name

    def _on_policy_terminate(self, e: int, reason: str) -> None:
        """The goal policy finished this env's run (a DFA reached ACCEPT / REJECT).

        Play keeps collecting: the chain is recorded and the env restarts from a fresh reset
        state with the policy reset -- the next unplayed one with ``one_chain_per_reset_state``,
        else a random one of its profile -- unless a plan budget retires it (the TAMP play
        loop's ``_on_policy_terminate``).
        """
        if reason == "REJECT":
            self.policy_rejects += 1
        else:
            self.policy_accepts += 1
        log.info("env %d: goal policy terminated (%s)", e, reason)
        budget = self._budget_reached(e)
        self._end_chain(e)  # one chain per reset state: moves on to the next state
        if self.play.one_chain_per_reset_state or budget or self.mode[e] == "reset":
            if self.mode[e] == "reset":
                self.policy_restarts += 1
            return
        total = self.play.max_plans_total
        if total is None or self.plans_installed < total:
            self._chain_rid[e] = None  # lets _chain_reset_state draw a fresh random state
            self.mode[e] = "reset"
            self.policy_restarts += 1

    def _poll_ff(self) -> None:
        for e, pend in list(self._pending.items()):
            if not pend.future.done():
                continue
            del self._pending[e]
            try:
                res: FFResult = pend.future.result()
            except Exception as exc:  # noqa: BLE001 - FF itself crashed
                res = FFResult(False, [], 0.0, repr(exc))
            prog: Program | None = None
            error = res.error
            if res.ok:
                try:
                    prog = program_from_plan(
                        res.actions, pend.problem.name, int(self._chain_rid[e]), res.states
                    )
                except UnsupportedAction as exc:
                    error = f"plan uses unsupported action {exc}"
                if prog is not None and len(prog) == 0:
                    prog, error = None, "empty plan (goal already holds)"
            if prog is None:
                self.plans_failed += 1
                self._plan_failures[e] += 1
                log.info(
                    "env %d: no plan for %s (%s) - %d/%d",
                    e,
                    pend.goal.text,
                    error[:160],
                    self._plan_failures[e],
                    self.play.max_plan_failures_per_env,
                )
                if self._budget_reached(e):
                    self._end_chain(e)
                else:
                    self._park(e, fresh_goal_wait=True)
                continue
            self._install_episode(e, prog, pend)

    def _install_episode(self, e: int, prog: Program, pend: _Pending) -> None:
        st = self.states[e]
        rid = int(self._chain_rid[e])
        pid = pend.problem.name
        self._problems[pid] = pend.problem
        self.ctx.problem_goals[pid] = list(pend.problem.goal)
        self._episode_problem[e] = pid
        self._episode_goal[e] = pend.goal.text
        st.assign(Program(list(prog.skills), pid, rid), self.next_episode)
        self.next_episode += 1
        self.episode_ticks[e] = 0
        self.phase_idx[e] = 0
        self._begun.discard(e)
        self._wait_since.pop(e, None)
        self.ctx.params[e] = {}
        self.rec.start_episode(e, st.episode_id, rid, prog.skills[0].hl_state if prog.skills else ())
        self.rec.set_goal(e, pend.goal.text)
        skills = program_skills(prog)
        self.executed_skills.update(skills)
        if hasattr(self.goal_policy, "on_plan_skills"):
            self.goal_policy.on_plan_skills(e, skills)
        self._abandon_at[e] = None
        play = self.play
        if (
            play.abandon_plan_prob > 0
            and any(call.name in PLACE_SKILLS for call in prog.skills)
            and self.rng.random() < play.abandon_plan_prob
        ):
            self._abandon_at[e] = self.rng.uniform(*play.abandon_fraction_range)
        self.plans_installed += 1
        self._plans_executed[e] += 1
        self._chain_plans[e] += 1
        self._plan_failures[e] = 0
        self.mode[e] = "episode"
        st.phase = Phase.READY
        log.info(
            "env %d: plan %d for %s: %s%s",
            e,
            self._plans_executed[e] - 1,
            pend.goal.text,
            " ; ".join(str(s) for s in prog.skills),
            f" (abandon at {self._abandon_at[e]:.0%} of the transport)"
            if self._abandon_at[e] is not None
            else "",
        )

    # ------------------------------------------------------------ abandon
    def _check_abandon(self) -> None:
        """Cut armed plans once their transport reached the drawn fraction."""
        for e in range(self.n):
            frac = self._abandon_at[e]
            if frac is None or self.mode[e] != "episode":
                continue
            if not bool(self.table.active[e]) or self.table.labels[e] != TRANSPORT_LABEL:
                continue
            length = int(self.table.length[e])
            cursor = int(self.table.cursor[e])
            if cursor < frac * length:
                continue
            self._abandon_at[e] = None
            held = int(self.ctx.held[e])
            # abandon only when the chain gets another plan to pick a new block with
            if held < 0 or self._budget_reached(e):
                continue
            name = self.ctx.block_names[held]
            self.table.cancel([e])
            self.ctx.params[e]["_abandon"] = name
            reason = (
                f"abandoned at {cursor}/{length} transport rows holding {name} "
                f"(goal: {self._episode_goal.get(e, '')})"
            )
            log.info("env %d: %s; next goal picks another block", e, reason)
            self._end_episode(e, True, reason)

    # ------------------------------------------------------------ episodes
    def _close_episode(self, e: int) -> None:
        params = self.ctx.params[e]
        reached, reason = params.get("_outcome", (False, "unknown"))
        abandoned = params.pop("_abandon", None)
        if abandoned is not None:
            self.plans_abandoned += 1
            self.goal_policy.on_episode_result(e, True)
            kept = True
        else:
            self.goal_policy.on_episode_result(e, bool(reached))
            if reached:
                self.goals_reached += 1
                self._chain_successes[e] += 1
            else:
                self.goals_missed += 1
            kept = bool(reached) or not self.play.missed_goals_to_failure
        # recorded success decides blocks_success vs blocks_failure; a missed goal's reason
        # goes to the error column, an abandoned episode is saved as a plain success (as
        # the TAMP play orchestrator does)
        params["_outcome"] = (kept, "" if reached or abandoned is not None else reason)
        super()._close_episode(e)
        pid = self._episode_problem.pop(e, None)
        if pid is not None:
            self._problems.pop(pid, None)
            self.ctx.problem_goals.pop(pid, None)
        self._episode_goal.pop(e, None)
        self._abandon_at[e] = None
        if self.mode[e] != "episode":
            return  # cut at the end of the run
        self._held_after_abandon[e] = abandoned
        if self._budget_reached(e):
            self._end_chain(e)
            if self.mode[e] == "reset" and abandoned is not None:
                self._held_after_abandon[e] = None
            return
        self._park(e)

    def _replan_live(self, e: int) -> Program | None:
        """FF from the live state toward the episode's own goal (after a skill's retries)."""
        st = self.states[e]
        if st.program is None or st.program.problem_id not in self._problems:
            return None
        prob = self._problems[st.program.problem_id]
        state, facts = read_symbolic_state(
            prob,
            self.ctx.block_index,
            self.ctx.block_poses()[e],
            self.ctx.sizes[e],
            self.ctx.table_pose()[e],
        )
        del state
        ids = torch.tensor([e], device=self.device)
        static = [
            f
            for f in self.problem_for(st.program.reset_state_id).init
            if f.pred not in DYNAMIC_PREDICATES
        ]
        live = Problem(
            prob.name,
            prob.domain,
            prob.objects,
            facts + static + self.derived_fn(prob, ids)[0],
            prob.goal,
        )
        res = run_ff(self.ff_domain, render_problem(live), timeout_s=self.play.ff_timeout_s)
        if not res.ok:
            log.info("env %d: FF re-plan failed: %s", e, res.error[:200])
            return None
        try:
            return program_from_plan(
                res.actions, prob.name, st.program.reset_state_id, res.states
            )
        except UnsupportedAction as exc:
            log.info("env %d: FF re-plan uses unsupported action %s", e, exc)
            return None

    # ------------------------------------------------------------ summary
    def summary(self) -> dict[str, Any]:
        """Run summary with the ``play_data`` block the play pipeline aggregates (as the TAMP play loop wrote)."""
        total_skills = sum(self.executed_skills.values())
        return {
            "total": len(self.records),
            "play_data": {
                "plans_submitted": self.plans_submitted,
                "plans_installed": self.plans_installed,
                "plans_failed": self.plans_failed,
                "goals_reached": self.goals_reached,
                "goals_missed": self.goals_missed,
                "plans_abandoned": self.plans_abandoned,
                "abandon_followup_goals": self.abandon_followup_goals,
                "policy_accepts": self.policy_accepts,
                "policy_rejects": self.policy_rejects,
                "policy_restarts": self.policy_restarts,
                "episodes_kept": sum(r.success for r in self.records),
                "plans_executed_per_env": list(self._plans_executed),
                "planning_failures_per_env": list(self._plan_failures),
                "goals_skipped_per_env": list(self._skipped_goals),
                "executed_skills": dict(sorted(self.executed_skills.items())),
                "executed_skill_shares": {
                    k: round(v / total_skills, 4) for k, v in sorted(self.executed_skills.items())
                }
                if total_skills
                else {},
                "chains": list(self.chains),
                "chains_below_min_success": sum(not c["reached_min_success"] for c in self.chains),
                "unplayed_reset_states": sorted(
                    {rid for states in self._rs_by_slot.values() for rid in states} - self._played
                )
                if self.play.one_chain_per_reset_state
                else [],
            },
        }
