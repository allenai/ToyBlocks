"""The lockstep engine: programs -> skills -> batched geometry -> segment tables -> sim.

Per tick: gather one target row per env, write all targets, step the sim, record.
At segment boundaries (only), envs are grouped by (skill, phase) and their next
segments are produced with batched IK / planning. At skill boundaries the effect is
verified on the live state; failing envs restore the pre-skill snapshot and retry with
freshly sampled parameters, or give up after ``max_attempts``.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from blocksuite.utils.reset_states_io import describe_spawn_profile
from blocksuite.vecgen.geometry import Geometry
from blocksuite.vecgen.program import EnvProgramState, Phase, Program
from blocksuite.vecgen.segments import Segment, SegmentTable, interpolate_rows
from blocksuite.vecgen.sim import VecSim
from blocksuite.vecgen.skills.base import PhaseRequest, Skill
from blocksuite.vecgen.symbolic.predicates import goal_facts_satisfied

log = logging.getLogger("vecgen")


@dataclass
class EngineConfig:
    max_attempts: int = 3
    reset_settle_ticks: int = 4
    done_hold_ticks: int = 25  # the orchestrator's DoneMarker (100 physics steps)
    segment_capacity: int = 1500
    target_ff: bool = True
    # cancel the arm's joint friction in motion-planned commands so it reaches the planned
    # grasp / place poses (VecSim.friction_comp_offset); RL skill policies are left as trained
    friction_comp: bool = True
    profile: str = "medium"
    verify_settle_ticks: int = 3
    max_episode_ticks: int = 6000
    max_replans: int = 2  # FF re-plans from the live state after a skill's retries run out
    async_workers: int = 4  # worker threads for skills' async_phases (0 = build on the sim thread)
    plan_gather_ticks: int = (
        8  # how long a plan request may wait for others to batch with (0 = never)
    )
    plan_gather_min: int = 4  # rows that end the wait early
    # keep every env busy until the episode target is met; episodes still in flight at
    # that point are closed as failures ("cut: episode target reached") instead of being
    # drained by a shrinking set of live envs
    cut_tail: bool = False
    # start every reset state at most once: an env whose scene profile has no unstarted
    # state left idles instead of cycling them (multi-problem packages, whose episode
    # target is one episode per reset state)
    reset_states_once: bool = False
    # solve transport plans on a planner thread while the sim keeps stepping the other envs
    # (the pool call is one CUDA-graph replay, so the thread releases the GIL while it runs)
    async_plans: bool = True
    # run FF for newly assigned envs on a thread while they settle (the tick never waits)
    async_programs: bool = True


@dataclass
class EpisodeRecord:
    env_id: int
    episode_id: int
    reset_state_id: int
    success: bool
    reason: str
    ticks: int
    skills: int


@dataclass
class EngineContext:
    """Everything skills may read or update between ticks."""

    sim: VecSim
    geo: Geometry
    device: torch.device
    block_names: list[str]
    block_index: dict[str, int]
    sizes: Tensor  # [N, K, 3]
    held: Tensor  # [N] block index held, -1 for none
    params: list[dict[str, Any]]
    rng: torch.Generator
    problem_goal: list[Any] = field(default_factory=list)
    # goal facts per problem id (problem streaming); ``problem_goal`` is the fallback
    problem_goals: dict[str, list[Any]] = field(default_factory=dict)
    # per-env program state (read-only for skills: e.g. a place looks at what comes next)
    states: list[Any] = field(default_factory=list)

    def block_poses(self) -> Tensor:
        return self.sim.object_poses(self.block_names)

    def table_pose(self) -> Tensor:
        return self.sim.object_pose("workspace_table")


class Recorder:
    """Optional recording hooks; the default implementation records nothing."""

    def start_episode(
        self, env_id: int, episode_id: int, reset_state_id: int, hl_state: tuple[str, ...] = ()
    ) -> None: ...

    def set_goal(self, env_id: int, goal_text: str) -> None:
        """The goal of the episode just started (play data picks one per episode)."""

    def symbolic(
        self, env_id: int, hl: str, hl_args: tuple[str, ...], ll: str, hl_state: tuple[str, ...] = ()
    ) -> None:
        """A skill starts: its PDDL action and args, the LL label, and ``hl_state``, the
        symbolic state that action starts from (empty when unknown: the last one stands)."""

    def step(self, env_ids: Tensor) -> None: ...

    def frame_mark(self, env_id: int) -> int:
        """A position in env ``env_id``'s recorded frames, for :meth:`drop_since`."""
        return 0

    def drop_since(self, env_id: int, mark: int) -> None:
        """Leave the frames recorded since ``mark`` out of the episode (an undone attempt)."""

    def end_episode(self, env_id: int, success: bool, error: str) -> None: ...

    def finalize(self) -> None: ...


def _slice_state(state: Any, i: int) -> Any:
    """Row ``i`` of a batched state dict (tensors keep a leading batch dim of 1)."""
    if isinstance(state, dict):
        return {k: _slice_state(v, i) for k, v in state.items()}
    if torch.is_tensor(state):
        return state[i : i + 1].clone()
    return state


class Engine:
    def __init__(
        self,
        sim: VecSim,
        geo: Geometry,
        skills: dict[str, Skill],
        program_fn: Callable[[int, int], Program | None],
        cfg: EngineConfig,
        recorder: Recorder | None = None,
        seed: int = 0,
        replan_fn: Callable[[int], Program | None] | None = None,
        programs_fn: Callable[[list[int], list[int]], list[Program | None]] | None = None,
    ) -> None:
        self.sim = sim
        self.replan_fn = replan_fn
        self.programs_fn = programs_fn
        self.geo = geo
        self.skills = skills
        self.program_fn = program_fn
        self.cfg = cfg
        self.rec = recorder or Recorder()
        n = sim.num_envs
        dev = sim.device
        self.n = n
        self.device = dev
        self.table = SegmentTable(n, geo.dof, cfg.segment_capacity, dev)
        self.states = [EnvProgramState(e) for e in range(n)]
        self.phase_idx = [0] * n
        self.snapshots: dict[int, dict[str, Any]] = {}
        self.episode_ticks = torch.zeros(n, dtype=torch.long, device=dev)
        self.next_episode = 0
        self.num_reset_states = sim.num_reset_states()
        # reset states per env spawn slot (matched by profile value), handed out in order:
        # every reset state runs once before any repeats, also when envs finish out of
        # step (streamed problems)
        self._env_slot: list[int] = sim.env_slots()
        self._rs_by_slot: dict[int, list[int]] = sim.reset_states_by_slot()
        self._rs_queue: dict[int, deque[int]] = {
            slot: deque(q) for slot, q in self._rs_by_slot.items()
        }
        # reset states already started (reset_states_once): a state whose profile values
        # match several slots sits in each of their queues
        self._started: set[int] = set()
        self.records: list[EpisodeRecord] = []
        self.ticks = 0
        self.timing: dict[str, float] = defaultdict(float)
        rng = torch.Generator(device="cpu")
        rng.manual_seed(seed)
        self.ctx = EngineContext(
            sim=sim,
            geo=geo,
            device=dev,
            block_names=sim.block_names,
            block_index={name: j for j, name in enumerate(sim.block_names)},
            sizes=geo.sizes,
            held=torch.full((n,), -1, dtype=torch.long, device=dev),
            params=[{} for _ in range(n)],
            states=self.states,
            rng=rng,
        )
        self._ids = torch.arange(n, device=dev)
        self._pending_boundary: list[int] = []
        self._begun: set[int] = set()
        self._wait_since: dict[int, int] = {}
        self._workers = ThreadPoolExecutor(
            max_workers=max(1, cfg.async_workers), thread_name_prefix="vecgen-build"
        )
        self._async: list[tuple[Future, Skill, str, list[int]]] = []
        # envs driven by a closed-loop policy instead of the segment table:
        # env -> (runner, settle ticks left, profile to restore or None)
        self._policy: dict[int, tuple[Any, int, str | None]] = {}
        # transport plans in flight: (future, plan rows, request owners)
        self._planner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vecgen-plan")
        self._planner.submit(geo.mark_main_thread).result()
        self._ff_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vecgen-ff")
        self._programs: list[tuple[Future, list[int], list[int]]] = []
        self._program_pending: set[int] = set()  # envs whose FF result is still due
        self._settled: set[int] = set()  # pending envs whose reset settle already ended
        self._plans: list[
            tuple[
                Future,
                list[tuple[int, int, PhaseRequest, Skill, str]],
                dict[int, tuple[PhaseRequest, Skill]],
            ]
        ] = []

    # ------------------------------------------------------------------ run
    def run(self, episodes: int, max_ticks: int | None = None) -> list[EpisodeRecord]:
        """Run until ``episodes`` episodes have ended (or ``max_ticks``)."""
        self.target_episodes = episodes
        self.sim.apply_profile(self.cfg.profile)
        t_start = time.time()
        while len(self.records) < episodes:
            if max_ticks is not None and self.ticks >= max_ticks:
                break
            if (
                self.cfg.reset_states_once
                and self._live_episodes() == 0
                and not any(self._unstarted(s) for s in self._rs_queue)
            ):
                log.info("every reset state has run once; stopping at %d episodes", len(self.records))
                break
            self.tick()
        if self.cfg.cut_tail:
            cut = [e for e in range(self.n) if self.states[e].phase is not Phase.IDLE]
            for e in cut:
                if "_outcome" not in self.ctx.params[e]:
                    self.ctx.params[e]["_outcome"] = (False, "cut: episode target reached")
                self._close_episode(e)
            log.info("cut %d in-flight episodes at the episode target", len(cut))
        wall = time.time() - t_start
        log.info(
            "engine: %d episodes in %d ticks, %.1f s (%.1f ms/tick), timing %s",
            len(self.records),
            self.ticks,
            wall,
            1000 * wall / max(1, self.ticks),
            {k: round(v, 1) for k, v in self.timing.items()},
        )
        return self.records

    # ----------------------------------------------------------------- tick
    def tick(self) -> None:
        t0 = time.perf_counter()
        pos, vel, acc, commanded, finished = self.table.gather(
            self.sim.joint_pos() if self.table.needs_joint_pos() else None
        )
        ids = self._ids[commanded]
        if ids.numel():
            ff = self.table.ff[ids]
            fc = self.cfg.friction_comp
            if self.cfg.target_ff and bool(ff.any()):
                self.sim.command_arm(pos[ids][ff], ids[ff], vel[ids][ff], acc[ids][ff], True, friction_comp=fc)
                if bool((~ff).any()):
                    self.sim.command_arm(pos[ids][~ff], ids[~ff], target_ff=False, friction_comp=fc)
            else:
                self.sim.command_arm(pos[ids], ids, target_ff=False, friction_comp=fc)
            self.sim.flush()
        policy_ids = self._step_policies()
        t1 = time.perf_counter()
        self.sim.step()
        t2 = time.perf_counter()
        self.ticks += 1
        if policy_ids.numel():
            ids = torch.cat([ids, policy_ids])
            commanded = commanded.clone()
            commanded[policy_ids] = True
        self.episode_ticks += commanded.long()
        self.rec.step(ids)
        t3 = time.perf_counter()
        finished = finished.clone()
        for e in self._policy_done():
            finished[e] = True
        done_ids = self._ids[finished]
        if done_ids.numel():
            self._on_segments_finished(done_ids)
        self._assign_idle()
        self._dispatch_ready()
        t4 = time.perf_counter()
        self.timing["command"] += t1 - t0
        self.timing["sim"] += t2 - t1
        self.timing["record"] += t3 - t2
        self.timing["dispatch"] += t4 - t3

    # ---------------------------------------------------------- policies
    def _step_policies(self) -> Tensor:
        """Command one closed-loop step for every env in a ``policy`` phase.

        The first ``settle`` ticks after activation hold the arm still (gripper opening
        and gains settle, as the old orchestrator's stiffness/gripper settle states);
        then each tick the runner returns absolute arm targets from the policy's
        action. Returns the ids of all policy-driven envs (stepped or settling).
        """
        if not self._policy:
            return torch.zeros(0, dtype=torch.long, device=self.device)
        by_runner: dict[int, tuple[Any, list[int]]] = {}
        for e, (runner, settle, profile) in list(self._policy.items()):
            if settle > 0:
                self._policy[e] = (runner, settle - 1, profile)
                continue
            by_runner.setdefault(id(runner), (runner, []))[1].append(e)
        t0 = time.perf_counter()
        for runner, envs in by_runner.values():
            ids = torch.tensor(envs, dtype=torch.long, device=self.device)
            targets = runner.step(envs)
            self.sim.command_arm(targets, ids, target_ff=False)
        if by_runner:
            self.sim.flush()
        self.timing["policy"] += time.perf_counter() - t0
        return torch.tensor(sorted(self._policy), dtype=torch.long, device=self.device)

    def _policy_done(self) -> list[int]:
        """Envs whose policy reached its goal or step cap this tick; they leave the policy."""
        if not self._policy:
            return []
        by_runner: dict[int, tuple[Any, list[int]]] = {}
        for e, (runner, settle, _profile) in self._policy.items():
            if settle <= 0 and runner.steps.get(e, 0) > 0:
                by_runner.setdefault(id(runner), (runner, []))[1].append(e)
        done: list[int] = []
        for runner, envs in by_runner.values():
            flags = runner.done(envs)
            finished = [e for e, f in zip(envs, flags.tolist(), strict=True) if f]
            if not finished:
                continue
            runner.stop(finished)
            done.extend(finished)
        for e in done:
            _runner, _settle, profile = self._policy.pop(e)
            if profile is not None:
                self.sim.apply_profile(self.cfg.profile, torch.tensor([e], device=self.device))
        return done

    # -------------------------------------------------------- boundaries
    def _on_segments_finished(self, env_ids: Tensor) -> None:
        """Advance phases; verify at skill end."""
        by_skill: dict[str, list[int]] = defaultdict(list)
        for e in env_ids.tolist():
            st = self.states[e]
            if "_special" in self.ctx.params[e]:  # post-reset settle / done hold
                self._finish_special(e)
            elif st.phase is Phase.EXECUTING and st.current is not None:
                by_skill[st.current.name].append(e)
        for name, envs in by_skill.items():
            skill = self.skills[name]
            ids = torch.tensor(envs, dtype=torch.long, device=self.device)
            phase = skill.phases[self.phase_idx[envs[0]]] if envs else ""
            # all envs in a group share the phase index only if they entered together;
            # regroup by phase index to be safe
            by_phase: dict[int, list[int]] = defaultdict(list)
            for e in envs:
                by_phase[self.phase_idx[e]].append(e)
            for pidx, envs_p in by_phase.items():
                ids = torch.tensor(envs_p, dtype=torch.long, device=self.device)
                phase = skill.phases[pidx]
                skill.after_phase(self.ctx, phase, ids)
                last = pidx == len(skill.phases) - 1
                for e in envs_p:
                    self.phase_idx[e] = pidx + 1
                    self.states[e].phase = Phase.VERIFY if last else Phase.READY
                if last:
                    self._pending_boundary.extend(envs_p)
        if self._pending_boundary:
            self._verify_pending()

    def _skip_noop_phases(self) -> None:
        """Move READY envs past the phases their skill reports as no-ops (``Skill.skips``).

        No segment is installed for a skipped phase, so it costs no tick and records no frame.
        An env skipped past its skill's last phase is verified right away.
        """
        for e in range(self.n):
            st = self.states[e]
            if st.phase is not Phase.READY or st.current is None or e not in self._begun:
                continue
            skill = self.skills.get(st.current.name)
            if skill is None:
                continue
            n_phases = len(skill.phases)
            while self.phase_idx[e] < n_phases and skill.skips(
                self.ctx, skill.phases[self.phase_idx[e]], e
            ):
                self.phase_idx[e] += 1
            if self.phase_idx[e] == n_phases:
                st.phase = Phase.VERIFY
                self._pending_boundary.append(e)
        if self._pending_boundary:
            self._verify_pending()

    def _verify_pending(self) -> None:
        envs = self._pending_boundary
        self._pending_boundary = []
        by_skill: dict[str, list[int]] = defaultdict(list)
        for e in envs:
            st = self.states[e]
            if st.current is not None:
                by_skill[st.current.name].append(e)
        for name, envs_s in by_skill.items():
            ids = torch.tensor(envs_s, dtype=torch.long, device=self.device)
            ok = self.skills[name].verify(self.ctx, ids)
            for i, e in enumerate(envs_s):
                st = self.states[e]
                if bool(ok[i]):
                    st.advance()
                    self.phase_idx[e] = 0
                    self._begun.discard(e)
                    if st.phase is Phase.SUCCESS:
                        unmet = self._unmet_goals(e)
                        if unmet:
                            self._end_episode(e, False, f"goal not satisfied: {unmet}")
                        else:
                            self._end_episode(e, True, "")
                else:
                    self._retry_or_fail(e, f"{name} effect check failed")

    def _unmet_goals(self, e: int) -> list[str]:
        """Goal facts of the problem that do not hold in env ``e`` (empty when no goal is set)."""
        st = self.states[e]
        goal = self.ctx.problem_goal
        if st.program is not None and st.program.problem_id in self.ctx.problem_goals:
            goal = self.ctx.problem_goals[st.program.problem_id]
        if not goal:
            return []
        ids = torch.tensor([e], device=self.device)
        ok = goal_facts_satisfied(
            goal,
            self.ctx.block_index,
            self.ctx.block_poses()[ids],
            self.ctx.sizes[ids],
            self.ctx.table_pose()[ids],
        )
        return [str(f) for f, good in zip(goal, ok, strict=True) if not bool(good[0])]

    def _retry_or_fail(self, e: int, reason: str) -> None:
        st = self.states[e]
        ids = torch.tensor([e], device=self.device)
        skill = self.skills.get(st.current.name) if st.current is not None else None
        from_live = bool(skill is not None and skill.retry_from_live)
        if st.retry(self.cfg.max_attempts, reason):
            snap = self.snapshots.get(e)
            if snap is not None and not from_live:
                state, held, spheres, mark = snap
                # the undone attempt never happened as far as the dataset is concerned: the
                # restored state continues from the frame recorded before it
                self.rec.drop_since(e, mark)
                self.sim.restore(state, ids)
                closed = torch.tensor([held >= 0], dtype=torch.bool, device=self.device)
                self.sim.command_gripper(closed, ids)
                self.sim.flush()
                self.ctx.held[e] = held
                self.geo.set_attachment_of(e, spheres)
            self.phase_idx[e] = 0
            self._begun.discard(e)
            log.info("env %d: retry %d of %s (%s)", e, st.attempts, st.current, reason)
        elif self._replan(e, reason):
            return
        else:
            if int(self.ctx.held[e]) >= 0:
                self.geo.detach(ids)
                self.ctx.held[e] = -1
            self._end_episode(e, False, reason)

    def _replan(self, e: int, reason: str) -> bool:
        """Re-plan env ``e`` from its live symbolic state; True when a new program is installed."""
        st = self.states[e]
        if self.replan_fn is None or st.replans >= self.cfg.max_replans or st.program is None:
            return False
        if int(self.ctx.held[e]) >= 0:  # drop whatever is in hand before re-planning
            self.geo.detach(torch.tensor([e], device=self.device))
            self.ctx.held[e] = -1
            self.sim.command_gripper(
                torch.zeros(1, dtype=torch.bool, device=self.device),
                torch.tensor([e], device=self.device),
            )
            self.sim.flush()
        prog = self.replan_fn(e)
        st.replans += 1
        if prog is None or len(prog) == 0:
            return False
        st.program = Program(list(prog.skills), st.program.problem_id, st.program.reset_state_id)
        st.pc = 0
        st.attempts = 0
        st.phase = Phase.READY
        self.phase_idx[e] = 0
        self._begun.discard(e)
        self.ctx.params[e] = {}
        log.info(
            "env %d: re-planned from live state after %s: %s",
            e,
            reason,
            " ; ".join(str(s) for s in prog.skills),
        )
        return True

    def _finish_special(self, e: int) -> None:
        st = self.states[e]
        kind = self.ctx.params[e].pop("_special", None)
        if kind == "settle":
            if e in self._program_pending:  # FF not back yet: hold until it is
                self._settled.add(e)
                st.phase = Phase.WAITING
            else:
                st.phase = Phase.READY
        elif kind == "done":
            self._close_episode(e)
        else:
            st.phase = Phase.READY

    # ---------------------------------------------------------- episodes
    def _assign_idle(self) -> None:
        idle = [e for e in range(self.n) if self.states[e].phase is Phase.IDLE]
        if self.cfg.reset_states_once:
            idle = [e for e in idle if self._unstarted(self._env_slot[e])]
        if not idle:
            return
        if self.cfg.cut_tail:
            assign = idle if len(self.records) < self.target_episodes else []
        else:
            if len(self.records) + self._live_episodes() >= self.target_episodes:
                return
            assign = idle[
                : max(0, self.target_episodes - len(self.records) - self._live_episodes())
            ]
        if not assign:
            return
        rids = [self._next_reset_state(e) for e in assign]
        if self.cfg.reset_states_once:  # envs of one profile may outnumber its unstarted states
            assign = [e for e, r in zip(assign, rids) if r is not None]
            rids = [r for r in rids if r is not None]
            if not assign:
                return
        ids = torch.tensor(assign, dtype=torch.long, device=self.device)
        inst = torch.tensor(rids, dtype=torch.long, device=self.device)
        self.sim.reset_to_instances(ids, inst)
        self.sim.command_gripper(
            torch.zeros(len(assign), dtype=torch.bool, device=self.device), ids
        )
        self.sim.flush()
        self.sim.forward()
        self.geo.detach(ids)
        self.ctx.held[ids] = -1
        q = self.sim.joint_pos()[ids]
        segs = []
        progs: list[Program | None]
        pending = self.programs_fn is not None and self.cfg.async_programs
        if pending:  # FF runs on a thread while the envs settle; placeholder programs now
            inst_list = inst.tolist()
            self._programs.append(
                (self._ff_pool.submit(self.programs_fn, assign, inst_list), assign, inst_list)
            )
            progs = [Program([], "pending", int(inst[i])) for i in range(len(assign))]
        elif self.programs_fn is not None:  # batched: facts from the live scene, FF in parallel
            progs = self.programs_fn(assign, inst.tolist())
        else:
            progs = [self.program_fn(e, int(inst[i])) for i, e in enumerate(assign)]
        for i, e in enumerate(assign):
            st = self.states[e]
            prog = progs[i]
            self.ctx.params[e] = {}
            if pending:
                self._program_pending.add(e)
                self._settled.discard(e)
            elif prog is None or len(prog) == 0:
                st.assign(Program([], "none", int(inst[i])), self.next_episode)
                self.next_episode += 1
                self._end_episode(e, False, "no program")
                continue
            st.assign(prog, self.next_episode)
            self.next_episode += 1
            self.episode_ticks[e] = 0
            self.phase_idx[e] = 0
            self._begun.discard(e)
            self._wait_since.pop(e, None)
            if not pending:  # a pending env opens its recording when its program (and state) arrives
                self.rec.start_episode(e, st.episode_id, int(inst[i]), prog.skills[0].hl_state)
            st.phase = Phase.EXECUTING
            self.ctx.params[e]["_special"] = "settle"
            segs.append(
                (
                    e,
                    Segment(
                        q[i : i + 1].expand(1, -1).clone(),
                        hold=self.cfg.reset_settle_ticks,
                        label="reset_settle",
                    ),
                )
            )
        if segs:
            self.table.install([e for e, _ in segs], [s for _, s in segs])

    def _live_episodes(self) -> int:
        return sum(1 for st in self.states if st.phase not in (Phase.IDLE,))

    def _next_reset_state(self, e: int) -> int | None:
        """The next unused reset state recorded under env ``e``'s spawned scene profile
        (baked into the env at spawn); a profile's states cycle once all have run, or with
        ``reset_states_once`` None is returned instead."""
        slot = self._env_slot[e]
        queue = self._rs_queue[slot]
        if not queue:
            if not self._rs_by_slot[slot]:
                raise RuntimeError(
                    f"no reset state was sampled under env {e}'s scene profile "
                    f"({describe_spawn_profile(self.sim.env_scene_profile(e))})"
                )
            if self.cfg.reset_states_once:
                return None
            queue.extend(self._rs_by_slot[slot])
        if self.cfg.reset_states_once:
            queue = self._unstarted(slot)
            if not queue:
                return None
            rid = queue.popleft()
            self._started.add(rid)
            return rid
        return queue.popleft()

    def _unstarted(self, slot: int) -> deque[int]:
        """``slot``'s queue with the reset states another slot already started dropped
        from its front (reset_states_once)."""
        queue = self._rs_queue[slot]
        while queue and queue[0] in self._started:
            queue.popleft()
        return queue

    def _end_episode(self, e: int, success: bool, reason: str) -> None:
        """Hold the arm for ``done_hold_ticks`` (recorded), then close the episode."""
        st = self.states[e]
        st.phase = Phase.EXECUTING
        self.ctx.params[e]["_special"] = "done"
        self.ctx.params[e]["_outcome"] = (success, reason)
        if success:  # the recorded hold after a solved task is the high-level "done" step
            self.rec.symbolic(e, "done", (), "done")
        q = self.sim.joint_pos()[e : e + 1]
        self.table.install([e], [Segment(q.clone(), hold=self.cfg.done_hold_ticks, label="done")])

    def _close_episode(self, e: int) -> None:
        st = self.states[e]
        success, reason = self.ctx.params[e].pop("_outcome", (False, "unknown"))
        self.records.append(
            EpisodeRecord(
                env_id=e,
                episode_id=st.episode_id,
                reset_state_id=st.program.reset_state_id if st.program else -1,
                success=success,
                reason=reason,
                ticks=int(self.episode_ticks[e]),
                skills=len(st.program) if st.program else 0,
            )
        )
        self.rec.end_episode(e, success, reason)
        log.info(
            "env %d episode %d: %s %s (%d ticks)",
            e,
            st.episode_id,
            "SUCCESS" if success else "FAILURE",
            reason,
            int(self.episode_ticks[e]),
        )
        if int(self.ctx.held[e]) >= 0:
            self.geo.detach(torch.tensor([e], device=self.device))
            self.ctx.held[e] = -1
        self.snapshots.pop(e, None)
        self._begun.discard(e)
        self._wait_since.pop(e, None)
        self._program_pending.discard(e)
        self._settled.discard(e)
        if e in self._policy:
            runner, _settle, profile = self._policy.pop(e)
            runner.stop([e])
            if profile is not None:
                self.sim.apply_profile(self.cfg.profile, torch.tensor([e], device=self.device))
        st.clear()

    def _poll_programs(self) -> None:
        """Install the programs FF finished; envs whose settle ended meanwhile become READY."""
        if not self._programs:
            return
        still = []
        for fut, envs, insts in self._programs:
            if not fut.done():
                still.append((fut, envs, insts))
                continue
            try:
                progs = fut.result()
            except Exception as exc:
                log.warning("FF thread raised: %r", exc)
                progs = [None] * len(envs)
            for e, rid, prog in zip(envs, insts, progs, strict=True):
                if e not in self._program_pending:  # episode already closed (cut)
                    continue
                self._program_pending.discard(e)
                st = self.states[e]
                if prog is None or len(prog) == 0:
                    self._settled.discard(e)
                    self._end_episode(e, False, "no program")
                    continue
                st.program = Program(list(prog.skills), prog.problem_id, int(rid))
                self.rec.start_episode(e, st.episode_id, int(rid), prog.skills[0].hl_state)
                if e in self._settled:
                    self._settled.discard(e)
                    st.phase = Phase.READY
        self._programs = still

    # ----------------------------------------------------------- dispatch
    def _poll_async(self) -> None:
        """Install the segments of finished worker-thread builds."""
        if not self._async:
            return
        still: list[tuple[Future, Skill, str, list[int]]] = []
        for fut, skill, phase, envs in self._async:
            if not fut.done():
                still.append((fut, skill, phase, envs))
                continue
            try:
                req = fut.result()
            except Exception as exc:  # the build itself failed: fail those envs' attempts
                log.warning("async %s/%s build raised: %r", skill.name, phase, exc)
                for e in envs:
                    self.states[e].phase = Phase.READY
                    self._retry_or_fail(e, f"{skill.name}/{phase}: build error")
                continue
            for e in envs:
                self.states[e].phase = Phase.READY
            self._resolve([(req, skill, phase, envs)])
        self._async = still

    def _dispatch_ready(self) -> None:
        """Group READY envs by (skill, phase); begin fresh attempts; resolve requests.

        Non-plan requests are resolved at once. Plan requests from all groups are solved
        in ONE batched pool call; when fewer than ``plan_gather_min`` rows are waiting, the
        envs wait up to ``plan_gather_ticks`` ticks for more to arrive (the pool round costs
        the same for 1 or 16 rows, and transports arrive staggered).
        """
        self._poll_async()
        self._poll_plans()
        self._poll_programs()
        self._skip_noop_phases()
        groups: dict[tuple[str, int], list[int]] = defaultdict(list)
        for e in range(self.n):
            st = self.states[e]
            if st.phase is not Phase.READY or st.current is None:
                continue
            if self.episode_ticks[e] > self.cfg.max_episode_ticks:
                self._end_episode(e, False, "episode tick limit")
                continue
            if st.current.name not in self.skills:
                self._end_episode(e, False, f"unsupported skill {st.current.name}")
                continue
            groups[(st.current.name, self.phase_idx[e])].append(e)
        plan_groups: list[tuple[PhaseRequest, Skill, str, list[int]]] = []
        for (name, pidx), envs in groups.items():
            skill = self.skills[name]
            fresh = [e for e in envs if pidx == 0 and e not in self._begun]
            if fresh:
                ids = torch.tensor(fresh, dtype=torch.long, device=self.device)
                snap = self.sim.snapshot(ids)  # one state read for the whole group
                for i, e in enumerate(fresh):
                    # sim state plus the attachment bookkeeping, so a retry of a skill that
                    # starts with a block in hand (place) restores the grasp too
                    self.snapshots[e] = (
                        _slice_state(snap, i),
                        int(self.ctx.held[e]),
                        self.geo.attachment_of(e),
                        self.rec.frame_mark(e),
                    )
                    self.ctx.params[e]["_attempt"] = self.states[e].attempts
                args = [self.states[e].current.args for e in fresh]  # type: ignore[union-attr]
                ok = skill.begin(self.ctx, ids, args)
                for i, e in enumerate(fresh):
                    if bool(ok[i]):
                        self._begun.add(e)
                    else:
                        self._retry_or_fail(e, f"{name} begin: no feasible parameters")
                envs = [e for e in envs if e in self._begun]
                if not envs:
                    continue
            ids = torch.tensor(envs, dtype=torch.long, device=self.device)
            phase = skill.phases[pidx]
            if phase in skill.async_phases and self.cfg.async_workers > 0:
                for e in envs:
                    self.states[e].phase = Phase.WAITING
                fut = self._workers.submit(skill.request, self.ctx, phase, ids)
                self._async.append((fut, skill, phase, envs))
                continue
            req = skill.request(self.ctx, phase, ids)
            if req.kind == "plan" and self.cfg.plan_gather_ticks > 0:
                plan_groups.append((req, skill, phase, envs))
            else:
                self._resolve([(req, skill, phase, envs)])
        if plan_groups:
            rows = sum(len(g[3]) for g in plan_groups)
            oldest = self.ticks
            for _, _, _, envs in plan_groups:
                for e in envs:
                    oldest = min(oldest, self._wait_since.setdefault(e, self.ticks))
            if rows < self.cfg.plan_gather_min and self.ticks - oldest < self.cfg.plan_gather_ticks:
                return  # keep gathering; the envs hold their last target meanwhile
            for _, _, _, envs in plan_groups:
                for e in envs:
                    self._wait_since.pop(e, None)
            self._resolve(plan_groups)

    def _resolve(self, batch: list[tuple[PhaseRequest, Skill, str, list[int]]]) -> None:
        """Resolve one or more requests; all ``plan`` requests share one pool call."""
        dev = self.device
        q_all = self.sim.joint_pos()
        failed: dict[int, str] = {}
        segs: dict[int, Segment] = {}
        owner: dict[int, tuple[PhaseRequest, Skill]] = {}
        plan_rows: list[
            tuple[int, int, PhaseRequest, Skill, str]
        ] = []  # (env, row idx in req, ...)
        for req, skill, phase, envs in batch:
            ok = (
                req.ok
                if req.ok is not None
                else torch.ones(len(envs), dtype=torch.bool, device=dev)
            )
            live = []
            for i, e in enumerate(envs):
                owner[e] = (req, skill)
                if bool(ok[i]):
                    live.append(i)
                else:
                    failed[e] = f"{skill.name}/{phase}: request not formed"
            if not live:
                continue
            sub = torch.tensor(live, dtype=torch.long, device=dev)
            ids = torch.tensor(envs, dtype=torch.long, device=dev)
            if req.kind in ("plan", "ik_interp"):
                t0 = time.perf_counter()
                self.geo.update_worlds(
                    self.ctx.block_poses(),
                    self.ctx.table_pose(),
                    exclude=self._full_mask(req.exclude, sub, envs),
                    pad_fraction=req.pad_fraction,
                    no_pad=self._full_mask(req.no_pad, sub, envs),
                    env_ids=ids[sub],
                )
                self.timing["worlds"] += time.perf_counter() - t0
            if req.kind == "plan":
                for i in live:
                    plan_rows.append((envs[i], i, req, skill, phase))
            elif req.kind == "ik_interp":
                sub_ids = ids[sub]
                goals = req.goals[sub]  # type: ignore[index]
                t0 = time.perf_counter()
                ok_ik, q_goal, perr, rerr = self.geo.ik(sub_ids, goals, seeds=q_all[sub_ids])
                self.timing["ik"] += time.perf_counter() - t0
                accept = ok_ik | ((perr <= 0.005) & (rerr <= 0.05))
                rows = interpolate_rows(q_all[sub_ids], q_goal, max(2, req.ticks))
                for k, i in enumerate(live):
                    e = envs[i]
                    if not bool(accept[k]):
                        failed[e] = f"{skill.name}/{phase}: IK failed ({perr[k]:.3f} m)"
                        continue
                    segs[e] = Segment(
                        rows[k], hold=req.hold, label=req.label, gripper_closed=req.closed
                    )
                self.geo.remember_q(sub_ids, q_goal)
            elif req.kind == "policy":
                runner = req.extra["runner"]
                live_envs = [envs[i] for i in live]
                sub_ids = ids[sub]
                runner.start(live_envs, {e: req.extra["env_data"][e] for e in live_envs})
                if req.gripper_target is not None:
                    self.sim.command_gripper_value(
                        torch.full((len(live_envs),), float(req.gripper_target), device=dev),
                        sub_ids,
                    )
                    self.sim.flush()
                profile = req.extra.get("profile")
                if profile is not None and profile != self.cfg.profile:
                    self.sim.apply_profile(profile, sub_ids)
                else:
                    profile = None
                for e in live_envs:
                    self._policy[e] = (runner, max(0, req.ticks), profile)
                    st = self.states[e]
                    st.phase = Phase.EXECUTING
                    if st.current is not None:
                        cur = st.current
                        self.rec.symbolic(
                            e,
                            cur.hl_name or skill.hl_name(),
                            cur.hl_args or cur.args,
                            req.label,
                            cur.hl_state,
                        )
            elif req.kind == "rows":
                for i in live:
                    e = envs[i]
                    r = req.rows[i] if req.rows is not None else None
                    if r is None or r.shape[0] == 0:
                        failed[e] = f"{skill.name}/{phase}: no rows"
                        continue
                    segs[e] = Segment(
                        r,
                        hold=req.hold,
                        label=req.label,
                        gripper_closed=req.closed,
                        settle_tol=req.settle_tol,
                    )
            else:  # gripper / hold: keep the current arm target
                ticks = max(1, req.ticks)
                for i in live:
                    e = envs[i]
                    segs[e] = Segment(
                        q_all[e : e + 1].expand(ticks, -1).clone(),
                        hold=req.hold,
                        label=req.label,
                        gripper_closed=req.closed,
                    )
        if plan_rows:
            ids = torch.tensor([r[0] for r in plan_rows], dtype=torch.long, device=dev)
            goals = torch.stack([r[2].goals[r[1]] for r in plan_rows])  # type: ignore[index]
            self.timing["plan_rows"] += len(plan_rows)
            self.timing["plan_calls"] += 1
            if self.cfg.async_plans:
                for e, *_ in plan_rows:
                    self.states[e].phase = Phase.WAITING
                fut = self._planner.submit(self.geo.plan, ids, q_all[ids].clone(), goals)
                self._plans.append((fut, plan_rows, {e: owner[e] for e, *_ in plan_rows}))
            else:
                t0 = time.perf_counter()
                plans = self.geo.plan(ids, q_all[ids], goals)
                self.timing["plan"] += time.perf_counter() - t0
                segs_p, failed_p = self._plan_segments(plan_rows, plans)
                segs.update(segs_p)
                failed.update(failed_p)
        self._install(segs, owner)
        for e, reason in failed.items():
            self._retry_or_fail(e, reason)

    def _plan_segments(
        self, plan_rows: list[tuple[int, int, PhaseRequest, Skill, str]], plans: list[Any]
    ) -> tuple[dict[int, Segment], dict[int, str]]:
        segs: dict[int, Segment] = {}
        failed: dict[int, str] = {}
        for (e, _i, req, skill, phase), p in zip(plan_rows, plans, strict=True):
            if not p.ok:
                failed[e] = f"{skill.name}/{phase}: plan {p.status}"
                continue
            segs[e] = Segment(
                p.position,
                p.velocity if self.cfg.target_ff else None,
                p.acceleration if self.cfg.target_ff else None,
                hold=req.hold,
                label=req.label,
                gripper_closed=req.closed,
                settle_tol=req.settle_tol,
            )
        return segs, failed

    def _poll_plans(self) -> None:
        """Install the transport plans the planner thread has finished."""
        if not self._plans:
            return
        still = []
        for fut, plan_rows, owner in self._plans:
            if not fut.done():
                still.append((fut, plan_rows, owner))
                continue
            for e, *_ in plan_rows:
                self.states[e].phase = Phase.READY
            try:
                plans = fut.result()
            except Exception as exc:
                log.warning("planner thread raised: %r", exc)
                for e, _i, _req, skill, phase in plan_rows:
                    self._retry_or_fail(e, f"{skill.name}/{phase}: plan error")
                continue
            segs, failed = self._plan_segments(plan_rows, plans)
            self._install(segs, owner)
            for e, reason in failed.items():
                self._retry_or_fail(e, reason)
        self._plans = still

    def _install(
        self, segs: dict[int, Segment], owner: dict[int, tuple[PhaseRequest, Skill]]
    ) -> None:
        if not segs:
            return
        dev = self.device
        install_ids = list(segs.keys())
        by_target: dict[float, list[int]] = defaultdict(list)
        closed_sel: list[int] = []
        closed_val: list[bool] = []
        for e in install_ids:
            req, _skill = owner[e]
            if req.gripper_target is not None:
                by_target[float(req.gripper_target)].append(e)
            elif segs[e].gripper_closed is not None:
                closed_sel.append(e)
                closed_val.append(bool(segs[e].gripper_closed))
        for target, envs in by_target.items():
            self.sim.command_gripper_value(
                torch.full((len(envs),), target, device=dev),
                torch.tensor(envs, dtype=torch.long, device=dev),
            )
        if closed_sel:
            self.sim.command_gripper(
                torch.tensor(closed_val, dtype=torch.bool, device=dev),
                torch.tensor(closed_sel, dtype=torch.long, device=dev),
            )
        if by_target or closed_sel:
            self.sim.flush()
        self.table.install(install_ids, [segs[e] for e in install_ids])
        for e in install_ids:
            req, skill = owner[e]
            st = self.states[e]
            st.phase = Phase.EXECUTING
            if st.current is not None:
                cur = st.current
                self.rec.symbolic(
                    e, cur.hl_name or skill.hl_name(), cur.hl_args or cur.args, req.label, cur.hl_state
                )

    def _full_mask(self, mask: Tensor | None, sub: Tensor, envs: list[int]) -> Tensor | None:
        """Expand a ``[B, K]`` request mask to ``[N, K]`` (rows outside ``envs`` unused)."""
        if mask is None:
            return None
        full = torch.zeros(self.n, mask.shape[1], dtype=torch.bool, device=self.device)
        full[torch.tensor(envs, device=self.device)[sub]] = mask.to(self.device)[sub]
        return full
