"""Vectorized datagen: sample -> FF -> batched skills -> verify, in one Isaac process.

Example (red_tower_3, 16 envs, 32 episodes):
    python scripts/vecgen/run.py --env-path src/blocksuite/tasks/autogen/red_tower_3 \
        --pddl src/blocksuite/tamp/blocks/problems/red_tower_3.pddl --num-envs 16 \
        --episodes 32 --headless

Play data (``--play``, see :mod:`blocksuite.vecgen.play`): each reset state plays a chain of
policy-chosen goals (balanced pnp / flip / rotate by default in the pipeline), each goal one
FF call from the live symbolic state, with no reset between plans; no TAMP:
    python scripts/vecgen/run.py --play --env-path <pkg> --pddl <scene.pddl> --num-envs 8 \
        --one-chain-per-reset-state --max-plans-per-env 20 --balanced-skills \
        --disallowed-actions push_to_right push_next_to --missed-goals-to-failure \
        --record <out> --summary <out>/summary.json --headless
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from blocksuite.cli.isaac_app import launch_app

simulation_app = launch_app()

import torch  # noqa: E402

from blocksuite.playdata.domain_filter import list_actions, write_filtered_domain  # noqa: E402
from blocksuite.playdata.goal_policy import RandomGoalPolicy  # noqa: E402
from blocksuite.playdata.goal_proposer import (  # noqa: E402
    GOAL_FAMILIES,
    GoalProposerCfg,
    RandomGoalProposer,
)
from blocksuite.playdata.skill_balance import (  # noqa: E402
    BalancedSkillCfg,
    BalancedSkillGoalPolicy,
)
from blocksuite.tamp.blocks.template.codegen import extract_goal_block  # noqa: E402
from blocksuite.vecgen.engine import Engine, EngineConfig  # noqa: E402
from blocksuite.vecgen.geometry import Geometry  # noqa: E402
from blocksuite.vecgen.play import PlayConfig, PlayEngine  # noqa: E402
from blocksuite.vecgen.program import Program  # noqa: E402
from blocksuite.vecgen.sim import PHYSICS_STEPS_PER_TICK, VecSim, make_env  # noqa: E402
from blocksuite.vecgen.skills.base import Skill  # noqa: E402
from blocksuite.vecgen.skills.flip import Flip  # noqa: E402
from blocksuite.vecgen.skills.pnp import Pick, Place  # noqa: E402
from blocksuite.vecgen.skills.rotate import Rotate  # noqa: E402
from blocksuite.vecgen.symbolic.derived import derived_facts  # noqa: E402
from blocksuite.vecgen.symbolic.ff import (  # noqa: E402
    domain_without_actions,
    plan_problems,
    run_ff,
)
from blocksuite.vecgen.symbolic.pddl import (  # noqa: E402
    Fact,
    Problem,
    parse_problem,
    render_problem,
)
from blocksuite.vecgen.symbolic.predicates import assert_goal_supported  # noqa: E402
from blocksuite.vecgen.symbolic.programs import (  # noqa: E402
    UNSUPPORTED,
    UnsupportedAction,
    program_from_plan,
)
from blocksuite.vecgen.symbolic.state import derive_init_facts  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-path", required=True)
    ap.add_argument(
        "--pddl",
        default=None,
        help="PDDL problem; omitted when --env-path holds problems.json (problem streaming)",
    )
    ap.add_argument("--domain", default="src/blocksuite/tamp/blocks/domain.pddl")
    ap.add_argument(
        "--exclude-actions",
        default="",
        help="comma list of domain actions FF may never plan, also when re-planning "
        "(e.g. reorient,rotate for pick-and-place-only data)",
    )
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", default="medium")
    ap.add_argument("--no-ff", action="store_true", help="disable target feed-forward")
    # Defaults: the frictionless, armature-free arm without friction compensation that the
    # Nov 2026 training sets (e.g. RI) were generated with.
    ap.add_argument("--friction-comp", action=argparse.BooleanOptionalAction, default=False,
                    help="joint-friction compensation of motion-planned arm targets")
    ap.add_argument("--arm-joint-friction-scale", type=float, default=0.0,
                    help="scale the arm's static / dynamic / viscous joint friction in sim; 0 = frictionless arm")
    ap.add_argument("--arm-armature-scale", type=float, default=0.0,
                    help="scale the arm's joint armature in sim; 0 = no armature")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--async-workers", type=int, default=4)
    ap.add_argument(
        "--no-derived",
        action="store_true",
        help="plan from the PDDL init only (no derived geometric facts)",
    )
    ap.add_argument(
        "--init-from-scene",
        action="store_true",
        help="plan each episode from its live scene (on / on_table / clear / ori_of read off "
        "the reset state, as a re-plan does) instead of the PDDL :init; for reset states "
        "that are not the problem's init, e.g. TAMPER's captured eval failures",
    )
    ap.add_argument("--plan-gather-ticks", type=int, default=8)
    ap.add_argument(
        "--pool-slots",
        type=int,
        default=None,
        help="cuRobo pool slots (default min(num_envs, 32)); the main thread keeps half, "
        "the async build workers share the rest",
    )
    ap.add_argument(
        "--profile-out", default=None, help="write a cProfile dump of the engine loop here"
    )
    ap.add_argument("--max-ticks", type=int, default=None)
    ap.add_argument(
        "--build-slots",
        type=int,
        default=32,
        help="slots of the separate pool used by the async skill builds (0 = share the main pool)",
    )
    ap.add_argument(
        "--sync-plans", action="store_true", help="solve transport plans on the tick thread"
    )
    ap.add_argument(
        "--sync-programs", action="store_true", help="run FF on the tick thread at assignment"
    )
    ap.add_argument(
        "--pool-ik-seeds", type=int, default=None, help="main pool IK seeds (cuRobo default 32)"
    )
    ap.add_argument(
        "--pool-trajopt-seeds",
        type=int,
        default=None,
        help="trajopt seeds per plan (cuRobo default 4)",
    )
    ap.add_argument(
        "--build-ik-seeds",
        type=int,
        default=None,
        help="IK seeds of the build pool (cuRobo default 32; 8 cost ~5 points of flip success)",
    )
    ap.add_argument(
        "--plan-gather-min",
        type=int,
        default=None,
        help="plan rows that end the gather wait early (default max(4, num_envs // 8))",
    )
    ap.add_argument(
        "--cut-tail",
        action="store_true",
        help="keep all envs busy until the episode target; cut in-flight episodes then",
    )
    ap.add_argument(
        "--reset-states-once",
        action="store_true",
        help="start every reset state at most once (envs idle instead of cycling a scene "
        "profile's states; the run also ends once all have run)",
    )
    ap.add_argument(
        "--reset-state-ids",
        type=int,
        nargs="+",
        default=None,
        help="run only these reset states, each once (implies --reset-states-once)",
    )
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--record", default=None, help="data dir: record a LeRobot dataset (no video)")
    ap.add_argument(
        "--record-videos",
        action="store_true",
        help="--record also writes the scene cameras as LeRobot videos (spawns the env package's "
        "cameras; pass --enable_cameras too so Isaac renders them)",
    )
    ap.add_argument(
        "--mark-complete",
        action="store_true",
        help="write results.json and .execution_complete into --record when done",
    )
    play = ap.add_argument_group(
        "play data (--play): chains of policy-chosen goals from each reset state, no reset "
        "between plans (TAMP-free successor of the former generate_play_data.py)"
    )
    play.add_argument("--play", action="store_true", help="collect play data instead of datagen")
    play.add_argument("--max-plans-per-env", type=int, default=None)
    play.add_argument("--max-plans-total", type=int, default=None)
    play.add_argument("--min-successful-plans-per-env", type=int, default=None)
    play.add_argument(
        "--one-chain-per-reset-state",
        action="store_true",
        help="play every reset state as its own chain; an env moves to the next unplayed "
        "reset state of its scene profile when its chain's budget is met",
    )
    play.add_argument("--max-plan-failures-per-env", type=int, default=8)
    play.add_argument(
        "--missed-goals-to-failure",
        action="store_true",
        help="record episodes whose goal was missed into blocks_failure (default: all success)",
    )
    play.add_argument("--allowed-actions", nargs="*", default=None)
    play.add_argument("--disallowed-actions", nargs="*", default=None)
    play.add_argument(
        "--balanced-skills",
        action="store_true",
        help="goals keep executed skills near 1/3 pnp, 1/3 flip, 1/3 rotate (else random goals)",
    )
    play.add_argument("--max-stack-height", type=int, default=3)
    play.add_argument("--balance-by-success", action="store_true")
    play.add_argument("--flip-min-x", nargs="*", default=None, metavar="DIR=M")
    play.add_argument("--abandon-plan-percent", type=float, default=0.0)
    play.add_argument("--abandon-fraction-range", type=float, nargs=2, default=(0.2, 0.8))
    play.add_argument("--goal-families", nargs="+", default=None, help="random goals only")
    play.add_argument("--num-goal-predicates", type=int, default=1, help="random goals only")
    play.add_argument("--ori-blocks", nargs="*", default=None, help="random goals only")
    play.add_argument("--settle-steps", type=int, default=60, help="physics steps, max settle")
    play.add_argument("--settle-min-steps", type=int, default=8, help="physics steps")
    play.add_argument(
        "--pre-episode-steps", type=int, default=50, help="physics steps after a reset"
    )
    play.add_argument(
        "--poll-steps", type=int, default=8, help="physics steps between goal retries"
    )
    play.add_argument("--summary", default=None, help="write the play summary JSON here")
    play.add_argument(
        "--dfa-spec",
        default=None,
        help="YAML/JSON DFA strategy (e.g. src/blocksuite/playdata/dfa/library/flip_search.yaml) "
        "choosing the goals instead of random / balanced sampling; on ACCEPT / REJECT the env "
        "restarts from a fresh reset state",
    )
    play.add_argument(
        "--dfa-param",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="override the DFA spec's params, values parsed as YAML (e.g. letter=Q flips_per_block=3)",
    )
    args, _ = ap.parse_known_args()
    if args.dfa_spec and args.balanced_skills:
        ap.error("--dfa-spec drives the goals; it cannot be combined with --balanced-skills")
    return args


def _parse_flip_min_x(pairs: list[str] | None) -> dict[str, float]:
    """``direction=metres`` pairs for --flip-min-x."""
    out: dict[str, float] = {}
    for pair in pairs or []:
        key, sep, raw = pair.partition("=")
        if not sep or key.strip() not in ("forward", "backward", "left_flip", "right_flip"):
            raise SystemExit(
                f"--flip-min-x expects forward|backward|left_flip|right_flip=<m>, got {pair!r}"
            )
        out[key.strip()] = float(raw)
    return out


def main() -> int:
    args = parse_args()
    handler = logging.StreamHandler(sys.__stdout__)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    for name in ("vecgen", "vecgen.run"):
        lg = logging.getLogger(name)
        lg.handlers = [handler]
        lg.propagate = False
        lg.setLevel(args.log_level)
    log = logging.getLogger("vecgen.run")
    dev = torch.device("cuda:0")
    t0 = time.time()
    env = make_env(
        args.env_path,
        num_envs=args.num_envs,
        device="cuda:0",
        cameras=args.record_videos,
    )
    env.reset()
    sim = VecSim(env, dev)
    if args.arm_joint_friction_scale != 1.0:
        sim.set_arm_joint_friction(args.arm_joint_friction_scale)
    if args.arm_armature_scale != 1.0:
        sim.set_arm_armature(args.arm_armature_scale)
    log.info("arm dynamics: %s | friction_comp=%s", sim.arm_dynamics_summary(), args.friction_comp)
    log.info(
        "env ready in %.1f s: %d envs, blocks %s", time.time() - t0, sim.num_envs, sim.block_names
    )
    # plan with the spawned colliders, not the package's nominal sizes (a letter block's
    # collider is ~4-6.5% smaller than its nominal size: the spawn scales the decal-inclusive
    # bounding box to it)
    t0 = time.time()
    sizes, centre = sim.measure_block_dims()
    nominal = sim.block_sizes().to(dev)
    ratio = sizes / nominal
    log.info(
        "block colliders measured in %.1f s: collider/nominal %.3f-%.3f, max centre offset %.2f mm",
        time.time() - t0,
        float(ratio.min()),
        float(ratio.max()),
        1000.0 * float(centre.abs().max()),
    )
    if float(centre.abs().max()) > 0.0005:
        log.warning("a block collider is off its body origin by > 0.5 mm; planning assumes centred")
    table_top = sim.measure_table_top()
    t0 = time.time()
    geo = Geometry(
        sim.num_envs,
        sim.block_names,
        sizes,
        dev,
        random_seed=args.seed,
        pool_slots=args.pool_slots,
        regions=2 if not args.sync_plans else 1,  # tick thread + planner thread
        build_slots=args.build_slots if args.async_workers > 0 else 0,
        build_regions=args.async_workers,
        ik_seeds=args.pool_ik_seeds,
        trajopt_seeds=args.pool_trajopt_seeds,
        build_ik_seeds=args.build_ik_seeds,
    )
    geo.table_top = table_top
    log.info("table collider top %.4f m above its origin", float(table_top.mean()))
    log.info("planner pool ready in %.1f s", time.time() - t0)

    # ---- the domain FF plans over (minus --exclude-actions)
    ff_domain = Path(args.domain)
    excluded = [a.strip() for a in args.exclude_actions.split(",") if a.strip()]
    planned_actions: list[str] = []
    if args.play:
        # the play goal policies need the actions FF may use; vecgen has no push / isolate skill
        known = list_actions(ff_domain.read_text())
        disallowed = list(args.disallowed_actions or []) + excluded
        disallowed += [a for a in sorted(UNSUPPORTED) if a in known and a not in disallowed]
        tmp = Path(tempfile.mkdtemp(prefix="vecgen_domain_"))
        planned_actions = write_filtered_domain(
            ff_domain, tmp / ff_domain.name, args.allowed_actions, disallowed
        )
        ff_domain = tmp / ff_domain.name
        log.info("play: FF plans with %s (%s)", planned_actions, ff_domain)
    elif excluded:
        tmp = Path(tempfile.mkdtemp(prefix="vecgen_domain_"))
        ff_domain = domain_without_actions(ff_domain, excluded, tmp)
        log.info("FF plans without %s (%s)", excluded, ff_domain)

    # ---- problems: one PDDL, or a superset package streaming several (problems.json)
    index_path = Path(args.env_path) / "problems.json"
    problems: dict[str, Problem] = {}
    # problem name -> its PDDL goal block verbatim: the recorded pddl_goal must be the text the
    # env package was compiled from (eval's set_episode_goal matches it modulo whitespace)
    goal_block: dict[str, str] = {}
    problem_of_rid: dict[int, str] = {}
    # multi-problem package: reset state -> (problem slug, id within the problem, parked blocks)
    episode_of_rid: dict[int, tuple[str, int, list[str]]] = {}
    if args.pddl:
        problem = parse_problem(args.pddl)
        problems[problem.name] = problem
        goal_block[problem.name] = extract_goal_block(Path(args.pddl).read_text())
    elif index_path.exists():
        index = json.loads(index_path.read_text())
        for entry in index["problems"]:
            pddl_path = Path(args.env_path) / entry["pddl"]
            prob = parse_problem(pddl_path)
            problems[prob.name] = prob
            goal_block[prob.name] = extract_goal_block(pddl_path.read_text())
            lo, hi = entry["reset_states"]
            parked = [b for b in index["blocks"] if b not in set(entry["blocks"])]
            for rid in range(lo, hi):
                problem_of_rid[rid] = prob.name
                episode_of_rid[rid] = (entry["slug"], rid - lo, parked)
        problem = next(iter(problems.values()))
        log.info("streaming %d problems from %s", len(problems), index_path)
    else:
        log.error("--pddl is required unless --env-path holds problems.json")
        return 1

    for prob in problems.values():
        assert_goal_supported(prob.goal, f"problem {prob.name}")

    def problem_for(reset_state_id: int) -> Problem:
        return problems[problem_of_rid.get(reset_state_id, problem.name)]

    base_program: Program | None = None
    if args.pddl and not args.play and not args.init_from_scene:  # one plan from :init
        ff = run_ff(ff_domain, render_problem(problem))
        if not ff.ok:
            log.error("FF failed: %s", ff.error)
            return 1
        base_program = program_from_plan(ff.actions, problem.name, 0, ff.states)
        log.info(
            "FF plan (%.0f ms): %s",
            ff.seconds * 1000,
            " ; ".join(str(s) for s in base_program.skills),
        )

    def program_fn(env_id: int, reset_state_id: int) -> Program | None:
        if base_program is None:
            return None
        return Program(list(base_program.skills), problem.name, reset_state_id)

    def _plan_from_facts(
        env_ids: list[int], inits: list[list[Fact]], prob: Problem
    ) -> list[Program | None]:
        results = plan_problems(
            ff_domain, prob, inits, workers=min(32, os.cpu_count() or 8)
        )
        out: list[Program | None] = []
        for e, res in zip(env_ids, results, strict=True):
            if not res.ok:
                log.info("env %d: FF failed: %s", e, res.error[:160])
                out.append(None)
                continue
            try:
                out.append(program_from_plan(res.actions, prob.name, 0, res.states))
            except UnsupportedAction as exc:
                log.info("env %d: FF plan uses unsupported action %s", e, exc)
                out.append(None)
        return out

    def _derived(prob: Problem, ids: torch.Tensor) -> list[list[Fact]]:
        """Derived facts over the problem's own blocks (a superset package parks the rest)."""
        if args.no_derived:
            return [[] for _ in range(int(ids.numel()))]
        cols = torch.tensor([engine.ctx.block_index[b] for b in prob.blocks], device=dev)
        return derived_facts(
            prob.blocks,
            engine.ctx.block_poses()[ids][:, cols],
            engine.ctx.sizes[ids][:, cols],
        )

    def programs_fn(env_ids: list[int], reset_state_ids: list[int]) -> list[Program | None]:
        """PDDL init plus geometric facts derived from each env's live scene, FF per env."""
        progs: list[Program | None] = [None] * len(env_ids)
        by_problem: dict[str, list[int]] = {}
        for i, rid in enumerate(reset_state_ids):
            by_problem.setdefault(problem_for(int(rid)).name, []).append(i)
        for name, rows in by_problem.items():
            prob = problems[name]
            ids = torch.tensor([env_ids[i] for i in rows], device=dev)
            derived = _derived(prob, ids)
            if args.init_from_scene:
                ctx = engine.ctx
                poses, sizes, table = ctx.block_poses()[ids], ctx.sizes[ids], ctx.table_pose()[ids]
                inits = [
                    derive_init_facts(prob, ctx.block_index, poses[k], sizes[k], table[k]) + d
                    for k, d in enumerate(derived)
                ]
            else:
                inits = [list(prob.init) + d for d in derived]
            sub = _plan_from_facts([env_ids[i] for i in rows], inits, prob)
            for i, pr in zip(rows, sub, strict=True):
                progs[i] = pr
        for i, rid in enumerate(reset_state_ids):
            if progs[i] is not None:
                progs[i] = Program(progs[i].skills, problem_for(int(rid)).name, int(rid))
        log.info(
            "assigned %d envs (%d problems): programs %s",
            len(env_ids),
            len(by_problem),
            [len(p) if p else None for p in progs],
        )
        return progs

    skills: dict[str, Skill] = {
        "pick": Pick(),
        "place_on": Place("on"),
        "place_table": Place("table"),
        "place_side": Place("side"),
        "flip": Flip(profile=args.profile),
        "rotate": Rotate(),
    }
    if args.reset_state_ids is not None:
        args.reset_states_once = True
        keep = set(args.reset_state_ids)
        by_slot = sim.reset_states_by_slot()

        def selected_by_slot() -> dict[int, list[int]]:
            return {slot: [r for r in rids if r in keep] for slot, rids in by_slot.items()}

        sim.reset_states_by_slot = selected_by_slot  # type: ignore[method-assign]
        log.info("running reset states %s once each", sorted(keep))
    cfg = EngineConfig(
        max_attempts=args.max_attempts,
        target_ff=not args.no_ff,
        friction_comp=args.friction_comp,
        profile=args.profile,
        async_workers=args.async_workers,
        plan_gather_ticks=args.plan_gather_ticks,
        plan_gather_min=args.plan_gather_min
        if args.plan_gather_min is not None
        else max(4, args.num_envs // 8),
        cut_tail=args.cut_tail,
        reset_states_once=args.reset_states_once,
        async_plans=not args.sync_plans,
        async_programs=not args.sync_programs,
    )
    recorder = None
    if args.record:
        from blocksuite.vecgen.recorder import LeRobotRecorder

        goal_text = goal_block[problem.name]
        recorder = LeRobotRecorder(
            sim,
            geo,
            args.record,
            problem.name,
            goal_text,
            # play data sets each episode's own goal (Recorder.set_goal)
            goal_of_reset_state=None
            if args.play
            else lambda rid: goal_block[problem_for(rid).name],
            episode_of_reset_state=episode_of_rid.__getitem__ if episode_of_rid else None,
            record_videos=args.record_videos,
        )
        log.info("recording to %s", args.record)

    def replan_fn(env_id: int) -> Program | None:
        ids = torch.tensor([env_id], device=dev)
        st_prog = engine.states[env_id].program
        prob = problems.get(st_prog.problem_id, problem) if st_prog is not None else problem
        facts = derive_init_facts(
            prob,
            engine.ctx.block_index,
            engine.ctx.block_poses()[ids][0],
            engine.ctx.sizes[ids][0],
            engine.ctx.table_pose()[ids][0],
        )
        facts = facts + _derived(prob, ids)[0]
        res = run_ff(ff_domain, render_problem(prob, facts))
        if not res.ok:
            log.info("env %d: FF re-plan failed: %s", env_id, res.error[:200])
            return None
        try:
            return program_from_plan(res.actions, prob.name, 0, res.states)
        except UnsupportedAction as exc:
            log.info("env %d: FF re-plan uses unsupported action %s", env_id, exc)
            return None

    if args.play:
        return _run_play(
            args, sim, geo, skills, cfg, recorder, problem_for, ff_domain, planned_actions, log
        )

    engine = Engine(
        sim,
        geo,
        skills,
        program_fn,
        cfg,
        recorder=recorder,
        seed=args.seed,
        replan_fn=replan_fn,
        programs_fn=programs_fn,
    )
    engine.ctx.problem_goal = list(problem.goal)
    engine.ctx.problem_goals = {name: list(pr.goal) for name, pr in problems.items()}
    t0 = time.time()
    if args.profile_out:
        import cProfile

        prof = cProfile.Profile()
        prof.enable()
        records = engine.run(args.episodes, max_ticks=args.max_ticks)
        prof.disable()
        prof.dump_stats(args.profile_out)
    else:
        records = engine.run(args.episodes, max_ticks=args.max_ticks)
    wall = time.time() - t0
    if recorder is not None:
        t1 = time.time()
        recorder.finalize()
        log.info("recorder finalized in %.1f s", time.time() - t1)
    n_cut = sum(r.reason.startswith("cut:") for r in records)
    records = [r for r in records if not r.reason.startswith("cut:")]  # --cut-tail leftovers
    n_ok = sum(r.success for r in records)
    reasons: dict[str, int] = {}
    for r in records:
        if not r.success:
            reasons[r.reason] = reasons.get(r.reason, 0) + 1
    log.info(
        "RESULT episodes=%d success=%d (%.1f%%) wall=%.1fs ticks=%d ms/tick=%.1f ep/min=%.1f",
        len(records),
        n_ok,
        100.0 * n_ok / max(1, len(records)),
        wall,
        engine.ticks,
        1000 * wall / max(1, engine.ticks),
        60.0 * len(records) / max(1e-9, wall),
    )
    log.info("failure reasons: %s", reasons)
    if n_cut:
        log.info("%d in-flight episodes cut at the target (recorded as failures)", n_cut)
    log.info("pool: %s", geo.pool.stats_summary())
    for name in ("place_table", "place_side"):
        log.info("%s placement: %s", name, getattr(skills[name], "stats", {}))
    if geo.build_pool is not None:
        log.info("build pool: %s", geo.build_pool.stats_summary())
    print(
        f"RESULT episodes={len(records)} success={n_ok} wall={wall:.1f}s ticks={engine.ticks} "
        f"ms_per_tick={1000 * wall / max(1, engine.ticks):.1f} reasons={reasons}",
        flush=True,
    )
    if args.record and args.mark_complete:
        out = Path(args.record)
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(
            json.dumps(
                {
                    "problem": problem.name,
                    "episodes": len(records),
                    "success": n_ok,
                    "wall_s": wall,
                    "ticks": engine.ticks,
                    "failure_reasons": reasons,
                    "records": [r.__dict__ for r in records],
                },
                indent=2,
            )
        )
        (out / ".execution_complete").write_text("")
    print("RUN_OK", flush=True)
    return 0


def _parse_dfa_params(pairs: list[str]) -> dict[str, Any]:
    """``key=value`` overrides for the DFA spec, values parsed as YAML scalars."""
    import yaml

    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--dfa-param must be key=value, got {pair!r}")
        key, _, raw = pair.partition("=")
        out[key.strip()] = yaml.safe_load(raw)
    return out


def _run_play(
    args: argparse.Namespace,
    sim: VecSim,
    geo: Geometry,
    skills: dict[str, Skill],
    cfg: EngineConfig,
    recorder: Any,
    problem_for: Callable[[int], Problem],
    ff_domain: Path,
    planned_actions: list[str],
    log: logging.Logger,
) -> int:
    """Play-data loop (``--play``): goal policy -> FF -> skills, chained per reset state."""
    rng = random.Random(args.seed)
    engine_ref: list[PlayEngine] = []

    def derived(prob: Problem, ids: torch.Tensor) -> list[list[Fact]]:
        """Derived facts over the problem's own blocks (as datagen's ``_derived``)."""
        if args.no_derived:
            return [[] for _ in range(int(ids.numel()))]
        ctx = engine_ref[0].ctx
        cols = torch.tensor([ctx.block_index[b] for b in prob.blocks], device=ids.device)
        return derived_facts(prob.blocks, ctx.block_poses()[ids][:, cols], ctx.sizes[ids][:, cols])

    if args.dfa_spec:
        from blocksuite.playdata.dfa import DFAGoalPolicy, glyph_of, load_dfa_spec

        spec = load_dfa_spec(args.dfa_spec, overrides=_parse_dfa_params(args.dfa_param))
        policy: Any = DFAGoalPolicy(spec, seed=args.seed, glyph_of=glyph_of)
        log.info(
            "play: DFA goal policy '%s' from %s: initial=%s, states=%s, params=%s",
            spec.name,
            args.dfa_spec,
            spec.initial,
            list(spec.states),
            spec.params,
        )
    elif args.balanced_skills:
        policy = BalancedSkillGoalPolicy(
            BalancedSkillCfg(
                allowed_actions=tuple(planned_actions),
                max_stack_height=args.max_stack_height,
                count_successes=args.balance_by_success,
                flip_min_x=_parse_flip_min_x(args.flip_min_x),
            ),
            rng=rng,
        )
        log.info("play: balanced-skill goals, target shares %s", policy.shares)
    else:
        policy = RandomGoalPolicy(
            RandomGoalProposer(
                GoalProposerCfg(
                    families=tuple(args.goal_families or GOAL_FAMILIES),
                    num_predicates=args.num_goal_predicates,
                    ori_blocks=tuple(args.ori_blocks) if args.ori_blocks else None,
                ),
                rng=rng,
            )
        )
        log.info("play: random goals")
    lo, hi = args.abandon_fraction_range
    if not 0.0 <= args.abandon_plan_percent <= 100.0 or not 0.0 <= lo <= hi <= 1.0:
        raise SystemExit("--abandon-plan-percent must be in [0, 100] and 0 <= lo <= hi <= 1")

    def ticks(steps: int) -> int:
        return max(1, steps // PHYSICS_STEPS_PER_TICK)

    play_cfg = PlayConfig(
        max_plans_per_env=args.max_plans_per_env,
        max_plans_total=args.max_plans_total,
        min_successful_plans_per_env=args.min_successful_plans_per_env,
        one_chain_per_reset_state=args.one_chain_per_reset_state,
        max_plan_failures_per_env=args.max_plan_failures_per_env,
        settle_ticks=ticks(args.settle_steps),
        settle_min_ticks=ticks(args.settle_min_steps),
        poll_ticks=ticks(args.poll_steps),
        pre_episode_ticks=ticks(args.pre_episode_steps),
        missed_goals_to_failure=args.missed_goals_to_failure,
        abandon_plan_prob=args.abandon_plan_percent / 100.0,
        abandon_fraction_range=(lo, hi),
        allowed_actions=tuple(planned_actions),
        max_stack_height=args.max_stack_height,
    )
    engine = PlayEngine(
        sim,
        geo,
        skills,
        cfg,
        play_cfg,
        policy,
        problem_for,
        ff_domain,
        derived,
        recorder=recorder,
        seed=args.seed,
    )
    engine_ref.append(engine)
    t0 = time.time()
    records = engine.run(max_ticks=args.max_ticks)
    wall = time.time() - t0
    if recorder is not None:
        t1 = time.time()
        recorder.finalize()
        log.info("recorder finalized in %.1f s", time.time() - t1)
    summary = engine.summary()
    summary.update(
        {
            "execution_time_seconds": wall,
            "num_envs": sim.num_envs,
            "ticks": engine.ticks,
            "data_dir": args.record,
            "env_path": args.env_path,
        }
    )
    play = summary["play_data"]
    log.info(
        "PLAY episodes=%d kept=%d goals_reached=%d missed=%d abandoned=%d plans failed=%d "
        "chains=%d wall=%.1fs ms/tick=%.1f skills=%s",
        len(records),
        play["episodes_kept"],
        play["goals_reached"],
        play["goals_missed"],
        play["plans_abandoned"],
        play["plans_failed"],
        len(play["chains"]),
        wall,
        1000 * wall / max(1, engine.ticks),
        play["executed_skill_shares"],
    )
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(summary, indent=2, default=str))
    if args.record and args.mark_complete:
        out = Path(args.record)
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(summary, indent=2, default=str))
        (out / ".execution_complete").write_text("")
    print("RUN_OK", flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    # simulation_app.close() can hang in headless runs; give it a moment, then exit hard.
    import threading

    threading.Timer(15.0, lambda: os._exit(rc)).start()
    try:
        simulation_app.close()
    finally:
        os._exit(rc)
