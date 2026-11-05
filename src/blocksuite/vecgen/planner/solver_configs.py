"""cuRobo-v1-style world-collision handling for the pool's IK and trajopt solver configs.

Ported from TAMP's ``tmp3.Backends.Curobo.motion_gen.apply_v1_world_collision_configs``.
"""

from __future__ import annotations

from typing import Any

from curobo._src.solver.solver_ik import IKSolverCfg

#: IK world collision as a soft cost (cuRobo v1): weight and activation distance
IK_WORLD_COST_WEIGHT = 5000.0
IK_WORLD_ACTIVATION = 0.01
#: trajopt world collision as a swept cost: weight and activation distance. 0.025 (v1's value)
#: keeps v2 plans ~2 cm closer to the blocks than v1's did; 0.04 matches v1's clearance and arc
#: height on replayed v1 pick queries.
TRAJOPT_WORLD_COST_WEIGHT = 100000.0
TRAJOPT_WORLD_ACTIVATION = 0.04


def apply_v1_world_collision_configs(
    planner_cfg: Any,
    robot: Any,
    device_cfg: Any,
    self_collision_check: bool,
    random_seed: int | None = None,
    ik_optimizer_configs: list[Any] | None = None,
) -> None:
    """Rebuild ``planner_cfg``'s IK and trajectory-optimization solver configs cuRobo-v1 style.

    cuRobo v1 handled world collision in IK as a soft cost (its IK task config has no
    constraint section), so a goal a few millimetres into a neighbouring obstacle still
    produced an IK solution and trajectory optimization decided feasibility. v2 makes it a hard
    IK constraint (and an IK success criterion), which fails grasp / place goals that v1
    planned. Likewise v1's trajopt config kept world collision as a swept cost with no hard
    constraint. Both are rebuilt that way here, for the pool's batch planner and its single
    fallback planner.
    """
    from curobo._src.solver.solver_core_cfg import get_task_configs_path
    from curobo._src.solver.solver_trajopt import TrajOptSolverCfg
    from curobo._src.util.config_io import join_path, resolve_config

    seed_kwargs: dict[str, Any] = {} if random_seed is None else {"random_seed": int(random_seed)}

    def _ik_task_cfg(name: str) -> dict[str, Any]:
        cfg_dict = resolve_config(join_path(get_task_configs_path(), name))
        scene_cfg = cfg_dict["rollout"].get("constraint_cfg", {}).pop("scene_collision_cfg", None)
        if scene_cfg is not None:
            cfg_dict["rollout"].setdefault("cost_cfg", {})["scene_collision_cfg"] = {
                "weight": IK_WORLD_COST_WEIGHT,
                "activation_distance": IK_WORLD_ACTIVATION,
                "use_sweep": False,
            }
        return cfg_dict

    ik_names = list(ik_optimizer_configs or ["ik/lbfgs_ik.yml"])
    ik_cfgs = [_ik_task_cfg(n) if isinstance(n, str) else n for n in ik_names]
    metrics_cfg = resolve_config(join_path(get_task_configs_path(), "metrics_base.yml"))
    metrics_cfg["rollout"].get("constraint_cfg", {}).pop("scene_collision_cfg", None)
    base_ik_cfg = planner_cfg.ik_solver_config
    planner_cfg.ik_solver_config = IKSolverCfg.create(
        robot=robot,
        optimizer_configs=ik_cfgs,
        metrics_rollout=metrics_cfg,
        device_cfg=device_cfg,
        num_seeds=base_ik_cfg.num_seeds,
        position_tolerance=base_ik_cfg.position_tolerance,
        orientation_tolerance=base_ik_cfg.orientation_tolerance,
        optimizer_collision_activation_distance=IK_WORLD_ACTIVATION,
        self_collision_check=self_collision_check,
        max_batch_size=int(getattr(base_ik_cfg, "max_batch_size", 1)),
        multi_env=bool(getattr(base_ik_cfg, "multi_env", False)),
        **seed_kwargs,
    )

    trajopt_cfg_dict = resolve_config(
        join_path(get_task_configs_path(), "trajopt/lbfgs_bspline_trajopt.yml")
    )
    scene_cfg = (
        trajopt_cfg_dict["rollout"].get("constraint_cfg", {}).pop("scene_collision_cfg", None)
    )
    if scene_cfg is not None:
        scene_cfg = dict(scene_cfg)
        scene_cfg.update(
            {"weight": TRAJOPT_WORLD_COST_WEIGHT, "activation_distance": TRAJOPT_WORLD_ACTIVATION}
        )
        trajopt_cfg_dict["rollout"].setdefault("cost_cfg", {})["scene_collision_cfg"] = scene_cfg
    trajopt_metrics = resolve_config(join_path(get_task_configs_path(), "metrics_base.yml"))
    trajopt_metrics["rollout"].get("constraint_cfg", {}).pop("scene_collision_cfg", None)
    base_to = planner_cfg.trajopt_solver_config
    planner_cfg.trajopt_solver_config = TrajOptSolverCfg.create(
        robot=robot,
        optimizer_configs=[trajopt_cfg_dict],
        metrics_rollout=trajopt_metrics,
        device_cfg=device_cfg,
        num_seeds=base_to.num_seeds,
        position_tolerance=base_to.position_tolerance,
        orientation_tolerance=base_to.orientation_tolerance,
        optimizer_collision_activation_distance=TRAJOPT_WORLD_ACTIVATION,
        self_collision_check=self_collision_check,
        minimum_trajectory_dt=base_to.minimum_trajectory_dt,
        maximum_trajectory_dt=base_to.maximum_trajectory_dt,
        interpolation_dt=base_to.interpolation_dt,
        interpolation_buffer_size=base_to.interpolation_buffer_size,
        max_batch_size=int(getattr(base_to, "max_batch_size", 1)),
        multi_env=bool(getattr(base_to, "multi_env", False)),
        **seed_kwargs,
    )
