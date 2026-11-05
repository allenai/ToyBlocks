"""cuRobo-v1-shaped result and config types returned by :mod:`toyblocks.vecgen.planner.pool`.

Ported from TAMP's ``tmp3.Backends.Curobo.types``. They mirror the fields of
``curobo.wrap.reacher.motion_gen.MotionGenPlanConfig`` / ``MotionGenResult`` /
``MotionGenStatus`` and ``curobo.wrap.reacher.ik_solver.IKResult`` and depend on torch only.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch


class MotionGenStatus(Enum):
    """Mirror of cuRobo v1 ``MotionGenStatus`` (same member names and values)."""

    IK_FAIL = "IK Fail"
    GRAPH_FAIL = "Graph Fail"
    TRAJOPT_FAIL = "TrajOpt Fail"
    FINETUNE_TRAJOPT_FAIL = "Finetune TrajOpt Fail"
    DT_EXCEPTION = "dt exceeded maximum allowed trajectory dt"
    INVALID_QUERY = "Invalid Query"
    INVALID_START_STATE_UNKNOWN_ISSUE = "Invalid Start State, unknown issue"
    INVALID_START_STATE_WORLD_COLLISION = "Start state is colliding with world"
    INVALID_START_STATE_SELF_COLLISION = "Start state is in self-collision"
    INVALID_START_STATE_JOINT_LIMITS = "Start state is out of joint limits"
    INVALID_PARTIAL_POSE_COST_METRIC = "Invalid partial pose metric"
    SUCCESS = "Success"
    NOT_ATTEMPTED = "Not Attempted"


@dataclass
class MotionGenPlanConfig:
    """Mirror of cuRobo v1 ``MotionGenPlanConfig`` (fields read by the pool are
    ``max_attempts``, ``enable_graph``, ``enable_graph_attempt`` and ``check_start_validity``).
    """

    enable_graph: bool = False
    enable_opt: bool = True
    use_nn_ik_seed: bool = False
    need_graph_success: bool = False
    max_attempts: int = 60
    timeout: float = 10.0
    enable_graph_attempt: int | None = 3
    disable_graph_attempt: int | None = None
    ik_fail_return: int | None = None
    partial_ik_opt: bool = False
    num_ik_seeds: int | None = None
    num_graph_seeds: int | None = None
    num_trajopt_seeds: int | None = None
    success_ratio: float = 1.0
    fail_on_invalid_query: bool = True
    use_start_state_as_retract: bool = True
    pose_cost_metric: Any = None
    enable_finetune_trajopt: bool = True
    parallel_finetune: bool = True
    finetune_dt_scale: float = 0.9
    finetune_attempts: int = 1
    finetune_dt_decay: float = 1.025
    check_start_validity: bool = True
    time_dilation_factor: float | None = None
    use_cuda_graph: bool | None = None

    def clone(self) -> MotionGenPlanConfig:
        return copy.copy(self)


@dataclass
class MotionGenResult:
    """Mirror of cuRobo v1 ``MotionGenResult``.

    ``interpolated_plan`` is already trimmed to the valid horizon, so
    :meth:`get_interpolated_plan` returns it unchanged.
    """

    success: torch.Tensor | None = None
    valid_query: bool = True
    optimized_plan: Any = None
    optimized_dt: torch.Tensor | None = None
    position_error: torch.Tensor | None = None
    rotation_error: torch.Tensor | None = None
    cspace_error: torch.Tensor | None = None
    solve_time: float = 0.0
    ik_time: float = 0.0
    graph_time: float = 0.0
    trajopt_time: float = 0.0
    finetune_time: float = 0.0
    total_time: float = 0.0
    interpolated_plan: Any = None
    interpolation_dt: float = 0.02
    path_buffer_last_tstep: list[int] | None = None
    debug_info: dict[str, Any] | None = None
    status: MotionGenStatus | None = None
    attempts: int = 1
    trajopt_attempts: int = 0
    used_graph: bool = False
    goalset_index: torch.Tensor | None = None

    def get_interpolated_plan(self) -> Any:
        return self.interpolated_plan

    def clone(self) -> MotionGenResult:
        return copy.copy(self)


@dataclass
class IKResult:
    """Mirror of cuRobo v1 ``IKResult`` for a single query.

    ``js_solution`` is a joint-state trajectory of shape ``[1, return_seeds, dof]`` so
    ``js_solution[0]`` indexes like the v1 result.
    """

    success: torch.Tensor
    js_solution: Any
    solution: torch.Tensor | None = None
    position_error: torch.Tensor | None = None
    rotation_error: torch.Tensor | None = None
    solve_time: float = 0.0
    goalset_index: torch.Tensor | None = None
    seed: torch.Tensor | None = None
    debug_info: dict[str, Any] = field(default_factory=dict)
