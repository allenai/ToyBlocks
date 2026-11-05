"""
Vectorized PDDL goal checks for :class:`~blocksuite.tasks.base_env.BaseEnv` subclasses.

The TAMP predicates under ``predicates/`` evaluate against ``IsaacLabLowLevelState`` and
often return a single bool. Generated Isaac Lab envs are **batched** (``num_envs``) and
expose poses via ``RigidObject.data`` / :meth:`blocksuite.tasks.base_env.BaseEnv.get_pose`.
This module ports the **relational geometry** of those predicates to per-env tensors so
``generate_task_from_pddl`` can emit a single dispatch pattern for arbitrary conjunctions.

Predicates that consult motion planners, IK, or finger contact (``IsValid*``, ``at`` with
bound poses, ``Grasped``) are **not** representable as pure tensor functions of world state;
they are intentionally omitted from :data:`SUPPORTED_GOAL_PREDICATES`.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Sequence

import torch

from blocksuite.tamp.blocks.goal_predicates import SUPPORTED_GOAL_PREDICATES

if TYPE_CHECKING:
    from blocksuite.tasks.base_env import BaseEnv

PDDL_LOGICAL_CONNECTIVES = frozenset({"and", "or", "not"})
_PDDL_QUANTIFIER_NOISE = frozenset({"=", "exists", "forall"})


def filter_goal_atoms(goal_preds: list[dict]) -> list[dict]:
    """Drop PDDL connectives if present as pseudo-literals; keep concrete goal atoms.

    Preserves the ``negated`` field added by the negation-aware goal parser so
    that callers can emit ``~tensor`` for negated conjuncts.
    """
    out: list[dict] = []
    for p in goal_preds:
        name = (p.get("name") or "").strip()
        if not name or name.lower() in PDDL_LOGICAL_CONNECTIVES:
            continue
        if name.lower() in _PDDL_QUANTIFIER_NOISE:
            continue
        args = p.get("args", [])
        if any(a.startswith("?") for a in args):
            continue
        out.append(dict(p))
    return out


# --- Constants aligned with src/blocksuite/tamp/blocks/predicates/*.py (cube width 0.045) ---
_CUBE = 0.045
_ON_XY_TOL = 0.02
_ON_DZ_MAX = 0.06
_ON_LRFB_OFFSET = 0.02  # on_left, on_right, on_back
_ON_FRONT_OFFSET = 0.005
_ADJACENCY_GAP = 0.005  # predicates/next_to.py ADJACENCY_GAP, on_right's gap
_ON_TABLE_TOL = 0.02  # tolerance only; table/block extents are read from spawned geometry
# predicates/ori_of.py check_face_up → check_quat(..., yaw_tol_deg=..., axis_tol_deg=...)
_ORI_OF_YAW_TOL_DEG = 40.0
_ORI_OF_AXIS_TOL_DEG = 5.0

# --- Success gating: scene must be settled and the gripper released -----------
# Goal success is otherwise pure relative pose, so a block still held in the
# gripper above the target satisfies "on". These thresholds require the scene to
# be at rest and the gripper open before any goal counts as achieved. Shared with
# the recovering oracle (see eval/oracle/recovering.py) so the two stay aligned.
# PhysX reports a steady few cm/s of linear speed on still, stacked blocks, growing
# with height in the stack (up to ~0.06 m/s on the top of a 3-stack), while carried
# blocks move at 0.15+ m/s; the linear limit sits between the two.
_SETTLED_LIN_VEL = 0.1  # m/s: max per-object linear speed to count as "settled"
_SETTLED_ANG_VEL = 0.5  # rad/s: max per-object angular speed to count as "settled"
_GRIPPER_OPEN_BELOW = 0.25  # finger_joint pos below this = open (matches sim_adapter.py)
_GRIPPER_JOINT_NAME = "finger_joint"
# When True on ``env``, success gates keep the velocity check but skip gripper-open
# (flip/ori_of evals often finish while the block is still grasped).
_SKIP_GRIPPER_OPEN_ATTR = "_success_gate_skip_gripper_open"

# Copy of tmp3.Utils.orientations.ORI_QUATERNIONS (w, x, y, z) for ori_of goals.
ORI_QUATERNIONS: dict[str, tuple[float, float, float, float]] = {
    "o1": (1.0, 0.0, 0.0, 0.0),
    "o2": (0.7071, 0, 0, -0.7071),
    "o3": (0.0, 0.0, 0.0, -1.0),
    "o4": (0.7071, 0, 0, 0.7071),
    "o5": (0.7071, 0, -0.7071, 0),
    "o6": (0.5, -0.5, -0.5, -0.5),
    "o7": (0, 0.7071, 0, 0.7071),
    "o8": (0.5, 0.5, -0.5, 0.5),
    "o9": (0.0, 0.0, 1.0, 0.0),
    "o10": (0, 0.7071, 0.7071, 0),
    "o11": (0.0, -1.0, 0.0, 0.0),
    "o12": (0.0, -0.7071, 0.7071, 0.0),
    "o13": (0.7071, 0.0, 0.7071, 0.0),
    "o14": (0.5, 0.5, 0.5, -0.5),
    "o15": (0, -0.7071, 0, 0.7071),
    "o16": (0.5, -0.5, 0.5, 0.5),
    "o17": (0.5, 0.5, -0.5, -0.5),
    "o18": (0, 0, 0.7071, 0.7071),
    "o19": (0.5, 0.5, 0.5, 0.5),
    "o20": (0.7071, 0.7071, 0, 0),
    "o21": (0.5, -0.5, 0.5, -0.5),
    "o22": (0, 0, 0.7071, -0.7071),
    "o23": (0.5, -0.5, -0.5, 0.5),
    "o24": (0.7071, -0.7071, 0, 0),
}


def _expand_pose_to_envs(env: BaseEnv, pose: torch.Tensor) -> torch.Tensor:
    """Ensure pose is (num_envs, 7)."""
    if pose.ndim == 1:
        return pose.unsqueeze(0).expand(env.num_envs, -1)
    if pose.shape[0] == 1 and env.num_envs > 1:
        return pose.expand(env.num_envs, -1)
    return pose


def resolve_root_pose_w(env: BaseEnv, name: str) -> torch.Tensor:
    """
    World-frame root pose (pos + quat), shape ``(num_envs, 7)``.
    Table PDDL names (``table1``, …) fall back to scene entity ``table`` when present.
    """
    candidates = [name]
    if name.lower().startswith("table") and name != "table":
        candidates.append("table")

    last_err: Exception | None = None
    for key in candidates:
        try:
            pose = env.get_pose(key)
            return _expand_pose_to_envs(env, pose)
        except (KeyError, ValueError) as e:
            last_err = e
            continue
    raise KeyError(
        f"Goal object {name!r} not found in scene (tried {candidates}). "
        f"Last error: {last_err}"
    ) from last_err


def resolve_asset(env: BaseEnv, name: str):
    """Return the scene ``RigidObject`` for a PDDL object name.

    Mirrors the ``table``/``table1`` alias fallback in :func:`resolve_root_pose_w`
    so callers that need the asset itself (e.g. for world-frame corner points)
    resolve the same entity the pose lookup would.
    """
    candidates = [name]
    if name.lower().startswith("table") and name != "table":
        candidates.append("table")

    last_err: Exception | None = None
    for key in candidates:
        try:
            return env.scene[key]
        except KeyError as e:
            last_err = e
            continue
    raise KeyError(
        f"Goal object {name!r} not found in scene (tried {candidates}). "
        f"Last error: {last_err}"
    ) from last_err


def resolve_root_vel_w(env: BaseEnv, name: str) -> torch.Tensor:
    """World-frame root velocity (lin + ang), shape ``(num_envs, 6)``.

    Mirrors :func:`resolve_root_pose_w` but reads ``root_state_w[:, 7:13]``. Only
    magnitudes are used downstream, so the world frame is sufficient.
    """
    return env.scene[name].data.root_state_w[:, 7:13]


def set_success_gate_skip_gripper_open(env: Any, skip: bool = True) -> None:
    """Opt out of the gripper-open limb of :func:`scene_settled_and_gripper_open`.

    Autogen envs call that helper from generated ``_compute_intermediate_values``;
    flipping this attribute relaxes success for eval runs (e.g. flip tasks that
    end with the block still grasped) without regenerating the task package.
    Opt-in only; datagen and ``run_eval`` leave the default (gripper must be open).
    """
    setattr(env, _SKIP_GRIPPER_OPEN_ATTR, bool(skip))


# Yaw tolerance (deg) of ``ori_of`` goal conjuncts on ``env`` (eval opt-in).
_GOAL_ORI_OF_YAW_TOL_ATTR = "_goal_ori_of_yaw_tol_deg"


def set_goal_ori_of_yaw_tol_deg(env: Any, yaw_tol_deg: float = _ORI_OF_YAW_TOL_DEG) -> None:
    """Override the yaw tolerance of every ``ori_of`` goal conjunct (default
    ``_ORI_OF_YAW_TOL_DEG``), e.g. to relax goal success on block rotation. The
    tilt tolerance is unchanged. Opt-in only; datagen leaves the default.
    """
    setattr(env, _GOAL_ORI_OF_YAW_TOL_ATTR, float(yaw_tol_deg))


# When True on ``env``, an ``ori_of`` goal on a glyph that reads the same upside
# down also passes turned 180 degrees about world x (eval opt-in).
_SYMMETRIC_GLYPH_ORI_OF_ATTR = "_symmetric_glyph_ori_of"
# block_h / block_i / block_o and their copies (block_o_2); not the legacy block_o1 / block_o2.
_UPSIDE_DOWN_SYMMETRIC_GLYPH = re.compile(r"^block_[hio](?:_\d+)?$")


def set_symmetric_glyph_ori_of(env: Any, enable: bool = True) -> None:
    """Let H, I and O blocks meet an ``ori_of`` goal in its canonical orientation
    or that orientation turned 180 degrees about world x, the axis toward the
    front camera, since those glyphs read the same upside down. Opt-in only;
    datagen leaves the default.
    """
    setattr(env, _SYMMETRIC_GLYPH_ORI_OF_ATTR, bool(enable))


def _goal_ori_of_upside_down(
    env: BaseEnv, args: tuple[str, ...], yaw_tol_deg: float = _ORI_OF_YAW_TOL_DEG
) -> torch.Tensor:
    """(ori_of block face_name) with the face turned 180 degrees about world x."""
    obj_n, face = args[0], args[1]
    if face not in ORI_QUATERNIONS:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    q_cur = resolve_root_pose_w(env, obj_n)[:, 3:7]
    w, x, y, z = ORI_QUATERNIONS[face]
    # (0, 1, 0, 0) * (w, x, y, z): R_x(180) applied in the world frame.
    exp = torch.tensor((-x, w, -z, y), device=env.device, dtype=q_cur.dtype).expand_as(q_cur)
    return _quat_close_batched(q_cur, exp, yaw_tol_deg=yaw_tol_deg)


_FRONT_CAMERA_SENSOR = "camera:front_camera"


def front_camera_pixels(env: BaseEnv, points_w: torch.Tensor) -> torch.Tensor:
    """Project world points ``(num_envs, 3)`` into each env's front camera, ``(num_envs, 2)``.

    The sensor's ``quat_w_world`` uses the world camera convention (x forward,
    z up), so image u grows with -y and v with -z of the camera-frame point.
    """
    from isaaclab.utils import math as math_utils

    cam = env.scene.sensors[_FRONT_CAMERA_SENSOR].data
    p = math_utils.quat_apply_inverse(cam.quat_w_world, points_w - cam.pos_w)
    k = cam.intrinsic_matrices
    u = k[:, 0, 0] * (-p[:, 1]) / p[:, 0] + k[:, 0, 2]
    v = k[:, 1, 1] * (-p[:, 2]) / p[:, 0] + k[:, 1, 2]
    return torch.stack([u, v], dim=-1)

class GoalSettleTerms(NamedTuple):
    """Inputs and limbs of the success settle gate, per env.

    ``lin`` / ``ang``: ``(num_envs, len(object_names))`` world-frame linear (m/s)
    and angular (rad/s) speed of each checked object, in ``object_names`` order.
    ``finger``: ``(num_envs,)`` ``finger_joint`` position; NaN when the
    gripper-open limb is skipped, since the joint is then not read.
    ``settled`` / ``gripper_open``: ``(num_envs,)`` bool limbs of the gate.
    """

    lin: torch.Tensor
    ang: torch.Tensor
    finger: torch.Tensor
    settled: torch.Tensor
    gripper_open: torch.Tensor


def goal_settle_terms(env: BaseEnv, object_names: Sequence[str]) -> GoalSettleTerms:
    """Speeds, finger position and limbs behind :func:`scene_settled_and_gripper_open`.

    ``settled & gripper_open`` is exactly that gate; the raw terms are returned so
    eval can log what the gate saw. With ``_success_gate_skip_gripper_open`` set,
    ``gripper_open`` is all True and the finger joint is not read.
    """
    if not object_names:
        raise ValueError(
            "scene_settled_and_gripper_open needs at least one object to check; an "
            "empty set would let success fire while the goal blocks are in motion."
        )
    lin_list: list[torch.Tensor] = []
    ang_list: list[torch.Tensor] = []
    for obj_name in object_names:
        vel = resolve_root_vel_w(env, obj_name)
        lin_list.append(torch.linalg.norm(vel[:, :3], dim=-1))
        ang_list.append(torch.linalg.norm(vel[:, 3:6], dim=-1))
    lin = torch.stack(lin_list, dim=-1)
    ang = torch.stack(ang_list, dim=-1)
    settled = ((lin <= _SETTLED_LIN_VEL) & (ang <= _SETTLED_ANG_VEL)).all(dim=-1)

    if getattr(env, _SKIP_GRIPPER_OPEN_ATTR, False):
        finger = torch.full((env.num_envs,), float("nan"), device=env.device)
        gripper_open = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    else:
        art = env.robot.articulation
        ids, _ = art.find_joints([_GRIPPER_JOINT_NAME])
        finger = art.data.joint_pos[:, ids[0]]
        gripper_open = finger < _GRIPPER_OPEN_BELOW
    return GoalSettleTerms(lin=lin, ang=ang, finger=finger, settled=settled, gripper_open=gripper_open)


def scene_settled_and_gripper_open(env: BaseEnv, object_names: Sequence[str]) -> torch.Tensor:
    """True per env when ``object_names`` are at rest and the gripper is open.

    Applied to a whole goal conjunction (not inside ``goal_on``) so it also gates
    the sibling stacking/orientation goals and does not corrupt ``goal_clear``
    (which composes ``~goal_on``). Returns a bool tensor ``(num_envs,)``.

    When ``env`` has ``_success_gate_skip_gripper_open`` set (see
    :func:`set_success_gate_skip_gripper_open`), only the velocity/settle check
    runs. Stacking goals should leave the default (gripper must be open).

    ``object_names`` is the set of blocks the goal actually references (codegen
    derives it per goal via ``goal_objects_from_conjuncts``). Checking *every*
    ``scene.rigid_objects`` instead would include the table and distractors
    parked off-table, so one unrelated rolling body would make success
    unreachable and bias the run toward timeout failures. Required, not optional:
    an autogen package generated before this signature change calls it with one
    argument and fails loudly, which is the signal to regenerate.
    """
    terms = goal_settle_terms(env, object_names)
    return terms.settled & terms.gripper_open


def _quat_close_batched(
    q_cur: torch.Tensor,
    q_tgt: torch.Tensor,
    *,
    yaw_tol_deg: float = _ORI_OF_YAW_TOL_DEG,
    axis_tol_deg: float = _ORI_OF_AXIS_TOL_DEG,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Same geometry as tmp3.quat_utils.check_quat; batched (N, 4) quaternions wxyz."""
    from isaaclab.utils import math as math_utils

    q_cur = q_cur / (q_cur.norm(dim=-1, keepdim=True) + eps)
    q_tgt = q_tgt / (q_tgt.norm(dim=-1, keepdim=True) + eps)
    q_err = math_utils.quat_mul(q_cur, math_utils.quat_inv(q_tgt))
    q_err = math_utils.quat_unique(q_err)
    aa = math_utils.axis_angle_from_quat(q_err)
    angle = aa.norm(dim=-1)
    axis = aa / (angle.unsqueeze(-1) + eps)
    z = torch.tensor([0.0, 0.0, 1.0], device=q_err.device, dtype=q_err.dtype).expand_as(axis)
    z_rot = math_utils.quat_apply(q_err, z)
    z_dot = (z_rot * z).sum(dim=-1).clamp(-1.0, 1.0)
    tilt = torch.acos(z_dot)
    tilt_ok = tilt <= torch.deg2rad(
        torch.tensor(axis_tol_deg, device=tilt.device, dtype=tilt.dtype)
    )
    yaw_ok = angle <= torch.deg2rad(torch.tensor(yaw_tol_deg, device=angle.device, dtype=angle.dtype))
    return tilt_ok & yaw_ok


def goal_on(env: BaseEnv, args: tuple[str, ...]) -> torch.Tensor:
    top, bot = args[0], args[1]
    p_top = resolve_root_pose_w(env, top)[:, :3]
    p_bot = resolve_root_pose_w(env, bot)[:, :3]
    dx = torch.abs(p_top[:, 0] - p_bot[:, 0])
    dy = torch.abs(p_top[:, 1] - p_bot[:, 1])
    dz = p_top[:, 2] - p_bot[:, 2]
    return (dx <= _ON_XY_TOL) & (dy <= _ON_XY_TOL) & (dz > 0.0) & (dz <= _ON_DZ_MAX)


def goal_on_table(env: BaseEnv, args: tuple[str, ...]) -> torch.Tensor:
    """Block rests on the table's top surface, within the table's XY footprint.

    Reads the actual spawned geometry via orientation-aware world-frame corner
    points (``cube_corner_points``), so it is correct for non-centered root
    frames (YCB/USD assets) and per-env ``cube_scale`` rather than assuming
    hardcoded slab/block heights.
    """
    from blocksuite.utils.obs_utils import cube_corner_points  # local import (matches next_to.py)

    block_n, table_n = args[0], args[1]
    block = resolve_asset(env, block_n)
    table = resolve_asset(env, table_n)

    tc = cube_corner_points(table)  # (num_envs, 8, 3) world frame
    bc = cube_corner_points(block)
    table_top_z = tc[..., 2].amax(dim=1)
    block_bottom_z = bc[..., 2].amin(dim=1)
    tmin = tc.amin(dim=1)  # world x/y/z lower corner
    tmax = tc.amax(dim=1)  # world x/y/z upper corner

    block_pos = resolve_root_pose_w(env, block_n)[:, :3]
    bx, by = block_pos[:, 0], block_pos[:, 1]
    within_x = (bx >= tmin[:, 0] - _ON_TABLE_TOL) & (bx <= tmax[:, 0] + _ON_TABLE_TOL)
    within_y = (by >= tmin[:, 1] - _ON_TABLE_TOL) & (by <= tmax[:, 1] + _ON_TABLE_TOL)
    on_surface = (block_bottom_z - table_top_z).abs() <= _ON_TABLE_TOL
    return within_x & within_y & on_surface


def _goal_pair_axis_aligned(
    env: BaseEnv,
    args: tuple[str, ...],
    *,
    axis_sep: int,
    align_axes: tuple[int, int],
    distance: torch.Tensor,
    sep_atol: float,
    align_atols: tuple[float, float],
) -> torch.Tensor:
    """Helper: separation along one axis, alignment on two others (indices 0,1,2 = x,y,z)."""
    a, b = args[0], args[1]
    pa = resolve_root_pose_w(env, a)[:, :3]
    pb = resolve_root_pose_w(env, b)[:, :3]
    d = torch.abs(pa[:, axis_sep] - pb[:, axis_sep])
    sep_ok = torch.isclose(d, distance.expand_as(d), atol=sep_atol)
    i0, i1 = align_axes
    al0 = torch.isclose(pa[:, i0], pb[:, i0], atol=align_atols[0])
    al1 = torch.isclose(pa[:, i1], pb[:, i1], atol=align_atols[1])
    return sep_ok & al0 & al1


def _bbox_extents(env: BaseEnv, name: str) -> torch.Tensor:
    """Per-env body-frame bounding-box extents ``(num_envs, 3)`` of the spawned asset, as the
    TAMP predicates read them (``predicates/next_to._get_obj_sizes``)."""
    from blocksuite.utils.obs_utils import get_bounding_box  # local import (matches next_to.py)

    bbox = get_bounding_box(resolve_asset(env, name))
    return (bbox[:, 1] - bbox[:, 0]).to(device=env.device, dtype=torch.float32)


def goal_on_left(env: BaseEnv, args: tuple[str, ...]) -> torch.Tensor:
    dist = torch.tensor((_CUBE + _CUBE) / 2.0 + _ON_LRFB_OFFSET, device=env.device, dtype=torch.float32)
    return _goal_pair_axis_aligned(env, args, axis_sep=1, align_axes=(0, 2), distance=dist, sep_atol=0.01, align_atols=(0.04, 0.01))


def goal_on_right(env: BaseEnv, args: tuple[str, ...]) -> torch.Tensor:
    """(on_right a b) — mirrors predicates/on_right.py: |dy| = half(a) + half(b) + 5 mm gap
    (spawned bounding boxes, the spacing on_right is sampled and pushed to) within 0.03,
    x within 0.04, z within 0.02. A fixed 6.5 cm (+-1 cm) like on_left would reject every
    on_right a 4 cm block is sampled or pushed into (4.5 cm apart)."""
    dist = (_bbox_extents(env, args[0])[:, 1] + _bbox_extents(env, args[1])[:, 1]) / 2.0 + _ADJACENCY_GAP
    return _goal_pair_axis_aligned(env, args, axis_sep=1, align_axes=(0, 2), distance=dist, sep_atol=0.03, align_atols=(0.04, 0.02))


def goal_on_back(env: BaseEnv, args: tuple[str, ...]) -> torch.Tensor:
    dist = torch.tensor((_CUBE + _CUBE) / 2.0 + _ON_LRFB_OFFSET, device=env.device, dtype=torch.float32)
    return _goal_pair_axis_aligned(env, args, axis_sep=0, align_axes=(1, 2), distance=dist, sep_atol=0.01, align_atols=(0.04, 0.01))


def goal_on_front(env: BaseEnv, args: tuple[str, ...]) -> torch.Tensor:
    dist = torch.tensor((_CUBE + _CUBE) / 2.0 + _ON_FRONT_OFFSET, device=env.device, dtype=torch.float32)
    return _goal_pair_axis_aligned(env, args, axis_sep=0, align_axes=(1, 2), distance=dist, sep_atol=0.01, align_atols=(0.04, 0.01))


def goal_next_to(env: BaseEnv, args: tuple[str, ...]) -> torch.Tensor:
    """(next_to a b) — a is cardinally adjacent to b along +/-x or +/-y.

    Mirrors predicates/next_to.py ``evaluate_vectorized``: center separation along one axis is
    half(a)+half(b) of the spawned bounding boxes within atol, with alignment on the perpendicular
    axis and z. Adjacency in *either* cardinal axis satisfies the goal.
    """
    half_sum = _bbox_extents(env, args[0]) / 2.0 + _bbox_extents(env, args[1]) / 2.0  # (num_envs, 3)
    x_adj = _goal_pair_axis_aligned(env, args, axis_sep=0, align_axes=(1, 2), distance=half_sum[:, 0], sep_atol=0.03, align_atols=(0.04, 0.02))
    y_adj = _goal_pair_axis_aligned(env, args, axis_sep=1, align_axes=(0, 2), distance=half_sum[:, 1], sep_atol=0.03, align_atols=(0.04, 0.02))
    return x_adj | y_adj


def goal_ori_of(
    env: BaseEnv, args: tuple[str, ...], yaw_tol_deg: float = _ORI_OF_YAW_TOL_DEG
) -> torch.Tensor:
    """(ori_of block face_name) — same tolerances as predicates/ori_of.py check_face_up."""
    obj_n, face = args[0], args[1]
    if face not in ORI_QUATERNIONS:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    pose = resolve_root_pose_w(env, obj_n)
    q_cur = pose[:, 3:7]
    exp = torch.tensor(ORI_QUATERNIONS[face], device=env.device, dtype=q_cur.dtype).expand_as(q_cur)
    return _quat_close_batched(q_cur, exp, yaw_tol_deg=yaw_tol_deg)


_PICK_OBSTRUCTS_XY_RANGE = 0.1   # blocks within this XY distance obstruct each other
_PICK_OBSTRUCTS_Z_ATOL = 0.02


def goal_clear(env: BaseEnv, args: tuple[str, ...]) -> torch.Tensor:
    """True when no rigid object in the scene is stacked on top of ``args[0]``."""
    target = args[0]
    result = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for obj_name in env.scene.rigid_objects:
        if obj_name == target:
            continue
        result = result & ~goal_on(env, (obj_name, target))
    return result


def goal_pick_obstructs(env: BaseEnv, args: tuple[str, ...]) -> torch.Tensor:
    """True when ``args[0]`` is within ``_PICK_OBSTRUCTS_XY_RANGE`` (0.1 m) of
    ``args[1]`` in XY and at the same table height.
    """
    b1, b2 = args[0], args[1]
    p1 = resolve_root_pose_w(env, b1)[:, :3]
    p2 = resolve_root_pose_w(env, b2)[:, :3]
    xy_dist = torch.sqrt((p1[:, 0] - p2[:, 0]) ** 2 + (p1[:, 1] - p2[:, 1]) ** 2)
    dz = torch.abs(p1[:, 2] - p2[:, 2])
    return (xy_dist <= _PICK_OBSTRUCTS_XY_RANGE) & (dz <= _PICK_OBSTRUCTS_Z_ATOL)


# ``BaseEnv`` is only imported under TYPE_CHECKING; use ``Any`` here because this RHS is evaluated at import time.
TensorGoalFn = Callable[[Any, tuple[str, ...]], torch.Tensor]

TENSOR_GOAL_DISPATCH: dict[str, TensorGoalFn] = {
    "on": goal_on,
    "on_table": goal_on_table,
    "on_left": goal_on_left,
    "on_right": goal_on_right,
    "on_front": goal_on_front,
    "on_back": goal_on_back,
    "next_to": goal_next_to,
    "ori_of": goal_ori_of,
    "clear": goal_clear,
    "pick_obstructs": goal_pick_obstructs,
}

# Keep the import-light predicate registry and the actual tensor dispatch in
# lock-step: a predicate listed as supported must have an implementation here,
# and vice versa.
assert frozenset(TENSOR_GOAL_DISPATCH.keys()) == SUPPORTED_GOAL_PREDICATES, (
    "goal_predicates.SUPPORTED_GOAL_PREDICATES is out of sync with "
    f"TENSOR_GOAL_DISPATCH: dispatch={sorted(TENSOR_GOAL_DISPATCH)} "
    f"supported={sorted(SUPPORTED_GOAL_PREDICATES)}"
)


def batched_goal_conjunct(env: BaseEnv, pred_name: str, args: tuple[str, ...]) -> torch.Tensor:
    """Evaluate one goal literal; returns bool tensor ``(num_envs,)``."""
    key = pred_name.lower()
    fn = TENSOR_GOAL_DISPATCH.get(key)
    if fn is None:
        raise ValueError(
            f"Unsupported goal predicate {pred_name!r}. "
            f"Vectorized goal checks exist for: {sorted(SUPPORTED_GOAL_PREDICATES)}. "
            f"Predicates that depend on IK, motion validation, or contact (IsValid*, at with poses, Grasped) "
            f"are not expressible as pure world-state tensors here."
        )
    if key == "ori_of":
        yaw_tol_deg = getattr(env, _GOAL_ORI_OF_YAW_TOL_ATTR, _ORI_OF_YAW_TOL_DEG)
        ok = goal_ori_of(env, args, yaw_tol_deg=yaw_tol_deg)
        if getattr(env, _SYMMETRIC_GLYPH_ORI_OF_ATTR, False) and _UPSIDE_DOWN_SYMMETRIC_GLYPH.match(args[0]):
            ok = ok | _goal_ori_of_upside_down(env, args, yaw_tol_deg=yaw_tol_deg)
        return ok
    return fn(env, args)


def assert_goal_atoms_supported(goal_atoms: list[dict]) -> None:
    """Raise before writing env.py if the PDDL goal uses an unsupported literal."""
    bad: list[str] = []
    for p in goal_atoms:
        n = (p.get("name") or "").lower()
        if n in PDDL_LOGICAL_CONNECTIVES:
            continue
        if n not in SUPPORTED_GOAL_PREDICATES:
            bad.append(f"({p.get('name')} {' '.join(p.get('args', []))})")
    if bad:
        raise ValueError(
            "PDDL goal contains literals without vectorized goal checks:\n  "
            + "\n  ".join(bad)
            + f"\nSupported: {sorted(SUPPORTED_GOAL_PREDICATES)}"
        )
