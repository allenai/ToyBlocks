"""Episode-queue HTTP client for queue-driven evaluation.

:class:`EpisodeQueueClient` is attached to ``TemplateEnv.queue_client``;
during ``_reset_idx`` the env calls :meth:`EpisodeQueueClient.fetch_reset_states`
to pull one setup per reset env from the HTTP queue server
(``scripts/serve_episode_queue.py``), then calls :func:`apply_setup` directly
to override sim state.  Setup dict keys (all optional): ``episode_id``,
``pddl_goal``, ``robot_joint_positions``, ``robot_gripper_position``,
``objects[{name}] = {"position": [x,y,z], "quaternion": [w,x,y,z]}``
(positions are local-frame; ``env_origins`` is added inside ``apply_setup``),
``table_pose`` (same local-frame shape; restores the datagen ``workspace_table``
height so recorded block Z lands correctly) or ``table_height_delta``
(recorded offset added to the preset table baseline, which is anchored by the
dataset's recorded ``scene_id``; see ``_apply_table_height``). An optional
``BaseEnvCfg.eval_table_z_offset`` (world Z, metres) is applied after that and
shifts block local Z by the same amount so blocks stay on the table,
``camera_intrinsics`` (per-camera first-frame K; applied per env so
randomized-camera rerenders keep the matching lens),
and ``pddl_problem_file`` (path to a full PDDL problem; read by
:class:`toyblocks.eval.oracle.OracleHLController` when ``--use_oracle`` is set,
overriding the run-wide ``--pddl_problem_file``).  Unknown keys are ignored by
:func:`apply_setup`.
The policy prompt is :func:`~toyblocks.utils.prompt_resolver.parse_prompt_for_setup`
on the setup: ``hl_prompt`` (parquet ``goal.hl_prompt``) when present, else
English from ``pddl_goal``.

DEBUG logging on this module dumps every write + post-step verification
diff (``# VERBOSE-QUEUE`` markers); ``scripts/evaluate.py`` enables it
automatically when ``--queue_url`` is set.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Callable, Optional

import requests
import torch

if TYPE_CHECKING:
    from toyblocks.tasks.base_env import BaseEnv

logger = logging.getLogger(__name__)

# PhysX may relax sub-mm / sub-degree pose violations on the first step;
# anything above this triggers a WARNING in verify_after_step.
_DEFAULT_VERIFY_TOL = 1e-2


def fetch_next_setup(
    queue_url: str,
    timeout: float = 30.0,
    slot: Optional[int] = None,
    slot_scales: Optional[Sequence[float]] = None,
) -> Optional[dict]:
    """GET ``queue_url`` and return the next setup dict (or ``None`` when empty).

    ``slot`` (when given) is sent as a ``?slot=`` query param so a cube-scale
    slot-aware queue can return only episodes whose recorded block size matches
    this env's spawned size (see ``scripts/serve_episode_queue.py``). A FIFO
    queue ignores it.

    ``slot_scales`` is the cube-scale spawn-slot *ordering* the task package was
    generated with (empty for tasks with no per-episode size matching). It is sent
    alongside so the queue can compare it against its own: both sides derive the
    ordering from ``distinct_cube_scales`` over their respective episode sets, so
    any difference means slot ``i`` denotes a different block size on each side.
    Sending the values rather than just how many there are is what catches two
    episode sets of equal size but different scales.
    """
    params: Optional[dict] = None
    if slot is not None:
        params = {"slot": slot}
        if slot_scales:
            params["slot_scales"] = ",".join(str(float(s)) for s in slot_scales)
    response = requests.get(queue_url, timeout=timeout, params=params)
    response.raise_for_status()
    return response.json().get("setup")


def filter_setup_to_scene(setup: dict, scene_obj_names: Sequence[str]) -> dict:
    """Narrow ``setup["objects"]`` to the objects this scene actually has.

    A dataset recorded against a richer scene carries poses for objects a smaller scene
    does not spawn; :func:`apply_setup` raises on any such name, so the setup is filtered
    first.  Objects the scene has but the setup omits are left alone -- they keep their
    post-``_reset_idx`` value, as :func:`apply_setup` documents.

    Copied from ``origin/bennedh/demo-2`` (commit 5834afa5) on 2026-09-02 so that
    ``agentic_robotics``'s ``ToyBlocksEnv`` imports against this branch too.

    :param setup: Setup dict as built by ``toyblocks.utils.episode_dataset.load_setups``
    :param scene_obj_names: Names of the scene's rigid objects
    :return: ``setup`` with unknown object poses removed (the input is not mutated)
    """
    objects = setup.get("objects")
    if not objects:
        return setup

    known = set(scene_obj_names)
    dropped = sorted(name for name in objects if name not in known)
    if dropped:
        logger.info(
            "filter_setup_to_scene episode_id=%s dropping %d object pose(s) absent from "
            "the scene: %s",
            setup.get("episode_id", "<unknown>"), len(dropped), dropped,
        )

    return {**setup, "objects": {n: p for n, p in objects.items() if n in known}}


def apply_setup(
    base_env,
    env_idx: int,
    setup: dict,
    *,
    apply_recorded_joint_velocity: bool = False,
) -> None:
    """Override a single env's robot + object state to match ``setup``.

    Writes via Isaac Lab ``write_*_to_sim``; caller commits (``scene.write_data_to_sim``).
    No physics settle: ``sim.step()`` is global, so settling here during the
    auto-reset path would re-step non-resetting envs. LeRobot first-frame poses
    are already settled. The placed objects are queued for
    :func:`park_placed_blocks`. Joints/objects not listed keep their post-``_reset_idx``
    value; velocities for written joints/objects are zeroed -- except the arm
    joints when ``apply_recorded_joint_velocity`` is set and the setup carries
    ``robot_joint_velocities`` (dataset replay restores the recorded frame-0
    motion; the gripper joint velocity stays zero).

    Raises ``ValueError`` on setup/scene mismatch (extra joints, missing
    gripper joint, unknown object name) — these would otherwise silently produce
    a wrong eval.
    """
    device = base_env.device
    env_ids = torch.tensor([env_idx], device=device, dtype=torch.int64)
    ep_id = setup.get("episode_id", "<unknown>")

    robot = base_env.robot.articulation
    num_joints = robot.data.joint_pos.shape[1]

    # VERBOSE-QUEUE: dump the requested setup so the diff later is obvious.
    logger.debug(
        "apply_setup env=%d episode_id=%s keys=%s num_joints=%d num_scene_objects=%d",
        env_idx, ep_id, sorted(setup.keys()), num_joints,
        len(base_env.scene.rigid_objects),
    )

    full_joint_pos = robot.data.joint_pos[env_idx].clone().unsqueeze(0)
    pre_joint_pos = full_joint_pos.clone()

    if "robot_joint_positions" in setup:
        jp = torch.tensor(setup["robot_joint_positions"], device=device, dtype=torch.float32)
        if jp.numel() > num_joints:
            raise ValueError(
                f"setup {ep_id!r} robot_joint_positions has {jp.numel()} entries "
                f"but the robot only has {num_joints} joints"
            )
        full_joint_pos[:, : jp.numel()] = jp
        logger.debug(
            "  env=%d joint override jp=%s (replacing first %d of %d)",
            env_idx, jp.tolist(), jp.numel(), num_joints,
        )

    # TODO (rmh) - hardcoded 7-DOF; derive from env config
    if "robot_gripper_position" in setup:
        if num_joints <= 7:
            raise ValueError(
                f"setup {ep_id!r} requests robot_gripper_position={setup['robot_gripper_position']} "
                f"but the robot has only {num_joints} joints (no gripper joint at index 7). "
                f"Either drop the gripper field from the dataset or use a robot with a gripper."
            )
        full_joint_pos[:, 7] = float(setup["robot_gripper_position"])
        logger.debug(
            "  env=%d gripper override -> %.4f", env_idx, float(setup["robot_gripper_position"]),
        )

    joint_vel = torch.zeros_like(full_joint_pos)
    recorded_vel = setup.get("robot_joint_velocities")
    if apply_recorded_joint_velocity and recorded_vel is not None:
        jv = torch.tensor(recorded_vel, device=device, dtype=torch.float32)
        n_arm = min(jv.numel(), 7, num_joints)
        joint_vel[:, :n_arm] = jv[:n_arm]
        logger.debug("  env=%d joint velocity override jv=%s", env_idx, jv[:n_arm].tolist())
    robot.write_joint_state_to_sim(full_joint_pos, joint_vel, env_ids=env_ids)
    robot.set_joint_position_target(full_joint_pos, env_ids=env_ids)
    # The velocity *target* stays zero: the controller holds the restored pose
    # while the written state carries the recorded motion.
    robot.set_joint_velocity_target(torch.zeros_like(full_joint_pos), env_ids=env_ids)

    # VERBOSE-QUEUE: write_*_to_sim updates the buffer synchronously; diff should be ~0.
    immediate_jp = robot.data.joint_pos[env_idx].clone()
    write_diff = (immediate_jp.unsqueeze(0) - full_joint_pos).abs().max().item()
    if write_diff > 1e-5:
        logger.warning(
            "  env=%d post-write joint diff=%.6f (expected ~0; data buffer not updated?)",
            env_idx, write_diff,
        )
    else:
        logger.debug(
            "  env=%d post-write joint diff=%.2e (was=%s, now=%s)",
            env_idx, write_diff, pre_joint_pos[0, :7].tolist(), immediate_jp[:7].tolist(),
        )

    # Restore the datagen table before the blocks. Recorded block Z is relative
    # to this table surface, so a mismatched eval table makes blocks float/sink
    # even when XY/orientation match.
    _check_spawned_block_size(base_env, env_idx, setup, ep_id)
    _apply_table_height(base_env, env_idx, env_ids, setup, device, ep_id)
    table_z_extra = _apply_eval_table_z_offset(base_env, env_idx, env_ids, device)
    _apply_recorded_friction(base_env, env_idx, env_ids, setup, ep_id)

    objects = setup.get("objects", {})
    for obj_name, obj_data in objects.items():
        if obj_name not in base_env.scene.rigid_objects:
            raise ValueError(
                f"setup {ep_id!r} references rigid object {obj_name!r} which is "
                f"not in the scene. Scene rigid objects: "
                f"{sorted(base_env.scene.rigid_objects.keys())}. "
                f"This usually means the dataset and the eval task config disagree "
                f"about the scene preset / object names."
            )
        obj = base_env.scene.rigid_objects[obj_name]
        local_pos = torch.tensor(obj_data["position"], device=device, dtype=torch.float32).unsqueeze(0)
        if table_z_extra != 0.0:
            local_pos = local_pos.clone()
            local_pos[0, 2] += table_z_extra
        world_pos = local_pos + base_env.scene.env_origins[env_idx]
        quat = torch.tensor(obj_data["quaternion"], device=device, dtype=torch.float32).unsqueeze(0)
        root_pose = torch.cat([world_pos, quat], dim=-1)
        obj.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=device), env_ids=env_ids)

        # VERBOSE-QUEUE: per-object write check (local frame for readability).
        actual_local = (obj.data.root_pos_w[env_idx] - base_env.scene.env_origins[env_idx]).clone()
        actual_quat = obj.data.root_quat_w[env_idx].clone()
        pos_diff = (actual_local - local_pos[0]).abs().max().item()
        quat_diff = _quat_distance(actual_quat, quat[0])
        logger.debug(
            "  env=%d object=%s local_pos=%s quat=%s post-write pos_diff=%.2e quat_diff=%.2e",
            env_idx, obj_name, local_pos[0].tolist(), quat[0].tolist(),
            pos_diff, quat_diff,
        )

    _park_unused_blocks(base_env, env_idx, env_ids, set(objects.keys()), device)

    pending: dict[int, list[str]] = getattr(base_env, _PENDING_PARK_ATTR, {})
    pending[env_idx] = list(objects)
    setattr(base_env, _PENDING_PARK_ATTR, pending)

    if "pddl_goal" in setup:
        base_env.set_episode_goal(env_idx, setup["pddl_goal"])

    cfg = getattr(base_env, "cfg", None)
    if cfg is not None and getattr(cfg, "use_calibrated_cameras", False):
        from toyblocks.utils.calibrated_camera_intrinsics import (
            apply_calibrated_camera_intrinsics_for_env,
        )

        apply_calibrated_camera_intrinsics_for_env(base_env, env_idx)
    else:
        camera_intrinsics = setup.get("camera_intrinsics")
        if camera_intrinsics:
            from toyblocks.utils.dataset_provenance import apply_camera_intrinsics_for_env

            apply_camera_intrinsics_for_env(base_env, env_idx, camera_intrinsics)

    # Recorded camera pose + appearance (--use_render_params). USD writes only:
    # the reset path renders afterwards (num_rerenders_on_reset). In wave mode the
    # record waits for the wave release, and with several envs the scene-global
    # lighting is left alone so a reset cannot relight envs mid-episode.
    render_replay = getattr(base_env, "render_replay", None)
    if render_replay is not None:
        render_replay.stage(env_idx, setup)
        if not getattr(base_env, "render_replay_wave", False):
            render_replay.apply_staged([env_idx], apply_globals=(base_env.num_envs == 1))

    if getattr(base_env, "replay_block_identity", False) is True:
        _apply_block_identity(base_env, env_idx, setup, ep_id)


# Attribute on ``base_env``: env index -> {"glyphs": blocks whose glyph was changed,
# "colors": {cuboid: authored inputs:diffuseColor}} for every block a glyph/colour
# rerender's identity was painted on, so a later episode can restore the rest.
_BLOCK_IDENTITY_ATTR = "_block_identity_painted"
# Attribute on ``base_env``: the generated env's ``block_specs`` (read once).
_BLOCK_SPECS_ATTR = "_block_identity_block_specs"


def _env_block_specs(base_env) -> dict[str, dict]:
    """The generated env's ``block_specs`` literal, parsed from its source file."""
    specs = getattr(base_env, _BLOCK_SPECS_ATTR, None)
    if specs is None:
        import inspect

        from toyblocks.domain_randomization.interpreter import _block_specs

        env_file = inspect.getsourcefile(type(base_env))
        specs = _block_specs({"env_file": env_file})
        if not specs:
            raise ValueError(
                f"--replay_block_identity: no block_specs literal in {env_file}; only "
                "generated (autogen) envs can have their block identities painted"
            )
        setattr(base_env, _BLOCK_SPECS_ATTR, specs)
    return specs


def _apply_block_identity(base_env, env_idx: int, setup: dict, ep_id: str) -> None:
    """Paint the setup's recorded glyphs/colours on env ``env_idx``'s blocks (USD writes only).

    Blocks painted for an earlier episode that this setup does not name get their
    authored look back. Writes ``block_identity_status`` (``applied`` / ``none``)
    into ``setup``. Raises when a block cannot be painted, since a silently
    unpainted block would contradict the rendered goal the policy is prompted with.
    """
    from toyblocks.domain_randomization import block_colors, glyphs
    from toyblocks.domain_randomization.block_variables import BlockIdentity

    identity = BlockIdentity.from_dict(setup.get("block_identity") or {})
    specs = _env_block_specs(base_env) if identity.glyphs or identity.colors else {}
    for name in sorted(set(identity.glyphs) | set(identity.colors)):
        if name not in base_env.scene.rigid_objects:
            raise ValueError(
                f"setup {ep_id!r} records a glyph/colour for block {name!r}, which is not "
                f"in the scene ({sorted(base_env.scene.rigid_objects)})"
            )
    for name in sorted(identity.glyphs):
        usd_path = (specs.get(name) or {}).get("usd_path")
        if not glyphs.library_supported(usd_path):
            raise ValueError(
                f"setup {ep_id!r}: block {name!r} spawns {usd_path!r}, but recorded glyphs "
                f"can only be painted on {glyphs.SUPPORTED_LIBRARY!r} blocks"
            )
    for name in sorted(identity.colors):
        if not block_colors.is_cuboid(specs.get(name)):
            raise ValueError(f"setup {ep_id!r}: block {name!r} has a recorded colour but is not a cuboid")

    painted: dict[int, dict] = getattr(base_env, _BLOCK_IDENTITY_ATTR, {})
    env_state = painted.setdefault(env_idx, {"glyphs": set(), "colors": {}})

    # Glyphs: the shader input is a pure function of (glyph, authored glyph), so the
    # authored glyph itself restores a block exactly.
    want_glyphs = dict(identity.glyphs)
    for name in env_state["glyphs"] - set(want_glyphs):
        want_glyphs[name] = glyphs.authored_glyph(name)
    if want_glyphs:
        applied = glyphs.apply_wave_glyphs({env_idx: want_glyphs}, verbose=False).get(env_idx, {})
        missed = sorted(set(want_glyphs) - set(applied))
        if missed:
            raise RuntimeError(
                f"setup {ep_id!r}: env {env_idx} could not set the glyphs of {missed} (no "
                "decals shader, or no authored glyph to offset from)"
            )
    env_state["glyphs"] = set(identity.glyphs)

    # Colours: snapshot each cuboid's authored colour the first time it is painted.
    for name in sorted(identity.colors):
        if name in env_state["colors"]:
            continue
        shader = block_colors.find_surface_shader(env_idx, name)
        attr = shader.GetAttribute("inputs:diffuseColor") if shader is not None else None
        authored = attr.Get() if attr else None
        if authored is None:
            raise RuntimeError(
                f"setup {ep_id!r}: env {env_idx} block {name!r} has no authored "
                "inputs:diffuseColor to restore later"
            )
        env_state["colors"][name] = authored
    if identity.colors:
        applied = block_colors.apply_wave_block_colors(
            {env_idx: dict(identity.colors)}, identity.palette, verbose=False
        ).get(env_idx, {})
        missed = sorted(set(identity.colors) - set(applied))
        if missed:
            raise RuntimeError(
                f"setup {ep_id!r}: env {env_idx} could not recolour {missed} (no surface "
                "shader, or a colour missing from the recorded palette)"
            )
    for name, authored in env_state["colors"].items():
        if name not in identity.colors:
            block_colors.find_surface_shader(env_idx, name).GetAttribute(
                "inputs:diffuseColor"
            ).Set(authored)
    setattr(base_env, _BLOCK_IDENTITY_ATTR, painted)

    setup["block_identity_status"] = "applied" if identity.glyphs or identity.colors else "none"
    logger.debug(
        "  env=%d block identity %s: %d glyph(s), %d colour(s)",
        env_idx, setup["block_identity_status"], len(identity.glyphs), len(identity.colors),
    )


_TABLE_OBJ_NAME = "workspace_table"


def _eval_table_z_offset_m(base_env) -> float:
    cfg = getattr(base_env, "cfg", None)
    if cfg is None:
        return 0.0
    return float(getattr(cfg, "eval_table_z_offset", 0.0) or 0.0)


def _apply_eval_table_z_offset(base_env, env_idx: int, env_ids, device) -> float:
    """Shift table root Z (and return the offset for block poses)."""
    dz = _eval_table_z_offset_m(base_env)
    if abs(dz) < 1e-9:
        return 0.0
    if _TABLE_OBJ_NAME not in base_env.scene.rigid_objects:
        return dz
    table = base_env.scene.rigid_objects[_TABLE_OBJ_NAME]
    pose = table.data.root_pose_w[env_idx].clone().unsqueeze(0)
    before_z = float(pose[0, 2].item())
    pose[0, 2] += dz
    _record_expected_table_z(base_env, env_idx, float(pose[0, 2].item()))
    table.write_root_pose_to_sim(pose, env_ids=env_ids)
    table.write_root_velocity_to_sim(torch.zeros(1, 6, device=device), env_ids=env_ids)
    logger.debug(
        "  env=%d eval_table_z_offset=%+.4f applied to %r (z %.4f -> %.4f)",
        env_idx,
        dz,
        _TABLE_OBJ_NAME,
        before_z,
        before_z + dz,
    )
    return dz


# Scene rigid objects that are the table rather than a manipulable block. Block
# friction is applied to everything else; the table gets its own recorded value.
_TABLE_OBJ_NAMES = frozenset((_TABLE_OBJ_NAME, "table"))

# Env-local pose used to hide blocks not referenced by an episode setup. Deep
# below the floor and spread along Y so parked blocks from different resets do
# not stack on top of each other in the physics scene. Any coordinate outside
# every camera's frustum works; the depth just has to stay clear of the robot
# base and the workspace table's underside.
_BLOCK_GRAVEYARD_LOCAL_Z = -5.0
_BLOCK_GRAVEYARD_Y_STRIDE = 0.1


def _park_unused_blocks(
    base_env,
    env_idx: int,
    env_ids: torch.Tensor,
    active_names: set[str],
    device,
) -> None:
    """Teleport rigid blocks not listed in ``setup['objects']`` to a graveyard pose.

    Autogen tasks spawn the *union* of every block name observed across the
    training dataset (Isaac cannot add or remove rigid bodies at runtime), but
    each episode setup only names the subset that was actually on the table
    during the recorded expert. Without this the unlisted blocks keep whatever
    pose they had -- their ``init_state`` on the first reset (some codegen
    defaults land in or near the workspace), or the physics-driven leftover
    from the previous episode -- which shows up as phantom distractors in the
    video and adds contact geometry the policy never saw during training.
    """
    parked = 0
    origin = base_env.scene.env_origins[env_idx]
    zero_vel = torch.zeros(1, 6, device=device)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
    for obj_name, obj in base_env.scene.rigid_objects.items():
        if obj_name in _TABLE_OBJ_NAMES or obj_name in active_names:
            continue
        local_pos = torch.tensor(
            [0.0, float(parked) * _BLOCK_GRAVEYARD_Y_STRIDE, _BLOCK_GRAVEYARD_LOCAL_Z],
            device=device, dtype=torch.float32,
        )
        world_pos = (local_pos + origin).unsqueeze(0)
        root_pose = torch.cat([world_pos, quat], dim=-1)
        obj.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        obj.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
        parked += 1
    if parked:
        logger.debug(
            "  env=%d parked %d unused block(s) to graveyard (z=%.1f)",
            env_idx, parked, _BLOCK_GRAVEYARD_LOCAL_Z,
        )


# Attributes on ``base_env``: env index -> names of the objects apply_setup
# placed and park_placed_blocks has yet to park, and env index ->
# [(name, world root pose)] for the objects parked until restore_placed_blocks.
_PENDING_PARK_ATTR = "_placed_blocks_pending_park"
_PARKED_ATTR = "_placed_blocks_parked"
# Env-local graveyard x for placed objects sitting out a substep; the unused
# blocks _park_unused_blocks parks occupy x = 0.
_PLACED_BLOCK_GRAVEYARD_LOCAL_X = 1.0


def park_placed_blocks(base_env: BaseEnv) -> None:
    """Move the objects :func:`apply_setup` placed into the graveyard for one physics substep.

    PhysX keeps a body's contact cache across a teleport, so a block placed on
    the face it ended the previous episode resting on is dragged ~0.4 mm toward
    where it lay on the first substep after the reset: a failed episode shifts
    the next one's start. Call before a substep and :func:`restore_placed_blocks`
    right after it. Away from the table for that substep, the blocks lose the
    cache, then rest where they were. Other envs and the robot step normally.
    """
    pending: dict[int, list[str]] = getattr(base_env, _PENDING_PARK_ATTR, {})
    if not pending:
        return
    device = base_env.device
    zero_vel = torch.zeros(1, 6, device=device)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
    parked: dict[int, list[tuple[str, torch.Tensor]]] = {}
    for env_idx, names in pending.items():
        env_ids = torch.tensor([env_idx], device=device, dtype=torch.int64)
        origin = base_env.scene.env_origins[env_idx]
        parked[env_idx] = []
        for slot, obj_name in enumerate(names):
            obj = base_env.scene.rigid_objects[obj_name]
            parked[env_idx].append((obj_name, obj.data.root_pose_w[env_idx].clone().unsqueeze(0)))
            local_pos = torch.tensor(
                [
                    _PLACED_BLOCK_GRAVEYARD_LOCAL_X,
                    float(slot) * _BLOCK_GRAVEYARD_Y_STRIDE,
                    _BLOCK_GRAVEYARD_LOCAL_Z,
                ],
                device=device,
                dtype=torch.float32,
            )
            root_pose = torch.cat([(local_pos + origin).unsqueeze(0), quat], dim=-1)
            obj.write_root_pose_to_sim(root_pose, env_ids=env_ids)
            obj.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
    setattr(base_env, _PENDING_PARK_ATTR, {})
    setattr(base_env, _PARKED_ATTR, parked)


def restore_placed_blocks(base_env: BaseEnv) -> None:
    """Return the objects :func:`park_placed_blocks` parked to their prior pose, at rest."""
    parked: dict[int, list[tuple[str, torch.Tensor]]] = getattr(base_env, _PARKED_ATTR, {})
    if not parked:
        return
    device = base_env.device
    zero_vel = torch.zeros(1, 6, device=device)
    for env_idx, poses in parked.items():
        env_ids = torch.tensor([env_idx], device=device, dtype=torch.int64)
        for obj_name, root_pose in poses:
            obj = base_env.scene.rigid_objects[obj_name]
            obj.write_root_pose_to_sim(root_pose, env_ids=env_ids)
            obj.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
    setattr(base_env, _PARKED_ATTR, {})


def _friction_static_from_setup(setup: dict, obj_name: str) -> float | None:
    """Recorded static friction for *obj_name*, or ``None`` when not recorded."""
    if obj_name in _TABLE_OBJ_NAMES:
        table_friction = setup.get("table_friction")
        return None if table_friction is None else float(table_friction)
    block_friction = setup.get("block_friction")
    return None if block_friction is None else float(block_friction[0])


# Spawn sizes are copied from the recorded cube_scale (rounded to
# CUBE_SCALE_DECIMALS), so a matching env agrees to well under this.
_SPAWNED_BLOCK_SIZE_TOL = 1e-5


def _check_spawned_block_size(base_env, env_idx: int, setup: dict, ep_id: str) -> None:
    """Raise unless env ``env_idx`` spawned its blocks at the setup's recorded size.

    Block size is fixed at spawn, so this is the one profile field the episode
    cannot restore (friction and table height are written per episode). The
    check compares values, so it holds however episodes were routed to envs.
    """
    cube_scale = setup.get("cube_scale")
    env_scene_profile = getattr(base_env, "env_scene_profile", None)
    if cube_scale is None or not callable(env_scene_profile):
        return
    block_sizes = env_scene_profile(env_idx)["block_sizes"]
    wrong = {
        name: size
        for name, size in block_sizes.items()
        if max(abs(float(v) - float(cube_scale)) for v in size) > _SPAWNED_BLOCK_SIZE_TOL
    }
    if wrong:
        raise ValueError(
            f"setup {ep_id!r} was recorded with cube_scale={float(cube_scale):.6f} but env "
            f"{env_idx} (spawn slot {base_env.env_slots[env_idx]}) spawned "
            f"{ {n: [round(float(v), 6) for v in sz] for n, sz in wrong.items()} }. "
            "Block size is fixed at spawn, so this episode would run on wrong-sized blocks."
        )


def _apply_recorded_friction(base_env, env_idx: int, env_ids, setup: dict, ep_id: str) -> None:
    """Restore the per-episode contact material datagen randomized.

    Block and table friction are sampled per episode at datagen and recorded,
    but block friction is baked at spawn in the generated task package (one
    fixed value for every env). Without this the eval reconstructs an episode's
    pose, block size, table height and goal, then runs it under contact physics
    the demo that proved the goal reachable never saw — and for a contact-rich
    task like flipping, friction is first-order.

    This is also the path dataset replay (``--replay``) restores friction
    through, since replay episodes come in as queue setups.
    Setups from datasets that record no friction leave the spawned values alone.
    """
    from toyblocks.utils.randomization_utils import set_rigid_body_friction

    block_friction = setup.get("block_friction")
    table_friction = setup.get("table_friction")
    if block_friction is None and table_friction is None:
        return

    for obj_name, obj in base_env.scene.rigid_objects.items():
        static = _friction_static_from_setup(setup, obj_name)
        if static is None:
            continue
        if obj_name in _TABLE_OBJ_NAMES:
            dynamic, restitution = static, 0.0
        else:
            # PhysX expects dynamic <= static; datagen's randomization can record
            # the pair the other way round. Same clamp the replay path applies.
            dynamic = min(static, float(block_friction[1]))
            restitution = float(block_friction[2]) if len(block_friction) > 2 else 0.0
        set_rigid_body_friction(
            asset=obj,
            static_friction=static,
            dynamic_friction=dynamic,
            restitution=restitution,
            env_ids=env_ids,
        )
        logger.debug(
            "  env=%d friction %s -> static=%.4f dynamic=%.4f restitution=%.4f",
            env_idx, obj_name, static, dynamic, restitution,
        )

# Attribute on ``base_env`` holding, per env, the world-frame table z that
# ``_apply_table_height`` last wrote. Read by
# ``toyblocks.eval.scene_verification`` to hard-check the reconstruction landed.
EXPECTED_TABLE_Z_ATTR = "_expected_table_z_w"


def _record_expected_table_z(base_env, env_idx: int, world_z: float) -> None:
    """Stash the world-frame table z that this reset is reconstructing."""
    expected = getattr(base_env, EXPECTED_TABLE_Z_ATTR, None)
    if expected is None:
        expected = [None] * base_env.num_envs
        setattr(base_env, EXPECTED_TABLE_Z_ATTR, expected)
    expected[env_idx] = world_z


def _apply_table_height(base_env, env_idx, env_ids, setup: dict, device, ep_id) -> None:
    """Reconstruct the eval table height for one env from the recorded setup.

    Precedence:
      1. ``table_pose`` (absolute, env-local) — exact, written verbatim.
      2. ``table_height_delta`` (recorded offset from the preset baseline) —
         added to the current table root z. The baseline is anchored by the
         dataset's recorded ``scene_id`` (``reconcile_env_cfg_with_dataset``
         defaults the env's ``scene_preset_id`` to it and warns at startup on
         an override mismatch), so this is exact when the preset matches.

    If the scene spawns a ``workspace_table`` but the setup carries neither,
    this raises — eval will not guess the table height from block positions
    (no weird fallbacks). See docs/source/user_guide/workflows/evals/launch_eval.md "Dataset scene provenance".
    """
    if _TABLE_OBJ_NAME not in base_env.scene.rigid_objects:
        # No table to place. Only an error if the dataset expected one.
        if setup.get("table_pose") is not None:
            raise ValueError(
                f"setup {ep_id!r} carries a table_pose but the active scene "
                f"preset has no {_TABLE_OBJ_NAME!r} rigid object. Scene rigid "
                f"objects: {sorted(base_env.scene.rigid_objects.keys())}."
            )
        return

    table = base_env.scene.rigid_objects[_TABLE_OBJ_NAME]
    origin = base_env.scene.env_origins[env_idx]

    table_pose = setup.get("table_pose")
    if table_pose is not None:
        local_pos = torch.tensor(table_pose["position"], device=device, dtype=torch.float32).unsqueeze(0)
        world_pos = local_pos + origin
        quat = torch.tensor(table_pose["quaternion"], device=device, dtype=torch.float32).unsqueeze(0)
        _record_expected_table_z(base_env, env_idx, float(world_pos[0, 2].item()))
        table.write_root_pose_to_sim(torch.cat([world_pos, quat], dim=-1), env_ids=env_ids)
        table.write_root_velocity_to_sim(torch.zeros(1, 6, device=device), env_ids=env_ids)
        logger.debug("  env=%d table absolute local_pos=%s", env_idx, local_pos[0].tolist())
        return

    delta = setup.get("table_height_delta")
    if delta is not None:
        pose = table.data.root_pose_w[env_idx].clone().unsqueeze(0)
        before_z = float(pose[0, 2].item())
        pose[0, 2] += float(delta)
        # Record the world z this reconstruction targets so scene verification can
        # hard-check that it actually took. Nothing downstream re-reads the preset
        # baseline, so without this the delta path can only be reported, never
        # checked -- and a later randomization overwriting the table would pass.
        _record_expected_table_z(base_env, env_idx, float(pose[0, 2].item()))
        table.write_root_pose_to_sim(pose, env_ids=env_ids)
        table.write_root_velocity_to_sim(torch.zeros(1, 6, device=device), env_ids=env_ids)
        # The delta is relative to the scene-preset baseline; that preset is
        # anchored by the dataset's recorded scene_id (reconcile_env_cfg_with_dataset
        # defaults the env to it and WARNs at startup on any override mismatch),
        # so this is a normal reconstruction, not a guess. DEBUG only.
        logger.debug(
            "  env=%d table_height_delta=%+.4f applied to %r (z %.4f -> %.4f)",
            env_idx, float(delta), _TABLE_OBJ_NAME, before_z, before_z + float(delta),
        )
        return

    # No absolute table_pose and no recorded table_height_delta: the eval table
    # height cannot be anchored to the datagen scene. We refuse to guess from
    # block positions (a block-surface estimate conflates root with surface,
    # assumes a resting block, and ignores tilt/size). Fail loudly so the
    # mismatch surfaces instead of silently placing blocks on the wrong table.
    raise ValueError(
        f"setup {ep_id!r} has neither a table_pose nor a table_height_delta, but "
        f"the active scene preset spawns {_TABLE_OBJ_NAME!r}. The recorded block Z "
        "is relative to the datagen table, so eval cannot reconstruct the table "
        "height. Record table_height_delta (or an absolute object_pose.table) at "
        "datagen, or pin scene_preset_id. See docs/source/user_guide/workflows/evals/launch_eval.md 'Dataset scene provenance'."
    )


def _quat_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Chord distance between unit quaternions: 1 - |<a, b>|. Sign-invariant.

    Kept separate from ``scene_verification._quat_distance`` (same formula over
    JSON-loaded ``list[float]``) because this one runs per step on live sim
    tensors that may be on GPU; routing them through numpy would force a device
    transfer into the step loop.
    """
    dot = torch.abs(torch.sum(a * b)).item()
    return 1.0 - min(dot, 1.0)


class EpisodeQueueClient:
    """HTTP client for the episode queue + setup/verification state tracker.

    Maintains :attr:`current_setups` (what's running in env i now, or ``None``
    if the queue is exhausted) and :attr:`previous_setups` (what just finished;
    used to mask final_info entries from envs running on default randomization).

    Does **not** write to sim — the caller (``template_env._reset_idx``) is
    responsible for calling :func:`apply_setup` and committing.

    Each applied setup is verified against actual sim state on the *next*
    :meth:`verify_after_step` (one step lets PhysX settle so rejection
    surfaces). ``verify=False`` disables; ``verify_tolerance`` is the WARN threshold.
    """

    def __init__(
        self,
        base_env,
        queue_url: str,
        num_envs: int,
        *,
        num_slots: int = 1,
        slot_scales: Optional[Sequence[float]] = None,
        env_slots: Optional[Sequence[int]] = None,
        verify: bool = True,
        verify_tolerance: float = _DEFAULT_VERIFY_TOL,
        on_setups_changed: Optional[Callable[[list[Optional[dict]]], None]] = None,
        apply_recorded_joint_velocity: bool = False,
        start_gripper_open: bool = False,
    ) -> None:
        self._base_env = base_env
        self._queue_url = queue_url
        self._verify = verify
        self._verify_tolerance = verify_tolerance
        # Forwarded by ``TemplateEnv._reset_idx`` to :func:`apply_setup`: restore
        # the recorded frame-0 arm joint velocity (``--replay``) instead of
        # starting every episode at rest.
        self.apply_recorded_joint_velocity: bool = bool(apply_recorded_joint_velocity)
        # Start every episode with the gripper open (finger_joint 0.0) instead of
        # the recorded frame-0 gripper: rewritten on each fetched setup, so the
        # reset, the scene verification and the policy inputs all agree.
        self._start_gripper_open = bool(start_gripper_open)
        # Cube-scale spawn slots: env i is spawned at size slot (i % num_slots),
        # so it must only run episodes whose recorded cube_scale matches that
        # slot. We pass that slot on every fetch; a FIFO queue ignores it.
        self._num_slots = max(1, int(num_slots))
        # The slot each env was spawned at (BaseEnvCfg.env_slots); None means
        # round-robin, env i at slot i % num_slots.
        self._env_slots: Optional[list[int]] = (
            [int(s) for s in env_slots] if env_slots is not None else None
        )
        # The scale each slot was spawned at, in slot order (empty when the task
        # bakes no per-episode sizes). Sent with every fetch so the queue can
        # reject a mismatched slot ordering; see fetch_next_setup.
        self._slot_scales = [float(s) for s in slot_scales] if slot_scales else []
        # Subscriber notified whenever current_setups changes (PromptResolver
        # uses this). Receives a shallow copy; treat dicts as read-only —
        # mutating them would corrupt verification state.
        self._on_setups_changed = on_setups_changed

        self.current_setups: list[Optional[dict]] = [None] * num_envs
        self.previous_setups: list[Optional[dict]] = [None] * num_envs

        # Two-stage pending-verification queue: fetch_reset_states() appends to
        # _pending_next; verify_after_step() promotes _pending_next -> _pending_now
        # and checks.  The one-step lag gives PhysX a chance to relax the written state.
        self._pending_now: list[tuple[int, dict]] = []
        self._pending_next: list[tuple[int, dict]] = []

        # Counters for the eval-end summary.
        self._fetches = 0
        self._exhausted_logged = False

        logger.info(
            "EpisodeQueueClient initialized num_envs=%d num_slots=%d slot_scales=%s "
            "queue_url=%s verify=%s tol=%.2e",
            num_envs, self._num_slots, [round(s, 4) for s in self._slot_scales],
            queue_url, verify, verify_tolerance,
        )

    @property
    def done(self) -> bool:
        """``True`` iff every env has run out of setups."""
        return all(s is None for s in self.current_setups)

    def fetch_reset_states(self, env_ids: Sequence[int]) -> list[Optional[dict]]:
        """Fetch the next setup for each env in ``env_ids`` from the HTTP queue.

        Updates ``current_setups`` / ``previous_setups`` bookkeeping and
        notifies subscribers via ``_on_setups_changed``.  Does **not** call
        :func:`apply_setup` or ``write_data_to_sim`` — the caller is
        responsible for applying the returned setups to the sim.

        Returns a list parallel to ``env_ids``: the fetched setup dict for
        each env, or ``None`` when the queue is exhausted.
        """
        if isinstance(env_ids, torch.Tensor):
            indices = env_ids.cpu().tolist()
        else:
            indices = list(env_ids)

        logger.debug("fetch_reset_states() called for env_ids=%s", indices)

        result: list[Optional[dict]] = []
        for i in indices:
            self.previous_setups[i] = self.current_setups[i]
            setup = fetch_next_setup(
                self._queue_url,
                slot=(
                    self._env_slots[i] if self._env_slots is not None else i % self._num_slots
                ),
                slot_scales=self._slot_scales,
            )
            self._fetches += 1
            if setup is not None and self._start_gripper_open and "robot_gripper_position" in setup:
                setup["robot_gripper_position"] = 0.0
            self.current_setups[i] = setup

            if setup is not None:
                logger.info(
                    "env=%d <- queue setup episode_id=%s (fetch #%d, has_expert_videos=%s)",
                    i, setup.get("episode_id", "<unknown>"), self._fetches,
                    bool(setup.get("expert_videos")),
                )
                if not setup.get("expert_videos"):
                    logger.warning(
                        "fetched setup episode_id=%s episode_index=%s has no "
                        "expert_videos; expert_first_frame cannot be written "
                        "for this episode.",
                        setup.get("episode_id", "<unknown>"),
                        setup.get("episode_index"),
                    )
                if self._verify:
                    self._pending_next.append((i, setup))
            else:
                if not self._exhausted_logged:
                    logger.info("queue exhausted (env=%d's fetch returned None)", i)
                    self._exhausted_logged = True
                else:
                    logger.debug("queue still empty for env=%d", i)

            result.append(setup)

        if self._on_setups_changed is not None:
            self._on_setups_changed(list(self.current_setups))

        return result

    def verify_after_step(self) -> None:
        """Verify setups whose physics has had at least one step to settle.

        Call once per ``env.step()``. Logs WARNING for any joint/object pose
        diff above ``verify_tolerance``, DEBUG otherwise. No-op if ``verify=False``.
        """
        if not self._verify:
            return

        for env_idx, setup in self._pending_now:
            self._verify_one(env_idx, setup)

        self._pending_now = self._pending_next
        self._pending_next = []

    def _verify_one(self, env_idx: int, setup: dict) -> None:
        base_env = self._base_env
        ep_id = setup.get("episode_id", "<unknown>")
        tol = self._verify_tolerance
        worst_joint = 0.0
        worst_obj = 0.0
        problems: list[str] = []

        if "robot_joint_positions" in setup:
            robot = base_env.robot.articulation
            expected = torch.tensor(
                setup["robot_joint_positions"], device=base_env.device, dtype=torch.float32,
            )
            actual = robot.data.joint_pos[env_idx, : expected.numel()]
            diff = (actual - expected).abs()
            worst_joint = max(worst_joint, diff.max().item())
            if diff.max().item() > tol:
                problems.append(
                    f"joint_pos: max|Δ|={diff.max().item():.4f} "
                    f"per_joint={diff.tolist()}"
                )

        if "robot_gripper_position" in setup:
            robot = base_env.robot.articulation
            actual_g = robot.data.joint_pos[env_idx, 7].item()
            diff_g = abs(float(setup["robot_gripper_position"]) - actual_g)
            worst_joint = max(worst_joint, diff_g)
            if diff_g > tol:
                problems.append(f"gripper: |Δ|={diff_g:.4f}")

        for obj_name, obj_data in setup.get("objects", {}).items():
            obj = base_env.scene.rigid_objects[obj_name]
            expected_local = torch.tensor(
                obj_data["position"], device=base_env.device, dtype=torch.float32,
            )
            actual_local = obj.data.root_pos_w[env_idx] - base_env.scene.env_origins[env_idx]
            pos_diff = (actual_local - expected_local).abs().max().item()

            expected_quat = torch.tensor(
                obj_data["quaternion"], device=base_env.device, dtype=torch.float32,
            )
            actual_quat = obj.data.root_quat_w[env_idx]
            quat_diff = _quat_distance(actual_quat, expected_quat)

            obj_worst = max(pos_diff, quat_diff)
            worst_obj = max(worst_obj, obj_worst)
            if pos_diff > tol or quat_diff > tol:
                problems.append(
                    f"{obj_name}: pos_diff={pos_diff:.4f} quat_diff={quat_diff:.4f}"
                )

        # Contact material, read back from PhysX. A friction write that silently
        # did not land leaves poses perfect and only shows up as an unexplained
        # drop in success, so check it alongside the poses rather than trusting
        # the write. Static only: it is what both branches of
        # _apply_recorded_friction set, and dynamic is derived from it.
        for obj_name, obj in base_env.scene.rigid_objects.items():
            expected_friction = _friction_static_from_setup(setup, obj_name)
            if expected_friction is None:
                continue
            materials = obj.root_physx_view.get_material_properties()
            actual_friction = materials[env_idx, :, 0]
            friction_diff = (actual_friction - expected_friction).abs().max().item()
            if friction_diff > tol:
                problems.append(
                    f"{obj_name} static_friction: expected={expected_friction:.4f} "
                    f"actual={actual_friction.min().item():.4f}..{actual_friction.max().item():.4f} "
                    f"diff={friction_diff:.4f}"
                )

        if problems:
            logger.warning(
                "verify env=%d episode=%s diverged after one physics step "
                "(worst_joint=%.4f worst_obj=%.4f tol=%.2e):\n  - %s",
                env_idx, ep_id, worst_joint, worst_obj, tol, "\n  - ".join(problems),
            )
        else:
            logger.debug(
                "verify env=%d episode=%s OK (worst_joint=%.2e worst_obj=%.2e tol=%.2e)",
                env_idx, ep_id, worst_joint, worst_obj, tol,
            )
