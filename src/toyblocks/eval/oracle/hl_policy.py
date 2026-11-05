"""Oracle high-level policy: an FF plan turned into per-tick pixel-coordinate action strings.

Ported from the former TAMP submodule's ``evals/oracle_hl_policy.py``.
The plan comes from FF (toyblocks's vendored ``vecgen/symbolic/ff_planner``), the camera
from the live front-camera sensor, and the semantic argument layout from
``tamp/blocks/action_config.yaml``. Step effects are checked by the PDDL success gate's own
``batched_goal_conjunct`` on the bound env (upstream evaluated tmp3 predicate classes).

``get_current_high_level_action(env_idx)`` returns a string suitable as language
conditioning for a low-level VLA model; the oracle advances on its own when the current
step's effect predicates (new true facts in FF's state after the step) hold on the sim.
For pick-and-place steps a grasp test moves the step from its *pick* to its *place* phase.

Output format examples (include_coords=True, matches dataloader format)::

    pick 320,180
    place 400,120
    place on table 410,200
    flip 310,195 left
    rotate 310,195 cw
    done

With include_coords=False the coordinates are dropped (``pick``, ``flip left``, ...).

All envs driven by one instance share the plan (one PDDL problem); only the per-env
progress (plan index, phase, grasp tracking) is per env. The eval harness builds one
instance per env with ``n_envs=1`` (see :class:`~toyblocks.eval.oracle.controller.OracleHLController`).
"""

from __future__ import annotations

import logging
import random
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

import toyblocks
from toyblocks.vecgen.symbolic.ff import default_ff_binary

logger = logging.getLogger(__name__)

#: Semantic argument names per action (``config_map.<action>.HL_ARGS``), same as TAMP's
#: ``test_domains/Blocks/ActionConfig.yaml``.
DEFAULT_ACTION_CONFIG = Path(toyblocks.__file__).parent / "tamp" / "blocks" / "action_config.yaml"

# ---------------------------------------------------------------------------
# Front-camera extrinsics fallback (the table / color_only scene presets), in the
# per-env local frame. Used only when the sensor cannot be read.
# ---------------------------------------------------------------------------
_FRONT_CAMERA_POS = np.array([2.0, 0.0, 1.0])
_FRONT_CAMERA_QUAT_WXYZ = np.array([-0.3535534, 0.6123724, 0.6123724, -0.3535534])


def _build_R_cw(quat_wxyz: np.ndarray) -> np.ndarray:
    """World-to-camera rotation from a wxyz quaternion (ROS convention offset)."""
    q = quat_wxyz
    quat_xyzw = np.array([q[1], q[2], q[3], q[0]])
    R_wc = Rotation.from_quat(quat_xyzw).as_matrix()
    return R_wc.T


_R_CW: np.ndarray = _build_R_cw(_FRONT_CAMERA_QUAT_WXYZ)

Intrinsics = tuple[float, float, float, float]  # fx, fy, cx, cy


def _project_world_to_pixel(
    point_world: np.ndarray, cam_pos: np.ndarray, R_cw: np.ndarray, intrinsics: Intrinsics
) -> np.ndarray:
    """Project a 3-D point to ``(u, v)`` pixel coords (NaN behind the camera)."""
    p_cam = R_cw @ (point_world - cam_pos)
    if p_cam[2] <= 0:
        return np.array([np.nan, np.nan], dtype=np.float32)
    fx, fy, cx, cy = intrinsics
    u = fx * p_cam[0] / p_cam[2] + cx
    v = fy * p_cam[1] / p_cam[2] + cy
    return np.array([u, v], dtype=np.float32)


# ---------------------------------------------------------------------------
# Table geometry defaults (BlocksSceneCfg)
# ---------------------------------------------------------------------------
_TABLE_TOP_Z = -0.035 + 0.025  # table pos z + half-height of 0.05 visual

_REACHABLE_X_MIN, _REACHABLE_X_MAX = 0.25, 0.75
_REACHABLE_Y_MIN, _REACHABLE_Y_MAX = -0.35, 0.35
_MIN_CLEARANCE = 0.08

_DEFAULT_IGNORED_HL_ACTIONS = frozenset(
    {"init", "move_to", "nudge", "stop", "STOP", "done", "reach-goal"}
)

#: Predicates the sim can check (the success gate's goal predicates). Purely symbolic
#: ones (handempty, clear, at, holding, initialized, ...) are not step effects.
_EVALUABLE_PREDICATES = frozenset(
    {"on", "on_table", "ori_of", "on_right", "on_front", "on_back", "on_left", "next_to"}
)


class _EnvState:
    """Mutable per-environment progress state."""

    __slots__ = ("plan_idx", "phase", "pick_phase_steps", "pick_init_z", "cached_table_pixel")

    def __init__(self) -> None:
        self.plan_idx: int = 0
        self.phase: str = "idle"
        self.pick_phase_steps: int = 0
        self.pick_init_z: float | None = None
        self.cached_table_pixel: tuple[int, int] | None = None


def _parse_prop(prop_str: str) -> tuple[str, list[str]] | None:
    """``'(on block_r block_blue)'`` -> ``('on', ['block_r', 'block_blue'])``; None if unparseable."""
    prop_str = prop_str.strip()
    if not prop_str.startswith("("):
        return None
    parts = prop_str.strip("() ").strip().split()
    if not parts:
        return None
    return (parts[0], parts[1:])


class OracleHighLevelPolicy:
    """Stateful oracle converting a PDDL plan into per-timestep action strings.

    Args:
        domain_file: Path to the PDDL domain file.
        problem_file: Path to the PDDL problem file (shared by all envs).
        simulator: The sim view the oracle reads (``get_pose``, ``robots``, ``base_env``,
            ``env_index``); in the eval harness a
            :class:`~toyblocks.eval.oracle.sim_adapter.BaseEnvOracleSimAdapter`.
        scene: The scene view (``sensors``, ``rigid_objects``, ``env_origins``).
        n_envs: Number of parallel environments to drive.
        action_config_path: Action config with the semantic ``HL_ARGS`` layouts (default
            :data:`DEFAULT_ACTION_CONFIG`).
        include_coords: Whether to append pixel coordinates to action strings.
    """

    def __init__(
        self,
        domain_file: str,
        problem_file: str,
        simulator: Any,
        scene: Any,
        n_envs: int = 1,
        action_config_path: str | None = None,
        include_coords: bool = True,
    ) -> None:
        self._simulator = simulator
        self._scene = scene
        self._include_coords = include_coords
        self._n_envs = int(n_envs)
        self._eval_errors_logged: set[tuple[str, str]] = set()  # (predicate, error type)

        self._action_cfg = self._load_action_config(
            action_config_path or str(DEFAULT_ACTION_CONFIG)
        )
        self._intrinsics = self._read_camera_intrinsics()
        self._cam_pos, self._cam_R_cw = self._read_camera_extrinsics()

        # Shared plan (same problem file for all envs).
        self._plan: list[dict[str, Any]] = []
        self._compute_plan(domain_file, problem_file)

        self._env_states: list[_EnvState] = [_EnvState() for _ in range(self._n_envs)]
        for env_idx in range(self._n_envs):
            self._init_phase(env_idx)

    @property
    def n_envs(self) -> int:
        return self._n_envs

    # ------------------------------------------------------------------
    # Action config
    # ------------------------------------------------------------------
    @staticmethod
    def _load_action_config(path: str) -> dict[str, Any]:
        with open(path) as f:
            return yaml.safe_load(f)

    def _ignored_actions(self) -> frozenset[str]:
        extra = self._action_cfg.get("ignore_hl_actions", [])
        return _DEFAULT_IGNORED_HL_ACTIONS | frozenset(s.lower() for s in extra)

    def _hl_args_names(self, action_name: str) -> list[str]:
        entry = self._action_cfg.get("config_map", {}).get(action_name, {})
        return [a.lower() for a in entry.get("HL_ARGS", [])]

    # ------------------------------------------------------------------
    # Robot and camera
    # ------------------------------------------------------------------
    def _get_robot(self) -> Any:
        """The robot view (keyed ``"robot"``) with ``_gripper_open``, ``_robot``, ``_ee_link_id``."""
        robots = self._simulator.robots
        for key in ("robot", "franka"):
            if key in robots:
                return robots[key]
        if robots:
            return next(iter(robots.values()))
        return None

    def _read_camera_intrinsics(self) -> Intrinsics:
        sensor = self._scene.sensors.get("front_camera")
        if sensor is not None and hasattr(sensor, "data"):
            try:
                K = sensor.data.intrinsic_matrices
                if K.ndim == 3:
                    K = K[0]
                K_np = K.cpu().numpy().astype(np.float64)
                fx, fy = float(K_np[0, 0]), float(K_np[1, 1])
                cx, cy = float(K_np[0, 2]), float(K_np[1, 2])
                print(
                    f"[OracleHL] Camera intrinsics from sensor: "
                    f"fx={fx:.2f} fy={fy:.2f} cx={cx:.1f} cy={cy:.1f}"
                )
                return (fx, fy, cx, cy)
            except Exception as e:
                print(f"[OracleHL] Could not read intrinsics from sensor ({e}); using fallback.")
        print("[OracleHL] Using fallback intrinsics (640x360, fl=30, ha=20.955)")
        fx = 30.0 * 640 / 20.955
        return (fx, fx, 320.0, 180.0)

    def _read_camera_extrinsics(self) -> tuple[np.ndarray, np.ndarray]:
        """``(cam_pos_local, R_cw)`` for the front camera.

        Read from the sensor rather than the module constants, which describe only the
        table / color_only scene presets (sim2real puts the front camera elsewhere).
        Positions are localized to the env origin because ``_get_object_pixel`` projects
        env-local object poses.
        """
        sensor = self._scene.sensors.get("front_camera")
        if sensor is not None and hasattr(sensor, "data"):
            # Imported here: annotation_utils pulls in OpenCV, which only the sensor path needs.
            from toyblocks.data_annotation.annotation_utils import camera_from_sensor

            try:
                pos_w = sensor.data.pos_w[0].cpu().numpy().astype(np.float64)
                quat_w = sensor.data.quat_w_world[0].cpu().numpy().astype(np.float64)
                origin = self._scene.env_origins[0].cpu().numpy().astype(np.float64)
                cam_pos, R_cw, _ = camera_from_sensor(pos_w - origin, quat_w, np.eye(3))
                print(
                    f"[OracleHL] Camera extrinsics from sensor: pos={np.round(cam_pos, 3).tolist()}"
                )
                return cam_pos, R_cw
            except Exception as e:
                print(f"[OracleHL] Could not read extrinsics from sensor ({e}); using fallback.")
        print("[OracleHL] Using fallback extrinsics (table preset)")
        return _FRONT_CAMERA_POS, _R_CW

    # ------------------------------------------------------------------
    # FF planner invocation + state parsing
    # ------------------------------------------------------------------
    def _compute_plan(self, domain_file: str, problem_file: str) -> None:
        ff_binary = default_ff_binary()
        if not ff_binary.exists():
            raise FileNotFoundError(f"FF planner binary not found at {ff_binary}")

        with open(problem_file) as f:
            content = f.read()

        # FF writes scratch files to its cwd; keep them (and the patched problem) out of
        # the caller's.
        with tempfile.TemporaryDirectory(prefix="oracle_ff_") as cwd:
            actual_problem_file = Path(problem_file).resolve()
            if "(gripper_at_init)" not in content:
                actual_problem_file = Path(cwd) / "problem_patched.pddl"
                actual_problem_file.write_text(
                    content.replace("(handempty", "(gripper_at_init)\n        (handempty", 1)
                )
            res = subprocess.run(
                [
                    str(ff_binary),
                    "-o",
                    str(Path(domain_file).resolve()),
                    "-f",
                    str(actual_problem_file),
                ],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=cwd,
            )
        raw = res.stdout + res.stderr

        marker = "found legal plan as follows"
        if marker not in raw:
            raise RuntimeError(f"FF planner failed.\n{raw[-500:]}")

        plan_section = raw.split(marker)[1].split("time")[0].replace("step", "")
        ignored = self._ignored_actions()

        raw_steps: list[tuple[str, list[str]]] = []
        raw_step_indices: list[int] = []  # indices into the full FF plan (1-based)
        ff_step_idx = 0
        for line in plan_section.strip().splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            ff_step_idx += 1
            parts = line.split(":", 1)[1].strip().lower().split()
            if not parts or parts[0] in ignored:
                continue
            raw_steps.append((parts[0], parts[1:]))
            raw_step_indices.append(ff_step_idx)

        ff_states = self._parse_ff_states(raw)
        # FF prints no state after the plan's last action; the goal stands in for it.
        goal_props = self._parse_goal_props(content)

        for i, (hl_action, hl_args_raw) in enumerate(raw_steps):
            effects = self._compute_effect_predicates(
                ff_states,
                raw_step_indices[i],
                goal_props=goal_props if i == len(raw_steps) - 1 else None,
            )
            self._plan.append(
                {
                    "hl_action": hl_action,
                    "hl_args": hl_args_raw,
                    "semantic": self._map_args_to_semantic(hl_action, hl_args_raw),
                    "effect_predicates": effects,
                }
            )

        print(
            f"[OracleHL] Plan has {len(self._plan)} actionable steps "
            f"(shared across {self._n_envs} envs):"
        )
        for i, step in enumerate(self._plan):
            eff_names = [e["name"] for e in step["effect_predicates"]]
            print(
                f"  {i}: {step['hl_action']} {step['hl_args']}  effects({len(eff_names)})={eff_names}"
            )

    @staticmethod
    def _parse_ff_states(raw_output: str) -> list[set[str]]:
        """The ``*** Plan 1 States ***`` section of FF's output: one fact set per state
        (index 0 = the initial state); empty when the section is absent."""
        state_marker = "*** Plan 1 States ***\n"
        if state_marker not in raw_output:
            return []
        states: list[set[str]] = []
        for block in raw_output.split(state_marker)[1].split("State \n\n"):
            block = block.strip()
            if not block:
                continue
            props: set[str] = set()
            for line in block.split("\n"):
                line = line.strip().lower()
                if not line or "not" in line:
                    continue
                props.add(("(" + line.replace("(", " ")).strip())
            states.append(props)
        return states

    @staticmethod
    def _parse_goal_props(problem_text: str) -> set[str]:
        """Positive ground literals of the problem's ``:goal``, as ``"(name arg ...)"``
        (negated and quantified ``?var`` literals are skipped)."""
        start = problem_text.lower().find("(:goal")
        if start < 0:
            return set()
        goal = problem_text[start + len("(:goal") :].lower()
        goal = re.sub(r"\(\s*not\s*\([^()]*\)\s*\)", " ", goal)
        props: set[str] = set()
        for name, args in re.findall(r"\(\s*([a-z_][\w-]*)((?:\s+[^\s()]+)*)\s*\)", goal):
            arg_list = args.split()
            if name in ("and", "or") or any(a.startswith("?") for a in arg_list):
                continue
            props.add("(" + " ".join([name, *arg_list]) + ")")
        return props

    def _compute_effect_predicates(
        self,
        ff_states: list[set[str]],
        step_ff_idx: int,
        goal_props: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """The sim-checkable facts a plan step makes true.

        New facts in the state after step ``step_ff_idx`` (1-based) that were false
        before it, filtered to :data:`_EVALUABLE_PREDICATES`. ``goal_props`` stands in
        for the state after the plan's last step when FF printed none. Returns
        ``{'name', 'args', 'prop_str'}`` dicts.
        """
        if not ff_states or step_ff_idx > len(ff_states):
            return []
        if step_ff_idx < len(ff_states):
            state_after = ff_states[step_ff_idx]
        elif goal_props is not None:
            state_after = goal_props
        else:
            return []
        state_before = ff_states[step_ff_idx - 1] if step_ff_idx > 0 else set()

        effects: list[dict[str, Any]] = []
        for prop_str in sorted(state_after - state_before):
            parsed = _parse_prop(prop_str)
            if parsed is None or parsed[0] not in _EVALUABLE_PREDICATES:
                continue
            effects.append({"name": parsed[0], "args": parsed[1], "prop_str": prop_str})
        return effects

    def _map_args_to_semantic(self, action_name: str, args: list[str]) -> dict[str, str]:
        names = self._hl_args_names(action_name)
        return {name: args[i] for i, name in enumerate(names) if i < len(args)}

    # ------------------------------------------------------------------
    # Effect checks (per env)
    # ------------------------------------------------------------------
    def _evaluate_effect_predicates(self, env_idx: int) -> bool:
        """True when every effect predicate of env_idx's current step holds."""
        st = self._env_states[env_idx]
        if st.plan_idx >= len(self._plan):
            return True
        effects = self._plan[st.plan_idx]["effect_predicates"]
        if not effects:
            # A pnp step with nothing checkable (a table-to-table relocation keeps
            # on_table and ori_of; only non-evaluable *_obstructs facts change)
            # completes once the placed block is released.
            if self._is_pnp_action(self._plan[st.plan_idx]["hl_action"]) and st.phase == "place":
                return not self._is_object_grasped(env_idx)
            return False
        return all(self._holds(e["name"], e["args"]) for e in effects)

    def _holds(self, pred: str, args: list[str]) -> bool:
        """``(pred args)`` on the bound env, by the success gate's own check. A check
        that cannot run is not a check that passed: it would advance (or finish) the
        plan without the step happening."""
        from toyblocks.tamp.blocks.pddl_goal_tensor_conversions import batched_goal_conjunct

        sim = self._simulator
        try:
            return bool(batched_goal_conjunct(sim.base_env, pred, tuple(args))[sim.env_index])
        except Exception as exc:
            key = (pred, type(exc).__name__)
            if key not in self._eval_errors_logged:
                self._eval_errors_logged.add(key)
                logger.error(
                    "[OracleHL] cannot evaluate (%s %s): %r -- treating it as NOT holding",
                    pred,
                    " ".join(args),
                    exc,
                )
            return False

    # ------------------------------------------------------------------
    # Phases (per env)
    # ------------------------------------------------------------------
    def _init_phase(self, env_idx: int) -> None:
        if not self._plan:
            self._env_states[env_idx].phase = "done"
            return
        self._set_phase_for_current_action(env_idx)

    def _set_phase_for_current_action(self, env_idx: int) -> None:
        st = self._env_states[env_idx]
        if st.plan_idx >= len(self._plan):
            st.phase = "done"
            return
        if self._is_pnp_action(self._plan[st.plan_idx]["hl_action"]):
            st.phase = "pick"
            st.pick_phase_steps = 0
            st.pick_init_z = self._get_pick_object_z(env_idx)
        else:
            st.phase = "active"
        st.cached_table_pixel = None

    def _get_pick_object_z(self, env_idx: int) -> float | None:
        """The target object's world Z at pick-phase entry."""
        st = self._env_states[env_idx]
        if st.plan_idx >= len(self._plan):
            return None
        target_obj = self._pick_object(self._plan[st.plan_idx])
        if not target_obj:
            return None
        try:
            obj_pose = self._simulator.get_pose(target_obj, env_idx=env_idx, local=False)
            return float(obj_pose[2].cpu().numpy())
        except Exception:
            return None

    @staticmethod
    def _is_pnp_action(action: str) -> bool:
        return action.startswith("pnp_") or action in ("move_to_side", "move_to_center")

    # ------------------------------------------------------------------
    # Grasp detection: is the step's object held by the gripper? (per env)
    # ------------------------------------------------------------------
    _GRASP_LIFT_DELTA = 0.02  # object must rise this far above its initial Z
    _GRASP_EE_DIST_THRESHOLD = 0.10  # object must be within this distance of EE

    def _is_object_grasped(self, env_idx: int) -> bool:
        """True when the gripper is closed, the object lifted above its initial Z, and near the EE."""
        st = self._env_states[env_idx]
        robot = self._get_robot()
        if robot is None:
            return False
        try:
            gripper_open = bool(robot._gripper_open[env_idx].item())
        except Exception:
            gripper_open = True
        if gripper_open:
            return False

        target_obj = self._pick_object(self._plan[st.plan_idx])
        if not target_obj:
            return False
        try:
            # World frame, comparable with the world-frame EE pose.
            obj_pose = self._simulator.get_pose(target_obj, env_idx=env_idx, local=False)
            obj_pos = obj_pose[:3].cpu().numpy().astype(np.float64)
        except Exception:
            return False

        obj_z = float(obj_pos[2])
        init_z = st.pick_init_z
        base_z = init_z if init_z is not None else _TABLE_TOP_Z
        lifted = obj_z > base_z + self._GRASP_LIFT_DELTA
        # The lift test only gates pick -> place (fingers closing on a resting block
        # are not a grasp). While placing, the block may be carried below its pick
        # height (unstacking onto the table), so held = closed + at EE.
        if st.phase == "place":
            lifted = True

        try:
            art = robot._robot
            ee_link = robot._ee_link_id
            ee_idx = ee_link[0] if isinstance(ee_link, (list, tuple)) else int(ee_link)
            ee_pos = art.data.body_link_pose_w[env_idx, ee_idx, :3].cpu().numpy().astype(np.float64)
            dist = float(np.linalg.norm(obj_pos[:3] - ee_pos[:3]))
        except Exception:
            return False

        if st.pick_phase_steps % 10 == 0:
            print(
                f"  [GRASP][env{env_idx}] step={st.pick_phase_steps} obj={target_obj} "
                f"obj_z={obj_z:.4f} init_z={init_z} lifted={lifted} ee_dist={dist:.4f} "
                f"gripper=closed"
            )
        if lifted and dist < self._GRASP_EE_DIST_THRESHOLD:
            print(
                f"  [GRASP][env{env_idx}] DETECTED! obj={target_obj} z={obj_z:.4f} ee_dist={dist:.4f}"
            )
            return True
        return False

    def _update_phase(self, env_idx: int) -> None:
        st = self._env_states[env_idx]
        if st.plan_idx >= len(self._plan):
            st.phase = "done"
            return

        min_pick_steps = 10
        if self._is_pnp_action(self._plan[st.plan_idx]["hl_action"]):
            if st.phase == "pick":
                st.pick_phase_steps += 1
            grasped = self._is_object_grasped(env_idx)
            if st.phase == "pick" and st.pick_phase_steps > min_pick_steps and grasped:
                st.phase = "place"
                st.cached_table_pixel = None
                print(
                    f"[OracleHL][env{env_idx}] Grasp detected after {st.pick_phase_steps} "
                    f"pick steps -> transitioning to 'place' phase"
                )
        if self._evaluate_effect_predicates(env_idx):
            self._advance(env_idx)

    # ------------------------------------------------------------------
    # Pixels (per env, local frame)
    # ------------------------------------------------------------------
    def _get_object_pixel(self, obj_name: str, env_idx: int) -> tuple[int, int]:
        try:
            pose = self._simulator.get_pose(obj_name, env_idx=env_idx, local=True)
            pos = pose[:3].cpu().numpy().astype(np.float64)
        except Exception:
            return (0, 0)
        uv = _project_world_to_pixel(pos, self._cam_pos, self._cam_R_cw, self._intrinsics)
        if np.any(np.isnan(uv)):
            return (0, 0)
        return (int(round(float(uv[0]))), int(round(float(uv[1]))))

    def _get_all_block_positions(self, env_idx: int) -> list[np.ndarray]:
        positions: list[np.ndarray] = []
        for name in self._scene.rigid_objects:
            if name in ("table", "table_collider", "table2"):
                continue
            try:
                pose = self._simulator.get_pose(name, env_idx=env_idx, local=True)
                pos = pose[:3].cpu().numpy().astype(np.float64)
            except Exception:
                continue
            # A non-finite pose would fail every clearance check (NaN > x is False).
            if np.all(np.isfinite(pos[:2])):
                positions.append(pos[:2])
        return positions

    def _sample_table_placement(self, env_idx: int) -> tuple[int, int]:
        """A free table spot's pixel (cached for the step), at least ``_MIN_CLEARANCE`` from every block."""
        st = self._env_states[env_idx]
        if st.cached_table_pixel is not None:
            return st.cached_table_pixel

        block_xys = self._get_all_block_positions(env_idx)
        for _ in range(200):
            candidate = np.array(
                [
                    random.uniform(_REACHABLE_X_MIN, _REACHABLE_X_MAX),
                    random.uniform(_REACHABLE_Y_MIN, _REACHABLE_Y_MAX),
                ]
            )
            if all(np.linalg.norm(candidate - bxy) > _MIN_CLEARANCE for bxy in block_xys):
                world_pt = np.array([candidate[0], candidate[1], _TABLE_TOP_Z], dtype=np.float64)
                uv = _project_world_to_pixel(
                    world_pt, self._cam_pos, self._cam_R_cw, self._intrinsics
                )
                if not np.any(np.isnan(uv)):
                    st.cached_table_pixel = (int(round(float(uv[0]))), int(round(float(uv[1]))))
                    return st.cached_table_pixel
        # No clear spot: project the reachable-area centre (world metres, not pixels).
        center = np.array(
            [
                (_REACHABLE_X_MIN + _REACHABLE_X_MAX) / 2,
                (_REACHABLE_Y_MIN + _REACHABLE_Y_MAX) / 2,
                _TABLE_TOP_Z,
            ],
            dtype=np.float64,
        )
        uv = _project_world_to_pixel(center, self._cam_pos, self._cam_R_cw, self._intrinsics)
        if np.any(np.isnan(uv)):
            uv = np.array(self._intrinsics[2:4], dtype=np.float32)  # image centre (cx, cy)
        st.cached_table_pixel = (int(round(float(uv[0]))), int(round(float(uv[1]))))
        return st.cached_table_pixel

    # ------------------------------------------------------------------
    # Step arguments
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_direction(raw_dir: str) -> tuple[str, str]:
        """``(verb, direction)`` from a PDDL direction constant: ``left_flip`` ->
        ``('flip', 'left')``, ``cw`` -> ``('rotate', 'cw')``, ``forward`` -> ``('flip', 'forward')``."""
        lower = raw_dir.lower().strip()
        if lower in ("cw", "ccw"):
            return "rotate", lower
        parts = lower.split("_")
        if len(parts) >= 2 and parts[-1] in ("flip", "rotate"):
            return parts[-1], "_".join(parts[:-1])
        return "flip", lower

    def _pick_object(self, step: dict[str, Any]) -> str:
        return step["semantic"].get("obj", step["hl_args"][1] if len(step["hl_args"]) > 1 else "")

    def _place_target(self, step: dict[str, Any]) -> str | None:
        if step["hl_action"] in ("pnp_table_block", "pnp_block_block"):
            return step["semantic"].get("target_obj")
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_current_high_level_action(self, env_idx: int = 0) -> str:
        """The action string for env_idx at the current tick (advancing the plan first
        when the current step's effects hold): ``pick u,v``, ``place u,v`` (onto a
        block), ``place on table u,v``, ``flip u,v <dir>``, ``rotate u,v <dir>``,
        ``act u,v`` (other actions) or ``done``."""
        st = self._env_states[env_idx]
        if st.plan_idx >= len(self._plan):
            return "done"

        self._update_phase(env_idx)

        if st.phase == "done" or st.plan_idx >= len(self._plan):
            return "done"

        step = self._plan[st.plan_idx]
        action = step["hl_action"]
        if self._is_pnp_action(action):
            return self._format_pnp(step, env_idx)
        if action in ("reorient", "rotate"):
            return self._format_reorient_rotate(step, env_idx)
        obj = self._pick_object(step)
        if self._include_coords:
            u, v = self._get_object_pixel(obj, env_idx) if obj else (0, 0)
            return f"act {u},{v}"
        return "act"

    def get_current_high_level_actions(self) -> list[str]:
        """The action string for every environment (length n_envs)."""
        return [self.get_current_high_level_action(i) for i in range(self._n_envs)]

    def advance_action(self, env_idx: int = 0) -> None:
        """Force-advance env_idx to the next step (the oracle normally advances itself)."""
        self._advance(env_idx)

    def current_plan_index_of(self, env_idx: int) -> int:
        return self._env_states[env_idx].plan_idx

    def phase_of(self, env_idx: int) -> str:
        return self._env_states[env_idx].phase

    def is_done_of(self, env_idx: int) -> bool:
        st = self._env_states[env_idx]
        return st.phase == "done" or st.plan_idx >= len(self._plan)

    @property
    def all_done(self) -> bool:
        return all(self.is_done_of(i) for i in range(self._n_envs))

    # Single-env (env 0) accessors.
    @property
    def current_plan_index(self) -> int:
        return self.current_plan_index_of(0)

    @property
    def phase(self) -> str:
        return self.phase_of(0)

    @property
    def is_done(self) -> bool:
        return self.is_done_of(0)

    @property
    def plan_length(self) -> int:
        return len(self._plan)

    def get_plan_summary(self) -> list[dict[str, Any]]:
        return [
            {
                "idx": i,
                "action": s["hl_action"],
                "args": s["hl_args"],
                "semantic": s["semantic"],
                "effect_predicates": [e["prop_str"] for e in s["effect_predicates"]],
            }
            for i, s in enumerate(self._plan)
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _advance(self, env_idx: int) -> None:
        st = self._env_states[env_idx]
        st.plan_idx += 1
        self._set_phase_for_current_action(env_idx)
        if st.plan_idx < len(self._plan):
            step = self._plan[st.plan_idx]
            print(
                f"[OracleHL][env{env_idx}] Advanced to step {st.plan_idx}: "
                f"{step['hl_action']} {step['hl_args']} phase={st.phase}"
            )
        else:
            print(f"[OracleHL][env{env_idx}] Plan complete.")

    def _format_pnp(self, step: dict[str, Any], env_idx: int) -> str:
        st = self._env_states[env_idx]
        if st.phase == "pick":
            if self._include_coords:
                u, v = self._get_object_pixel(self._pick_object(step), env_idx)
                return f"pick {u},{v}"
            return "pick"
        target = self._place_target(step)
        # Datagen labels a table placement "place on table" (goal.action) and a
        # placement onto a block "place"; the LL was trained on those phrases.
        verb = "place" if target is not None else "place on table"
        if self._include_coords:
            if target is not None:
                u, v = self._get_object_pixel(target, env_idx)
            else:
                u, v = self._sample_table_placement(env_idx)
            return f"{verb} {u},{v}"
        return verb

    def _format_reorient_rotate(self, step: dict[str, Any], env_idx: int) -> str:
        verb, direction = self._parse_direction(step["semantic"].get("direction", ""))
        if self._include_coords:
            u, v = self._get_object_pixel(self._pick_object(step), env_idx)
            return f"{verb} {u},{v} {direction}"
        return f"{verb} {direction}"
