"""Build the cuRobo v2 robot configuration used by the planner pool.

Ported from TAMP's ``tmp3.Backends.Curobo.robot_config`` with the FR3 + Robotiq 2F-85
overrides of ``tmp3.Config.RobotConfig`` inlined: the stock cuRobo Robotiq config
(``configs/robot/franka_robotiq_2f_85_test.yml``) is patched with the FR3 model-full URDF,
collision spheres, link names and self-collision tables, then converted from the v1
kinematics dict to the v2 ``KinematicsLoaderCfg`` format:

* ``ee_link`` becomes ``tool_frames``; USD-only keys are dropped;
* ``collision_spheres`` is inlined (v2 resolves relative sphere files against its own
  content tree);
* ``cspace`` keeps only the active (non-locked) joints and uses ``default_joint_position``.

The FR3 kinematics URDF and sphere file (``TAMP/assets/fr3_robotiq_2f85/curobo``) are
vendored under ``configs/robot``.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent / "configs" / "robot"
_ROBOTIQ_YML = _CONFIG_DIR / "franka_robotiq_2f_85_test.yml"
FR3_ROBOTIQ_2F85_URDF_PATH = _CONFIG_DIR / "fr3_model_full_robotiq_2f85_kinematics.urdf"
FR3_ROBOTIQ_2F85_CUROBO_SPHERES_PATH = (
    _CONFIG_DIR / "fr3_model_full_robotiq_2f85_collision_spheres.yml"
)
ROBOT_URDF_PATH = str(FR3_ROBOTIQ_2F85_URDF_PATH)
ROBOT_CUROBO_SPHERES_PATH = str(FR3_ROBOTIQ_2F85_CUROBO_SPHERES_PATH)

FR3_ROBOTIQ_2F85_COLLISION_LINK_NAMES = [
    "panda_link0",
    "panda_link1",
    "panda_link2",
    "panda_link3",
    "panda_link4",
    "panda_link5",
    "panda_link6",
    "panda_link7",
    "robotiq_arg2f_base_link",
    "left_outer_knuckle",
    "right_outer_knuckle",
    "left_outer_finger",
    "right_outer_finger",
    "left_inner_knuckle",
    "right_inner_knuckle",
    "left_inner_finger",
    "right_inner_finger",
    "attached_object",
]
FR3_ROBOTIQ_2F85_SELF_COLLISION_IGNORE: dict[str, list[str]] = {
    "panda_link0": ["panda_link1", "panda_link2"],
    "panda_link1": ["panda_link2", "panda_link3"],
    "panda_link2": ["panda_link3", "panda_link4"],
    "panda_link3": ["panda_link4", "panda_link5"],
    "panda_link4": ["panda_link5", "panda_link6", "panda_link7"],
    "panda_link5": ["panda_link6", "panda_link7"],
    "panda_link6": ["panda_link7", "robotiq_arg2f_base_link"],
    "panda_link7": ["robotiq_arg2f_base_link"],
    "robotiq_arg2f_base_link": [
        "left_outer_knuckle",
        "right_outer_knuckle",
        "left_outer_finger",
        "right_outer_finger",
        "left_inner_knuckle",
        "right_inner_knuckle",
        "left_inner_finger",
        "right_inner_finger",
        "attached_object",
    ],
    "left_outer_knuckle": [
        "right_outer_knuckle",
        "left_outer_finger",
        "left_inner_knuckle",
        "right_inner_knuckle",
        "left_inner_finger",
        "attached_object",
    ],
    "right_outer_knuckle": [
        "right_outer_finger",
        "left_inner_knuckle",
        "right_inner_knuckle",
        "right_inner_finger",
        "attached_object",
    ],
    "left_outer_finger": ["left_inner_knuckle", "left_inner_finger", "attached_object"],
    "right_outer_finger": ["right_inner_knuckle", "right_inner_finger", "attached_object"],
    "left_inner_knuckle": ["right_inner_knuckle", "left_inner_finger", "attached_object"],
    "right_inner_knuckle": ["right_inner_finger", "attached_object"],
    "left_inner_finger": ["attached_object"],
    "right_inner_finger": ["attached_object"],
    "attached_object": [],
}
FR3_ROBOTIQ_2F85_SELF_COLLISION_BUFFER: dict[str, float] = {
    link_name: 0.0 for link_name in FR3_ROBOTIQ_2F85_COLLISION_LINK_NAMES
}
FR3_ROBOTIQ_2F85_SELF_COLLISION_BUFFER.update({"panda_link0": 0.1, "panda_link1": 0.05})

_V1_ONLY_KEYS = (
    "use_usd_kinematics",
    "isaac_usd_path",
    "usd_path",
    "usd_robot_root",
    "usd_flip_joints",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_spheres(spec: Any) -> dict[str, Any]:
    """Return the ``collision_spheres`` mapping for a path or an inline dict."""
    if isinstance(spec, dict):
        return spec.get("collision_spheres", spec)
    path = Path(str(spec))
    if not path.is_absolute():
        candidate = _CONFIG_DIR / path
        if candidate.exists():
            path = candidate
    if not path.exists():
        raise FileNotFoundError(f"collision sphere file not found: {spec}")
    return _load_yaml(path)["collision_spheres"]


def apply_fr3_curobo_kinematics_overrides(
    kinematics_cfg: dict[str, Any], finger_joint_lock: float = 0.0
) -> None:
    """Patch cuRobo's stock Robotiq config to match the FR3 model-full link names."""
    kinematics_cfg["urdf_path"] = ROBOT_URDF_PATH
    kinematics_cfg["asset_root_path"] = os.path.dirname(ROBOT_URDF_PATH)
    kinematics_cfg["collision_spheres"] = ROBOT_CUROBO_SPHERES_PATH
    kinematics_cfg["collision_link_names"] = list(FR3_ROBOTIQ_2F85_COLLISION_LINK_NAMES)
    kinematics_cfg["mesh_link_names"] = list(FR3_ROBOTIQ_2F85_COLLISION_LINK_NAMES)
    kinematics_cfg["self_collision_ignore"] = {
        link_name: list(ignored_links)
        for link_name, ignored_links in FR3_ROBOTIQ_2F85_SELF_COLLISION_IGNORE.items()
    }
    kinematics_cfg["self_collision_buffer"] = dict(FR3_ROBOTIQ_2F85_SELF_COLLISION_BUFFER)
    kinematics_cfg["lock_joints"] = {"finger_joint": finger_joint_lock}
    kinematics_cfg["extra_links"] = {
        "attached_object": {
            "parent_link_name": "robotiq_arg2f_base_link",
            "link_name": "attached_object",
            "fixed_transform": [0, 0, 0, 1, 0, 0, 0],
            "joint_type": "FIXED",
            "joint_name": "attach_joint",
        },
    }


def _v1_robotiq_kinematics(num_attached_spheres: int) -> dict[str, Any]:
    cfg = _load_yaml(_ROBOTIQ_YML)["robot_cfg"]["kinematics"]
    cfg["base_link"] = "panda_link0"
    cfg["ee_link"] = "robotiq_arg2f_base_link"
    cfg["extra_collision_spheres"] = {"attached_object": num_attached_spheres}
    apply_fr3_curobo_kinematics_overrides(cfg)
    if not cfg.get("urdf_path") or not os.path.isabs(str(cfg["urdf_path"])):
        raise RuntimeError(
            f"the cuRobo v2 planner needs an absolute robot URDF; got {cfg.get('urdf_path')!r}"
        )
    return cfg


def _v2_stock_franka_kinematics(num_attached_spheres: int) -> dict[str, Any]:
    from curobo._src.util.config_io import join_path, load_yaml
    from curobo.content import get_robot_configs_path

    cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]["kinematics"]
    cfg["extra_collision_spheres"] = {"attached_object": num_attached_spheres}
    return cfg


def _to_v2(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    for key in _V1_ONLY_KEYS:
        cfg.pop(key, None)
    ee_link = cfg.pop("ee_link", None)
    if ee_link is not None and not cfg.get("tool_frames"):
        cfg["tool_frames"] = [ee_link]
    cfg["collision_spheres"] = _resolve_spheres(cfg["collision_spheres"])
    locked: list[str] = list((cfg.get("lock_joints") or {}).keys())
    cspace = cfg.get("cspace")
    if isinstance(cspace, dict):
        names: list[str] = list(cspace["joint_names"])
        keep = [i for i, n in enumerate(names) if n not in locked]
        new_cspace: dict[str, Any] = {"joint_names": [names[i] for i in keep]}
        retract = cspace.get("retract_config", cspace.get("default_joint_position"))
        if retract is not None:
            new_cspace["default_joint_position"] = [retract[i] for i in keep]
        for key in ("null_space_weight", "cspace_distance_weight", "max_acceleration", "max_jerk"):
            val = cspace.get(key)
            if isinstance(val, list):
                new_cspace[key] = [val[i] for i in keep]
            elif val is not None:
                new_cspace[key] = val
        cfg["cspace"] = new_cspace
    cfg["format_version"] = 2.0
    return cfg


def build_robot_cfg_dict(robotiq: bool = True, num_attached_spheres: int = 32) -> dict[str, Any]:
    """Robot config dict (``{"kinematics": {...}}``) for ``curobo.types.RobotCfg.create``."""
    if robotiq:
        kin = _to_v2(_v1_robotiq_kinematics(num_attached_spheres))
    else:
        kin = _v2_stock_franka_kinematics(num_attached_spheres)
    return {"kinematics": kin}
