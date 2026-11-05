import json
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets.asset_base_cfg import AssetBaseCfg
from isaaclab.assets.rigid_object import RigidObjectCfg
from isaaclab.sensors.camera.tiled_camera_cfg import TiledCameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from blocksuite import SIM2REAL_SCENE_ASSETS_DIR, SIM2REAL_SCENE_FILEPATH
from blocksuite.scenes.base import BaseSceneConfig

# Asset archives do not always ship the exact USD flavor scene.json names (e.g. the
# 20260901 toyblocks_real release lists robot_mount.usda but contains the binary
# robot_mount.usd), so fall back to a sibling with another USD extension.
_USD_EXTENSIONS: tuple[str, ...] = (".usda", ".usd", ".usdc")


def _resolve_usd_path(usd_path: Path) -> Path:
    """Return *usd_path*, or a sibling with a different USD extension if only that exists."""
    if usd_path.exists():
        return usd_path
    for ext in _USD_EXTENSIONS:
        candidate: Path = usd_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return usd_path  # let Isaac Lab raise its own "USD file not found" error


def _load_asset_cfgs_from_json() -> dict:
    with open(SIM2REAL_SCENE_FILEPATH, "r") as f:
        scene = json.load(f)

    cfgs = {}
    for name, obj in scene["objects"].items():
        if not obj.get("enabled", True):
            continue
        path = obj["path"]
        if path.startswith("http"):
            continue  # skip remote assets (e.g. FlatGrid)

        usd_path: Path = _resolve_usd_path(SIM2REAL_SCENE_ASSETS_DIR / path)
        t = obj["transform"]["translate"]
        o = obj["transform"]["orient"]

        pos = (t["x"], t["y"], t["z"])
        rot = (o["w"], o["x"], o["y"], o["z"])  # JSON is xyzw, Isaac Lab is wxyz

        if obj.get("rigidbody", False):
            cfgs[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
                spawn=UsdFileCfg(
                    usd_path=usd_path.resolve().as_posix(),
                    # Apply a (kinematic) rigid body so RigidObject init always finds one, even when
                    # the resolved USD asset only ships a collider (e.g. toyblocks_real table.usda
                    # in some asset versions). Mirrors the droid_with_box_table preset. Fixed
                    # furniture -> kinematic + no gravity; harmless if the USD already has a body.
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=True,
                        kinematic_enabled=True,
                    ),
                ),
            )
        else:
            cfgs[name] = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
                spawn=UsdFileCfg(usd_path=usd_path.resolve().as_posix()),
            )
    return cfgs

K_front = [
    786.14367676,
    0.0,
    723.64068604,
    0.0,
    786.14367676,
    406.79891968,
    0.0,
    0.0,
    1.0,
]


@configclass
class Sim2RealSceneConfig(BaseSceneConfig):
    env_spacing: float | None = 5.0
    asset_cfgs = _load_asset_cfgs_from_json()
    camera_cfgs = dict(
        front_camera=TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/FrontCamera",
            update_period=0.04,
            # Record the live camera pose each frame. Default (False) returns the
            # init-time pose, which can mismatch the actual render transform and
            # left datasets with a wrong observation.camera_pose (breaking the
            # goal.position projection). See droid_franka.py for the same setting.
            update_latest_camera_pose=True,
            height=360,
            width=640,
            data_types=[
                "rgb",
                "distance_to_image_plane",
                "instance_segmentation_fast",
                "semantic_segmentation",
            ],
            colorize_instance_id_segmentation=True,
            colorize_instance_segmentation=True,
            spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                intrinsic_matrix=K_front,
                width=1280,
                height=720,
                focus_distance=400.0,
                clipping_range=(0.05, 1.0e2),
            ),
            offset=TiledCameraCfg.OffsetCfg(
                # Calibrated front camera extrinsics: T_base_cam (cam → robot base).
                pos=(0.944958, 0.000714, 0.48414),
                #rot=(-0.2841564, 0.6454526, 0.6476641, -0.2884047), # wxyz, ROS
                 # Base: Rx(-125°) then Rz(90°) → q = Rz(90) * Rx(-125) (wxyz, ROS).
                rot=(-0.3265056, 0.6272114, 0.6272114, -0.3265056),
                convention="ros",
            ),
        ),
        left_camera=TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/LeftCamera",
            update_period=0.1,
            # Record the live camera pose each frame (see front_camera note).
            update_latest_camera_pose=True,
            height=360,
            width=640,
            data_types=[
                "rgb",
                "distance_to_image_plane",
                "instance_segmentation_fast",
                "semantic_segmentation",
            ],
            colorize_instance_id_segmentation=True,
            colorize_instance_segmentation=True,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.12,
                focus_distance=0.6,
                horizontal_aperture=5.139,
                clipping_range=(0.1, 3.0),
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.15, 0.40, 0.45),
                rot=(0.151496, -0.317428, 0.844820, -0.403198),
                convention="ros",
            ),
        ),
    )
