from dataclasses import MISSING
from isaaclab.utils import configclass

from isaaclab.sensors.camera.camera_cfg import CameraCfg
from isaaclab.assets.asset_base_cfg import AssetBaseCfg

@configclass
class BaseSceneConfig:
    """Base scene configuration. Scene configs are used to spawn any presets of assets and cameras. Typically this includes
    a ground plane, table, and cameras"""
    asset_cfgs: dict[str, AssetBaseCfg] = MISSING
    camera_cfgs: dict[str, CameraCfg] = MISSING