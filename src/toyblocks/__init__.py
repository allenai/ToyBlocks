from pathlib import Path

__version__ = "0.0.1"

TOP_LEVEL_TOYBLOCKS_DIR = Path(__file__).parent.parent.parent
ASSET_DIR = TOP_LEVEL_TOYBLOCKS_DIR / "assets"
ROBOT_ASSET_DIR = ASSET_DIR / "robots"
OBJECT_ASSET_DIR = ASSET_DIR / "objects"
SCENE_ASSET_DIR = ASSET_DIR / "scenes"

# new sim2real usd model for the franka+robotiq. This model has the correct orientation of the
# gripper, as well as the inertial properties and other stuff brought from the mjcf model we used
# for molmobot and molmospaces, which was tested on both sim and real
FRANKA_ROBOTIQ_NEW_USD_FILEPATH = (
    ROBOT_ASSET_DIR / "franka_droid" / "usd" / "fr3_model_full_robotiq_2f85.usd"
)

# this is the same as the model above, but it just adds some overrides to match the old orientation
# of the gripper, similar to the old usd model used before for tamp and rl skills training
FRANKA_ROBOTIQ_NEW_USD_VISUAL_ALIGNED_FILEPATH = (
    ROBOT_ASSET_DIR
    / "franka_droid"
    / "usd"
    / "fr3_model_full_robotiq_2f85_visual_aligned.usd"
)

# this is the old usd model generated using omni's urdf converter and an old urdf that was also
# used for curobo and datagen with tamp
FRANKA_ROBOTIQ_OLD_USD_FILEPATH = (
    ROBOT_ASSET_DIR
    / "franka_droid_legacy"
    / "usd"
    / "franka_robotiq_2f_85"
    / "usd"
    / "updated_franka.usd"
)

# Franka urdf model used for viser to work fine
FRANKA_ROBOTIQ_URDF_FILEPATH = (
    ROBOT_ASSET_DIR / "franka_droid" / "usd" / "fr3_franka_hand.urdf"
)

# NOTE: for now, these scene json files will be tracked on the repo instead of the
# resources manager, which for now it's the simplest move. Note that the assets are still under
# the resources manager, but the json is not, so we just have to use the right paths from now on

PACKAGE_ASSETS_DIR = Path(__file__).parent / "assets"

SIM2REAL_SCENE_FILEPATH = PACKAGE_ASSETS_DIR / "scene.json"
SIM2REAL_SCENE_ASSETS_DIR = SCENE_ASSET_DIR / "toyblocks_real" / "usd"

# HDRI skybox datasets under `assets/environments/<dataset>/` (each a dataset.json manifest + *.hdr);
# polyhaven comes with `get-toyblocks-assets --asset-manifest asset_manifest.json`, see
# docs/source/user_guide/workflows/datagen/local.md#hdri-backdrops. Single source of
# truth so the pool can be relocated without touching code/yamls; render resolves skybox `set:`
# routes under this dir and it is overridable via the RENDER_HDRI_DIR env var.
ENVIRONMENT_ASSET_DIR = ASSET_DIR / "environments"

# Table material library (grayscale table textures + per-episode HSV randomization) consumed by
# render's material system; lives next to the table USD under the sim2real scene assets.
TABLE_MATERIALS_FILEPATH = SIM2REAL_SCENE_ASSETS_DIR / "table" / "materials.json"
