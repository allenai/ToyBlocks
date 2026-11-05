from toyblocks.scenes.droid_with_box_table import DroidWithBoxTableSceneConfig
from toyblocks.scenes.color_only_blocks_table import ColorOnlyBlocksTableSceneConfig
from toyblocks.scenes.table import TableSceneConfig
from toyblocks.scenes.sim2real import Sim2RealSceneConfig

SCENE_PRESETS = {
    "droid_with_box_table": DroidWithBoxTableSceneConfig(),
    "color_only_blocks_table": ColorOnlyBlocksTableSceneConfig(),
    "table": TableSceneConfig(),
    "sim2real": Sim2RealSceneConfig()
}
