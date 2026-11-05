from blocksuite.scenes.droid_with_box_table import DroidWithBoxTableSceneConfig
from blocksuite.scenes.color_only_blocks_table import ColorOnlyBlocksTableSceneConfig
from blocksuite.scenes.table import TableSceneConfig
from blocksuite.scenes.sim2real import Sim2RealSceneConfig

SCENE_PRESETS = {
    "droid_with_box_table": DroidWithBoxTableSceneConfig(),
    "color_only_blocks_table": ColorOnlyBlocksTableSceneConfig(),
    "table": TableSceneConfig(),
    "sim2real": Sim2RealSceneConfig()
}
