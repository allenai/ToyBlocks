"""Shared test helpers for ToyBlocks simulation tests."""

SCENE_ROBOT_MAP = {
    "droid_with_box_table": "fr3_robotiq_gripper_soft",
    "sim2real": "fr3_robotiq_gripper_soft",
}


def teardown_sim():
    """Tear down SimulationContext singleton between tests."""
    from isaaclab.sim import SimulationContext

    sim = SimulationContext.instance()
    if sim is None:
        return
    sim.stop()
    sim.clear_all_callbacks()
    sim.clear_instance()
