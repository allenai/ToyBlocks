"""scene_swap must read a problem's block set from its generated env.py without importing it.

The parse is what decides whether the scene needs mutating at a problem boundary, so a wrong or
absent answer either skips a renderable problem or renders it into the wrong scene. None must mean
"unknown" and never "no blocks".
"""
import textwrap
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rerender"))
import scene_swap as ss


def _env(tmp_path, body: str) -> str:
    p = tmp_path / "env.py"
    p.write_text(textwrap.dedent(body))
    return str(p)


TWO_BLOCK = """
    class Foo:
        def _setup_scene_objects(self):
            from blocksuite.utils import rigid_objects
            block_names = ['block_a', 'block_blue']
            block_specs = {'block_a': {'type': 'usd', 'usd_path': 'train_blocks_large/usd/A.usda'},
                           'block_blue': {'type': 'cuboid', 'usd_path': None}}
            return
    """


def test_reads_names_and_specs(tmp_path):
    f = _env(tmp_path, TWO_BLOCK)
    assert ss.signature(f) == frozenset({"block_a", "block_blue"})
    specs = ss.block_specs(f)
    assert specs["block_a"]["type"] == "usd"
    assert specs["block_blue"]["type"] == "cuboid"
    assert specs["block_a"]["usd_path"].startswith("train_blocks_large/")


def test_single_block(tmp_path):
    f = _env(tmp_path, TWO_BLOCK.replace("['block_a', 'block_blue']", "['block_a']"))
    assert ss.signature(f) == frozenset({"block_a"})


def test_unknown_is_none_not_empty(tmp_path):
    """Every failure mode must be None; an empty set would read as 'this problem has no blocks'."""
    assert ss.signature(None) is None
    assert ss.signature(str(tmp_path / "missing.py")) is None
    assert ss.signature(_env(tmp_path, "x = 1\n")) is None                    # no setup fn
    assert ss.signature(_env(tmp_path, """
        class Foo:
            def _setup_scene_objects(self):
                block_names = compute_names()
                block_specs = {}
                return
        """)) is None                                                          # not a literal


def test_names_not_in_specs_are_dropped(tmp_path):
    f = _env(tmp_path, TWO_BLOCK.replace("'block_blue': {'type': 'cuboid', 'usd_path': None}",
                                         "'block_other': {'type': 'cuboid', 'usd_path': None}"))
    assert ss.signature(f) == frozenset({"block_a"})


# -- block sizes: an episode renders only in an env spawned at its recorded size --

SIZED = """
    class Foo:
        def _setup_scene_objects(self):
            from blocksuite.utils import rigid_objects
            block_names = ['block_1', 'block_blue']
            block_specs = {'block_1': {'type': 'usd', 'usd_path': 'train_blocks_large/usd/1.usda'},
                           'block_blue': {'type': 'cuboid', 'usd_path': None}}
            randomizations = {'block_sizes': {'block_1': [[0.0375, 0.0375, 0.0375], [0.049, 0.049, 0.049], [0.0331, 0.0331, 0.0331], [0.0331, 0.0331, 0.0331]],
                                              'block_blue': [[0.0375, 0.0375, 0.0375], [0.049, 0.049, 0.049], [0.0331, 0.0331, 0.0331], [0.0331, 0.0331, 0.0331]]},
                              'table_height': [0.0, 0.0, 0.0, 0.0]}
            return
    """


def test_size_table_is_the_first_blocks_edge_per_slot(tmp_path):
    assert ss.env_size_table(_env(tmp_path, SIZED)) == [0.0375, 0.049, 0.0331, 0.0331]


def test_size_table_absent_is_none(tmp_path):
    assert ss.env_size_table(_env(tmp_path, TWO_BLOCK)) is None
    assert ss.env_size_table(str(tmp_path / "missing.py")) is None


def test_match_size_slots_by_value():
    import numpy as np

    table = [0.0375, 0.049, 0.0331, 0.0331]
    rec = [float(np.float32(0.049)), float(np.float32(0.0375)), 0.0331, 0.0446]
    # float32 round-off matches; shared edges go to the lowest slot; an unknown size is None
    assert ss.match_size_slots(table, rec) == [1, 0, 2, None]


def test_round_robin_slots_is_the_template_default():
    assert ss.round_robin_slots(8, 8) == list(range(8))
    assert ss.round_robin_slots(32, 8)[:10] == [0, 1, 2, 3, 4, 5, 6, 7, 0, 1]
    assert sorted(set(ss.round_robin_slots(4, 8))) == [0, 1, 2, 3]  # fewer envs than slots
