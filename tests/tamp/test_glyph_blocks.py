from blocksuite.tamp.blocks.glyph_blocks import (
    DIGIT_TYPE,
    LETTER_TYPE,
    block_name,
    block_type,
    glyph_usd_subpath,
)


def test_block_10_resolves_to_the_large_library_usd() -> None:
    assert block_name("10") == "block_10"
    assert block_type("10") == DIGIT_TYPE
    assert glyph_usd_subpath("block_10", "train_blocks_large") == "train_blocks_large/usd/10.usda"
    assert glyph_usd_subpath("block_10_2", "train_blocks_large") == "train_blocks_large/usd/10.usda"


def test_single_character_glyphs_are_unchanged() -> None:
    assert glyph_usd_subpath("block_1", "train_blocks_large") == "train_blocks_large/usd/1.usda"
    assert glyph_usd_subpath("block_1_2", "train_blocks_large") == "train_blocks_large/usd/1.usda"
    assert glyph_usd_subpath("block_a") == "train_blocks_large/usd/A.usda"
    assert glyph_usd_subpath("block_a", "train_blocks") == "train_blocks/usd/A.usd"
    assert block_type("a") == LETTER_TYPE
    assert block_type("1") == DIGIT_TYPE
    # color cuboids and other numbers are not glyphs
    assert glyph_usd_subpath("block_red1") is None
    assert glyph_usd_subpath("block_11") is None


def test_every_glyph_and_legacy_alias_spawns_from_the_large_library_by_default() -> None:
    """block_r / block_t / block_b and the block_o1 / block_o2 aliases used to be pinned
    to train_blocks, mixing asset sets in a scene even with --usd-library train_blocks_large."""
    from blocksuite.tamp.blocks.template.codegen import Block

    names = [f"block_{c}" for c in "abcdefghijklmnopqrstuvwxyz123456789"] + ["block_10", "block_o1", "block_o2"]
    for library in (None, "train_blocks_large"):
        for block in Block.from_names(names, usd_library=library):
            assert block.type == "usd" and block.usd_path.startswith("train_blocks_large/usd/"), (library, block.name)
    assert [b.type for b in Block.from_names(["block_red1", "block_pink"])] == ["cuboid", "cuboid"]
