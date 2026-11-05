"""Passthrough must not overwrite the text randomize_task_letters rewrites.

`--passthrough_source_columns` defaults to True and its contract is "this render differs from its
source only in pixels", so it copies every non-OWNED source column verbatim after rendering. With
the glyph swap on, that silently restored the original `pddl_goal`/`task` and dropped `glyph_map`
entirely: the blocks showed new letters, the labels showed the old ones, and the parquet was
internally consistent so nothing downstream could detect it. Observed on a real render before the
fix -- render_params.json carried the assignment while the parquet carried none of it.

Loaded by AST because rerender.py imports isaaclab at module scope.
"""
import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "scripts" / "rerender" / "rerender.py"


def _load():
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    want = {"PASSTHROUGH_OWNED_BASE", "PASSTHROUGH_OWNED_TASK_LETTERS",
            "passthrough_owned_columns"}
    keep = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in want)
            or (isinstance(n, ast.Assign)
                and any(getattr(t, "id", None) in want for t in n.targets))]
    assert keep, "passthrough_owned_columns is gone from rerender.py"
    m = type(sys)("pt")
    exec(compile(ast.Module(body=keep, type_ignores=[]), "pt", "exec"), m.__dict__)
    return m


M = _load()


def test_flag_off_owns_only_the_two_index_columns():
    """Default behaviour must be untouched: everything else still comes from the source."""
    assert M.passthrough_owned_columns(False) == {"episode_index", "index"}


def test_flag_on_owns_nothing_extra():
    """OWNED means "take the RENDER's value", and for this feature nothing qualifies.

    Every text column is REBUILT from the source instead, because in a re-render the recorder's
    own values are placeholders:
      * pddl_goal  -- the recorder has no PDDL problem file, so its goal is empty; owning it
                      produced an all-null column on a real run.
      * task_index -- the recorder writes a ONE-ROW tasks.parquet holding the generic string
                      "manipulation", so owning the index silently discarded every real task
                      string. Observed: a swapped render whose whole task table was
                      ["manipulation"] while the unswapped control kept all four real tasks.
      * glyph_map  -- owning cannot help; the rebuild iterates src_schema.names and the source
                      has no such column, so it is appended explicitly.
    """
    owned = M.passthrough_owned_columns(True)
    assert owned == {"episode_index", "index"}
    assert "pddl_goal" not in owned, "owning pddl_goal takes the recorder's EMPTY value"
    assert "glyph_map" not in owned, "owning cannot add a column the source lacks"


def test_flag_on_still_owns_the_base_columns():
    assert M.passthrough_owned_columns(False) <= M.passthrough_owned_columns(True)


def test_task_index_is_not_owned():
    """Rewriting creates new task strings, so the source's task_index no longer resolves -- but
    the fix is to RE-POINT it at a rebuilt table, not to take the recorder's.

    Owning it swapped one wrong answer for another: the recorder's index points into its own
    one-row "manipulation" table, so the dataset ended up with no instruction text at all.
    """
    assert "task_index" not in M.passthrough_owned_columns(True)


def test_flag_on_and_off_own_the_same_set():
    """The flag changes how text columns are REBUILT, never which are owned."""
    assert M.passthrough_owned_columns(True) == M.passthrough_owned_columns(False)


def test_source_lacking_columns_cannot_be_rescued_by_owning():
    """The rebuild walks src_schema.names, so OWNED only chooses WHICH value a shared column
    takes -- it can never introduce one. Documented so the next person does not re-try it."""
    owned = M.passthrough_owned_columns(True)
    assert "glyph_map" not in owned


def test_nothing_unrelated_is_claimed():
    """Owning a column means the source's value is discarded -- keep the set minimal."""
    owned = M.passthrough_owned_columns(True)
    for col in ("observation.state.object_pose.block_a", "action.joint_position",
                "observation.images.camera:front_camera", "is_episode_successful"):
        assert col not in owned
