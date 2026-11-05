"""A `__sNNN` seed variant must resolve its env from the BASE problem name.

Seed variants (`problem_x__s001`) share the scene of `problem_x` and usually ship no env dir of
their own. render already falls back to the base name, and discover_jobs records why: without
it "an item made up only of variants resolves no env at all and the whole item dies at 'No env
found' -- 4 items of aug27_pnp_2r2g_soft failed 8x into the DLQ exactly this way".

Nothing pinned that behaviour. It is load-bearing for the pending render of
sep3_ai2_stack_PARTIAL5k: of its 167 source problems only 83 have an exactly-named env under
`sep3_ai2_reorient_fixed/envs`; the other 84 resolve solely through this fallback. A regression
would not fail loudly per problem -- it would silently drop half the dataset.

Pure path/AST work: no Isaac, no GPU, no weka.
"""
import ast
import re
from pathlib import Path
from typing import Optional

import pytest

RENDER = Path(__file__).resolve().parents[2] / "scripts" / "render" / "render.py"


@pytest.fixture(scope="module")
def mod():
    """Exec only the two resolver helpers; render's module body launches Isaac."""
    tree = ast.parse(RENDER.read_text(encoding="utf-8"))
    want = {"_base_problem", "_resolve_env_file"}
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in want]
    assigns = [n for n in tree.body if isinstance(n, ast.Assign)
               and any(getattr(t, "id", "") == "_SEED_SUFFIX" for t in n.targets)]
    assert len(defs) == len(want), f"missing: {want - {d.name for d in defs}}"
    # the helpers carry runtime-evaluated annotations (Optional[str])
    ns = {"re": re, "Optional": Optional}
    exec(compile(ast.Module(body=assigns + defs, type_ignores=[]), "render", "exec"), ns)
    return ns


@pytest.mark.parametrize("name,base", [
    ("problem_x__s001", "problem_x"),
    ("problem_x__s017", "problem_x"),
    ("problem_letter_color_10_20260903_174015_621857_6883__s001",
     "problem_letter_color_10_20260903_174015_621857_6883"),
    ("problem_x", "problem_x"),                      # no suffix: unchanged
])
def test_base_problem_strips_the_seed_suffix(mod, name, base):
    assert mod["_base_problem"](name) == base


def test_a_name_that_merely_contains_s_digits_is_not_stripped(mod):
    """Only a trailing `__sNNN` is a seed suffix."""
    for name in ("problem_s001", "problem_x_s001"):
        assert mod["_base_problem"](name) == name


def test_variant_resolves_through_the_base_env(mod, tmp_path):
    """The real P9 shape: the variant ships no env dir, the base does."""
    envs = tmp_path / "envs"
    (envs / "problem_x").mkdir(parents=True)
    (envs / "problem_x" / "env.py").write_text("# base env\n")
    variant = "problem_x__s001"
    assert not (envs / variant).exists()

    candidates = [envs / variant / "env.py",                        # exact: absent
                  envs / mod["_base_problem"](variant) / "env.py"]  # fallback: present
    assert mod["_resolve_env_file"](candidates, None) == str(envs / "problem_x" / "env.py")


def test_exact_name_wins_over_the_base(mod, tmp_path):
    """A variant that DOES ship its own env must use it, not the base's."""
    envs = tmp_path / "envs"
    for p in ("problem_x", "problem_x__s001"):
        (envs / p).mkdir(parents=True)
        (envs / p / "env.py").write_text(f"# {p}\n")
    candidates = [envs / "problem_x__s001" / "env.py", envs / "problem_x" / "env.py"]
    assert mod["_resolve_env_file"](candidates, None) == str(envs / "problem_x__s001" / "env.py")


def test_explicit_env_file_is_the_last_resort(mod, tmp_path):
    """With neither on disk, the --env_file fallback is returned rather than None."""
    assert mod["_resolve_env_file"]([tmp_path / "nope" / "env.py"], "/x/env.py") == "/x/env.py"
    assert mod["_resolve_env_file"]([None, tmp_path / "nope" / "env.py"], None) is None
