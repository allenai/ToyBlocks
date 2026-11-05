"""The glyph swap must be applied on EVERY render branch, not only the camera-DR one.

`randomize_task_letters` sampled correctly, rewrote pddl_goal/task/glyph_map correctly, and
rendered the ORIGINAL letters, because the only call to the applier lived inside
`apply_wave_cameras`, which render.py reaches from a single `elif randomize_cams:` arm. A flags
file carrying `randomize_task_letters: true` and no camera DR took the `else` (pure render) arm,
so the shaders were never written.

Nothing caught it. Every metadata check passed -- the parquet was perfectly self-consistent,
because glyph_map, pddl_goal and render_params all derive from the same SAMPLING call. The
applier is the only code that warns, so with it never called the log showed zero
'SHADER NOT FOUND' and zero 'MISMATCH': a clean run and untouched pixels look identical.

This pins the structural invariant: at least one call to the applier must be reachable without
any camera-DR or physics-mode condition being true.
"""
import ast

import pytest
from pathlib import Path

RENDER = Path(__file__).resolve().parents[2] / "scripts" / "render" / "render.py"
APPLIER = "apply_wave_task_letters"

#: The two tests below read TOYBLOCKS's interpreter, not TAMP's. The glyph feature spans both
#: repos and its ToyBlocks half may not be on the branch TAMP is pinned by, in which case these
#: assert a feature that is legitimately absent rather than a regression. Skip with a reason:
#: a red suite that is red for a known, unrelated reason trains people to ignore it.
_BS_INTERP = (Path(__file__).resolve().parents[2] / "src" / "toyblocks"
              / "domain_randomization" / "interpreter.py")
_HAS_BS_GLYPHS = _BS_INTERP.is_file() and APPLIER in _BS_INTERP.read_text(encoding="utf-8")
_needs_bs = pytest.mark.skipif(
    not _HAS_BS_GLYPHS,
    reason=f"this ToyBlocks checkout has no {APPLIER}; the glyph feature's ToyBlocks half is "
           f"not on this branch (see tests/test_glyph_cross_repo_pairing.py)")
# Conditions that must NOT gate the glyph swap. `t == 0` is fine (once per wave); these are not.
BRANCH_NAMES = {"randomize_cams", "use_physics"}


def _tree():
    return ast.parse(RENDER.read_text(encoding="utf-8"))


def _names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _applier_calls(tree):
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
        if name == APPLIER:
            out.append(node)
    return out


def _gating_names(tree, target):
    """Every Name appearing in an `if`/`elif` test that encloses `target`."""
    gates, stack = set(), []

    class V(ast.NodeVisitor):
        def visit_If(self, node):
            stack.append(node.test)
            for child in node.body:
                self.visit(child)
            stack.pop()
            # An elif chain's `orelse` holds another If whose own test is pushed when visited;
            # the outer test does not gate it, so walk orelse outside this stack frame.
            for child in node.orelse:
                self.visit(child)

        def generic_visit(self, node):
            if node is target:
                for t in stack:
                    gates.update(_names(t))
            super().generic_visit(node)

    V().visit(tree)
    return gates


def test_applier_is_called_in_render():
    assert _applier_calls(_tree()), (
        f"render.py never calls {APPLIER}; randomize_task_letters would rewrite the text and "
        f"still render the authored glyphs"
    )


def test_at_least_one_call_is_not_gated_by_camera_dr_or_physics():
    tree = _tree()
    ungated = [c for c in _applier_calls(tree) if not (_gating_names(tree, c) & BRANCH_NAMES)]
    assert ungated, (
        f"every call to {APPLIER} is nested under one of {sorted(BRANCH_NAMES)}. A flags file "
        f"with randomize_task_letters and no camera DR then renders the ORIGINAL letters while "
        f"pddl_goal/task/glyph_map all claim the new ones -- self-consistent metadata, untouched "
        f"pixels, and a silent log."
    )


@_needs_bs
def test_applier_is_public_on_the_interpreter():
    """The call must go through a public entry point, not a module-private helper."""
    src = (Path(__file__).resolve().parents[2] / "src" / "toyblocks"
           / "domain_randomization" / "interpreter.py").read_text(encoding="utf-8")
    fns = {n.name for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef)}
    assert APPLIER in fns, f"interpreter.py must define a public {APPLIER}()"


@_needs_bs
def test_public_applier_is_not_self_recursive():
    """Guards the slip that turned the first version of this fix into infinite recursion."""
    src = (Path(__file__).resolve().parents[2] / "src" / "toyblocks"
           / "domain_randomization" / "interpreter.py").read_text(encoding="utf-8")
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.FunctionDef) and n.name == APPLIER:
            inner = {c.func.id for c in ast.walk(n)
                     if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
            assert APPLIER not in inner, f"{APPLIER} calls itself -- RecursionError at render time"
            return
    raise AssertionError(f"{APPLIER} not found in interpreter.py")
