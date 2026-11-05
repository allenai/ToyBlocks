"""randomize_task_letters spans BOTH repos; the pairing must fail loudly, not per-fragment.

TAMP is pinned by toyblocks's submodule pointer, so the two halves of the glyph feature can land
at different times. On 2026-10-06 ToyBlocks main pinned a TAMP whose rerender.py calls
interpreter.sample_wave_task_letters, while main's own interpreter had no such function and no
glyphs.py at all. Result: EVERY fragment of EVERY run -- glyph or not, because the applier is
called at t == 0 before the use_physics/randomize_cams branch -- died with

    AttributeError: module 'toyblocks.domain_randomization.interpreter'
                    has no attribute 'sample_wave_task_letters'

and the worker then wedged at SM 2% holding 29.5 GiB of VRAM instead of exiting.

Two properties are pinned here:
  1. a run that does NOT ask for glyphs never touches the glyph API, so an older ToyBlocks works;
  2. a run that DOES ask for it fails at STARTUP with a message naming the fix.
"""
import ast
import sys
import types
from pathlib import Path

import pytest

RERENDER = Path(__file__).resolve().parents[2] / "scripts" / "rerender" / "rerender.py"
SRC = RERENDER.read_text(encoding="utf-8")


def _names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


@pytest.fixture(scope="module")
def startup_check():
    """Compile the LIVE pairing check out of __main__ and run it against a fake toyblocks."""
    tree = ast.parse(SRC)
    hits = [n for n in ast.walk(tree)
            if isinstance(n, ast.If) and "_early" in _names(n.test)
            and "sample_wave_task_letters" in ast.dump(n)]
    assert len(hits) == 1, f"expected one pairing check, found {len(hits)}"
    wrapper = ast.parse("def _chk():\n    pass\n")
    wrapper.body[0].body = [hits[0]]
    ast.fix_missing_locations(wrapper)
    code = compile(wrapper, "rerender", "exec")

    def run(*, flag, has_api):
        mod = types.ModuleType("toyblocks.domain_randomization.interpreter")
        if has_api:
            mod.sample_wave_task_letters = lambda *a, **k: {}
            mod.apply_wave_task_letters = lambda *a, **k: {}
        pkg = types.ModuleType("toyblocks.domain_randomization")
        pkg.interpreter = mod
        saved = {k: sys.modules.get(k) for k in
                 ("toyblocks", "toyblocks.domain_randomization",
                  "toyblocks.domain_randomization.interpreter")}
        root = types.ModuleType("toyblocks"); root.domain_randomization = pkg
        sys.modules["toyblocks"] = root
        sys.modules["toyblocks.domain_randomization"] = pkg
        sys.modules["toyblocks.domain_randomization.interpreter"] = mod
        try:
            ns = {"_early": {"randomize_task_letters": flag} if flag is not None else {}}
            exec(code, ns)
            ns["_chk"]()
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
    return run


def test_old_toyblocks_is_fine_when_glyphs_are_not_requested(startup_check):
    """THE MAIN CASE: an ordinary rerender on a ToyBlocks that predates glyphs.py must run."""
    startup_check(flag=None, has_api=False)
    startup_check(flag=False, has_api=False)


def test_old_toyblocks_fails_loudly_when_glyphs_ARE_requested(startup_check):
    with pytest.raises(SystemExit) as e:
        startup_check(flag=True, has_api=False)
    msg = str(e.value)
    assert "sample_wave_task_letters" in msg and "glyphs.py" in msg
    assert "submodule" in msg or "ToyBlocks" in msg, "must name the fix, not just the symptom"


def test_new_toyblocks_passes(startup_check):
    startup_check(flag=True, has_api=True)
    startup_check(flag={"glyphs": ["A-C"]}, has_api=True)


def test_both_call_sites_are_gated_on_the_flag():
    """Neither entry point may be reached unconditionally -- that is what broke main."""
    for fn in ("sample_wave_task_letters", "apply_wave_task_letters"):
        i = SRC.index(f"_dr_interp.{fn}(")
        window = SRC[max(0, i - 400):i + 120]
        assert 'getattr(_ir, "task_letters", False)' in window, \
            f"{fn} is not gated on the flag"
