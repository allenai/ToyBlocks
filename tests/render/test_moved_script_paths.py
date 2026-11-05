"""Every in-repo reference to a MOVED render script must resolve.

The move out of TAMP left six hardcoded ``TAMP/scripts/render/`` paths behind, and a repo-wide
sweep found 27 more outside ``scripts/render/``. Four were functional, and the worst of them
--- ``run_datagen_pipeline.py``'s live subprocess argv --- is invisible to every other test:
unit tests never exercise that constant, so datagen would have gone on launching TAMP's copy.

These tests are cheap and would have caught all four.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: Helpers that genuinely did NOT move and are still only in TAMP. A reference to one of these
#: is correct and must not be "fixed".
STAYED_IN_TAMP = {
    "attach_data_from_source.sh",
    "submit_partial_dataset.py",
    "render_flags.example.yaml",
    "download_hdris.sh",
}

SEARCH_DIRS = ("src", "scripts", "tests", "docs")
SUFFIXES = (".py", ".yaml", ".yml", ".md", ".sh")


def _repo_files():
    for d in SEARCH_DIRS:
        for p in (ROOT / d).rglob("*"):
            if p.is_file() and p.suffix in SUFFIXES:
                yield p


def test_no_reference_to_a_moved_script_still_points_into_tamp():
    """A TAMP/scripts/render/<x> reference is only allowed when <x> really stayed there."""
    offenders = []
    for p in _repo_files():
        for n, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            for ref in re.findall(r"TAMP/scripts/render/([A-Za-z0-9_.*-]+)", line):
                if ref in STAYED_IN_TAMP or "*" in ref:
                    continue
                if (ROOT / "scripts/render" / ref).exists() or (
                    ROOT / "src/toyblocks/domain_randomization/flags" / ref
                ).exists():
                    offenders.append(f"{p.relative_to(ROOT)}:{n} -> {ref} (it moved here)")
    assert not offenders, "stale TAMP paths for scripts that moved:\n  " + "\n  ".join(offenders)


def test_ci_collects_the_render_suite():
    """tests/render must be named in a workflow, or the moved tests never run."""
    wf = ROOT / ".github/workflows"
    assert any("tests/render" in f.read_text() for f in wf.glob("*.y*ml")), (
        "no workflow runs tests/render -- CI invokes pytest with explicit paths"
    )
