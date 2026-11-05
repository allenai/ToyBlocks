"""Guard: blocksuite does not depend on the removed TAMP submodule.

Datagen (vecgen), play data, TAMPER and the eval oracles all run without TAMP; this keeps
it that way. Every Python file under ``src/``, ``scripts/`` and ``tests/`` is
parsed and any import of TAMP's packages (``tmp3``, ``evals``, ``test_domains``) fails,
including imports inside functions. ``.gitmodules`` must not declare the submodule.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCANNED = ("src", "scripts", "tests")
TAMP_PACKAGES = frozenset({"tmp3", "evals", "test_domains"})

#: Files still on TAMP, pending their port (none left; add an entry only with a port planned).
PENDING_PORT: frozenset[str] = frozenset()


def _tamp_imports(path: Path) -> list[str]:
    """``"line: module"`` for every import of a TAMP package in ``path``."""
    tree = ast.parse(path.read_text(), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules = [node.module]
        found += [f"{node.lineno}: {m}" for m in modules if m.split(".")[0] in TAMP_PACKAGES]
    return found


def _python_files() -> list[Path]:
    files: list[Path] = []
    for top in SCANNED:
        for path in (ROOT / top).rglob("*.py"):
            parts = path.relative_to(ROOT).parts
            if "__pycache__" in parts or "autogen" in parts:  # generated env packages
                continue
            files.append(path)
    return files


def test_no_tamp_imports() -> None:
    offenders: dict[str, list[str]] = {}
    for path in _python_files():
        rel = path.relative_to(ROOT).as_posix()
        if rel in PENDING_PORT:
            continue
        hits = _tamp_imports(path)
        if hits:
            offenders[rel] = hits
    assert not offenders, f"TAMP imports (the submodule is gone): {offenders}"


def test_pending_port_list_is_current() -> None:
    """An allow-listed file that no longer imports TAMP (or no longer exists) must leave the list."""
    stale = sorted(
        rel
        for rel in PENDING_PORT
        if not (ROOT / rel).is_file() or not _tamp_imports(ROOT / rel)
    )
    assert not stale, f"ported or removed, drop from PENDING_PORT: {stale}"


def test_no_tamp_submodule_declared() -> None:
    gitmodules = (ROOT / ".gitmodules").read_text()
    assert 'submodule "TAMP"' not in gitmodules
    assert "allenai/TAMP" not in gitmodules
