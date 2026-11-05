"""Installation checks, fast tier: run ``pytest tests/install`` after following the installation guide.

Nothing here needs a GPU or starts Isaac Sim, so the whole tier takes seconds. It checks the extras
this environment was installed with (see ``conftest.EXTRA_ANCHORS``) and skips the others:

- this checkout is the ``toyblocks`` that Python imports;
- every requirement of the base install and of each installed extra is installed, in the version
  range ``pyproject.toml`` (or its uv override) allows;
- what each installed extra's workflows import, imports (in a fresh interpreter, so a module that
  would start Isaac Sim fails instead);
- the submodules an installed extra needs are checked out at the commit ToyBlocks pins, and their
  packages are imported from them;
- the assets are downloaded.

The GPU tier (``test_install_gpu.py``, ``--install-gpu``) runs Isaac Sim, cuRobo and vecgen.
"""

from __future__ import annotations

import json
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest

from tests.install.helpers import (
    EXTRA_ANCHORS,
    PYPROJECT,
    REPO_ROOT,
    dist_version,
    extra_installed,
    overrides,
    run,
    run_python,
    tail,
    unmet,
)

EXTRA_IMPORTS: dict[str, tuple[str, ...]] = {
    "base": ("toyblocks", "toyblocks.tamp.blocks.template.codegen"),
    "vecgen": (
        "curobo",
        "warp",
        "av",
        "lerobot.datasets.lerobot_dataset",
        "toyblocks.vecgen.symbolic.pddl",
        "toyblocks.vecgen.recording._video_encoder",
        "toyblocks.vecgen.recording.merge",
    ),
    "maniflow": ("hvla", "olmo", "toyblocks.serving"),
    "agentic": ("mcp", "toyblocks.serving", "agentic_robotics.adapters.toyblocks.cli"),
}
"""Modules each installed extra's workflows import that load without starting Isaac Sim.
(``isaaclab`` itself starts Isaac Sim on import; the GPU tier covers it.)"""

EXTRA_SNIPPETS: dict[str, str] = {
    # The uv overrides move starlette past what Isaac Sim's fastapi pin accepts; an incompatible
    # pair fails here ("Router.__init__() got an unexpected keyword argument 'on_startup'").
    "isaac": "from fastapi import FastAPI\nFastAPI()\n",
}
"""Extra code to run after an extra's imports."""

CONSOLE_SCRIPTS: dict[str, tuple[str, ...]] = {
    "base": ("get-toyblocks-assets",),
    "agentic": ("agentic-robotics",),
}

SUBMODULES: dict[str, tuple[tuple[str, ...], str]] = {
    "curobo": (("vecgen",), "curobo"),
    "maniflow-tb": (("maniflow",), "hvla"),
    "molmo2": (("maniflow",), "olmo"),
    "agentic_robotics": (("agentic",), "agentic_robotics"),
}
"""Submodule path -> (the extras that need it, the package it provides)."""


def _installed_or_skip(extra: str) -> None:
    if extra != "base" and not extra_installed(extra):
        pytest.skip(f"the {extra} extra is not installed")


def _git(*args: str, cwd: Path = REPO_ROOT) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()


def test_toyblocks_is_this_checkout() -> None:
    spec = find_spec("toyblocks")
    assert spec is not None and spec.origin is not None, (
        "toyblocks is not installed: uv pip install -e ."
    )
    origin = Path(spec.origin).resolve()
    assert origin.is_relative_to(REPO_ROOT / "src" / "toyblocks"), (
        f"Python imports toyblocks from {origin}, not from this checkout ({REPO_ROOT}). "
        "Activate this checkout's .venv, or run uv pip install -e . here."
    )


def test_base_requirements() -> None:
    problems = unmet(PYPROJECT["project"]["dependencies"])
    assert not problems, "The base install is incomplete:\n  " + "\n  ".join(problems)


@pytest.mark.parametrize("extra", sorted(EXTRA_ANCHORS))
def test_extra_requirements(extra: str) -> None:
    _installed_or_skip(extra)
    problems = unmet(PYPROJECT["project"]["optional-dependencies"][extra])
    assert not problems, (
        f"The {extra} extra is incomplete:\n  "
        + "\n  ".join(problems)
        + "\nReinstall with every extra you use in one command, e.g. "
        f'uv pip install -e ".[dev,isaac,torch-cu128,{extra}]"'
    )


def test_overrides_hold() -> None:
    problems = [
        f"{requirement.name} {version} is outside {requirement.specifier}"
        for requirement in overrides().values()
        if (version := dist_version(requirement.name)) is not None
        and requirement.specifier
        and not requirement.specifier.contains(version, prereleases=True)
    ]
    assert not problems, (
        "Installed versions are outside pyproject's [tool.uv] override-dependencies. The environment was "
        'installed from an older pyproject.toml, or without uv; rerun uv pip install -e ".[...]" with '
        "your extras:\n  " + "\n  ".join(problems)
    )


@pytest.mark.parametrize("extra", sorted(EXTRA_IMPORTS.keys() | EXTRA_SNIPPETS.keys()))
def test_extra_imports(extra: str) -> None:
    _installed_or_skip(extra)
    modules = EXTRA_IMPORTS.get(extra, ())
    code = (
        "import importlib\n"
        f"for name in {modules!r}:\n"
        "    importlib.import_module(name)\n"
        + EXTRA_SNIPPETS.get(extra, "")
        + "print('IMPORTS_OK')\n"
    )
    process = run_python(code, timeout=300)
    assert process.returncode == 0 and "IMPORTS_OK" in process.stdout, (
        f"The {extra} extra's imports fail ({', '.join(modules) or 'snippet'}):\n{tail(process)}"
    )


@pytest.mark.parametrize("extra", sorted(CONSOLE_SCRIPTS))
def test_console_scripts(extra: str) -> None:
    _installed_or_skip(extra)
    bin_dir = Path(sys.executable).parent
    missing = [name for name in CONSOLE_SCRIPTS[extra] if not (bin_dir / name).exists()]
    assert not missing, f"Missing from {bin_dir}: {', '.join(missing)}"


def test_agentic_lists_tasks() -> None:
    _installed_or_skip("agentic")
    process = run([str(Path(sys.executable).parent / "agentic-robotics"), "tasks"], timeout=120)
    assert process.returncode == 0 and "ai2_stacked" in process.stdout, tail(process)


def test_every_submodule_is_checked() -> None:
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    listed = set(_git("config", "-f", ".gitmodules", "--get-regexp", r"\.path$").split()[1::2])
    assert listed == set(SUBMODULES), (
        f".gitmodules lists {sorted(listed)}; update SUBMODULES to match"
    )


@pytest.mark.parametrize("path", sorted(SUBMODULES))
def test_submodule(path: str) -> None:
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    extras, package = SUBMODULES[path]
    needed_by = [extra for extra in extras if extra_installed(extra)]
    if not (REPO_ROOT / path / ".git").exists():
        if needed_by:
            pytest.fail(
                f"{path} is not checked out, but the {needed_by[0]} extra needs it: "
                f"git submodule update --init {path}"
            )
        pytest.skip(f"{path} is not checked out and no installed extra needs it")
    pinned = _git("ls-files", "-s", "--", path).split()[1]
    checked_out = _git("rev-parse", "HEAD", cwd=REPO_ROOT / path)
    assert checked_out == pinned, (
        f"{path} is at {checked_out[:12]}, but ToyBlocks pins {pinned[:12]}: git submodule update --init {path}"
    )
    if needed_by:
        spec = find_spec(package)
        location: Path | None = None
        if spec is not None and spec.origin:
            location = Path(spec.origin).resolve()
        elif (
            spec is not None and spec.submodule_search_locations
        ):  # a namespace package (cuRobo v2)
            location = Path(next(iter(spec.submodule_search_locations))).resolve()
        assert location is not None and location.is_relative_to((REPO_ROOT / path).resolve()), (
            f"Python imports {package} from {location}, not from this checkout's {path}/: reinstall with "
            'uv pip install -e ".[...]" from this checkout'
        )


def test_assets_downloaded() -> None:
    manifest: dict[str, dict[str, object]] = json.loads(
        (REPO_ROOT / "asset_manifest.json").read_text()
    )
    missing = [
        f"assets/{category}/{name}"
        for category, entries in manifest.items()
        for name in entries
        if not ((path := REPO_ROOT / "assets" / category / name).is_dir() and any(path.iterdir()))
    ]
    assert not missing, (
        "Assets are missing: "
        + ", ".join(missing)
        + ". With the venv active, run: get-toyblocks-assets --asset-manifest asset_manifest.json"
    )
