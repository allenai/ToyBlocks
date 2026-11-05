"""Helpers for the installation checks (``test_install.py`` and ``test_install_gpu.py``).

An optional extra counts as installed when its anchor distribution is (:data:`EXTRA_ANCHORS`), so
the checks cover exactly the extras this environment was installed with. Requirements are read from
``pyproject.toml``; where ``[tool.uv] override-dependencies`` overrides a package, the override's
version range is the one that has to hold, because that is what ``uv pip install`` resolved.
"""

from __future__ import annotations

import importlib.metadata as md
import os
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement
from packaging.utils import NormalizedName, canonicalize_name

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
PYPROJECT: dict[str, Any] = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
PROJECT_NAME: NormalizedName = canonicalize_name(PYPROJECT["project"]["name"])

EXTRA_ANCHORS: dict[str, str] = {
    "isaac": "isaaclab",
    "torch-cu128": "torch",  # and only a CUDA 12.8 build counts, see extra_installed
    "vecgen": "nvidia-curobo",
    "maniflow": "maniflow-tb",
    "agentic": "agentic-robotics",
}
"""The distribution whose presence means an extra was installed."""


def dist_version(name: str) -> str | None:
    """The installed version of a distribution, or None when it is not installed."""
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def extra_installed(extra: str) -> bool:
    """Whether this environment was installed with `extra` (judged by its anchor distribution)."""
    version = dist_version(EXTRA_ANCHORS[extra])
    if extra == "torch-cu128":
        return version is not None and version.endswith("+cu128")
    return version is not None


def _applies(requirement: Requirement, extra: str = "") -> bool:
    return requirement.marker is None or requirement.marker.evaluate({"extra": extra})


def overrides() -> dict[NormalizedName, Requirement]:
    """``[tool.uv] override-dependencies`` that apply on this platform, by package name."""
    entries = PYPROJECT.get("tool", {}).get("uv", {}).get("override-dependencies", [])
    parsed = (Requirement(entry) for entry in entries)
    return {canonicalize_name(r.name): r for r in parsed if _applies(r)}


def unmet(requirements: list[str]) -> list[str]:
    """Describe every requirement (and every requirement of its requested extras) that does not hold.

    :param requirements: PEP 508 requirement strings, as in ``pyproject.toml``
    :return: One line per missing package or out-of-range version; empty when all hold
    """
    pinned = overrides()
    problems: list[str] = []

    def check(requirement: Requirement, needed_for: str) -> None:
        name = canonicalize_name(requirement.name)
        if name == PROJECT_NAME:
            return
        override = pinned.get(name)
        specifier = override.specifier if override is not None else requirement.specifier
        version = dist_version(requirement.name)
        if version is None:
            problems.append(f"{requirement.name} is not installed{needed_for}")
        elif specifier and not specifier.contains(version, prereleases=True):
            source = "the pyproject override" if override is not None else "pyproject"
            problems.append(
                f"{requirement.name} {version} is outside {specifier} ({source}){needed_for}"
            )

    for entry in requirements:
        requirement = Requirement(entry)
        if not _applies(requirement):
            continue
        check(requirement, "")
        if not requirement.extras or dist_version(requirement.name) is None:
            continue
        for sub_entry in md.requires(requirement.name) or []:
            sub = Requirement(sub_entry)
            # Only what the requested extras add (applies with the extra, not without it).
            if not _applies(sub) and any(_applies(sub, extra) for extra in requirement.extras):
                check(
                    sub, f" (needed by {requirement.name}[{','.join(sorted(requirement.extras))}])"
                )
    return problems


def run_python(
    code: str, timeout: float, accept_eula: bool = False
) -> subprocess.CompletedProcess[str]:
    """Run `code` in a fresh interpreter of this environment, from the repository root.

    The code runs as a script file, not with ``python -c``: Isaac Sim crashes (SIGSEGV) while
    starting when the main module is ``-c``.

    :param code: Python source to run
    :param timeout: Seconds before the run is killed
    :param accept_eula: Keep ``OMNI_KIT_ACCEPT_EULA``; without it, anything that starts Isaac Sim
        fails at the EULA prompt instead of starting it (stdin is closed)
    :return: The finished process, with text stdout and stderr
    """
    with tempfile.TemporaryDirectory(prefix="toyblocks_install_check_") as tmp:
        script = Path(tmp) / "check.py"
        script.write_text(code)
        return run([sys.executable, str(script)], timeout, accept_eula)


def run(
    argv: list[str], timeout: float, accept_eula: bool = False
) -> subprocess.CompletedProcess[str]:
    """Run a command from the repository root (see :func:`run_python`)."""
    env = dict(os.environ)
    if not accept_eula:
        env.pop("OMNI_KIT_ACCEPT_EULA", None)
    return subprocess.run(
        argv,
        cwd=REPO_ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def tail(process: subprocess.CompletedProcess[str], lines: int = 40) -> str:
    """The last lines of a finished process's combined output, for assertion messages."""
    output = (process.stdout or "") + (process.stderr or "")
    return "\n".join(output.splitlines()[-lines:])
