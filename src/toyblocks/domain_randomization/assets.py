"""Default asset locations for domain randomization, resolved from toyblocks constants with the
same env-var overrides + repo-relative fallback rerender.py used. This replaces rerender.py's
module-level ``DEFAULT_HDRI_DIR`` / ``DEFAULT_TABLE_MATERIALS`` globals with an explicit, passable
config object so nothing relies on import-time module state.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# Repo-relative fallbacks (only hit if `toyblocks` is somehow not importable, e.g. a standalone
# checkout). This file lives at src/toyblocks/domain_randomization/assets.py -> three parents up is
# the repo root.
_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")


def default_hdri_dir() -> str:
    """HDRI/skybox pool root. Single source of truth = toyblocks.ENVIRONMENT_ASSET_DIR
    (asset-manager managed); overridable via the RERENDER_HDRI_DIR env var. Identical resolution to
    the former rerender.DEFAULT_HDRI_DIR so paths never change under the refactor."""
    try:
        from toyblocks import ENVIRONMENT_ASSET_DIR  # noqa: WPS433 (local import: optional dep)
        base = str(ENVIRONMENT_ASSET_DIR)
    except Exception:
        base = os.path.join(_REPO_ROOT, "assets", "environments")
    return os.environ.get("RERENDER_HDRI_DIR") or base


def default_table_materials() -> str:
    """Default table-material library. Single source of truth = toyblocks.TABLE_MATERIALS_FILEPATH;
    overridable via RERENDER_TABLE_MATERIALS. Identical to the former rerender.DEFAULT_TABLE_MATERIALS."""
    try:
        from toyblocks import TABLE_MATERIALS_FILEPATH  # noqa: WPS433
        base = str(TABLE_MATERIALS_FILEPATH)
    except Exception:
        base = os.path.join(_REPO_ROOT, "assets", "scenes", "toyblocks_real", "usd", "table",
                            "materials.json")
    return os.environ.get("RERENDER_TABLE_MATERIALS") or base


@dataclass(frozen=True)
class AssetPaths:
    """Where the DR asset pools live. Pass an instance instead of reading globals. ``hdri_dirs`` is an
    ORDERED tuple of HDRI/skybox roots (first wins, repo default last) so a run can prefer a fast or
    shared store (e.g. ``/tmp/assets/environments``) and fall back to the repo default -- enabling
    slim Docker images + a unified asset store."""
    hdri_dirs: tuple
    table_materials: str

    @property
    def hdri_dir(self) -> str:
        """Primary (highest-priority) HDRI root -- back-compat for single-dir readers."""
        return self.hdri_dirs[0] if self.hdri_dirs else ""

    @classmethod
    def default(cls) -> "AssetPaths":
        return cls(hdri_dirs=(default_hdri_dir(),), table_materials=default_table_materials())

    @classmethod
    def with_search_paths(cls, search_paths=None, table_materials=None) -> "AssetPaths":
        """Build with an ordered HDRI search: the given ``search_paths`` (in order) THEN the repo
        default (last). Each entry may be an assets-root or an environments-root (auto-detected by
        the resolver)."""
        roots = tuple([str(p) for p in (search_paths or []) if p] + [default_hdri_dir()])
        return cls(hdri_dirs=roots, table_materials=table_materials or default_table_materials())
