"""PURE HDRI / skybox path resolution -- no Isaac, no module-level globals. Moved verbatim from
rerender.py (the ``_expand_path`` / ``_resolve_*`` / ``_discover_datasets`` family) so rerender and
eval share exactly one resolver.

MULTI-ROOT: every resolver accepts either a single root (str) OR an ORDERED list of roots. HDRIs are
resolved against a unified ``{basename -> path}`` index built across the roots where the FIRST root
that has a given file WINS and the default/last root only fills gaps. This lets a run pull skyboxes
from a fast/local dir (e.g. ``/tmp/assets/environments``) first and fall back to the repo default --
useful for slim Docker images + a shared asset store. Each root auto-detects its level: a path that
contains an ``environments`` subdir is descended into it (so ``/tmp/assets`` and
``/tmp/assets/environments`` both work).
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List

# per-dataset manifest filename rerender scans for in each subfolder
SKYBOX_MANIFEST = "dataset.json"


def expand_path(p: str) -> str:
    """Expand ~ and ${ENV_VAR}/$VAR in an asset path so configs can use full/relocatable paths
    (absolute, home-relative, or env-var-based). Relative paths are left relative."""
    if not p:
        return p
    return os.path.expanduser(os.path.expandvars(str(p)))


def normalize_roots(dirs) -> List[str]:
    """Turn a single dir or a list of dirs into an ORDERED list of HDRI roots (dedup, order kept).
    For each entry, if it contains an ``environments`` subdir, descend into it -- so callers may pass
    an assets-root (``/tmp/assets``) or an environments-root (``/tmp/assets/environments``)."""
    if dirs is None:
        return []
    if isinstance(dirs, (str, bytes, os.PathLike)):
        dirs = [dirs]
    out, seen = [], set()
    for d in dirs:
        if not d:
            continue
        d = expand_path(str(d))
        env = os.path.join(d, "environments")
        root = env if os.path.isdir(env) else d
        if root not in seen:
            seen.add(root)
            out.append(root)
    return out


_MULTI_INDEX_CACHE: Dict[tuple, Dict[str, str]] = {}


def build_basename_index(dirs) -> Dict[str, str]:
    """Unified ``{basename -> path}`` HDRI index across the ordered roots. The FIRST root that has a
    given basename WINS (so higher-priority roots override the default). Recursive; cached per
    roots-tuple."""
    roots = tuple(normalize_roots(dirs))
    idx = _MULTI_INDEX_CACHE.get(roots)
    if idx is None:
        idx = {}
        for r in roots:
            for ext in ("hdr", "exr"):
                for p in sorted(glob.glob(os.path.join(r, "**", f"*.{ext}"), recursive=True)):
                    idx.setdefault(os.path.basename(p), p)      # first root wins
        _MULTI_INDEX_CACHE[roots] = idx
    return idx


def asset_source_summary(dirs) -> str:
    """One-line human summary of the resolved asset map: how many HDRIs came from each root (in
    priority order). For logging which dir served which assets."""
    roots = normalize_roots(dirs)
    idx = build_basename_index(dirs)
    counts = {r: 0 for r in roots}
    for p in idx.values():
        for r in roots:
            if p == r or p.startswith(r.rstrip(os.sep) + os.sep):
                counts[r] += 1
                break
    parts = [f"{r} -> {counts[r]}" for r in roots]
    return f"{len(idx)} HDRI(s) across {len(roots)} root(s): " + "; ".join(parts)


def resolve_hdri_names(names, dirs) -> List[str]:
    """Resolve HDRI env ids (Poly Haven ids, e.g. 'cabin') to file paths via the unified multi-root
    basename index, matching <name> or <name>_<res>. First root with a match wins."""
    idx = build_basename_index(dirs)
    out = []
    for nm in names:
        base = os.path.basename(str(nm))       # match the leaf id whether nm is bare or carries a subpath
        exact, prefix = [], []
        for bn, p in idx.items():
            stem = bn.rsplit(".", 1)[0]
            if stem == base:
                exact.append((bn, p))
            elif stem.startswith(base + "_"):
                prefix.append((bn, p))
        hit = (sorted(exact) or sorted(prefix))
        if hit:
            out.append(hit[0][1])
        else:
            print(f"[dr] WARNING skybox: HDRI name '{nm}' not found across {normalize_roots(dirs)}")
    return out


def discover_datasets(root: str) -> Dict[str, dict]:
    """Scan `root`'s immediate subfolders for per-dataset manifests. A subfolder is a dataset iff it
    contains SKYBOX_MANIFEST ('dataset.json'); the dataset name is the folder name. Returns
    {folder_name: {"dir": <subfolder>, "sets": {<subset>: [<filename>, ...]}}}."""
    out: Dict[str, dict] = {}
    try:
        entries = sorted(os.listdir(root))
    except OSError:
        return out
    for name in entries:
        d = os.path.join(root, name)
        mf = os.path.join(d, SKYBOX_MANIFEST)
        if os.path.isdir(d) and os.path.isfile(mf):
            try:
                data = json.load(open(mf))
            except Exception as e:
                print(f"[dr] WARNING skybox: bad manifest {mf}: {e}")
                continue
            out[name] = {"dir": d, "sets": data.get("sets", {})}
    return out


def discover_datasets_multi(dirs) -> Dict[str, dict]:
    """Datasets across the ordered roots. A dataset (subfolder with dataset.json) is taken from the
    FIRST root that defines it; its set definitions come from that root's dataset.json."""
    out: Dict[str, dict] = {}
    for r in normalize_roots(dirs):
        for name, ds in discover_datasets(r).items():
            out.setdefault(name, ds)            # first root wins for a given dataset name
    return out


def files_under(folder: str, files) -> List[str]:
    """Join each HDRI filename to its dataset `folder`. Skips (and warns once about) files not on
    disk, so the returned pool is always valid."""
    out, missing = [], []
    for f in files:
        p = expand_path(os.path.join(folder, str(f)))
        (out if os.path.exists(p) else missing).append(p)
    if missing:
        print(f"[dr] WARNING skybox: {len(missing)} file(s) not found under {folder} "
              f"(e.g. {os.path.basename(missing[0])}); skipped")
    return out


def resolve_set_route(route: str, dirs) -> List[str]:
    """Resolve a skybox `set` selector to HDRI paths. Datasets are auto-discovered across the ordered
    roots (first root that defines a dataset wins for its set list); each listed file is then resolved
    via the unified multi-root basename index, so an individual HDRI may come from a higher-priority
    root than the dataset.json it was listed in. `route` = '<dataset>/<subset>' or bare '<subset>'."""
    datasets = discover_datasets_multi(dirs)
    idx = build_basename_index(dirs)
    roots = normalize_roots(dirs)
    if not datasets:
        print(f"[dr] WARNING skybox: no datasets (*/{SKYBOX_MANIFEST}) found under {roots}")
        return []

    def _files(ds_dir, files):
        out, missing = [], []
        for f in files:
            p = idx.get(os.path.basename(str(f))) or expand_path(os.path.join(ds_dir, str(f)))
            (out if os.path.exists(p) else missing).append(p)
        if missing:
            print(f"[dr] WARNING skybox: {len(missing)} set file(s) not found "
                  f"(e.g. {os.path.basename(missing[0])}); skipped")
        return out

    route = str(route)
    if "/" in route:
        ds_name, subset = route.split("/", 1)
        ds = datasets.get(ds_name)
        if ds is None:
            print(f"[dr] WARNING skybox set: dataset '{ds_name}' not found across {roots} "
                  f"(have: {sorted(datasets)})")
            return []
        if subset not in ds["sets"]:
            print(f"[dr] WARNING skybox set: subset '{subset}' not in dataset '{ds_name}' "
                  f"(have: {sorted(ds['sets'])})")
            return []
        print(f"[dr] skybox set '{route}' -> {len(ds['sets'][subset])} file(s) "
              f"(dataset.json from {ds['dir']}, files via multi-root index)")
        return _files(ds["dir"], ds["sets"][subset])
    # bare subset name: first discovered dataset that defines it (pre-route-syntax compatibility)
    for ds_name, ds in datasets.items():
        if route in ds["sets"]:
            print(f"[dr] skybox set '{route}' -> {len(ds['sets'][route])} file(s) from "
                  f"dataset '{ds_name}' ({ds['dir']})")
            return _files(ds["dir"], ds["sets"][route])
    have = sorted(f"{n}/{s}" for n, ds in datasets.items() for s in ds["sets"])
    print(f"[dr] WARNING skybox set '{route}' not found across {roots} (have: {have})")
    return []


def resolve_skybox_textures(sky: dict, default_hdri_dirs) -> List[str]:
    """Build the final list of HDRI file paths for a skybox/env_sphere flag, from any of (merged):
      * textures:  explicit list of file paths (back-compat; NOT existence-filtered)
      * names:     env names (Poly Haven ids) resolved across the roots
      * set:       a '<dataset>/<subset>' route or bare '<subset>' (datasets auto-discovered)
      * textures_dir: base dir; glob ALL *.hdr/*.exr under it ONLY when no explicit selection.
    `default_hdri_dirs` is a single root or an ORDERED list of roots (first wins, default last). A
    per-flag `textures_dir` overrides with that explicit root."""
    roots = normalize_roots(sky.get("textures_dir") or default_hdri_dirs)
    paths = [expand_path(str(t)) for t in (sky.get("textures") or [])]
    names = list(sky.get("names") or [])
    set_name = sky.get("set")
    if set_name:
        paths += resolve_set_route(set_name, roots)
    if names:
        paths += resolve_hdri_names(names, roots)
    # whole-folder shorthand only when nothing was selected explicitly. RECURSIVE + across all roots.
    if sky.get("textures_dir") and not paths and not names and not set_name:
        for r in roots:
            paths += sorted(glob.glob(os.path.join(r, "**", "*.hdr"), recursive=True)
                            + glob.glob(os.path.join(r, "**", "*.exr"), recursive=True))
    # de-dup preserving order (skip with dedup: false)
    if not sky.get("dedup", True):
        return paths
    seen, uniq = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def resolve_basename(basename: str, dirs) -> str:
    """Resolve an HDRI *basename* (e.g. 'mosaic_tunnel_2k.hdr', as recorded in render_params) back to
    its full path via the unified multi-root index (first root wins). Used by the render_params
    re-render path (values, not yaml). Returns the basename unchanged if not found (caller decides)."""
    if not basename:
        return basename
    if os.path.isabs(basename) and os.path.exists(basename):
        return basename
    return build_basename_index(dirs).get(os.path.basename(basename), basename)


def shuffled_texture_order(textures: List[str], rng) -> List[str]:
    """Deterministic shuffle for without-replacement HDRI cycling (wraps after exhaustion)."""
    order = list(textures)
    rng.shuffle(order)
    return order
