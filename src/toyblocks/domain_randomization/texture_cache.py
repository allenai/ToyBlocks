"""RAM-backed HDRI cache: a path rewriter, not a pixel cache.

WHY A PATH REWRITER. The renderer opens the HDRI itself -- this package only ever hands USD an
asset path (``skybox.py`` dome ``CreateTextureFileAttr``, ``env_sphere.py`` ``inputs:file``). So a
Python-side pixel cache cannot intercept the read; there is no API to hand Omniverse decoded bytes
for a DomeLight. What we CAN control is the path. Copy the file into a tmpfs (``/dev/shm``, which
is RAM) once and hand out that path instead, and every subsequent open is a memory read.

WHY IT IS WORTH DOING. Measured with ``strace -e trace=openat`` on the shipped ``scene_spot_shadows``
at 8 envs: 17 HDRI opens per wave (1 dome + 1 per env sphere), each a FULL re-read of the file
(measured 4 opens of one file -> 27.3 MB of ``pread64``, i.e. ~6.8 MB every time). Reuse does not
help: a file served two waves ago is opened and read again. At the production 128-256 envs that is
~129-257 opens and ~0.9-1.7 GB of file I/O PER WAVE, forever. On a network filesystem that is the
bottleneck.

SIZING (measured through the REAL resolver, not by globbing the directory -- the `set:` routes are
much smaller than the folders they live in):
  polyhaven/rooms    197 files   1.21 GiB raw   2048x1024
  polyhaven/all      134 files   0.78 GiB raw   2048x1024
  indoor-2018/all   2113 files  43.84 GiB raw   4096x2048   (NOT resolvable from every asset root)
  union addressable by the shipped scene yamls: 294 files, 1.76 GiB
Raw bytes are what tmpfs holds, and raw is 3.7-4.7x smaller than decoded, which is the second
reason to cache files rather than pixels. Real working sets are ~0.8 GiB (dagger) and <=1.0 GiB per
rerender process -- far below the folder totals, which is why the cache is specified as a PLAN
rather than a byte budget (see PLAN-DRIVEN WARM-UP below).

/dev/shm GOTCHA: tmpfs defaults to half of RAM on a bare host, but Docker defaults it to **64 MB**
unless the job spec raises it. If the cache dir is too small the copy fails; that is handled -- a
failed copy falls back to the original path, so the render is never broken by a cache problem.

CROSS-PROCESS SHARING: entries are named ``<stem>-<size>-<mtime_ns><ext>`` and published by atomic
rename, so N worker processes on one node share one copy and never observe a partial file. That
makes the cache cheaper the more workers you pack: 8 rerender workers on a node pay for one copy,
not eight.

NOT IMPLEMENTED HERE (deliberately -- design only, see README): lookahead preloading with an
``offset``/``max_size`` window that materialises the HDRIs of upcoming waves while the current one
renders. The hooks for it are ``preload()`` and ``pin()``; nothing calls them yet.
"""
from __future__ import annotations

import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional

__all__ = ["configure", "resolve", "stats", "reset", "preload", "pin", "TextureCache"]

# ENVIRONMENT OVERRIDE. Turning the cache on is a DEPLOYMENT decision (does this node have tmpfs?
# how big?), not a scene-config one -- the same argument as the per-worker slice. So it is settable
# without touching a shared yaml, which also means a job can enable it on a flags file it does not
# own (e.g. one living in a submodule):
#     TOYBLOCKS_TEXTURE_CACHE=1                 enable with defaults
#     TOYBLOCKS_TEXTURE_CACHE=0                 force OFF, overriding the yaml
#     TOYBLOCKS_TEXTURE_CACHE_DIR=/dev/shm/foo
#     TOYBLOCKS_TEXTURE_CACHE_MAX_GB=2
#     TOYBLOCKS_TEXTURE_CACHE_PRELOAD=auto      (or a comma-separated list of set routes)
# The env wins over the yaml, so an operator can always overrule a config they cannot edit.
_DEFAULT_DIR = "/dev/shm/toyblocks_texture_cache"


class TextureCache:
    """Copy-on-first-use file cache in a directory that is expected to be tmpfs.

    ``max_bytes=0`` means unbounded (never evict) -- correct on a 440 GB node where both sets fit.
    Eviction is LRU by last *use*, skipping anything pinned.
    """

    def __init__(self, root: str = _DEFAULT_DIR, max_bytes: int = 0, verbose: bool = False):
        self.root = root
        self.max_bytes = int(max_bytes)
        self.verbose = verbose
        self._lock = threading.Lock()
        self._entries: Dict[str, str] = {}      # src abspath -> cached abspath
        self._size: Dict[str, int] = {}         # cached abspath -> bytes
        self._order: List[str] = []             # src abspaths, LRU first
        self._pinned: set = set()               # src abspaths that must not be evicted
        self.hits = self.misses = self.evictions = self.errors = 0
        self.bytes_in = 0
        os.makedirs(self.root, exist_ok=True)

    # -- internals -----------------------------------------------------------
    def _key(self, src: str) -> str:
        """Content-ish key: size+mtime, so a re-downloaded asset bundle invalidates naturally."""
        st = os.stat(src)
        stem, ext = os.path.splitext(os.path.basename(src))
        return f"{stem}-{st.st_size}-{st.st_mtime_ns}{ext}"

    def _touch(self, src: str) -> None:
        try:
            self._order.remove(src)
        except ValueError:
            pass
        self._order.append(src)

    def _evict_for(self, need: int) -> bool:
        """Free space for `need` bytes. Returns False if it cannot be freed."""
        if self.max_bytes <= 0:
            return True
        used = sum(self._size.values())
        if used + need <= self.max_bytes:
            return True
        for src in list(self._order):
            if used + need <= self.max_bytes:
                break
            if src in self._pinned:
                continue
            dst = self._entries.pop(src, None)
            if dst is None:
                continue
            used -= self._size.pop(dst, 0)
            self._order.remove(src)
            self.evictions += 1
            try:
                os.unlink(dst)
            except OSError:
                pass       # another process may share this entry; losing the race is harmless
        return used + need <= self.max_bytes

    # -- public --------------------------------------------------------------
    def resolve(self, src: str) -> str:
        """Return a path to use in USD: the cached copy if possible, else `src` unchanged.

        Never raises. Any failure (missing file, full tmpfs, permissions) falls back to the
        original path, so a cache problem degrades performance and never correctness.
        """
        src = os.path.abspath(src)
        try:
            with self._lock:
                hit = self._entries.get(src)
                if hit is not None and os.path.exists(hit):
                    self.hits += 1
                    self._touch(src)
                    return hit
                key = self._key(src)
                dst = os.path.join(self.root, key)
                need = os.path.getsize(src)
                if os.path.exists(dst):
                    # another process already published it -- adopt without copying
                    self._entries[src] = dst
                    self._size[dst] = need
                    self._touch(src)
                    self.hits += 1
                    return dst
                if not self._evict_for(need):
                    self.misses += 1
                    return src                      # cannot fit even after eviction
            # Copy outside the lock so threads fetch in parallel (env_sphere's luminance pass).
            # Two threads racing on one src each copy to their own tmp; the second replace wins
            # with identical bytes. A max_bytes budget can overshoot by the in-flight copies.
            tmp = f"{dst}.tmp{os.getpid()}.{threading.get_ident()}"
            shutil.copyfile(src, tmp)
            os.replace(tmp, dst)                    # atomic publish
            with self._lock:
                self._entries[src] = dst
                self._size[dst] = need
                self._touch(src)
                self.misses += 1
                self.bytes_in += need
                if self.verbose:
                    print(f"[texcache] + {os.path.basename(src)} ({need/1e6:.1f} MB), "
                          f"{len(self._entries)} entries, {sum(self._size.values())/2**30:.2f} GiB")
            return dst
        except Exception as e:
            with self._lock:
                self.errors += 1
            if self.verbose:
                print(f"[texcache] fallback for {os.path.basename(src)}: {e}")
            return src

    def pin(self, srcs: Iterable[str]) -> None:
        """Mark paths un-evictable (the current wave's working set)."""
        with self._lock:
            self._pinned = {os.path.abspath(s) for s in srcs}

    def preload(self, srcs: Iterable[str]) -> int:
        """Materialise paths ahead of use. Returns how many are now resident.

        DESIGN HOOK: this is what a lookahead thread would call with the next window of the
        queue. Nothing calls it yet.
        """
        srcs = list(srcs)
        if not srcs:
            return 0
        with ThreadPoolExecutor(max_workers=min(16, len(srcs))) as ex:
            got = list(ex.map(self.resolve, srcs))
        return sum(g != os.path.abspath(s) for g, s in zip(got, srcs))

    def stats(self) -> dict:
        with self._lock:
            return {"entries": len(self._entries),
                    "resident_gib": round(sum(self._size.values()) / 2**30, 3),
                    "hits": self.hits, "misses": self.misses,
                    "evictions": self.evictions, "errors": self.errors,
                    "copied_gib": round(self.bytes_in / 2**30, 3),
                    "root": self.root, "max_gib": round(self.max_bytes / 2**30, 2)}


# -- module singleton, so the two call sites need no plumbing --------------------
_CACHE: Optional[TextureCache] = None


def configure(spec) -> Optional[TextureCache]:
    """Install the process-wide cache from a flags['texture_cache'] block, env-overridable.

    spec: None/False -> disabled (resolve() becomes a plain abspath, zero overhead).
          True        -> defaults.
          dict        -> {enabled, dir, max_gb, verbose, preload}
    """
    global _CACHE
    env = os.environ.get("TOYBLOCKS_TEXTURE_CACHE")
    if env is not None:
        if env.strip().lower() in ("0", "false", "no", "off", ""):
            _CACHE = None
            print("[texcache] disabled by TOYBLOCKS_TEXTURE_CACHE")
            return None
        spec = dict(spec) if isinstance(spec, dict) else {}
        spec["enabled"] = True
        if env.strip().lower() not in ("1", "true", "yes", "on"):
            spec["dir"] = env.strip()          # a non-boolean value is taken as the cache dir
    if not spec:
        _CACHE = None
        return None
    if spec is True:
        spec = {}
    spec = dict(spec)
    if not spec.get("enabled", True):
        _CACHE = None
        return None
    _e = os.environ.get
    if _e("TOYBLOCKS_TEXTURE_CACHE_DIR"):
        spec["dir"] = _e("TOYBLOCKS_TEXTURE_CACHE_DIR")
    if _e("TOYBLOCKS_TEXTURE_CACHE_MAX_GB"):
        spec["max_gb"] = float(_e("TOYBLOCKS_TEXTURE_CACHE_MAX_GB"))
    if _e("TOYBLOCKS_TEXTURE_CACHE_PRELOAD"):
        v = _e("TOYBLOCKS_TEXTURE_CACHE_PRELOAD").strip()
        spec["preload"] = "auto" if v.lower() == "auto" else {"sets": [s for s in v.split(",") if s]}
    _CACHE = TextureCache(root=str(spec.get("dir") or _DEFAULT_DIR),
                          max_bytes=int(float(spec.get("max_gb", 0)) * 2**30),
                          verbose=bool(spec.get("verbose", False)))
    print(f"[texcache] enabled: root={_CACHE.root} "
          f"max={'unbounded' if _CACHE.max_bytes <= 0 else '%.1f GiB' % (_CACHE.max_bytes / 2**30)}")
    return _CACHE


def resolve(path: str) -> str:
    """The one call the write sites make. Identical to os.path.abspath when disabled."""
    if _CACHE is None or not path:
        return os.path.abspath(path) if path else path
    return _CACHE.resolve(path)


def pin(srcs: Iterable[str]) -> None:
    if _CACHE is not None:
        _CACHE.pin(srcs)


def preload(srcs: Iterable[str]) -> int:
    return _CACHE.preload(srcs) if _CACHE is not None else 0


def stats() -> Optional[dict]:
    return _CACHE.stats() if _CACHE is not None else None


def reset() -> None:
    global _CACHE
    _CACHE = None


# =============================================================================
# PLAN-DRIVEN WARM-UP
# =============================================================================
# A size ceiling is the wrong knob for a rerender worker: we already KNOW which HDRIs it will
# touch. But WHICH ones is a property of the worker's lease, not of the yaml -- an orchestrator
# hands each worker a different slice, and the yaml is shared by all of them. So the slice is
# never written in the yaml; the plan is derived automatically from the episode keys this process
# was actually given.
#
#   texture_cache:
#     enabled: true
#     preload: auto              # <- derive from THIS worker's episode keys. No slice in yaml.
#     # preload: {sets: [polyhaven/rooms]}   # or name whole routes explicitly
#     # preload: {auto: true, sets: [...]}   # or both (union)
#     max_gb: 0                  # optional ceiling; see below for when it is the right knob
#
# `auto` is exact for the rerender path, where setup() is handed the job's whole episode list up
# front, so the plan is precisely what this worker will draw and nothing else.
#
# It works because _sample_episode_env_spheres is a pure function of (spec, keys, seed): it builds
# its own rng and mutates nothing, so projecting the worker's own keys cannot disturb the live
# stream. The dome adds one path per wave from a separate bucket whose cursor we must not touch;
# that is one file against N spheres per wave, so the planner covers the spheres exactly and lets
# the dome warm on demand.
#
# WHEN max_gb IS THE RIGHT KNOB INSTEAD (dagger/GRPO). There the wrapper is installed without
# episode_keys, so setup() sees an empty list and there is no finite episode list to project --
# training runs indefinitely and keys arrive per reset. `auto` degrades to a no-op there, by
# design, and the cache fills on demand. Bound it with max_gb sized to the live working set:
# num_envs + 1 files per wave (128 envs of 2k polyhaven ~= 0.8 GiB), so max_gb ~= 2 is ample.
#
# THE PLAN IS ADVISORY, NEVER A LIMIT. Anything outside it is still cached on demand, so a problem
# folder that grew more episodes than the lease anticipated keeps working -- it just warms lazily
# for the extras. That is why max_gb defaults to 0: an unplanned texture must never be refused.


def plan_paths(flags: dict, hdri_dirs, *, episode_keys=None, indexed="hash",
               prob_salt: int = 0) -> List[str]:
    """Resolve the concrete HDRI list this worker will touch. Pure -- no rng or cache side effects.

    `preload: auto` (or `{auto: true}`) projects the episode keys THIS process was handed; the
    slice is never named in the yaml because it belongs to the worker's lease, not the config.
    """
    from .resolve import resolve_skybox_textures
    spec = ((flags or {}).get("texture_cache") or {})
    pre = spec.get("preload") if isinstance(spec, dict) else None
    if not pre:
        return []
    if pre is True or (isinstance(pre, str) and pre.lower() == "auto"):
        pre = {"auto": True}
    if not isinstance(pre, dict):
        print(f"[texcache] plan: ignoring unrecognised preload spec {pre!r}")
        return []
    out: List[str] = []
    for route in (pre.get("sets") or []):
        route = {"set": route} if isinstance(route, str) else dict(route)
        try:
            out += list(resolve_skybox_textures(route, hdri_dirs))
        except Exception as e:
            print(f"[texcache] plan: set {route} did not resolve: {e}")
    # `auto` defaults on when no sets were named, so `preload: {}` still does the useful thing.
    auto = pre.get("auto", not pre.get("sets"))
    es_spec = (flags or {}).get("env_sphere")
    if auto and es_spec:
        keys = list(episode_keys or [])
        if not keys:
            # dagger: no finite episode list exists at bind time. Not an error -- use max_gb.
            print("[texcache] plan: preload auto but this process has no episode keys "
                  "(dagger-style per-reset sampling); relying on on-demand caching + max_gb")
        else:
            try:
                from . import env_sphere as _es
                seed = int(es_spec.get("seed", 0)) + int(prob_salt)
                drawn = _es._sample_episode_env_spheres(es_spec, keys, seed, hdri_dirs,
                                                        indexed=indexed)
                out += [v["texture"] for v in drawn.values() if v.get("texture")]
            except Exception as e:
                print(f"[texcache] plan: episode projection failed ({e}); falling back on demand")
    seen, uniq = set(), []
    for p in out:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            uniq.append(ap)
    return uniq


def warm(flags: dict, hdri_dirs, *, episode_keys=None, indexed="hash", prob_salt: int = 0) -> int:
    """Materialise the plan up front. No-op when the cache is off or no preload block is given."""
    if _CACHE is None:
        return 0
    paths = plan_paths(flags, hdri_dirs, episode_keys=episode_keys, indexed=indexed,
                       prob_salt=prob_salt)
    if not paths:
        return 0
    want = sum(os.path.getsize(p) for p in paths if os.path.exists(p))
    print(f"[texcache] preloading {len(paths)} planned textures ({want/2**30:.2f} GiB)...")
    n = _CACHE.preload(paths)
    _CACHE.pin(paths)          # the plan is the working set; never evict it for an ad-hoc miss
    s = _CACHE.stats()
    print(f"[texcache] preloaded {n}/{len(paths)}; resident {s['resident_gib']} GiB, "
          f"errors {s['errors']}")
    return n
