"""``SkyBuckets`` -- PURE, self-contained multi-bucket HDRI queue picker (no Isaac, no globals).
Moved verbatim from render.py. All state (per-bucket queues + cursors) lives on the instance; the
caller owns the rng and passes it to ``next()`` / construction, so two consumers can each hold their
own independent picker. The only former global it relied on (``DEFAULT_HDRI_DIR``, via
``resolve_skybox_textures``) is now the explicit ``default_hdri_dir`` constructor argument.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

from .resolve import resolve_skybox_textures, shuffled_texture_order


def bucket_weight(s: dict) -> float:
    """Selection weight for one skybox bucket: prob == probability == weight (first present wins),
    default 1.0. RELATIVE + normalized across buckets (need not sum to 1), mirroring scenarios."""
    for k in ("prob", "probability", "weight"):
        if k in s:
            return float(s[k])
    return 1.0


def bucket_seq(s: dict):
    """Explicit ordered-selection index for a bucket (`order: <int>`), or None. In ordered mode the
    lowest-`order` not-yet-exhausted bucket is always chosen (requires exhaust_before_repeat)."""
    v = s.get("order")
    return int(v) if v is not None else None


class SkyBuckets:
    """Multi-bucket HDRI selection, shared by the skybox DOME and the env_sphere. A `sets:` list in
    the flag defines N buckets; each is resolved independently via resolve_skybox_textures
    (set/names/textures/textures_dir) and keeps its OWN without-replacement shuffled queue. next()
    first picks a bucket, then advances THAT bucket's queue (wrap on exhaustion).

    Bucket-selection modes (per next() call):
      * default          : weighted random by prob/probability/weight (equal by default).
      * exhaust_before_repeat: true  -> a bucket that has served ALL its uniques since the last reset
                           is SUPPRESSED until EVERY bucket is exhausted; then all reset (queues
                           re-shuffled) -> every HDRI used once before any repeat (super-cycle).
      * order: <int> per bucket (requires exhaust_before_repeat) -> ORDERED selection: always take
                           the lowest-`order` not-yet-exhausted bucket. Weights ignored in this mode.

    With NO `sets:` list it degenerates to a SINGLE bucket over the whole resolved pool. With
    exhaust_before_repeat off, a lone bucket consumes no rng in next() and its cursor advances once
    per call (next() == order[i % n]) -> fully back-compatible with the old sky_order[wi%n] behavior.
    """

    def __init__(self, flag: dict, rng, default_hdri_dir: str):
        self.buckets: List[dict] = []
        specs = flag.get("sets")
        multi = isinstance(specs, list) and len(specs) > 0
        self.exhaust = bool(flag.get("exhaust_before_repeat", False))
        if multi:
            # buckets inherit the flag's shared resolution defaults unless they override them
            base = {k: flag[k] for k in ("textures_dir", "shuffle", "dedup") if k in flag}
            entries = specs
        else:
            base, entries = {}, [flag]  # single implicit bucket = the whole flag
        for s in entries:
            is_bucket = s is not flag
            merged = {**base, **s} if is_bucket else flag
            tex = resolve_skybox_textures(merged, default_hdri_dir)
            if not tex:
                if is_bucket:
                    sel = {k: s.get(k) for k in ("set", "names", "textures") if k in s}
                    print(f"[dr] WARNING skybox bucket resolved 0 textures: {sel}")
                continue
            do_shuffle = bool(merged.get("shuffle", True))
            order = shuffled_texture_order(tex, rng) if do_shuffle else list(tex)
            name = s.get("set") or s.get("name") or ("inline" if is_bucket else "pool")
            self.buckets.append({"name": str(name), "weight": bucket_weight(s) if is_bucket else 1.0,
                                 "seq": bucket_seq(s) if is_bucket else None, "shuffle": do_shuffle,
                                 "order": order, "cursor": 0, "served": 0})
        w = np.array([b["weight"] for b in self.buckets], dtype=float)
        self.p = (w / w.sum()) if (w.size and w.sum() > 0) else None
        # ordered mode: any bucket carries an explicit `order` int. Requires exhaust_before_repeat.
        self.ordered = any(b["seq"] is not None for b in self.buckets)
        if self.ordered and not self.exhaust:
            print("[dr] WARNING skybox: bucket `order` needs `exhaust_before_repeat: true` "
                  "-> ignoring order (falling back to weighted selection)")
            self.ordered = False

    def __bool__(self) -> bool:
        return bool(self.buckets)

    @property
    def multi(self) -> bool:
        return len(self.buckets) > 1

    @property
    def pool(self) -> List[str]:
        """Combined candidate pool across all buckets (for the render_params record + existence check)."""
        return [t for b in self.buckets for t in b["order"]]

    def _reset(self, rng) -> None:
        """Super-cycle reset: clear served counts + cursors and re-shuffle each shuffled queue."""
        for b in self.buckets:
            b["served"] = 0
            b["cursor"] = 0
            if b["shuffle"]:
                rng.shuffle(b["order"])

    def next(self, rng) -> Tuple[Optional[str], Optional[str]]:
        """Pick a bucket, then return its next texture (advancing that bucket's queue). Returns
        (path, bucket_name). Default: weighted (no rng draw for a single bucket -> back-compat)."""
        if not self.buckets:
            return None, None
        if not self.exhaust:
            bi = int(rng.choice(len(self.buckets), p=self.p)) if len(self.buckets) > 1 else 0
        else:
            avail = [i for i, b in enumerate(self.buckets) if b["served"] < len(b["order"])]
            if not avail:
                self._reset(rng)
                avail = list(range(len(self.buckets)))
            if self.ordered:
                bi = min(avail, key=lambda i: ((self.buckets[i]["seq"] if self.buckets[i]["seq"]
                                                is not None else 1 << 30), i))
            elif len(avail) > 1:
                w = np.array([self.buckets[i]["weight"] for i in avail], dtype=float)
                p = (w / w.sum()) if w.sum() > 0 else None
                bi = int(rng.choice(avail, p=p))
            else:
                bi = avail[0]
        b = self.buckets[bi]
        tex = b["order"][b["cursor"] % len(b["order"])]
        b["cursor"] += 1
        b["served"] += 1
        return tex, b["name"]

    def summary(self) -> str:
        tot = sum(b["weight"] for b in self.buckets) or 1.0
        mode = ("ordered" if self.ordered else "exhaust" if self.exhaust else "weighted")
        parts = []
        for b in self.buckets:
            tag = (f"#{b['seq']}" if (self.ordered and b["seq"] is not None)
                   else f"@{b['weight'] / tot:.2f}")
            parts.append(f"{b['name']}:{len(b['order'])}{tag}")
        return f"{mode}[" + ", ".join(parts) + "]"

    def metadata(self) -> List[dict]:
        """Per-bucket record for render_params.json (reproducible even if manifests change later)."""
        tot = sum(b["weight"] for b in self.buckets) or 1.0
        out = []
        for b in self.buckets:
            m = {"name": b["name"], "weight": b["weight"], "prob": round(b["weight"] / tot, 4),
                 "count": len(b["order"]), "order": [os.path.basename(t) for t in b["order"]]}
            if b["seq"] is not None:
                m["seq"] = b["seq"]
            out.append(m)
        return out
