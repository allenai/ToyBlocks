"""Reusable domain-randomization modules (skybox/HDRI GI, env-sphere backdrop, table materials,
scene lights/sun, exposure, camera) extracted from scripts/render/render.py so they can be
consumed outside the render pipeline (e.g. toyblocks.eval) as plain, importable modules with NO
module-level mutable state -- every rng, queue cursor, cache and created-prim handle lives on an
instance passed in by the caller.

Layering:
  * assets.py       -- default asset locations (replaces render's DEFAULT_* globals)
  * resolve.py      -- PURE HDRI/manifest path resolution (no Isaac)
  * buckets.py      -- SkyBuckets: PURE stateful multi-bucket HDRI queue picker
  (further modules -- samplers, stage appliers, interpreter, session -- are added
   incrementally; see the package's build tasks.)

Pure layers import cleanly without a launched Isaac app; the stage/* appliers require a live sim.
"""
from __future__ import annotations

from .assets import AssetPaths
from . import resolve
from .buckets import SkyBuckets, bucket_weight, bucket_seq

# The client-facing entry point is .session.DRSession -- it wraps .interpreter (the working
# implementation, a verbatim lift of render's replay_job body) and fixes the call order.
# It is NOT eager-imported here so a lightweight consumer can `import toyblocks.domain_randomization`
# for the pure resolvers without pulling torch (camera.py needs it). Import explicitly:
#     from toyblocks.domain_randomization.session import DRSession

__all__ = ["AssetPaths", "resolve", "SkyBuckets", "bucket_weight", "bucket_seq"]
