"""Skybox DOME + fill-light STAGE appliers (the pure resolver + SkyBuckets picker live in
resolve.py / buckets.py). Moved verbatim from render.py. No module globals. pxr/omni/carb imports
are function-local so this imports without a launched Isaac app; the functions require a live stage."""
from __future__ import annotations

import os

from . import texture_cache as _texcache
from typing import Optional

def _dome_intensity(value: Optional[float] = None) -> Optional[float]:
    """Get (value=None) or set the /World/Light dome intensity. Returns the value or None."""
    try:
        from pxr import UsdLux
        import omni.usd
        prim = omni.usd.get_context().get_stage().GetPrimAtPath("/World/Light")
        if not (prim and prim.IsValid()):
            return None
        dome = UsdLux.DomeLight(prim)
        if value is not None:
            dome.GetIntensityAttr().Set(float(value))
            return float(value)
        v = dome.GetIntensityAttr().Get()
        return float(v) if v is not None else None
    except Exception:
        return None


SPHERE_FILL_PATH = "/World/ground/SphereLight"


def _sphere_fill_intensity(value: Optional[float] = None) -> Optional[float]:
    """Get (value=None) or set the ground-plane fill SphereLight intensity. Returns value/None."""
    try:
        from pxr import UsdLux
        import omni.usd
        prim = omni.usd.get_context().get_stage().GetPrimAtPath(SPHERE_FILL_PATH)
        if not (prim and prim.IsValid()):
            return None
        light = UsdLux.LightAPI(prim)
        if value is not None:
            light.GetIntensityAttr().Set(float(value))
            return float(value)
        v = light.GetIntensityAttr().Get()
        return float(v) if v is not None else None
    except Exception:
        return None


def _apply_skybox(texture_file: Optional[str], texture_format: str = "latlong",
                  rotation_deg: Optional[float] = None, intensity: Optional[float] = None,
                  visible_background: bool = False) -> bool:
    """Set the /World/Light DomeLight to an HDRI (image-based lighting) with optional Z-axis
    (world up) rotation and intensity. `visible_background=False` (default) makes the dome
    lighting-only (visibleInPrimaryRay off), so the HDRI relights the scene but is NOT drawn
    where the camera sees past the backdrop (avoids the skybox 'leaking' into the frame).
    Returns True on success."""
    try:
        from pxr import UsdLux, UsdGeom, Sdf
        import omni.usd
        prim = omni.usd.get_context().get_stage().GetPrimAtPath("/World/Light")
        if not (prim and prim.IsValid()):
            print("[render] skybox: /World/Light not found")
            return False
        dome = UsdLux.DomeLight(prim)
        if texture_file is not None:
            # texture_cache.resolve == os.path.abspath when the cache is off
            dome.CreateTextureFileAttr().Set(_texcache.resolve(texture_file))
            dome.CreateTextureFormatAttr().Set(texture_format)
        if intensity is not None:
            dome.GetIntensityAttr().Set(float(intensity))
        if rotation_deg is not None:
            xf = UsdGeom.Xformable(prim)
            zop = next((op for op in xf.GetOrderedXformOps()
                        if op.GetOpType() == UsdGeom.XformOp.TypeRotateZ), None)
            if zop is None:
                zop = xf.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble)
            zop.Set(float(rotation_deg))
        # primary-ray (background) visibility of the dome
        a = prim.GetAttribute("visibleInPrimaryRay")
        if not a:
            a = prim.CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool)
        a.Set(bool(visible_background))
        return True
    except Exception as e:
        print(f"[render] skybox apply failed: {e}")
        return False
