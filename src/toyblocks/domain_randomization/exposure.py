"""RTX post-process EXPOSURE + carb settings + render-mode -- moved verbatim from render.py.
Pure carb/USD appliers (no module globals). carb imports are function-local so this imports without a
launched Isaac app; the _apply_*/_set_* functions require a running kit process to take effect."""
from __future__ import annotations

from typing import Optional



# ---------------------------------------------------------------------------
# RTX post-process EXPOSURE control (flags['exposure']).
# Lowering the dome/skybox intensity mostly changes shadow CONTRAST, not overall brightness,
# because the RTX histogram AUTO-EXPOSURE re-normalizes the frame. To get a real dark<->bright
# (low-key<->high-key) dial we must disable auto-exposure and drive the tonemapper's manual
# camera exposure (ISO / f-number / shutter) directly. Global (per-batch), like the dome.
# NOTE: whether these carb settings affect the offscreen TILED-camera render (vs only the
# viewport) is verified empirically -- `_dump_exposure_settings` prints them so we can confirm.
# ---------------------------------------------------------------------------
_EXPOSURE_KEYS = {
    "iso": "/rtx/post/tonemap/filmIso",
    "fstop": "/rtx/post/tonemap/fNumber",
    "shutter": "/rtx/post/tonemap/cameraShutter",
}


def _dump_exposure_settings(tag: str = "") -> dict:
    """Print + return the current RTX post exposure/tonemap settings (verify keys exist + the
    auto-exposure state)."""
    out = {}
    try:
        import carb
        s = carb.settings.get_settings()
        keys = ["/rtx/post/histogram/enabled", "/rtx/post/tonemap/op",
                "/rtx/post/tonemap/filmIso", "/rtx/post/tonemap/fNumber",
                "/rtx/post/tonemap/cameraShutter", "/rtx/post/tonemap/cameraExposure"]
        print(f"[render] RTX exposure settings {tag}:")
        for k in keys:
            out[k] = s.get(k)
            print(f"    {k} = {out[k]}")
    except Exception as e:
        print(f"[render] could not read RTX settings: {e}")
    return out


def _apply_exposure(auto: Optional[bool] = None, iso=None, fstop=None, shutter=None) -> None:
    """Set RTX post exposure. auto=False disables histogram auto-exposure so manual ISO/f-stop/
    shutter take effect. Higher ISO / longer shutter / lower f-number -> brighter."""
    try:
        import carb
        s = carb.settings.get_settings()
        if auto is not None:
            s.set_bool("/rtx/post/histogram/enabled", bool(auto))
        for val, key in ((iso, _EXPOSURE_KEYS["iso"]), (fstop, _EXPOSURE_KEYS["fstop"]),
                         (shutter, _EXPOSURE_KEYS["shutter"])):
            if val is not None:
                s.set_float(key, float(val))
    except Exception as e:
        print(f"[render] could not set exposure: {e}")


def _apply_carb_settings(settings: Optional[dict]) -> None:
    """Set arbitrary carb settings at RUNTIME (post-processing/lighting settings the tiled camera
    honors, e.g. '/rtx/directLighting/sampledLighting/maxRayIntensity' to lift the firefly clamp so a
    small bright light isn't clipped). Chooses set_float/set_bool/set_int/set_string by value type."""
    if not isinstance(settings, dict) or not settings:
        return
    try:
        import carb
        s = carb.settings.get_settings()
        for k, v in settings.items():
            k = str(k)
            if isinstance(v, bool):
                s.set_bool(k, v)
            elif isinstance(v, int):
                s.set_int(k, v)
            elif isinstance(v, float):
                s.set_float(k, v)
            else:
                s.set_string(k, str(v))
            print(f"[render] carb setting {k} = {v} (now {s.get(k)})")
    except Exception as e:
        print(f"[render] could not apply carb_settings: {e}")


def _set_render_mode(mode: Optional[str]) -> None:
    """Set the RTX render mode via carb. 'RealTimePathTracing' = RTX Real-Time 2.0 (the mode that
    HONORS USD light-linking, needed for per-env suns in a batch); 'RaytracedLighting' = legacy RT
    (ignores light-linking); 'PathTracing' = interactive path tracer. Logs the mode before/after."""
    if not mode:
        return
    try:
        import carb
        s = carb.settings.get_settings()
        before = s.get("/rtx/rendermode")
        s.set("/rtx/rendermode", str(mode))
        print(f"[render] render mode: {before} -> {s.get('/rtx/rendermode')}")
    except Exception as e:
        print(f"[render] WARNING could not set render mode: {e}")
