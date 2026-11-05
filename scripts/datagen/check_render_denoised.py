#!/usr/bin/env python3
"""Detect an RTX RealTimePathTracing render that came out UNDENOISED.

RT2 renders at spp=1 and depends entirely on the DLSS ray-reconstruction denoiser (NGX).
`libnvidia-ngx.so.1` ships with the DRIVER, and libnvidia-container injects it only when the
`ngx` capability is requested -- `graphics` does NOT imply it. A container that does not set
`NVIDIA_DRIVER_CAPABILITIES=all` therefore produces frames covered in single-pixel fireflies.

Nothing else notices. The episodes record, the parquet validates, every metadata check passes --
an entire fleet can render a unusable dataset and report success. This measures the pixels.

Immerkaer noise sigma on sampled frames; measured on this pipeline:
    container RT2, capability unset   ~26 - 31      (unusable)
    container RT2, CAPABILITIES=all    ~0.65        (correct)
    local RTX 5090, no container       ~0.35 - 0.56 (correct)
The two populations are ~50x apart, so the threshold is not delicate.

    python scripts/datagen/check_render_denoised.py --dataset <root> [--max-sigma 5.0]
"""
import argparse
import json
import sys
from pathlib import Path

import av
import cv2
import numpy as np

RGBK = "observation.images.camera:front_camera"
# Far above any correctly denoised render (~0.7) and far below an undenoised one (~26).
DEFAULT_MAX_SIGMA = 5.0


def immerkaer_sigma(img) -> float:
    """Fast per-frame noise estimate: |img * L| through a Laplacian-like kernel."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    c = cv2.filter2D(g, -1, k)
    h, w = g.shape
    return float(np.sqrt(np.pi / 2) / (6 * (w - 2) * (h - 2)) * np.abs(c).sum())


def frames(root: Path, per_clip: int):
    clips = sorted(root.glob(f"videos/{RGBK}/chunk-*/*.mp4"))
    for mp4 in clips:
        with av.open(str(mp4)) as c:
            want = set(np.linspace(30, 240, per_clip).astype(int).tolist())
            for i, f in enumerate(c.decode(video=0)):
                if i in want:
                    yield mp4.name, i, f.to_ndarray(format="rgb24")
                if i > max(want):
                    break


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--max-sigma", type=float, default=DEFAULT_MAX_SIGMA)
    ap.add_argument("--per-clip", type=int, default=2)
    a = ap.parse_args()
    root = Path(a.dataset)

    rp = root / "meta" / "render_params.json"
    rt2 = None
    if rp.is_file():
        try:
            f = (json.loads(rp.read_text()).get("flags") or {}).get("carb_settings") or {}
            rt2 = bool(f.get("/persistent/rtx/modes/rt2/enabled")) or \
                "PathTracing" in str(f.get("/rtx/rendermode", ""))
        except (OSError, ValueError):
            pass
    print(f"  RT2 path tracing in flags: {rt2}")

    sig = [(n, i, immerkaer_sigma(im)) for n, i, im in frames(root, a.per_clip)]
    if not sig:
        print(f"FAIL: no front-camera clips under {root}")
        return 1
    vals = [s for _, _, s in sig]
    med = float(np.median(vals))
    print(f"  frames sampled: {len(sig)}   sigma median={med:.2f} min={min(vals):.2f} max={max(vals):.2f}")
    for n, i, s in sorted(sig, key=lambda r: -r[2])[:3]:
        print(f"    worst: {n} frame {i}  sigma={s:.2f}")

    if med > a.max_sigma:
        print(f"\n  FAIL: median sigma {med:.2f} > {a.max_sigma}. This render is UNDENOISED.")
        if rt2:
            print("  RT2 is on, so the DLSS denoiser (NGX) did not initialise. In a container that means\n"
                  "  it did not set NVIDIA_DRIVER_CAPABILITIES=all (graphics does NOT imply ngx).")
        print("  The metadata is probably fine -- only the pixels are ruined, which is why no other\n"
              "  check catches this.")
        return 1
    print(f"\n  PASS: median sigma {med:.2f} <= {a.max_sigma}; the denoiser ran.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
