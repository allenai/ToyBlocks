"""Bit-exactness check: our rg8_224 stream vs the REFERENCE conversion of the same uint12 stream.

Reference chain, exactly what the consumer does:
  reencode_to_rg8_batch.py  : decode gray12le, R = low byte, G = high byte, B = 0
  franka_dataset._decode_depth_raw : mm -> float32 -> Image.fromarray(...).resize((S,S), NEAREST)
Anything nonzero here means the dataloader's two depth paths disagree.
"""
import subprocess, sys, collections
import numpy as np
from PIL import Image

FRAME_BYTES_G12 = lambda w, h: w * h * 2

def decode(path, pix, w, h, chan):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo",
                          "-pix_fmt", pix, "-"], capture_output=True, check=True).stdout
    if chan == 3:
        return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    return np.frombuffer(raw, "<u2").reshape(-1, h, w)

def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height", "-of", "csv=p=0:s=x", path],
                         capture_output=True, text=True, check=True).stdout.strip()
    return tuple(int(x) for x in out.split("x"))

def check(uint12_mp4, rg8_mp4, label):
    sw, sh = probe(uint12_mp4)
    dw, dh = probe(rg8_mp4)
    mm = decode(uint12_mp4, "gray12le", sw, sh, 1).astype(np.uint16)

    # g12 stores the uint12 value as a gray12le sample; rg8 splits it across R and G. Decoding a
    # g12 sidecar as rgb24 would compare noise, so pick the decode from the SIDECAR, not a
    # hardcoded packing. Only rg8 has a byte-level reference to check as well.
    is_g12 = "_g12" in rg8_mp4
    if is_g12:
        got_mm = decode(rg8_mp4, "gray12le", dw, dh, 1).astype(np.uint16)
        n = min(len(mm), len(got_mm))
        got_mm = got_mm[:n]
        rgb = db = None
    else:
        rgb = decode(rg8_mp4, "rgb24", dw, dh, 3)
        n = min(len(mm), len(rgb))
        got_mm = (rgb[:n, ..., 0].astype(np.uint16) | (rgb[:n, ..., 1].astype(np.uint16) << 8))

    # --- the reference resize, verbatim from reencode_to_rg8_batch.py (float32 in, NEAREST) ---
    ref_small = np.stack([
        np.array(Image.fromarray(mm[i].astype(np.float32)).resize((dw, dh), Image.NEAREST))
        for i in range(n)]).astype(np.uint16)
    dv = got_mm.astype(np.int64) - ref_small.astype(np.int64)
    if not is_g12:
        ref_rgb = np.zeros((n, dh, dw, 3), np.uint8)
        ref_rgb[..., 0] = (ref_small & 0xFF).astype(np.uint8)
        ref_rgb[..., 1] = ((ref_small >> 8) & 0xFF).astype(np.uint8)
        db = rgb[:n].astype(np.int16) - ref_rgb.astype(np.int16)

    kind = "g12" if is_g12 else "rg8"
    print(f"\n  === {label} ===")
    print(f"    uint12 {sw}x{sh} x{len(mm)}f   {kind} {dw}x{dh} x{n}f")
    print(f"    VALUE  mismatched {np.count_nonzero(dv)}/{dv.size} "
          f"({100*np.count_nonzero(dv)/dv.size:.4f}%)  max|diff| {np.abs(dv).max()} mm")
    if db is not None:
        print(f"    BYTES  mismatched {np.count_nonzero(db)}/{db.size} "
              f"({100*np.count_nonzero(db)/db.size:.4f}%)")
    if np.count_nonzero(dv):
        print(f"    diffs {collections.Counter(dv[dv != 0].tolist()).most_common(5)}")
    ok = np.count_nonzero(dv) == 0 and (db is None or np.count_nonzero(db) == 0)
    print(f"    -> {'BIT-EXACT' if ok else 'MISMATCH'}")
    return ok

# Every --extra_depth variant this can check. The suffix is also the directory suffix.
_VARIANTS = ("rg8_224", "g12_224")

if __name__ == "__main__":
    root = sys.argv[1]
    import glob, os
    allok = True
    n_checked = 0
    for u in sorted(glob.glob(f"{root}/**/observation.depth.camera:*/chunk-*/file-*.mp4",
                              recursive=True)):
        if any(f"_{v}" in u for v in _VARIANTS):
            continue                                   # this IS a sidecar, not a base stream
        d = os.path.dirname(os.path.dirname(u))
        for v in _VARIANTS:
            r = u.replace(d, f"{d}_{v}")
            if os.path.exists(r):
                allok &= check(u, r, f"{os.path.basename(d)}_{v}  {os.path.basename(u)}")
                n_checked += 1
    # Comparing NOTHING is not a pass. Before this, a dataset with no sidecars at all -- or one
    # written with a variant this tool did not know, which is exactly what g12_224 was on the day
    # it landed -- skipped every file and still printed "ALL BIT-EXACT" and exited 0.
    if n_checked == 0:
        print(f"\n  OVERALL: NOTHING VERIFIED -- no depth sidecar found under {root}. "
              f"Looked for {' / '.join('<cam>_' + v for v in _VARIANTS)} beside each base depth "
              f"stream. This is NOT a pass: either the render wrote no --extra_depth, or it wrote "
              f"a variant this tool does not know.")
        sys.exit(2)
    print(f"\n  OVERALL: {'ALL BIT-EXACT' if allok else 'MISMATCHES PRESENT'} "
          f"({n_checked} sidecar file(s) compared)")
    sys.exit(0 if allok else 1)
