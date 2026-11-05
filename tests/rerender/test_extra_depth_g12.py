"""    python tests/rerender/test_extra_depth_g12.py

--extra_depth g12_224 must decode to the SAME depth as the HVLA reference produces from rg8_224.

g12 stores the uint12 depth as what it is -- a lossless gray12le HEVC stream with temporal
prediction and a short GOP -- instead of rg8's byte-split intra-only RGB. maniflow-tb's
scripts/depth_conversion/reencode_to_g12_batch.py produces it today by TRANSCODING the _rg8_224
videos, and its README measures ~3.3 KB/frame against rg8's 16.0, with cold-file reads of 3-80 ms
against 100-6000 ms.

Writing it directly from the recorder is only legitimate if it lands on the same depth values.
That holds for one reason: the reference never resizes -- it reads the already-224 rg8 frames and
unpacks them -- so the recorder's g12 path MUST reuse the identical _resize_nearest call rg8_224
uses. Change one without the other and the two silently diverge at depth edges.

This test encodes the same frames both ways and compares DECODED DEPTH, not file bytes: the
container metadata differs (~10 bytes) because PyAV and the ffmpeg CLI write different encoder
tags, and that is not part of the contract.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import av  # noqa: E402
from blocksuite.vecgen.recording._video_encoder import _VideoEncoderProcess as VP  # noqa: E402

_failures = []


def _check(name, got, want):
    if got == want:
        print(f"  PASS  {name}")
        return
    print(f"  FAIL  {name}\n        got : {got}\n        want: {want}")
    _failures.append(name)


def _frames(n=40, w=640, h=360):
    """Moving uint12 depth. A static clip compresses to nothing and would hide GOP differences."""
    rng = np.random.default_rng(7)
    base = rng.integers(300, 3000, (h, w)).astype(np.uint16)
    out = []
    for i in range(n):
        f = base.copy()
        f[:, (i * 7) % w:] = np.clip(f[:, (i * 7) % w:] + 120, 0, 4095)
        f[(i * 3) % h, :] = 4095                       # saturate the 12-bit ceiling
        f[:, (i * 11) % w] = 0                         # and the floor
        out.append(f.astype(np.uint16))
    return out


def _encode(variant, path, fkey, frames):
    vp = VP.__new__(VP)
    vp._fps, vp._depth_gop = 25, 1000
    spec = VP._EXTRA_DEPTH[variant]
    vp._extra_spec, vp._extra_size = spec, spec["size"]
    vp._extra_cfg = VP._DEPTH_CODECS[spec["codec"]]
    gop = VP._extra_depth_gop(spec, fkey)
    c, st = VP._open_video_stream(vp, path, 224, 224, is_depth=True,
                                  depth_cfg=vp._extra_cfg, extra_gop=gop)
    for f in frames:
        small = VP._resize_nearest(f, 224, 224)
        fr = (av.VideoFrame.from_ndarray(small.astype(np.uint16), format="gray12le")
              if vp._extra_cfg["pix_fmt"] == "gray12le"
              else av.VideoFrame.from_ndarray(VP._pack_rg8(small), format="rgb24"))
        for pk in st.encode(fr):
            c.mux(pk)
    for pk in st.encode():
        c.mux(pk)
    c.close()
    return gop


def _decode(path, gray):
    fmt = "gray12le" if gray else "rgb24"
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo",
                          "-pix_fmt", fmt, "-"], capture_output=True).stdout
    if gray:
        return np.frombuffer(raw, dtype="<u2").reshape(-1, 224, 224)
    a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 224, 224, 3)
    return a[..., 0].astype(np.uint16) | (a[..., 1].astype(np.uint16) << 8)


def _probe(p, field):
    return subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                           "-show_entries", f"stream={field}", "-of", "csv=p=0:nk=1", p],
                          capture_output=True, text=True).stdout.strip()


def main():
    frames = _frames()

    print("A: the variant carries the reference's recipe")
    spec = VP._EXTRA_DEPTH["g12_224"]
    cfg = VP._DEPTH_CODECS[spec["codec"]]
    _check("224x224", spec["size"], (224, 224))
    _check("libx265", cfg["vcodec"], "libx265")
    _check("gray12le", cfg["pix_fmt"], "gray12le")
    _check("preset medium", cfg["options"].get("preset"), "medium")
    for p in ("lossless=1", "bframes=0", "open-gop=0", "scenecut=0"):
        _check(f"x265-params carries {p}", p in cfg["options"]["x265-params"], True)
    print("\nB: GOP is per camera -- 8 static, 4 for the moving gripper")
    _check("front camera", VP._extra_depth_gop(spec, "observation.depth.camera:front_camera"), 8)
    _check("gripper camera", VP._extra_depth_gop(spec, "observation.depth.camera:gripper_camera"), 4)
    _check("left camera falls back to the default",
           VP._extra_depth_gop(spec, "observation.depth.camera:left_camera"), 8)
    _check("rg8_224 has no gop key -> stays intra-only (GOP 1)",
           VP._extra_depth_gop(VP._EXTRA_DEPTH["rg8_224"], "anything"), 1)
    _check("keyint AND min-keyint both get it",
           VP._depth_options(cfg, 4)["x265-params"].count("keyint=4"), 2)

    with tempfile.TemporaryDirectory() as td:
        print("\nC: g12 and rg8 decode to the SAME depth, per camera")
        for cam, want_gop in (("observation.depth.camera:front_camera", 8),
                              ("observation.depth.camera:gripper_camera", 4)):
            short = cam.split(":")[-1]
            rg8 = os.path.join(td, f"{short}_rg8.mp4")
            g12 = os.path.join(td, f"{short}_g12.mp4")
            _encode("rg8_224", rg8, cam, frames)
            gop = _encode("g12_224", g12, cam, frames)
            a, b = _decode(rg8, gray=False), _decode(g12, gray=True)
            _check(f"{short}: gop {want_gop}", gop, want_gop)
            _check(f"{short}: same frame count", a.shape[0], b.shape[0])
            _check(f"{short}: DEPTH IDENTICAL to rg8_224", np.array_equal(a, b), True)
            _check(f"{short}: 12-bit ceiling survives", int(b.max()), 4095)
            _check(f"{short}: zero floor survives", int(b.min()), 0)
            _check(f"{short}: written as hevc/gray12le",
                   (_probe(g12, "codec_name"), _probe(g12, "pix_fmt")), ("hevc", "gray12le"))
            ra, rb = os.path.getsize(rg8), os.path.getsize(g12)
            print(f"        {short}: rg8 {ra:,d} B -> g12 {rb:,d} B ({rb / ra:.3f}x)")
            _check(f"{short}: g12 is smaller than rg8", rb < ra, True)

    print("\nD: the resize is shared with rg8_224, which is what makes C possible")
    src = (Path(__file__).resolve().parents[2] / "src" / "blocksuite" / "vecgen" / "recording"
           / "_video_encoder.py").read_text()
    seg = src[src.index("if is_depth and self._extra_suffix:"):]
    seg = seg[:seg.index("if not is_depth")] if "if not is_depth" in seg else seg[:2000]
    _check("one _resize_nearest call feeds both variants", seg.count("_resize_nearest"), 1)
    _check("the gray12le branch is chosen from the codec, not the variant name",
           'self._extra_cfg["pix_fmt"] == "gray12le"' in seg, True)

    print()
    if _failures:
        print(f"{len(_failures)} FAILURE(S): {_failures}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())


def test_main():
    import shutil

    import pytest

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("needs the ffmpeg and ffprobe CLIs")
    assert main() == 0
