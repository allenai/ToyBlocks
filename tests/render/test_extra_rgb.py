"""    python tests/render/test_extra_rgb.py

--extra_rgb 224 writes the sidecar the HVLA training dataloader reads.

The contract is not "a 224x224 video": it is *the array the dataloader would have produced*.
`hvla.dataloader.franka_dataset._pil_to_tensor` resizes with Pillow BILINEAR, and no other kernel
agrees with it -- on these 640x360 clips ffmpeg `scale=flags=bilinear` differs from Pillow on ~70%
of the 224x224 pixels (max abs 35). So the resize is asserted against an independent transcription
of the dataloader call, not against "some bilinear".

The encoder settings must match maniflow-tb/scripts/rgb_conversion/reencode_to_224_batch.py,
which produced the existing sidecars: libsvtav1, crf=26, preset=8, yuv420p, and NO -g.
That last one is not cosmetic -- this recorder's own _GOP applied to a 224 sidecar inflated it
5.81x (1,781,024 B against the stored 306,410 B) before it was caught. PyAV's OutputContainer is
immutable, so the options dict cannot be spied on; section C asserts the *observable* consequence
instead -- keyframe cadence and file size against a deliberately GOP-forced encode of the same
frames -- which is the thing that actually went wrong.
"""
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import av
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
# The encoder imports LeRobot at module level, and the CPU CI lane does not install it; skip there
# (like the ffmpeg/ffprobe check in test_main) instead of failing collection for every test.
pytest.importorskip("lerobot")
from toyblocks.vecgen.recording._video_encoder import _VideoEncoderProcess as VP  # noqa: E402

_failures = []


def _check(name, got, want):
    if got == want:
        print(f"  PASS  {name}")
        return
    print(f"  FAIL  {name}\n        got : {got}\n        want: {want}")
    _failures.append(name)


def _dataloader_resize_ref(img, size=224):
    """Independent transcription of _pil_to_tensor's first two lines. Deliberately NOT a call to
    the helper under test -- a shared implementation would make this check tautological."""
    return np.asarray(Image.fromarray(img).resize((size, size), Image.BILINEAR), dtype=np.uint8)


def _probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=width,height,codec_name,pix_fmt,nb_read_packets",
         "-of", "default=nw=1:nk=0", path],
        capture_output=True, text=True, check=True).stdout
    return dict(ln.split("=", 1) for ln in out.strip().splitlines())


def _keyframes(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "packet=flags", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True).stdout
    return sum(1 for ln in out.split() if ln.startswith("K"))


def _frames(n):
    """Moving content -- a static clip compresses to nothing and hides GOP differences."""
    out = []
    for i in range(n):
        f = np.zeros((360, 640, 3), dtype=np.uint8)
        f[:, (i * 13) % 640:] = 200
        f[i % 360, :, 0] = 255
        f[:, :, 2] = (i * 7) % 256
        out.append(f)
    return out


def _encode(vp, path, w, h, imgs, rgb_cfg, resize):
    cont, stream = VP._open_video_stream(vp, path, w, h, rgb_cfg=rgb_cfg)
    for img in imgs:
        a = VP._resize_rgb_pil(img, resize) if resize else img
        for pk in stream.encode(av.VideoFrame.from_ndarray(a, format="rgb24")):
            cont.mux(pk)
    for pk in stream.encode():
        cont.mux(pk)
    cont.close()
    return path


def main():
    print("A: the 224 resize is the dataloader's resize, bit for bit")
    rng = np.random.default_rng(0)
    for i, frame in enumerate([
            rng.integers(0, 256, (360, 640, 3), dtype=np.uint8),                  # noise
            np.tile(np.arange(640, dtype=np.uint8)[None, :, None], (360, 1, 3)),  # ramp
            np.zeros((360, 640, 3), dtype=np.uint8)]):                            # flat
        mine = VP._resize_rgb_pil(frame, 224)
        _check(f"frame {i}: identical to Image.resize(BILINEAR)",
               np.array_equal(mine, _dataloader_resize_ref(frame)), True)
        _check(f"frame {i}: uint8 224x224x3", (mine.dtype, mine.shape), (np.uint8, (224, 224, 3)))

    print("\nB: the sidecar's encoder settings are reencode_to_224_batch.py's, not the recorder's")
    cfg = VP._EXTRA_RGB["224"]
    _check("size", cfg["size"], (224, 224))
    _check("crf 26 (reencode default)", cfg["crf"], "26")
    _check("preset 8 (reencode default)", cfg["preset"], "8")
    _check("g=None -> let SVT-AV1 pick its own keyframe interval", cfg["g"], None)

    vp = VP.__new__(VP)              # no Process.__init__: only the stream helpers are exercised
    vp._fps, vp._depth_gop = 25, 1
    vp._extra_rgb, vp._extra_rgb_suffix = "224", "_224"
    vp._extra_rgb_cfg, vp._extra_rgb_size = cfg, cfg["size"]

    n = 60
    imgs = _frames(n)
    with tempfile.TemporaryDirectory() as td:
        main_p = _encode(vp, os.path.join(td, "main.mp4"), 640, 360, imgs, None, None)
        side_p = _encode(vp, os.path.join(td, "side.mp4"), 224, 224, imgs, cfg, 224)
        # The bug: the recorder's own GOP applied to the sidecar.
        gop_p = _encode(vp, os.path.join(td, "gop.mp4"), 224, 224, imgs,
                        dict(cfg, g=VP._GOP), 224)

        # Section B only reads _EXTRA_RGB. If _open_video_stream ignored rgb_cfg and fell back to
        # the recorder's own _CRF/_PRESET, every assertion above would still pass. This control
        # encodes the SAME frames asking for a quality only the cfg can carry: if the dict is
        # honoured the files differ, if it is ignored they come out byte-identical.
        ctrl_p = _encode(vp, os.path.join(td, "ctrl.mp4"), 224, 224, imgs,
                         dict(cfg, crf="50", preset="12"), 224)

        print("\nC0: crf/preset from the sidecar cfg actually reach the encoder")
        s_ctrl = os.path.getsize(ctrl_p)
        print(f"        crf=26 preset=8  -> {os.path.getsize(side_p):,d} B")
        print(f"        crf=50 preset=12 -> {s_ctrl:,d} B")
        _check("a different crf/preset changes the file (cfg is not ignored)",
               s_ctrl != os.path.getsize(side_p), True)
        _check("the worse crf really is the smaller file",
               s_ctrl < os.path.getsize(side_p), True)

        print("\nC: the sidecar does NOT inherit the recorder's keyframe interval")
        k_side, k_gop = _keyframes(side_p), _keyframes(gop_p)
        s_side, s_gop = os.path.getsize(side_p), os.path.getsize(gop_p)
        print(f"        sidecar  {k_side} keyframes / {n}f, {s_side:,d} B")
        print(f"        GOP={VP._GOP}  {k_gop} keyframes / {n}f, {s_gop:,d} B  "
              f"({s_gop / s_side:.2f}x the sidecar)")
        _check("sidecar keyframes fewer than the GOP-forced encode", k_side < k_gop, True)
        _check("sidecar is not keyframe-per-frame", k_side < n, True)
        _check("forcing the recorder's GOP would inflate the file", s_gop > s_side, True)

        print("\nD: both streams take the SAME source frames and stay aligned")
        f, s = _probe(main_p), _probe(side_p)
        _check("sidecar is 224x224", (s["width"], s["height"]), ("224", "224"))
        _check("main is the render resolution", (f["width"], f["height"]), ("640", "360"))
        _check("same codec as the main RGB stream", s["codec_name"], f["codec_name"])
        _check("codec is av1", s["codec_name"], "av1")
        _check("pix_fmt yuv420p", s["pix_fmt"], "yuv420p")
        _check("frame counts match", s["nb_read_packets"], f["nb_read_packets"])
        _check(f"all {n} frames present", int(s["nb_read_packets"]), n)
        _check("sidecar is smaller than the full-size video",
               os.path.getsize(side_p) < os.path.getsize(main_p), True)
        print(f"        sizes: main {os.path.getsize(main_p):,d} B, "
              f"sidecar {s_side:,d} B ({s_side / os.path.getsize(main_p):.3f}x)")

        print("\nF: the sidecar matches what reencode_to_224_batch.py itself would write")
        # The strongest available check short of a GPU run: encode the SAME resized frames through
        # the reference implementation's own argv (ffmpeg CLI, the exact flags encode_argv builds
        # for codec=av1) and compare. PyAV and the CLI mux slightly differently, so sizes agree to
        # within ~10% rather than exactly -- measured 1.10x here and 0.944x against the real
        # existing sidecar. The band is wide enough for that and far too tight for the failures
        # that matter: forcing the recorder's GOP came in at 1.50x here and 5.81x on a real clip.
        ref_p = os.path.join(td, "ref.mp4")
        raw = b"".join(VP._resize_rgb_pil(im, 224).tobytes() for im in imgs)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", "224x224", "-r", "25", "-i", "pipe:0",
             # reencode_to_224_batch.encode_argv(codec="av1"): no -g, by design.
             "-c:v", "libsvtav1", "-crf", cfg["crf"], "-preset", cfg["preset"],
             "-pix_fmt", "yuv420p", ref_p],
            input=raw, check=True, capture_output=True)
        r = _probe(ref_p)
        s_ref, k_ref = os.path.getsize(ref_p), _keyframes(ref_p)
        ratio = s_side / s_ref
        print(f"        mine {s_side:,d} B / {k_side} keyframes   "
              f"reference {s_ref:,d} B / {k_ref} keyframes   ratio {ratio:.3f}")
        _check("same codec as the reference encoder", s["codec_name"], r["codec_name"])
        _check("same pix_fmt as the reference encoder", s["pix_fmt"], r["pix_fmt"])
        _check("same dimensions as the reference encoder",
               (s["width"], s["height"]), (r["width"], r["height"]))
        _check("same frame count as the reference encoder",
               s["nb_read_packets"], r["nb_read_packets"])
        _check("same keyframe cadence as the reference encoder", k_side, k_ref)
        _check(f"size within muxer tolerance of the reference (ratio {ratio:.3f})",
               0.75 < ratio < 1.30, True)

    print("\nE: the feature name matches the existing sidecars")
    # Asserting f"observation.images.{cam}_224" == "observation.images.camera:front_camera_224"
    # would compare an f-string to its own expansion and pass no matter what the recorder does.
    # Read the key the RECORDER actually builds, and evaluate it against a real camera name.
    rec = (Path(__file__).resolve().parents[2] / "src" / "toyblocks" / "vecgen" / "recording"
           / "lerobot_recorder.py").read_text()
    # Specifically the SIDECAR declaration -- the plain RGB feature is declared first and a
    # loose pattern matches that one instead.
    m = re.search(r'features\[f"(observation\.images\.[^"]*\{self\._extra_rgb\})"\]\s*=', rec)
    _check("recorder declares an observation.images.* sidecar feature", bool(m), True)
    if m:
        tmpl = m.group(1)
        cam_name, _extra_rgb = "camera:front_camera", "224"      # the names used in that f-string
        key = tmpl.replace("{cam_name}", cam_name).replace("{self._extra_rgb}", _extra_rgb)
        _check("recorder's key is the existing sidecar name", key,
               "observation.images.camera:front_camera_224")
        _check("no unexpanded placeholder left (the f-string uses these exact names)",
               "{" in key, False)
        _check("no codec prefix (rg8 is a DEPTH packing, meaningless for colour)",
               "_rg8_" in key, False)

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
