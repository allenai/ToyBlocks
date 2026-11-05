"""    python tests/render/test_verify_extra_rgb.py

verify_extra_rgb.py must CATCH each way a sidecar goes wrong, not just pass on good data.

This repo has already shipped a verifier that reported "ALL BIT-EXACT" and exited 0 having
compared zero files, so a checker is not trusted here until it has been shown failing. Every
section below plants a specific corruption in a synthetic tree and asserts the exact verdict:

  good      -> ok            a correct sidecar
  missing   -> missing       the case HVLA turns into ZEROS at collate
  empty     -> empty         0-byte file (the backfill's own coverage test would call it covered)
  short     -> short         fewer frames than the source; survives every later sweep, because
                             problem_jobs() only asks "exists and size > 0"
  black     -> black         decodes fine, all frames black
  wrongsize -> dims          not 224x224
  swapped   -> mismatch      a valid 224 clip of the WRONG episode (tier C)

and the empty-input case must exit 2, never 0.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[2] / "scripts" / "render"
VERIFY = HERE / "verify_extra_rgb.py"


def ff(*args, check=True):
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *map(str, args)],
                       capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr[-400:]}")
    return r


def make_source(path: Path, frames=30, seed=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Episodes must differ in LUMA, not just chroma. The first version of this test varied them
    # with hue= alone, which rotates chroma and leaves Y untouched -- so the "swapped" case scored
    # a high PSNR and the tier-C check could not have failed no matter how broken it was.
    ff("-f", "lavfi", "-i", f"testsrc=size=640x360:rate=10:duration={frames/10}:decimals=2",
       "-vf", f"hue=h={seed*47},drawbox=x={seed*70}:y={seed*30}:w=200:h=200:"
              f"color=white@1:t=fill,eq=brightness={(seed - 3) * 0.10:.2f}",
       "-frames:v", frames, "-pix_fmt", "yuv420p", path)


def make_sidecar(src: Path, dst: Path, frames=None, black=False, size=224):
    dst.parent.mkdir(parents=True, exist_ok=True)
    args = ["-i", str(src), "-vf", f"scale={size}:{size}:flags=bilinear"]
    if black:
        args = ["-f", "lavfi", "-i", f"color=c=black:s={size}x{size}:r=10:d=3"]
    if frames is not None:
        args += ["-frames:v", str(frames)]
    ff(*args, "-pix_fmt", "yuv420p", dst)


def build(tmp: Path):
    """One problem, one camera, seven episodes -- each a different planted defect."""
    vids = tmp / "problem_a" / "blocks_success" / "videos"
    # PRODUCTION NAMING, colon included -- the real dirs are observation.images.camera:front_camera.
    # The first version of this test used colon-free names and passed while the checker truncated
    # every real path at the colon inside the movie= filtergraph, calling all 29k clips unreadable.
    # Test data that does not look like production data does not test production.
    cam = vids / "observation.images.camera:front_camera"
    side = vids / "observation.images.camera:front_camera_224"
    cases = ["good", "missing", "empty", "short", "black", "wrongsize", "swapped"]
    for i, name in enumerate(cases):
        make_source(cam / f"episode_{i:06d}.mp4", frames=30, seed=i)
    for i, name in enumerate(cases):
        s, d = cam / f"episode_{i:06d}.mp4", side / f"episode_{i:06d}.mp4"
        if name == "missing":
            continue
        if name == "empty":
            d.parent.mkdir(parents=True, exist_ok=True); d.write_bytes(b""); continue
        if name == "short":
            make_sidecar(s, d, frames=12); continue
        if name == "black":
            make_sidecar(s, d, black=True); continue
        if name == "wrongsize":
            make_sidecar(s, d, size=112); continue
        if name == "swapped":
            make_sidecar(cam / "episode_000000.mp4", d); continue   # valid clip, wrong episode
        make_sidecar(s, d)
    return cases


def run(src: Path, *extra):
    rep = src / "report.json"
    r = subprocess.run([sys.executable, str(VERIFY), "--src", str(src), "--workers", "4",
                        "--report", str(rep), *map(str, extra)],
                       capture_output=True, text=True)
    data = json.loads(rep.read_text()) if rep.exists() else {}
    return r, data


def main():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        cases = build(tmp)
        print(f"A. planted {len(cases)} cases: {', '.join(cases)}")

        # tier C on every clip so the swapped episode is exercised too
        r, data = run(tmp, "--deep_sample", "-1")
        print(r.stdout[-700:])
        assert r.returncode == 1, f"expected exit 1 (failures found), got {r.returncode}"

        got = {}
        for b in data["bad"]:
            ep = b["detail"].split("episode_")[1][:6]
            got[int(ep)] = b["status"]
        expect = {1: "missing", 2: "empty", 3: "short", 4: "black", 5: "dims", 6: "mismatch"}
        for idx, want in expect.items():
            assert got.get(idx) == want, \
                f"episode {idx} ({cases[idx]}): expected '{want}', got '{got.get(idx)}'"
            print(f"   {cases[idx]:10s} -> {want}")
        assert 0 not in got, f"the GOOD sidecar was flagged: {got.get(0)}"
        print("   good       -> not flagged")
        assert data["checked"] == len(cases), f"checked {data['checked']} of {len(cases)}"

        print("B. a clean tree passes")
        clean = Path(td) / "clean"
        vids = clean / "problem_a" / "blocks_success" / "videos"
        cam = vids / "observation.images.camera:front_camera"
        side = vids / "observation.images.camera:front_camera_224"
        for i in range(3):
            make_source(cam / f"episode_{i:06d}.mp4", frames=20, seed=i)
            make_sidecar(cam / f"episode_{i:06d}.mp4", side / f"episode_{i:06d}.mp4")
        r2, d2 = run(clean, "--deep_sample", "-1")
        assert r2.returncode == 0, f"clean tree should pass, got {r2.returncode}: {r2.stdout[-500:]}"
        assert d2["checked"] == 3, d2
        print(f"   3 clips, exit 0, tier C ran on all 3")

        print("C. an empty tree must NOT report success")
        empty = Path(td) / "empty"; (empty / "problem_a" / "blocks_success" / "videos").mkdir(parents=True)
        r3, _ = run(empty)
        assert r3.returncode == 2, f"empty tree must exit 2, got {r3.returncode}"
        print("   exit 2, not 0")

    print("\nPASS")


if __name__ == "__main__":
    sys.exit(main())


def test_main():
    import shutil

    import pytest

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("needs the ffmpeg and ffprobe CLIs")
    main()  # asserts on failure
