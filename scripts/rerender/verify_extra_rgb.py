#!/usr/bin/env python3
"""Verify the 224 RGB sidecars are REAL -- present, full length, not black, same scene.

    python verify_extra_rgb.py --src <rerender_out_root> --workers 56
    python verify_extra_rgb.py --src <merged_dataset>   --workers 56   # after the merge too

WHY THIS EXISTS. HVLA's dataloader resolves the sidecar BY PATH and a missing variant directory
makes the collate fill with ZEROS instead of raising -- training silently regresses toward black
frames for those episodes and nothing anywhere reports an error. So "did the backfill work" cannot
be answered by "the sweep exited 0".

It also cannot be answered by the backfill's own coverage test, which is exactly:

    if not (dst.exists() and dst.stat().st_size > 0):        # problem_jobs

A one-byte file passes that. So does a sidecar that is genuinely shorter than its source, which is
the realistic corruption here: _encode_one pipes a decoder into an encoder, and if the DECODER dies
part-way (a truncated or moov-less source -- this project has hit both) the loop simply ends, the
encoder finalizes a valid SHORT clip, returncode is 0, and the result is published as good. Every
later sweep then skips it forever. Frame count is what catches that, and nothing else does.

Atomicity is already handled upstream -- the encode writes a pid-unique temp and os.replace()s it,
so a killed encoder leaves a .tmp and never a torn .mp4 -- so this checks the failure modes that
DO survive: absent, short, black, or the wrong scene.

THREE TIERS, cheapest first. Tier A and B run over every clip; C samples, because it decodes the
full-resolution source as well.

  A  existence + 224x224 + packet count == source packet count      (2 ffprobes, no decode)
  B  luma statistics on sampled frames -- an all-black sidecar      (decodes a few frames)
  C  PSNR against the bilinear-scaled source, on a random sample    (decodes both clips in full)

Tier C's threshold is deliberately loose. The sidecar is Pillow BILINEAR and this compares against
ffmpeg's bilinear, which do NOT agree (~70% of pixels differ on these clips, max abs 35), so the
question asked is "is this the same scene", not "is this bit-exact". To prove the check can even
tell, every sampled clip is ALSO scored against a different episode's source: a pass requires
self-PSNR to beat other-PSNR by --psnr_margin. A check that cannot fail is not a check.

EXIT CODES
  0  everything verified clean          1  failures found (report lists them)
  2  NOTHING WAS VERIFIED -- treated as failure, never as success. A previous verifier in this
     repo printed "ALL BIT-EXACT" and exited 0 having compared zero files.
"""
from __future__ import annotations

import argparse, json, os, random, re, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

SIZE = 224
SUFFIX = f"_{SIZE}"


def _run(cmd, timeout=120):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _probe_stream(p: Path):
    """(width, height, frames). Header first, full packet count only as a fallback.

    -count_packets reads the ENTIRE file; over weka that pinned the first live run at 10 clips/s
    (~50 min for 29k clips). A properly finalized mp4 carries nb_frames in its moov, which costs
    one small read -- and the corruption this hunts (a SHORT clip whose encoder finalized cleanly)
    is reported faithfully there, because the moov is written from what was actually encoded. Only
    when the header has no usable count do we pay for the full scan.
    """
    r = _run(["ffprobe", "-v", "error", "-select_streams", "v:0",
              "-show_entries", "stream=width,height,nb_frames",
              "-of", "csv=p=0:nk=1", str(p)])
    if r.returncode == 0:
        parts = r.stdout.strip().split(",")
        if len(parts) >= 3:
            try:
                w, h, n = int(parts[0]), int(parts[1]), int(parts[2])
                if n > 0:
                    return w, h, n
            except ValueError:
                pass
    r = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
              "-show_entries", "stream=width,height,nb_read_packets",
              "-of", "csv=p=0:nk=1", str(p)])
    if r.returncode != 0:
        return None
    parts = [x for x in r.stdout.strip().split(",") if x != ""]
    if len(parts) < 3:
        return None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None


def _luma_max(p: Path, frames: int):
    """Max luma seen over the first `frames` frames. 0/None means black or undecodable."""
    r = _run(["ffprobe", "-v", "error", "-f", "lavfi",
              "-i", f"movie={_esc(p)},signalstats",
              "-show_entries", "frame_tags=lavfi.signalstats.YMAX",
              "-read_intervals", f"%+#{frames}", "-of", "csv=p=0:nk=1"])
    if r.returncode != 0:
        return None
    vals = [int(v) for v in r.stdout.split() if v.strip().isdigit()]
    return max(vals) if vals else None


def _esc(p: Path) -> str:
    r"""Escape a path for the movie= filtergraph source. NOTE THE DOUBLED BACKSLASHES.

    Filtergraph text is unescaped TWICE -- once by the graph parser, once by the filter's own
    argument parser -- so a literal ':' must arrive as '\\:'. The real camera directories in this
    dataset are named 'observation.images.camera:front_camera', so a single-escaped path gets
    truncated at the colon and every probe fails with

        Failed to avformat_open_input '.../observation.images.camera'

    which reads as "every sidecar is corrupt" rather than "the checker is broken". That is exactly
    what the first live run reported: 6000 of 6000 clips bad, every one of them actually fine.
    """
    out = str(p)
    for ch in ("\\", ":", ",", "'", "[", "]"):
        out = out.replace(ch, "\\\\" + ch)
    return out


def _psnr(sidecar: Path, source: Path):
    """Average PSNR (dB) of the sidecar against the bilinear-scaled source. None if it cannot run.

    loglevel MUST stay at info: the psnr filter prints its summary at info, so "-v error" -- the
    obvious choice for a quiet subprocess -- silently yields no output at all and every clip then
    looks unreadable. setrange=tv on both legs removes ffmpeg's "different color ranges
    (unknown != tv)" warning, which otherwise leaves the comparison's scaling assumption implicit.
    """
    r = _run(["ffmpeg", "-nostdin", "-loglevel", "info", "-nostats",
              "-i", str(sidecar), "-i", str(source),
              "-lavfi", f"[0:v]setrange=tv[m];[1:v]scale={SIZE}:{SIZE}:flags=bilinear,"
                        f"setrange=tv[s];[m][s]psnr", "-f", "null", "-"],
             timeout=600)
    m = re.search(r"average:([0-9.]+|inf)", r.stderr or "")
    if not m:
        return None
    return 99.0 if m.group(1) == "inf" else float(m.group(1))


def check_clip(job):
    """(status, detail) for one sidecar. status: ok | missing | empty | unreadable | dims |
    short | black | mismatch."""
    src, dst, deep, luma_frames, psnr_min, psnr_margin, other = job
    src, dst = Path(src), Path(dst)
    if not dst.exists():
        return ("missing", f"{dst}")
    if dst.stat().st_size == 0:
        return ("empty", f"{dst}")
    d = _probe_stream(dst)
    if d is None:
        return ("unreadable", f"{dst}")
    w, h, n_dst = d
    if (w, h) != (SIZE, SIZE):
        return ("dims", f"{dst} is {w}x{h}")
    s = _probe_stream(src)
    if s is None:
        return ("src_unreadable", f"{src}")       # the SOURCE is bad; sidecar is not to blame
    n_src = s[2]
    if n_dst != n_src:
        return ("short", f"{dst} has {n_dst} frames, source has {n_src}")
    if luma_frames > 0:
        ym = _luma_max(dst, luma_frames)
        if ym is None:
            return ("unreadable", f"{dst} (luma probe failed)")
        if ym <= 16:                               # 16 == limited-range black
            return ("black", f"{dst} max luma {ym} over {luma_frames} frames")
    if deep:
        self_db = _psnr(dst, src)
        if self_db is None:
            return ("unreadable", f"{dst} (psnr failed)")
        if self_db < psnr_min:
            return ("mismatch", f"{dst} psnr {self_db:.1f} dB < {psnr_min}")
        if other:
            other_db = _psnr(dst, Path(other))
            if other_db is not None and self_db - other_db < psnr_margin:
                return ("mismatch", f"{dst} psnr self {self_db:.1f} vs other {other_db:.1f} dB "
                                    f"(margin < {psnr_margin})")
        return ("ok_deep", f"{dst} psnr {self_db:.1f} dB")
    return ("ok", "")


def video_roots(src: Path):
    """Every 'videos' dir under src: works on an unmerged tree and on a merged dataset alike."""
    if (src / "videos").is_dir():
        yield src / "videos"
        return
    for prob in sorted(p for p in src.iterdir() if p.is_dir()):
        for cand in (prob / "blocks_success" / "videos", prob / "videos"):
            if cand.is_dir():
                yield cand
                break


def enumerate_pairs(src: Path):
    """(source_clip, expected_sidecar) for every RGB clip that should have one."""
    pairs = []
    for vids in video_roots(src):
        for cam in sorted(vids.iterdir()):
            if not cam.is_dir() or not cam.name.startswith("observation.images.") \
               or cam.name.endswith(SUFFIX):
                continue
            if "depth" in cam.name:
                continue
            for mp4 in sorted(cam.rglob("*.mp4")):
                pairs.append((str(mp4), str(vids / (cam.name + SUFFIX) / mp4.relative_to(cam))))
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--luma_frames", type=int, default=5,
                    help="frames to luma-probe per sidecar (0 disables tier B)")
    ap.add_argument("--deep_sample", type=int, default=300,
                    help="clips to PSNR-check against their source (tier C; 0 disables, -1 all)")
    ap.add_argument("--psnr_min", type=float, default=25.0)
    ap.add_argument("--psnr_margin", type=float, default=5.0,
                    help="self-PSNR must beat a different episode's by this much")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--max_report", type=int, default=50)
    a = ap.parse_args()

    if not a.src.is_dir():
        print(f"[verify] --src {a.src} is not a directory", file=sys.stderr)
        return 2

    t0 = time.time()
    pairs = enumerate_pairs(a.src)
    if a.num_shards > 1:
        pairs = [p for i, p in enumerate(pairs) if i % a.num_shards == a.shard]
    print(f"[verify] {len(pairs)} clip(s) to check under {a.src}", flush=True)
    if not pairs:
        print("[verify] NOTHING TO VERIFY -- refusing to report success", file=sys.stderr)
        return 2

    rng = random.Random(a.seed)
    deep_idx = set(range(len(pairs))) if a.deep_sample == -1 else set(
        rng.sample(range(len(pairs)), min(a.deep_sample, len(pairs))) if a.deep_sample > 0 else [])
    jobs = []
    for i, (s, d) in enumerate(pairs):
        other = pairs[(i + len(pairs) // 2) % len(pairs)][0] if i in deep_idx else None
        if other == s:
            other = None
        jobs.append((s, d, i in deep_idx, a.luma_frames, a.psnr_min, a.psnr_margin, other))

    counts, bad = {}, []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(check_clip, j) for j in jobs]
        for n, f in enumerate(as_completed(futs), 1):
            st, detail = f.result()
            counts[st] = counts.get(st, 0) + 1
            if not st.startswith("ok"):
                bad.append({"status": st, "detail": detail})
            if n % 2000 == 0 or n == len(futs):
                el = time.time() - t0
                print(f"[verify] {n}/{len(futs)} {n/max(el,1e-9):.0f} clips/s "
                      f"bad={len(bad)}", flush=True)

    checked = sum(counts.values())
    print(f"\n[verify] checked {checked} clip(s) in {(time.time()-t0)/60:.1f}m")
    for k in sorted(counts):
        print(f"[verify]   {k:15s} {counts[k]}")
    deep_done = counts.get("ok_deep", 0) + sum(1 for b in bad if b["status"] == "mismatch")
    print(f"[verify] tier C (psnr vs source) ran on {deep_done} clip(s)")
    if checked == 0:
        print("[verify] NOTHING WAS VERIFIED", file=sys.stderr)
        return 2
    if a.report:
        a.report.write_text(json.dumps({"checked": checked, "counts": counts, "bad": bad}, indent=2))
        print(f"[verify] report -> {a.report}")
    if bad:
        print(f"\n[verify] {len(bad)} PROBLEM(S), first {min(len(bad), a.max_report)}:")
        for b in bad[:a.max_report]:
            print(f"   {b['status']:15s} {b['detail']}")
        return 1
    print("[verify] PASS: every sidecar present, full length, non-black"
          + (", and matching its source" if deep_done else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
