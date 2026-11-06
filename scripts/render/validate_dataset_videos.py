"""Validate every mp4 in a (local) dataset directory before it is trusted / uploaded.

Catches the `moov atom not found` corruption (writer killed before close) AND short videos (killed
mid-episode -> fewer frames than the episode length). Meant to run:
  * post-write, right after a render/datagen run finishes, and
  * as the UPLOAD GATE (upload_dataset_s3.sh calls this and refuses to push a dataset with bad videos).

Exit code 0 = all good; 1 = at least one invalid video (list written to --out).

Usage:
  python validate_dataset_videos.py <dataset_dir> [--frames] [--workers 32] [--out bad.txt]
    --frames : also decode-count every video and check it == the episode length from meta/episodes
               (slower but catches short/truncated-but-has-moov videos). Without it, only the fast
               structural moov check runs.
"""
import argparse, concurrent.futures as cf, glob, json, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from toyblocks.vecgen.recording._video_validate import has_moov, count_frames  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dataset_dir")
ap.add_argument("--frames", action="store_true", help="also check decoded frame count == episode length")
ap.add_argument("--workers", type=int, default=32)
ap.add_argument("--out", default=None)
args = ap.parse_args()

VID_RE = re.compile(r"file-(\d+)\.mp4$")


def episode_lengths(split_root):
    """Map episode_index -> length from meta/episodes/*.parquet (empty if unavailable)."""
    out = {}
    try:
        import pandas as pd
        pqs = glob.glob(os.path.join(split_root, "meta", "episodes", "**", "*.parquet"), recursive=True)
        for pq in pqs:
            df = pd.read_parquet(pq, columns=["episode_index", "length"])
            for _, r in df.iterrows():
                out[int(r["episode_index"])] = int(r["length"])
    except Exception as e:
        print(f"[validate] WARN could not read episode lengths under {split_root}: {e}")
    return out


# discover split roots (dirs that contain a videos/ subdir)
splits = sorted({os.path.dirname(p) for p in glob.glob(
    os.path.join(args.dataset_dir, "**", "videos"), recursive=True)})
if not splits:
    # maybe dataset_dir IS a split root
    if os.path.isdir(os.path.join(args.dataset_dir, "videos")):
        splits = [args.dataset_dir]
print(f"[validate] {len(splits)} split root(s) under {args.dataset_dir}")

tasks = []  # (path, expected_or_None)
for sr in splits:
    lengths = episode_lengths(sr) if args.frames else {}
    for vp in glob.glob(os.path.join(sr, "videos", "**", "*.mp4"), recursive=True):
        m = VID_RE.search(os.path.basename(vp))
        exp = lengths.get(int(m.group(1))) if (m and args.frames) else None
        tasks.append((vp, exp))
print(f"[validate] {len(tasks)} videos to check (frames-check={'on' if args.frames else 'off'})")


def check(t):
    path, exp = t
    if not has_moov(path):
        return (path, "no-moov")
    if args.frames:
        n = count_frames(path)
        if n <= 0:
            return (path, "undecodable")
        if exp is not None and n != exp:
            return (path, f"short:{n}/{exp}")
    return None


bad = []
with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
    for i, res in enumerate(ex.map(check, tasks), 1):
        if res:
            bad.append(res)
        if i % 2000 == 0:
            print(f"  .. {i}/{len(tasks)} ({len(bad)} bad)")

if args.out and bad:
    with open(args.out, "w") as f:
        for p, why in bad:
            f.write(f"{p}\t{why}\n")

print(f"\n[validate] {len(bad)} INVALID / {len(tasks)} videos")
for p, why in bad[:20]:
    print(f"  {why:16} {p}")
if len(bad) > 20:
    print(f"  ... and {len(bad) - 20} more" + (f" (full list -> {args.out})" if args.out else ""))
sys.exit(1 if bad else 0)
