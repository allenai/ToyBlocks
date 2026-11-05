#!/usr/bin/env python3
"""Is each episode's camera actually ITS camera, or another episode's?

Camera columns can be correct values attached to the wrong episodes. That passes every
self-consistency check -- the values are a genuine DR draw and match the reconstruction -- while
being wrong for the pixels the episode actually contains.

For a target episode, score EVERY episode's camera against the target's own depth map by counting
how many blocks project onto themselves. If the target's own camera wins, the assignment is right.
If some other episode's camera wins clearly, it is misassigned.

  python verify_camera_assignment.py --dataset data/datasets/merged/train/<name> --episode 16
"""
import argparse, json, sys
from pathlib import Path

import av, numpy as np, pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from blocksuite.data_annotation.annotation_utils import extrinsics_from_camera_pose  # noqa: E402

DEPK = "observation.depth.camera:front_camera"
MM = 1000.0
HIT_LO, HIT_HI = -0.045, 0.006


def frames(mp4, idxs):
    want, out = sorted(set(idxs)), {}
    with av.open(str(mp4)) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i in want:
                a = fr.to_ndarray()
                out[i] = (a[..., 0] if a.ndim == 3 else a).astype(np.float64) / MM
                if len(out) == len(want):
                    break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--candidates", type=int, default=40)
    a = ap.parse_args()
    root = Path(a.dataset)
    info = json.loads((root / "meta" / "info.json").read_text()); fps = info.get("fps", 25)

    dfile = sorted(root.glob("data/chunk-*/file-*.parquet"))[0]
    cols = pq.ParquetFile(dfile).schema_arrow.names
    obj = [c for c in cols if c.startswith("observation.state.object_pose.")
           and "workspace_table" not in c]
    keep = ["episode_index", "frame_index"] + obj + \
           ["observation.camera_pose.camera:front_camera",
            "observation.camera_intrinsics.camera:front_camera"]
    df = pq.read_table(dfile, columns=keep).to_pandas()
    eps = df["episode_index"].to_numpy(); fidx = df["frame_index"].to_numpy()

    cams = {}
    for e, p, k in zip(eps.tolist(), df["observation.camera_pose.camera:front_camera"],
                       df["observation.camera_intrinsics.camera:front_camera"]):
        if int(e) not in cams and p is not None:
            K = np.asarray(k, float)
            cams[int(e)] = (np.asarray(p, float), float(K[0]), float(K[2]), float(K[5]))

    em = pq.read_table(sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))[0],
                       columns=["episode_index", f"videos/{DEPK}/chunk_index",
                                f"videos/{DEPK}/file_index", f"videos/{DEPK}/from_timestamp"]
                       ).to_pandas()
    vmap = {int(r[0]): (int(r[1]), int(r[2]), float(r[3])) for r in em.itertuples(index=False)}
    ch, fi, ts = vmap[a.episode]
    mp4 = root / "videos" / DEPK / f"chunk-{ch:03d}" / f"file-{fi:03d}.mp4"
    rows = np.flatnonzero(eps == a.episode)
    picks = [int(rows[int(p * (len(rows) - 1))]) for p in np.linspace(0.2, 0.85, a.frames)]
    base = int(round(ts * fps))
    F = frames(mp4, [base + int(fidx[r]) for r in picks])

    pts = []
    for r in picks:
        D = F.get(base + int(fidx[r]))
        if D is None:
            continue
        for c in obj:
            v = df[c].iloc[r]
            if v is not None and len(v) == 7:
                pts.append((np.asarray(v[:3], float), D))
    print(f"[assign] episode {a.episode}: {len(pts)} block observations over {len(F)} depth frames")

    def hit_rate(cam):
        pose, fx, cx, cy = cam
        cpos, Rcw = extrinsics_from_camera_pose(pose.reshape(1, 7))
        cpos, Rcw = cpos[0], Rcw[0]
        hits = tot = 0
        for p3, D in pts:
            pc = Rcw @ (p3 - cpos)
            if pc[2] <= 0.05:
                continue
            H, W = D.shape
            u = int(round(fx * pc[0] / pc[2] + cx)); v = int(round(fx * pc[1] / pc[2] + cy))
            if not (1 <= u < W - 1 and 1 <= v < H - 1):
                continue
            tot += 1
            d = float(np.median(D[v - 1:v + 2, u - 1:u + 2]))
            if HIT_LO <= d - pc[2] <= HIT_HI:
                hits += 1
        return (hits / tot if tot else 0.0), tot

    own, own_n = hit_rate(cams[a.episode])
    print(f"[assign] OWN camera (ep{a.episode}): hit rate {own*100:5.1f}%  (n={own_n})")
    others = [e for e in sorted(cams) if e != a.episode][: a.candidates]
    scored = sorted(((hit_rate(cams[e])[0], e) for e in others), reverse=True)
    print(f"[assign] best OTHER episodes' cameras on this episode's frames:")
    for h, e in scored[:6]:
        print(f"            ep{e:4d}: {h*100:5.1f}%")
    best_h, best_e = scored[0]
    rank = 1 + sum(1 for h, _ in scored if h > own)
    print(f"[assign] own camera ranks {rank} of {len(scored)+1}")
    if own >= best_h:
        print("[assign] VERDICT: the episode's own camera fits best -- assignment looks correct")
    else:
        print(f"[assign] VERDICT: ep{best_e}'s camera fits BETTER ({best_h*100:.1f}% vs "
              f"{own*100:.1f}%) -- possible misassignment")
    return 0


if __name__ == "__main__":
    sys.exit(main())
