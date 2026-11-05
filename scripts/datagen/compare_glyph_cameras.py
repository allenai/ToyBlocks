#!/usr/bin/env python3
"""Same episode, same frame, every RGB camera -- the glyph swap must look identical in all of them.

The swap is written to the block's USD material, so every camera should see it. That is the
expectation, not a guarantee: a stream captured before the per-wave apply, or a camera whose
render product is stale, would show the ORIGINAL letters while the others show the new ones --
and no metadata check could tell, because the mapping is recorded once per episode.

Worth running whenever the camera set changes. The allrand config renders front, gripper AND
left, where the glyph-only runs skipped left entirely.

    python scripts/datagen/compare_glyph_cameras.py <dataset> <episode> <out.png>
"""
import json, sys
from pathlib import Path
import av, cv2, numpy as np, pyarrow.parquet as pq
D, EP, OUT = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
fps = json.loads((D/"meta/info.json").read_text()).get("fps", 25)
cams = sorted(p.name for p in (D/"videos").iterdir()
              if p.is_dir() and p.name.startswith("observation.images.camera:"))
em_f = sorted((D/"meta/episodes").glob("chunk-*/file-*.parquet"))[0]
gm = json.loads(pq.read_table(sorted(D.glob("data/chunk-*/file-*.parquet"))[0],
                              columns=["episode_index","glyph_map"]).to_pandas()
                .query(f"episode_index=={EP}")["glyph_map"].iloc[0])
panels=[]
for cam in cams:
    cols = ["episode_index","length",f"videos/{cam}/chunk_index",
            f"videos/{cam}/file_index",f"videos/{cam}/from_timestamp"]
    try: em = pq.read_table(em_f, columns=cols).to_pandas()
    except Exception: continue
    r = em[em.episode_index==EP].iloc[0]
    mp4 = D/"videos"/cam/f"chunk-{int(r[f'videos/{cam}/chunk_index']):03d}"/f"file-{int(r[f'videos/{cam}/file_index']):03d}.mp4"
    if not mp4.exists(): continue
    want = int(round(float(r[f"videos/{cam}/from_timestamp"])*fps)) + max(0,int(int(r["length"])*0.5)-1)
    img=None
    with av.open(str(mp4)) as c:
        for i,f in enumerate(c.decode(video=0)):
            if i==want: img=f.to_ndarray(format="rgb24").copy(); break
            if i>want: break
    if img is None: continue
    img = cv2.resize(img, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)
    panels.append((cam.split(":")[-1], img))
H=max(p[1].shape[0] for p in panels); W=max(p[1].shape[1] for p in panels)
sheet=np.full((H+56, (W+10)*len(panels)+10, 3), 18, np.uint8)
for k,(nm,im) in enumerate(panels):
    x0=10+k*(W+10); sheet[10:10+im.shape[0], x0:x0+im.shape[1]]=im
    cv2.putText(sheet, nm, (x0+4, H+34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120,230,255),2,cv2.LINE_AA)
swap="  ".join(f"{b.replace('block_','')}->{g}" for b,g in sorted(gm.items()))
cv2.putText(sheet, f"ep{EP}  {swap}", (10, H+54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200),1,cv2.LINE_AA)
cv2.imwrite(OUT, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
print(f"  wrote {OUT}  cameras={[p[0] for p in panels]}  map={gm}")
