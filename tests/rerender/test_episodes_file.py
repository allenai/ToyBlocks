"""    python tests/rerender/test_episodes_file.py

--episodes_file parsing. It decides WHICH episodes a patch pass re-renders; a mis-parse either
patches the wrong episodes or silently patches none while reporting success."""
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "rerender"))
import rerender as rr

fails=[]
def chk(n,got,want):
    if got==want: print(f"  PASS  {n}")
    else: print(f"  FAIL  {n}\n        got : {got}\n        want: {want}"); fails.append(n)
def w(txt):
    f=tempfile.NamedTemporaryFile("w",suffix=".tsv",delete=False); f.write(txt); f.close(); return f.name

print("1. the TSV the orchestrator itself writes (write_patch_tsv shape)")
got=rr._load_episode_filter(w("problem\tsplit\tepisode\nproblem_000\tblocks_success\t3\nproblem_000\tblocks_success\t7\nproblem_001\tblocks_failure\t0\n"))
chk("parsed", {k:sorted(v) for k,v in got.items()},
    {("problem_000","blocks_success"):[3,7], ("problem_001","blocks_failure"):[0]})

print("2. CSV and column order must not matter")
got=rr._load_episode_filter(w("episode,problem,split\n5,problem_000,blocks_success\n"))
chk("csv, reordered columns", {k:sorted(v) for k,v in got.items()},
    {("problem_000","blocks_success"):[5]})

print("3. a plain list of video paths")
got=rr._load_episode_filter(w(
 "unmerged/out/problem_000/blocks_success/videos/observation.images.camera:front_camera/chunk-000/file-012.mp4\n"
 "unmerged/out/problem_002/blocks_failure/videos/observation.depth.camera:gripper_camera/chunk-001/file-003.mp4\n"))
chk("paths -> (problem, split): episode", {k:sorted(v) for k,v in got.items()},
    {("problem_000","blocks_success"):[12], ("problem_002","blocks_failure"):[1003]})

print("4. degenerate inputs must not explode or silently mean 'everything'")
chk("None -> None (filter disabled)", rr._load_episode_filter(None), None)
chk("empty file -> {} (patch nothing, not everything)", rr._load_episode_filter(w("")), {})
chk("header only -> {}", rr._load_episode_filter(w("problem\tsplit\tepisode\n")), {})
got=rr._load_episode_filter(w("problem\tsplit\tepisode\nproblem_000\tblocks_success\t1\nbroken row\n"))
chk("short row skipped, good row kept", {k:sorted(v) for k,v in got.items()},
    {("problem_000","blocks_success"):[1]})

print("5. duplicates collapse (a set, not a list)")
got=rr._load_episode_filter(w("problem\tsplit\tepisode\nproblem_000\tblocks_success\t4\nproblem_000\tblocks_success\t4\n"))
chk("deduped", {k:sorted(v) for k,v in got.items()}, {("problem_000","blocks_success"):[4]})

print()
print("ALL PASS" if not fails else f"{len(fails)} FAILURE(S): {fails}")
if __name__ == "__main__":
    sys.exit(1 if fails else 0)


def test_main():
    assert not fails
