#!/usr/bin/env python3
"""Build (and optionally serve) a video gallery of the generated task families.

Scans the unmerged LeRobot datasets produced by ``execute_plans.py``, picks one
successful episode per task family, and writes a self-contained directory with
an ``index.html`` plus one clip per family.

The recorder writes one mp4 per episode per camera, under
``blocks_success/videos/observation.images.camera:<cam>/chunk-XXX/file-NNN.mp4``,
and records the chunk/file indices in ``meta/episodes``. Those indices are used
to pick the clip, falling back to the first mp4 for the camera when the metadata
lacks the columns. Should a dataset ever pack several episodes per file, the
recorded offsets are honoured by cutting with ffmpeg (or, without ffmpeg, an
HTML ``#t=from,to`` media fragment).

``--serve`` exposes the gallery over HTTP on the given port.

Usage:
    python scripts/task_gallery.py \\
        --datasets-root data/datasets/unmerged --prefix <experiment-id> --out-dir data/gallery
    python scripts/task_gallery.py --out-dir <dir> --no-build --serve --port 8080
"""

from __future__ import annotations

import argparse
import functools
import html
import http.server
import json
import shutil
import socket
import socketserver
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CAMERA = "front_camera"

#: family -> (display name, what the robot is asked to do, why it is interesting)
TASK_INFO: dict[str, tuple[str, str, str]] = {
    "spell_stack": (
        "Spell a word (stack)",
        'Stack letter blocks so they spell a word top-to-bottom, e.g. "Stack the blocks to spell CAT".',
        "Ordering plus orientation: each letter must be reoriented readable before it is stacked.",
    ),
    "spell_stack_distract": (
        "Spell a word (stack, with distractors)",
        "Same as the spelling stack, but 2-4 unrelated letter blocks share the table.",
        "Tests grounding: the planner must ignore letters that are not part of the word.",
    ),
    "spell_line": (
        "Spell a word (line)",
        'Arrange letter blocks left-to-right in reading order, e.g. "Arrange the blocks to spell RUN".',
        "Spatial ordering with a left-right convention, built from chained push actions.",
    ),
    "make_number": (
        "Make a number (stack)",
        'Stack digit blocks to compose a multi-digit number, e.g. "Spell 137".',
        "Multi-digit sequencing over digit labels rather than letters.",
    ),
    "make_number_line": (
        "Make a number (line)",
        "Push digit blocks into a row reading the requested number left-to-right.",
        "The line form of number composition; 3-digit lines are the reliable ceiling.",
    ),
    "letters_orientation": (
        "Reorient letters",
        "Make every letter readable from the front camera without relocating it.",
        "Isolates the reorient skill: no stacking, no placement, orientation only.",
    ),
    "color_zebra_stack": (
        "Zebra stack",
        "Stack cubes in alternating colours, e.g. red-green-red-green bottom-to-top.",
        "Pattern rule induction where colour class matters but block identity does not.",
    ),
    "color_alternating_line": (
        "Alternating colour line (4-5)",
        "Push cubes into a row alternating two colours.",
        "Superseded by the 3-block variant: longer push chains do not survive execution.",
    ),
    "color_alternating_line_l3": (
        "Alternating colour line (3)",
        "Push three cubes into a row alternating two colours, e.g. red-blue-red.",
        "The reliable line length for chained pushes.",
    ),
    "color_rainbow": (
        "Rainbow stack",
        'Stack cubes in an explicitly specified colour order, e.g. "red on the bottom, then green, then blue".',
        "Direct instruction-to-order mapping, with no rule to infer.",
    ),
    "color_mirror_line": (
        "Mirrored colour line (5)",
        "Build a palindromic colour row that reads the same from both ends.",
        "Superseded by the 3-block variant for the same push-chain reason.",
    ),
    "color_mirror_line_l3": (
        "Mirrored colour line (3)",
        "Build a three-cube palindromic row, e.g. red-green-red.",
        "Symmetry rule at a length that executes reliably.",
    ),
    "split_into_groups": (
        "Split into groups",
        "Partition loose cubes into several equal stacks.",
        "Goal decomposition: one instruction implies several independent sub-stacks.",
    ),
    "stack_exactly_n": (
        "Stack exactly N",
        "Stack exactly N cubes while extra cubes remain on the table.",
        "Counting with a stopping condition; the spare blocks must be left alone.",
    ),
    "conditional_r_on_red": (
        "Conditional placement",
        "Place the R block onto the red cube.",
        "Goal conditioned on a specific object pair among several candidates.",
    ),
    "push_next_to": (
        "Push next to",
        "Push one block until it rests beside another.",
        "Non-prehensile manipulation: the target is moved by pushing, not picking.",
    ),
    "chained_conjunctions": (
        "Chained goals",
        "Satisfy two or three independent goal predicates at once.",
        "Multi-goal planning where the sub-goals can interfere with one another.",
    ),
    "isolate": (
        "Isolate a block",
        "Separate one block away from the surrounding clutter.",
        "Currently unsolved: the planner exhausts its search budget on every instance.",
    ),
    # Cluttered stacking sweep. Same goals as the families above, but the scene
    # is packed with goal-irrelevant blocks to force collision avoidance and to
    # stop the instruction from being groundable by elimination.
    "stack_distract_med": (
        "Stack with 6 distractors",
        "Stack the red cubes while six unrelated blocks share the table.",
        "The low rung of the clutter sweep, for measuring how yield scales with scene density.",
    ),
    "stack_distract_heavy": (
        "Stack with 14 distractors",
        "Stack the red cubes on a table crowded with fourteen unrelated blocks.",
        "Heavy clutter costs only a few points of planning yield, so density alone is cheap.",
    ),
    "stack_letters_distract": (
        "Stack among letter blocks",
        "Stack the red cubes while fourteen letter blocks crowd the table.",
        "The hardest clutter variant: letter meshes pack tighter than cuboids, and whole "
        "problems can fail on start-state collision when they settle too close together.",
    ),
    "stack_n5_distract": (
        "Five-high stack with clutter",
        "Build a five-cube stack with ten distractors present.",
        "Longer horizon: five placements must all survive before the goal check passes.",
    ),
    "stack_buried": (
        "Buried target stack",
        "Stack the red cubes, some of which start underneath distractor blocks.",
        "Lookahead: the plan has to clear blocks off the targets before it can build, so the "
        "action count exceeds what the instruction literally names.",
    ),
    "split_groups_distract": (
        "Split into groups with clutter",
        "Partition the red cubes into equal stacks amid eight to twelve distractors.",
        "Goal decomposition under clutter, with the sub-stacks competing for free table space.",
    ),
    "zebra_stack_distract": (
        "Zebra stack with clutter",
        "Build an alternating-colour stack with distractors of other colours present.",
        "Pattern induction where the distractors rule out grounding the rule by elimination.",
    ),
    "rainbow_stack_distract": (
        "Rainbow stack with clutter",
        "Stack cubes in a specified colour order with distractors present.",
        "Explicit colour ordering that must be grounded against a crowded scene.",
    ),
}


@dataclass
class FamilyEntry:
    """One card in the gallery."""

    family: str
    title: str
    instruction: str
    rationale: str
    episodes: int = 0
    clip: str | None = None
    fragment: str = ""
    problems: int = 0
    source: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def describe(family: str) -> tuple[str, str, str]:
    """Look up a family's copy, tolerating the ``topup_`` prefix."""
    base = family.removeprefix("topup_")
    if base in TASK_INFO:
        title, instruction, rationale = TASK_INFO[base]
        if family.startswith("topup_"):
            title = f"{title} (top-up)"
        return title, instruction, rationale
    return family.replace("_", " ").title(), "", ""


def _read_episode_meta(dataset: Path) -> list[dict[str, Any]]:
    """Return episode metadata rows, or [] when the parquets are unreadable."""
    ep_dir = dataset / "meta" / "episodes"
    if not ep_dir.is_dir():
        return []
    try:
        import pandas as pd
    except ImportError:
        return []
    rows: list[dict[str, Any]] = []
    for pq in sorted(ep_dir.glob("chunk-*/file-*.parquet")):
        try:
            rows.extend(pd.read_parquet(pq).to_dict("records"))
        except Exception:  # noqa: BLE001 - a partial dataset must not abort the gallery
            continue
    return rows


def _video_key(camera: str) -> str:
    """Video column/dir name. The recorder prefixes the sensor with ``camera:``."""
    return f"observation.images.camera:{camera}"


def _video_for_episode(
    dataset: Path, row: dict[str, Any], camera: str
) -> tuple[Path, float, float] | None:
    """Resolve the mp4 and [from, to] offsets holding one episode.

    This recorder writes one mp4 per episode, so the offsets are usually the
    whole file; they are still honoured in case a dataset packs episodes.
    """
    key = _video_key(camera)
    chunk = row.get(f"videos/{key}/chunk_index")
    file_idx = row.get(f"videos/{key}/file_index")
    if chunk is None or file_idx is None:
        ep = int(row.get("episode_index", 0))
        chunk, file_idx = ep // 1000, ep % 1000
    path = dataset / "videos" / key / f"chunk-{int(chunk):03d}" / f"file-{int(file_idx):03d}.mp4"
    if not path.exists():
        return None
    start = float(row.get(f"videos/{key}/from_timestamp", 0.0) or 0.0)
    end = float(row.get(f"videos/{key}/to_timestamp", 0.0) or 0.0)
    return path, start, end


def _any_video(dataset: Path, camera: str) -> tuple[Path, float, float] | None:
    """First mp4 for the camera, used when episode metadata is missing columns."""
    cam_dir = dataset / "videos" / _video_key(camera)
    if not cam_dir.is_dir():
        return None
    for mp4 in sorted(cam_dir.glob("chunk-*/file-*.mp4")):
        return mp4, 0.0, 0.0
    return None


def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _extract_clip(src: Path, dst: Path, start: float, end: float) -> bool:
    """Cut [start, end] out of src. Returns False if ffmpeg is unavailable/fails."""
    if not _have_ffmpeg() or end <= start:
        return False
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(src),
        "-c", "copy", str(dst),
    ]
    if subprocess.run(cmd, capture_output=True).returncode == 0 and dst.exists():
        return True
    # Stream copy can fail when the cut lands off a keyframe; re-encode instead.
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(src),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", str(dst),
    ]
    return subprocess.run(cmd, capture_output=True).returncode == 0 and dst.exists()


def collect_family(family: str, family_dir: Path, out_dir: Path, camera: str) -> FamilyEntry:
    """Count episodes for a family and pull one representative clip."""
    title, instruction, rationale = describe(family)
    entry = FamilyEntry(family=family, title=title, instruction=instruction, rationale=rationale)

    problem_dirs = sorted(p for p in family_dir.iterdir() if p.is_dir())
    chosen: tuple[Path, dict[str, Any]] | None = None
    for problem in problem_dirs:
        dataset = problem / "blocks_success"
        if not dataset.is_dir():
            continue
        rows = _read_episode_meta(dataset)
        if not rows:
            continue
        entry.problems += 1
        entry.episodes += len(rows)
        if chosen is None:
            for row in rows:
                if _video_for_episode(dataset, row, camera):
                    chosen = (dataset, row)
                    break
            else:
                if _any_video(dataset, camera):
                    chosen = (dataset, {})

    if chosen is None:
        return entry

    dataset, row = chosen
    resolved = _video_for_episode(dataset, row, camera) if row else None
    if resolved is None:
        resolved = _any_video(dataset, camera)
    if resolved is None:
        return entry
    src, start, end = resolved
    entry.source = str(src)

    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    dst = videos_dir / f"{family}.mp4"
    if _extract_clip(src, dst, start, end):
        entry.clip = f"videos/{dst.name}"
    else:
        shutil.copy2(src, dst)
        entry.clip = f"videos/{dst.name}"
        if end > start:
            entry.fragment = f"#t={start:.2f},{end:.2f}"
    return entry


PAGE_CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; padding: 2.5rem 2rem 4rem; background: #0f1115; color: #e6e8ee;
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, sans-serif; }
header { max-width: 1200px; margin: 0 auto 2rem; }
h1 { margin: 0 0 .4rem; font-size: 1.7rem; letter-spacing: -.02em; }
.sub { color: #98a0b3; font-size: .95rem; }
.grid { max-width: 1200px; margin: 0 auto; display: grid; gap: 1.25rem;
  grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); }
.card { background: #171a21; border: 1px solid #262b36; border-radius: 12px; overflow: hidden;
  display: flex; flex-direction: column; }
.card video { width: 100%; aspect-ratio: 4 / 3; background: #000; display: block; }
.no-video { width: 100%; aspect-ratio: 4 / 3; display: grid; place-items: center;
  background: #12141a; color: #6b7488; font-size: .9rem; }
.body { padding: .9rem 1rem 1.1rem; }
.title { font-weight: 600; font-size: 1.02rem; margin-bottom: .35rem; }
.instr { color: #c3c9d6; margin-bottom: .5rem; }
.why { color: #8b93a7; font-size: .87rem; }
.meta { margin-top: .7rem; display: flex; gap: .45rem; flex-wrap: wrap; }
.tag { font-size: .76rem; padding: .16rem .5rem; border-radius: 999px;
  background: #222735; color: #9aa3b8; border: 1px solid #2d3444; }
.tag.warn { background: #2e2318; color: #e0a367; border-color: #4a3620; }
code { font-size: .8rem; color: #7f8798; word-break: break-all; }
"""


def render_page(entries: list[FamilyEntry], title: str) -> str:
    total_eps = sum(e.episodes for e in entries)
    with_video = sum(1 for e in entries if e.clip)
    cards: list[str] = []
    for e in sorted(entries, key=lambda x: (-x.episodes, x.family)):
        if e.clip:
            media = (
                f'<video controls muted loop preload="metadata" '
                f'src="{html.escape(e.clip + e.fragment)}"></video>'
            )
        else:
            media = '<div class="no-video">no successful episode</div>'
        tags = [f'<span class="tag">{e.episodes} episodes</span>']
        if e.episodes == 0:
            tags.append('<span class="tag warn">no data</span>')
        elif e.episodes < 128:
            tags.append('<span class="tag warn">below 128</span>')
        cards.append(
            '<div class="card">'
            f"{media}"
            '<div class="body">'
            f'<div class="title">{html.escape(e.title)}</div>'
            f'<div class="instr">{html.escape(e.instruction)}</div>'
            f'<div class="why">{html.escape(e.rationale)}</div>'
            f'<div class="meta">{"".join(tags)}</div>'
            "</div></div>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>{PAGE_CSS}</style></head>
<body>
<header>
  <h1>{html.escape(title)}</h1>
  <div class="sub">{len(entries)} task families &middot; {total_eps} successful episodes &middot;
  {with_video} with sample video. Each clip is one successful demonstration from the front camera.</div>
</header>
<div class="grid">{"".join(cards)}</div>
</body></html>
"""


def build(args: argparse.Namespace) -> list[FamilyEntry]:
    root = Path(args.datasets_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = args.family_suffix
    prefixes = args.prefix if isinstance(args.prefix, list) else [args.prefix]
    family_dirs: dict[str, Path] = {}
    for prefix in prefixes:
        for d in sorted(root.glob(f"{prefix}_*{suffix}")):
            if d.is_dir():
                family_dirs[d.name[len(prefix) + 1: -len(suffix)] if suffix else d.name] = d
    if not family_dirs:
        raise SystemExit(
            f"no dataset dirs matching {prefixes} with suffix {suffix!r} under {root}"
        )

    entries: list[FamilyEntry] = []
    for family, d in family_dirs.items():
        entry = collect_family(family, d, out_dir, args.camera)
        entries.append(entry)
        print(f"  {family:28} episodes={entry.episodes:4} clip={'yes' if entry.clip else 'NO'}")

    (out_dir / "index.html").write_text(render_page(entries, args.title))
    (out_dir / "gallery.json").write_text(
        json.dumps([e.__dict__ for e in entries], indent=2, default=str) + "\n"
    )
    print(f"\nwrote {out_dir / 'index.html'} ({len(entries)} families)")
    return entries


def serve(out_dir: Path, port: int) -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(out_dir))

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    host = socket.gethostname()
    with Server(("0.0.0.0", port), handler) as httpd:
        print(f"serving {out_dir} at http://{host}:{port}/  (ctrl-c to stop)", flush=True)
        httpd.serve_forever()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--datasets-root",
        default="data/datasets/unmerged",
        help="Parent of the per-family dataset dirs",
    )
    ap.add_argument(
        "--prefix",
        nargs="+",
        default=None,
        help="Experiment prefix(es) to scan for; families from all of them share one page "
        "(required unless --no-build)",
    )
    ap.add_argument(
        "--family-suffix", default="_3b", help="Suffix the execution stage appends to family dirs"
    )
    ap.add_argument("--out-dir", default="gallery", help="Where to write index.html and clips")
    ap.add_argument("--camera", default=DEFAULT_CAMERA, help="Camera to pull the clip from")
    ap.add_argument("--title", default="ToyBlocks TAMP task gallery")
    ap.add_argument("--serve", action="store_true", help="Serve the gallery over HTTP")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--no-build", action="store_true", help="Serve an existing gallery dir as-is")
    args = ap.parse_args()

    if not args.no_build:
        if not args.prefix:
            ap.error("--prefix is required to build the gallery (pass --no-build to only serve it)")
        build(args)
    if args.serve:
        serve(Path(args.out_dir), args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
