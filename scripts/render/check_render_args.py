#!/usr/bin/env python3
"""Validate a render.py command line WITHOUT launching Isaac.

render.py builds its parser after importing isaaclab, so the only way to find out that a flag
is misspelled is normally to schedule a GPU job, wait for the image, stage the data, and read
the traceback. That cost a 3.8 GB stage and two job slots for a stray `--seed`.

This extracts the accepted option strings straight from the source with AST and checks a command
line against them, in milliseconds, on any box.

    python scripts/render/check_render_args.py -- <the args you plan to pass>
    python scripts/render/check_render_args.py --list
"""
import argparse
import ast
import sys
from pathlib import Path

RENDER = Path(__file__).resolve().parent / "render.py"

# Added to the parser by isaaclab's AppLauncher.add_app_launcher_args(), not by any
# add_argument call in render.py.
APP_LAUNCHER_OPTS = {
    "--headless", "--livestream", "--enable_cameras", "--xr", "--device", "--verbose",
    "--info", "--experience", "--rendering_mode", "--kit_args", "--anim_recording_enabled",
    "--anim_recording_start_time", "--anim_recording_stop_time",
}


def accepted_options(path: Path = RENDER) -> set:
    """Every ``--flag`` render.py's parser accepts, read from the source."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    opts = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr == "add_argument"):
            continue
        for a in node.args:
            if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith("-"):
                opts.add(a.value)
        # argparse also accepts --foo-bar for dest foo_bar only if declared; do not invent it.
    if "AppLauncher.add_app_launcher_args" in path.read_text(encoding="utf-8"):
        # Isaac injects these into the same parser at runtime, so they are valid even though no
        # add_argument call for them exists in this file. Omitting them made the checker reject
        # a correct command line, which is worse than the error it exists to catch.
        opts |= APP_LAUNCHER_OPTS
    return opts


def check(argv, opts):
    """Return the list of unknown options in ``argv``."""
    bad = []
    for tok in argv:
        if not tok.startswith("-") or tok == "--":
            continue
        name = tok.split("=", 1)[0]
        if name in opts:
            continue
        # a negative number or a bare value like -1 is not a flag
        try:
            float(tok)
            continue
        except ValueError:
            pass
        bad.append(name)
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print every accepted option and exit")
    ap.add_argument("--source", default=str(RENDER))
    ap.add_argument("args", nargs="*", help="the command line to check (after --)")
    a = ap.parse_args()

    opts = accepted_options(Path(a.source))
    if a.list:
        for o in sorted(opts):
            print(" ", o)
        print(f"  {len(opts)} options")
        return 0
    bad = check(a.args, opts)
    if bad:
        print(f"[check] UNKNOWN option(s): {bad}")
        for b in bad:
            near = sorted(o for o in opts if b.lstrip('-')[:4] in o)
            if near:
                print(f"         {b} -> did you mean {near[:4]}?")
        return 1
    print(f"[check] ok: {len(a.args)} token(s), all options recognised "
          f"({len(opts)} accepted by render.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
