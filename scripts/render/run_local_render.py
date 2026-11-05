#!/usr/bin/env python3
"""Run render.py to completion and then actually make it exit.

render.py prints ``[render] all jobs done`` immediately before a teardown that routinely
hangs -- Isaac/Kit does not reliably return from ``simulation_app.close()``. The datagen and
play pipelines run it inside a ``bash -c a && b`` chain, where a hang does not just waste a GPU, it blocks ``run_local_stage`` forever and the pipeline never
reaches merge. Observed: 16/16 episodes rendered and finalized, then 53 minutes of nothing.

So success is decided by the log marker and the on-disk result, never by the exit status.

    python scripts/render/run_local_render.py -- <the full render.py argv>
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

MARKER = "all jobs done"   # render.py prints this just before its teardown hang
GRACE_SECONDS = 30         # let it close cleanly after the marker before we SIGKILL the pgid


def _killpg(pgid: int) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        time.sleep(2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stall-timeout", type=int, default=3600,
                    help="SIGKILL if the render prints nothing for this long (0 disables)")
    ap.add_argument("--hard-timeout", type=int, default=0,
                    help="SIGKILL after this many seconds regardless (0 disables)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="-- followed by the render.py command line")
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        print("[run_local_render] no command given", file=sys.stderr)
        return 2

    started = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            start_new_session=True, text=True, errors="replace", bufsize=1)
    pgid = os.getpgid(proc.pid)
    saw_marker = False
    last_out = time.time()

    try:
        for line in proc.stdout:                      # stream through so the pipeline log keeps it
            sys.stdout.write(line)
            sys.stdout.flush()
            last_out = time.time()
            if MARKER in line:
                saw_marker = True
                print(f"[run_local_render] marker seen; {GRACE_SECONDS}s grace then SIGKILL "
                      f"(render.py hangs in teardown)", flush=True)
                break
    except Exception:                                  # pipe closed by our own kill is normal
        pass

    if saw_marker:
        deadline = time.time() + GRACE_SECONDS
        while time.time() < deadline and proc.poll() is None:
            time.sleep(1)
        if proc.poll() is None:
            _killpg(pgid)
        print(f"[run_local_render] done in {int(time.time()-started)}s", flush=True)
        return 0

    # No marker: let it finish, or kill it on a stall / hard timeout.
    while proc.poll() is None:
        now = time.time()
        if args.stall_timeout and now - last_out > args.stall_timeout:
            print(f"[run_local_render] no output for {int(now-last_out)}s -> SIGKILL", flush=True)
            _killpg(pgid)
            return 3
        if args.hard_timeout and now - started > args.hard_timeout:
            print(f"[run_local_render] hard timeout {args.hard_timeout}s -> SIGKILL", flush=True)
            _killpg(pgid)
            return 4
        time.sleep(2)
    rc = proc.returncode
    print(f"[run_local_render] exited rc={rc} WITHOUT the '{MARKER}' marker", flush=True)
    return rc if rc else 5


if __name__ == "__main__":
    sys.exit(main())
