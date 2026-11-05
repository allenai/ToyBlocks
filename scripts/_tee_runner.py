#!/usr/bin/env python3
"""Wrapper that runs a target Python script while teeing stdout/stderr to a log file.

Used by VS Code launch configurations that want to capture a script's output to
disk (debugpy does not run ``args`` through a shell, so ``>`` / ``| tee`` cannot
be used there).

Implementation note: the tee is done at the OS file-descriptor level (``os.dup2``
onto fd 1 / fd 2) rather than by reassigning ``sys.stdout`` / ``sys.stderr``.
That way *child processes* spawned via ``multiprocessing`` /
``ProcessPoolExecutor`` (including the ``'spawn'`` start method, which re-execs
Python while inheriting fd 1 / fd 2) also flow through the tee and end up in
the log file.

Usage:
    python _tee_runner.py --log <path> [--append] -- <script.py> [script args...]

Environment overrides:
    TEE_LOG_FILE  Default log path if --log is not provided.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
import threading
from pathlib import Path
from typing import IO


def _pump(read_fd: int, mirror_fd: int, log_file: IO[bytes]) -> None:
    """Read from ``read_fd`` and write each chunk to both the terminal fd and the log."""
    try:
        while True:
            try:
                chunk = os.read(read_fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            try:
                os.write(mirror_fd, chunk)
            except OSError:
                pass
            try:
                log_file.write(chunk)
                log_file.flush()
            except Exception:
                pass
    finally:
        try:
            os.close(read_fd)
        except OSError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--log", default=os.environ.get("TEE_LOG_FILE"))
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to log instead of overwriting",
    )
    args, rest = parser.parse_known_args()

    if rest and rest[0] == "--":
        rest = rest[1:]
    if not rest:
        sys.stderr.write("usage: _tee_runner.py --log <path> -- <script.py> [args...]\n")
        sys.exit(2)

    target, *target_args = rest

    log_path = Path(args.log or "run.log").expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "ab" if args.append else "wb", buffering=0)

    sys.__stdout__.flush()
    sys.__stderr__.flush()

    orig_stdout_fd = os.dup(1)
    orig_stderr_fd = os.dup(2)

    out_r, out_w = os.pipe()
    err_r, err_w = os.pipe()

    os.dup2(out_w, 1)
    os.dup2(err_w, 2)
    os.close(out_w)
    os.close(err_w)

    sys.stdout = os.fdopen(1, "w", buffering=1, encoding="utf-8", closefd=False)
    sys.stderr = os.fdopen(2, "w", buffering=1, encoding="utf-8", closefd=False)

    t_out = threading.Thread(
        target=_pump, args=(out_r, orig_stdout_fd, log_file), name="tee-stdout"
    )
    t_err = threading.Thread(
        target=_pump, args=(err_r, orig_stderr_fd, log_file), name="tee-stderr"
    )
    t_out.start()
    t_err.start()

    print(f"[tee_runner] logging to {log_path}", file=sys.stderr)

    sys.argv = [target, *target_args]
    exit_code = 0
    try:
        runpy.run_path(target, run_name="__main__")
    except SystemExit as e:
        exit_code = int(e.code) if isinstance(e.code, int) else (0 if e.code is None else 1)
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass

        os.dup2(orig_stdout_fd, 1)
        os.dup2(orig_stderr_fd, 2)
        os.close(orig_stdout_fd)
        os.close(orig_stderr_fd)

        t_out.join(timeout=5)
        t_err.join(timeout=5)

        try:
            log_file.flush()
            log_file.close()
        except Exception:
            pass

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
