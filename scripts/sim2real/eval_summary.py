"""Success rate and average time per skill, from the eval logs the GUI writes.

Reads ``eval_log_<date>.json`` (see ``toyblocks.serving.sim2real.rollout.eval_log``)
and reduces the graded commands to one row per skill.

A verdict means one of two different things, and the two are never pooled:

  per-skill     a run without the HL. One verdict covers one skill -- "flip
                left", "rotate cw" -- so the rows are those skill names.
  long-horizon  a run with Molmo2. One verdict covers the whole sequence the HL
                drove, so the rows are HL prompts ("Stack G V L") and an 80%
                success rate means 4 of 5 *sequences* finished, not 4 of 5
                skills.

Which one a verdict came from is read off its run header: ``run_metadata``
fills in ``molmo_checkpoint`` only under ``use_molmo``, so its presence is an
exact discriminator rather than a guess.

Time is ``duration_s``: submission to the moment the command stopped running,
excluding however long the operator took to grade it.

Usage:
    python scripts/sim2real/eval_summary.py ~/checkpoints/maniflow
    python scripts/sim2real/eval_summary.py LOG_OR_DIR [...]
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PER_SKILL = "per-skill"
LONG_HORIZON = "long-horizon"
EVAL_LOG_GLOB = "eval_log_*.json"


@dataclass
class Verdict:
    """One graded command, plus the run header it was graded under.

    Deliberately parsed here rather than through
    ``toyblocks.eval.skill_success.verdicts``: that reader carries the clip
    export pipeline with it (numpy, the recordings reader), and a success-rate
    table should not need any of that, nor break if that package goes away.
    """

    skill: str
    success: bool
    duration_s: float | None
    run: dict[str, Any] = field(default_factory=dict)

    @property
    def mode(self) -> str:
        return LONG_HORIZON if self.run.get("molmo_checkpoint") else PER_SKILL


def read_eval_logs(roots) -> list[Verdict]:
    """Graded commands from every eval log at or under ``roots``.

    Malformed lines are skipped rather than failing the read: a log is appended
    to live, so a crashed run can leave a half-written final line.
    """
    logs: set[Path] = set()
    for root in roots:
        root = Path(root).expanduser()
        if root.is_file():
            logs.add(root.resolve())
        elif root.is_dir():
            logs.update(p.resolve() for p in root.rglob(EVAL_LOG_GLOB))
        else:
            logger.warning("no such path: %s", root)

    verdicts: list[Verdict] = []
    for path in sorted(logs):
        run: dict[str, Any] = {}
        for lineno, line in enumerate(open(path), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("%s:%d is not JSON; skipping", path, lineno)
                continue
            if record.get("kind") == "run":
                run = record
            elif record.get("kind") == "verdict":
                duration = record.get("duration_s")
                verdicts.append(
                    Verdict(
                        skill=str(record.get("command", "")),
                        success=bool(record.get("success")),
                        duration_s=None if duration is None else float(duration),
                        run=run,
                    )
                )
    return verdicts


def print_table(verdicts: list[Verdict], mode: str) -> None:
    """One row per skill for ``mode``, worst success rate first."""
    rows = [v for v in verdicts if v.mode == mode]
    if not rows:
        return

    skills: dict[str, list[Verdict]] = {}
    for verdict in rows:
        skills.setdefault(verdict.skill, []).append(verdict)

    def line(name: str, group: list[Verdict]) -> tuple[str, str, str, str]:
        successes = sum(v.success for v in group)
        durations = [v.duration_s for v in group if v.duration_s is not None]
        mean = f"{statistics.fmean(durations):.1f}" if durations else "-"
        rate = f"{round(100 * successes / len(group))}%"
        return name, f"{successes}/{len(group)}", rate, mean

    body = [
        line(name, group)
        for name, group in sorted(
            skills.items(), key=lambda kv: (sum(v.success for v in kv[1]) / len(kv[1]), kv[0])
        )
    ]
    body.append(line("ALL", rows))

    header = ("skill" if mode == PER_SKILL else "HL prompt", "graded", "success", "mean s")
    widths = [max(len(r[i]) for r in [header, *body]) for i in range(4)]
    rule = "  ".join("-" * w for w in widths)

    print(f"\n{mode} ({len(rows)} graded)")
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(header)))
    print(rule)
    for cells in body[:-1]:
        print("  ".join(cells[i].ljust(widths[i]) for i in range(4)))
    print(rule)
    print("  ".join(body[-1][i].ljust(widths[i]) for i in range(4)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "roots",
        nargs="+",
        type=Path,
        help="eval_log_<date>.json files, or directories searched for them",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    verdicts = read_eval_logs(args.roots)
    if not verdicts:
        # Not an error: pointing at a checkpoint folder with no grades yet is a
        # normal thing to do.
        print("no graded commands found", file=sys.stderr)
        return

    for mode in (PER_SKILL, LONG_HORIZON):
        print_table(verdicts, mode)


if __name__ == "__main__":
    main()
