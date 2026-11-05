"""Per-skill success rates for an eval run, written to ``<eval_dir>/skill_success.json``.

Each episode in ``eval_episodes.json`` is labelled by the skill its source dataset
episode recorded: the sequence of ``action.high_level_action`` commands with
repeats collapsed and no-op actions dropped. A single-command episode gets
``flip <dir>``, ``rotate <dir>``, ``stack`` (``pnp_table_block``), ``unstack``
(``pnp_block_table``) or ``move on table`` (``pnp_table_table``); anything else is
labelled by its command sequence joined with ``" + "``.

An episode whose eval scored sub-goals (oracle / Molmo HL runs) is counted by its
sub-goals instead of its skill, with ``attempts`` / ``success`` (and
``success_ignore_collision`` under ``--collision penalize``), into two groups:
``aggregate`` (``flip``, ``rotate``, ``pick``, ``place``) and ``detailed`` (``flip <dir>``,
``rotate <dir>``, ``pick (table)``, ``pick (from block)``, ``place`` (on a block),
``place on table``). A pick's source comes from a single pick-and-place episode's skill
(``unstack`` picks from a block); other picks count only toward ``aggregate``. Both
groups end with the episode-level ``all`` row; episodes without sub-goals add their
skill rows to ``detailed``.

Usage:
    python scripts/skill_success_from_eval_run.py --eval-dir <run_dir>
"""

from __future__ import annotations

import collections
import dataclasses
import json
from pathlib import Path

import tyro

from blocksuite.utils.wandb_logging import read_episode_rows

# HL actions that carry no skill of their own (TAMP bookkeeping / free motion).
_IGNORED_HL_ACTIONS = frozenset({"", "init", "move_to", "nudge", "stop", "done", "reach-goal"})
_PNP_SKILLS = {
    "pnp_table_block": "stack",
    "pnp_block_table": "unstack",
    "pnp_table_table": "move on table",
}
# The table a place sub-goal names when it puts the block on the table.
_TABLE = "workspace_table"


# Fixed column order for spreadsheet paste (``--atomic-copy-row``).
_ATOMIC_SUCCESS_ONCE_ROW_KEYS: tuple[str, ...] = (
    "all",
    "flip backward",
    "flip forward",
    "flip left",
    "flip right",
    "rotate ccw",
    "rotate cw",
    "stack",
    "unstack",
)


def format_atomic_success_once_row(summary: dict[str, dict[str, float | int]]) -> str:
    """Tab-separated ``success_once`` rates as percentages (one decimal), fixed column order."""
    parts: list[str] = []
    for key in _ATOMIC_SUCCESS_ONCE_ROW_KEYS:
        entry = summary.get(key)
        if entry is None:
            parts.append("")
            continue
        rate = float(entry["success_once_rate"])
        parts.append(f"{100 * rate:.1f}%")
    return "\t".join(parts)


@dataclasses.dataclass
class Args:
    eval_dir: Path
    """Eval run directory holding ``eval_episodes.json``."""

    atomic_copy_row: bool = False
    """Print a tab-separated ``success_once`` %% row for atomic eval spreadsheets."""


def _skill_label(commands: list[tuple[str, str]]) -> str:
    """Skill name for one episode's collapsed ``(action, args_json)`` command sequence."""
    if len(commands) != 1:
        return " + ".join(action for action, _ in commands) or "<none>"
    action, args_json = commands[0]
    if action in ("reorient", "rotate"):
        args = json.loads(args_json)
        verb = "flip" if action == "reorient" else "rotate"
        return f"{verb} {args[2].replace('_flip', '')}"
    return _PNP_SKILLS.get(action, action)


def _sub_goal_labels(name: str, skill: str) -> tuple[str, str | None] | None:
    """``(aggregate, detailed)`` rows for an eval sub-goal name (``pick block_a``,
    ``place block_a on workspace_table``, ``flip block_a left``) of an episode with ``skill``;
    None for other sub-goals. The detailed row is None for a pick whose source is unknown
    (any episode but a single pick-and-place)."""
    verb, *rest = name.split()
    if verb == "pick":
        if skill == "unstack":
            return "pick", "pick (from block)"
        return "pick", "pick (table)" if skill in ("stack", "move on table") else None
    if verb == "place":
        return "place", "place on table" if rest[-1] == _TABLE else "place"
    if verb in ("flip", "rotate"):
        return verb, f"{verb} {rest[1].replace('_flip', '')}" if len(rest) > 1 else verb
    return None


def episode_skills(dataset_path: str, episode_indices: set[int]) -> dict[int, str]:
    """``episode_index -> skill`` for the given episodes of a LeRobot dataset."""
    commands: dict[int, list[tuple[str, str]]] = collections.defaultdict(list)
    for ep, action, args in read_episode_rows(
        dataset_path, episode_indices, ["action.high_level_action", "action.high_level_args"]
    ):
        action = (action or "").strip().lower()
        if action in _IGNORED_HL_ACTIONS:
            continue
        seq = commands[ep]
        if not seq or seq[-1][0] != action:
            seq.append((action, args))
    return {ep: _skill_label(commands.get(ep, [])) for ep in episode_indices}


def main(args: Args) -> None:
    episodes = json.loads((args.eval_dir / "eval_episodes.json").read_text())
    by_dataset: dict[str, set[int]] = collections.defaultdict(set)
    for ep in episodes:
        by_dataset[ep["source_dataset_path"]].add(int(ep["source_dataset_episode_idx"]))
    skills = {
        (path, idx): skill
        for path, indices in by_dataset.items()
        for idx, skill in episode_skills(path, indices).items()
    }

    metrics: list[str] = ["success_at_end", "success_once"]
    # --collision penalize also records the successes with collisions ignored.
    if any("success_once_ignore_collision" in ep for ep in episodes):
        metrics += ["success_at_end_ignore_collision", "success_once_ignore_collision"]
    counts: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"episodes": 0, **{m: 0 for m in metrics}}
    )
    # Episodes with scored sub-goals (oracle / Molmo HL runs) are counted by their
    # sub-goals instead of their skill, into aggregate and detailed rows.
    sub_metrics: list[str] = ["success"]
    if any("sub_goal_success_ignore_collision" in ep for ep in episodes):
        sub_metrics.append("success_ignore_collision")

    def _sub_counter() -> collections.defaultdict[str, dict[str, int]]:
        return collections.defaultdict(lambda: {"attempts": 0, **{m: 0 for m in sub_metrics}})

    agg_counts = _sub_counter()
    detail_counts = _sub_counter()
    for ep in episodes:
        skill = skills[(ep["source_dataset_path"], int(ep["source_dataset_episode_idx"]))]
        if ep.get("sub_goals"):
            oks = ep["sub_goal_success"]
            dones = ep.get("sub_goal_success_ignore_collision", oks)
            for name, ok, done in zip(ep["sub_goals"], oks, dones):
                labels = _sub_goal_labels(name, skill)
                if labels is None:
                    continue
                for sub_counts, label in zip((agg_counts, detail_counts), labels):
                    if label is None:
                        continue
                    c = sub_counts[label]
                    c["attempts"] += 1
                    c["success"] += bool(ok)
                    if "success_ignore_collision" in c:
                        c["success_ignore_collision"] += bool(done)
        keys: tuple[str, ...] = ("all",) if ep.get("sub_goals") else (skill, "all")
        for key in keys:
            c = counts[key]
            c["episodes"] += 1
            for m in metrics:
                c[m] += bool(ep.get(m))

    def _sub_rows(sub_counts: dict[str, dict[str, int]]) -> dict[str, dict]:
        return {
            label: {**c, **{f"{m}_rate": c[m] / c["attempts"] for m in sub_metrics}}
            for label, c in sorted(sub_counts.items())
        }

    episode_rows = {
        skill: {**c, **{f"{m}_rate": c[m] / c["episodes"] for m in metrics}}
        for skill, c in sorted(counts.items(), key=lambda kv: (kv[0] == "all", kv[0]))
    }
    # Skill rows only come from episodes without sub-goals (e.g. plain policy runs).
    summary = {
        "aggregate": {**_sub_rows(agg_counts), "all": episode_rows["all"]},
        "detailed": {**_sub_rows(detail_counts), **episode_rows},
    }
    out = args.eval_dir / "skill_success.json"
    out.write_text(json.dumps(summary, indent=2))

    def _print(unit: str, ms: list[str], rows: dict[str, dict]) -> None:
        width = max(len(m) for m in ms) + 2
        print(f"{'skill':28s} {unit:>8s}" + "".join(f" {m:>{width}s}" for m in ms))
        for skill, s in rows.items():
            cells = [f"{s[m]:5d} ({100 * s[m + '_rate']:5.1f}%)" for m in ms]
            print(f"{skill:28s} {s[unit]:8d}" + "".join(f" {c:>{width}s}" for c in cells))

    for group, sub_counts in (("aggregate", agg_counts), ("detailed", detail_counts)):
        if sub_counts:
            print(f"[{group}]")
            _print("attempts", sub_metrics, _sub_rows(sub_counts))
            print()
    _print("episodes", metrics, episode_rows)
    print(f"wrote {out}")

    if args.atomic_copy_row:
        row = format_atomic_success_once_row(summary)
        labels = (
            "all, flip bwd, flip fwd, flip l, flip r, rot ccw, rot cw, stack, unstack"
        )
        print(f"===> atomic skill success_once (%), tab-separated ({labels})")
        print(row)


if __name__ == "__main__":
    main(tyro.cli(Args))
