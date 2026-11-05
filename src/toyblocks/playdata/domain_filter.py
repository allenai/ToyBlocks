"""Restrict a PDDL domain to a subset of its high-level actions.

Play data can leave actions out of planning (e.g. no push / align, so no RL
skill is ever needed). ``ignore_hl_actions`` in ``action_config.yaml`` is not
the right switch for that: it only means "no low-level spec needed", and the
planner still chooses such an action and then executes it as a no-op. Removing
the action from the domain is what actually keeps the planner from using it.

Pure text processing, importable without Isaac Sim or the planner.
"""
from __future__ import annotations

import re
from pathlib import Path

#: Bookkeeping actions every plan relies on; never filtered out.
ALWAYS_KEPT_ACTIONS: frozenset[str] = frozenset({"init", "move_to"})

_ACTION_RE = re.compile(r"\(:action\s+([A-Za-z0-9_\-]+)", re.IGNORECASE)


def _strip_comment(line: str) -> str:
    """``line`` without its ``;`` comment (PDDL comments run to end of line)."""
    idx = line.find(";")
    return line if idx < 0 else line[:idx]


def _action_spans(text: str) -> list[tuple[str, int, int]]:
    """``(name, start, end)`` character spans of every live ``(:action ...)`` block.

    Commented-out actions are skipped: the scan works on a copy with comments
    blanked out, so character offsets still line up with ``text``.
    """
    # Blank comments but keep every character position (and newline) intact.
    live = "".join(
        _strip_comment(line).ljust(len(line)) if not line.endswith("\n")
        else _strip_comment(line[:-1]).ljust(len(line) - 1) + "\n"
        for line in text.splitlines(keepends=True)
    )
    spans: list[tuple[str, int, int]] = []
    for match in _ACTION_RE.finditer(live):
        start = match.start()
        depth = 0
        for pos in range(start, len(live)):
            if live[pos] == "(":
                depth += 1
            elif live[pos] == ")":
                depth -= 1
                if depth == 0:
                    spans.append((match.group(1).lower(), start, pos + 1))
                    break
        else:
            raise ValueError(f"unbalanced parentheses in action {match.group(1)!r}")
    return spans


def list_actions(text: str) -> list[str]:
    """Names of the live (not commented-out) actions in a domain, in file order."""
    return [name for name, _, _ in _action_spans(text)]


def resolve_allowed_actions(
    all_actions: list[str],
    allowed: list[str] | tuple[str, ...] | None,
    disallowed: list[str] | tuple[str, ...] | None,
) -> list[str]:
    """Actions to keep: ``allowed`` (default: all) minus ``disallowed``, plus bookkeeping.

    Raises ``ValueError`` for a name the domain does not define, so a typo does not
    silently leave the domain unrestricted.
    """
    known = set(all_actions)
    requested = {a.lower() for a in (allowed or [])} | {a.lower() for a in (disallowed or [])}
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(f"unknown high-level action(s) {unknown}; the domain defines {sorted(known)}")
    keep = set(all_actions) if not allowed else {a.lower() for a in allowed}
    keep -= {a.lower() for a in (disallowed or [])}
    keep |= ALWAYS_KEPT_ACTIONS & known
    return [a for a in all_actions if a in keep]


def filter_domain_text(text: str, keep: list[str] | tuple[str, ...] | set[str]) -> str:
    """``text`` with every live action not in ``keep`` removed."""
    keep_set = {a.lower() for a in keep}
    out: list[str] = []
    cursor = 0
    for name, start, end in _action_spans(text):
        if name in keep_set:
            continue
        out.append(text[cursor:start])
        out.append(f"; (:action {name} ...) removed: not in the allowed high-level actions\n")
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


def write_filtered_domain(
    domain_file: Path,
    out_file: Path,
    allowed: list[str] | tuple[str, ...] | None,
    disallowed: list[str] | tuple[str, ...] | None,
) -> list[str]:
    """Write ``domain_file`` restricted to the allowed actions to ``out_file``.

    Returns the actions kept. The original file is never modified.
    """
    text = Path(domain_file).read_text()
    keep = resolve_allowed_actions(list_actions(text), allowed, disallowed)
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(filter_domain_text(text, keep))
    return keep
