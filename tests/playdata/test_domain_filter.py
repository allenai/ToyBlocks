"""blocksuite.playdata.domain_filter: restricting a domain to allowed actions."""
from pathlib import Path

import pytest

from blocksuite.playdata.domain_filter import (
    filter_domain_text,
    list_actions,
    resolve_allowed_actions,
    write_filtered_domain,
)

DOMAIN = Path(__file__).resolve().parents[2] / "src/blocksuite/tamp/blocks/domain.pddl"


def _balanced(text: str) -> bool:
    depth = 0
    for line in text.splitlines():
        for ch in line.split(";", 1)[0]:
            depth += (ch == "(") - (ch == ")")
            if depth < 0:
                return False
    return depth == 0


def test_lists_live_actions_only() -> None:
    actions = list_actions(DOMAIN.read_text())
    assert {"init", "move_to", "rotate", "reorient", "push_to_right"} <= set(actions)
    assert "isolate" not in actions  # commented out in the domain


def test_disallowed_actions_are_removed_and_domain_stays_valid() -> None:
    text = DOMAIN.read_text()
    keep = resolve_allowed_actions(list_actions(text), None, ["push_to_right", "push_next_to"])
    out = filter_domain_text(text, keep)
    assert "push_to_right" not in list_actions(out) and "push_next_to" not in list_actions(out)
    assert set(list_actions(out)) == set(list_actions(text)) - {"push_to_right", "push_next_to"}
    assert _balanced(out)


def test_allowed_list_keeps_bookkeeping() -> None:
    keep = resolve_allowed_actions(list_actions(DOMAIN.read_text()), ["rotate", "reorient"], None)
    assert keep == ["init", "move_to", "rotate", "reorient"]


def test_unknown_action_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown high-level action"):
        resolve_allowed_actions(list_actions(DOMAIN.read_text()), ["rotat"], None)


def test_write_leaves_original_untouched(tmp_path: Path) -> None:
    before = DOMAIN.read_text()
    kept = write_filtered_domain(DOMAIN, tmp_path / "domain.pddl", None, ["rotate"])
    assert "rotate" not in kept
    assert "rotate" not in list_actions((tmp_path / "domain.pddl").read_text())
    assert DOMAIN.read_text() == before
