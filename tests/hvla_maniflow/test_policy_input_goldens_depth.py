"""Depth policy-input golden (requires maniflow-tb)."""

from __future__ import annotations

import pytest

from tests.serving import test_policy_input_goldens as pig


def test_policy_input_state9_depth_matches_golden(request: pytest.FixtureRequest) -> None:
    pig.test_policy_input_matches_golden("policy_input_state9_depth", request)
