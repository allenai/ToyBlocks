"""Client-side oracle high-level policy integration for the eval harness.

When ``--use_oracle`` is set, :class:`OracleHLController` runs a per-env symbolic
oracle (FF plan + live-sim step checks, :mod:`.hl_policy` / :mod:`.recovering`) next
to the live sim and injects ``task_name`` + ``target_heatmap`` into the observation,
replacing the learned Molmo HL model.

``OracleHLController`` is exposed lazily (PEP 562) so importing this package (e.g. for
the pure ``parse_oracle_action``) does not pull in the controller's serving deps.
"""

from __future__ import annotations

from typing import Any

from blocksuite.eval.oracle.parse import parse_oracle_action

__all__ = ["OracleHLController", "parse_oracle_action"]


def __getattr__(name: str) -> Any:  # PEP 562 lazy attribute access
    if name == "OracleHLController":
        from blocksuite.eval.oracle.controller import OracleHLController

        return OracleHLController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
