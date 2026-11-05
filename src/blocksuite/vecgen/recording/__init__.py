"""LeRobot dataset recording (ported from TAMP's ``tmp3.DataRecorder``).

The package stays import-light: ``_video_encoder`` is unpickled in a forkserver child
that must not import isaaclab, so the recorder class is loaded on first access.
"""

from __future__ import annotations

from typing import Any

__all__ = ["IsaacDataRecorderLeRobot", "scan_shards"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from blocksuite.vecgen.recording import lerobot_recorder

        return getattr(lerobot_recorder, name)
    raise AttributeError(name)
