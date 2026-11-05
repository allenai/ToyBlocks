#!/usr/bin/env python3
"""Extract A's (ToyBlocks) CuRobo final interpolated joint trajectories into a pickle.

For each ToyBlocks plan `parallel_policy_instance_{i}.pkl` (i=0..7), lift the two
ManipTrajectory segments' interpolated CuRobo waypoints as plain (T,7) float32
arrays, keyed by (a_instance, reference_frame). The classic ManipTrajectory
executor (env-gated by RG_ATRAJ_PKL) loads this and replays these waypoints
instead of planning, so B executes A's exact arm trajectory.

    .venv/bin/python scripts/debug/extract_a_trajs.py
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
A_PLANS = REPO / "scratchpad/red_green_2block/red_on_green_2block/plans"
OUT = REPO / "scratchpad/red_green_2block/a_interp_trajs.pkl"
N_INSTANCES = 8


class _Stub:
    """Tolerant placeholder for cross-branch classes we don't need to fully load."""

    def __init__(self, *a, **k) -> None:
        pass

    def __setstate__(self, s) -> None:
        self.__dict__.update(s if isinstance(s, dict) else {})


class _TolerantUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        try:
            return super().find_class(module, name)
        except Exception:
            return type(name, (_Stub,), {"__module__": module})


def _traj_from_segment(seg: dict) -> np.ndarray:
    v = seg["value"]
    mp = v.motion_plan
    ip = mp.interpolated_plan
    n = mp.path_buffer_last_tstep
    n0 = int(n[0]) if hasattr(n, "__getitem__") else int(n)
    pos = ip.position
    try:
        pos = pos.detach().cpu().numpy()
    except Exception:
        pos = np.asarray(pos)
    return np.ascontiguousarray(pos[:n0], dtype=np.float32)


def main() -> int:
    out: Dict[Tuple[int, str], np.ndarray] = {}
    for i in range(N_INSTANCES):
        p = A_PLANS / f"parallel_policy_instance_{i}.pkl"
        with open(p, "rb") as f:
            policy = _TolerantUnpickler(f).load()
        for seg in policy:
            if not isinstance(seg, dict) or seg.get("type") != "ManipTrajectory":
                continue
            ref = getattr(seg["value"], "reference_frame", None)
            if ref is None:
                continue
            traj = _traj_from_segment(seg)
            out[(i, ref)] = traj
        got = {r: out[(i, r)].shape for (ii, r) in out if ii == i}
        print(f"instance {i}: {got}")
    with open(OUT, "wb") as f:
        pickle.dump(out, f)
    print(f"\nwrote {OUT}  ({len(out)} trajectories, keys=(a_instance, ref_frame))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
