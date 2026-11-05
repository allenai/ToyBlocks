#!/usr/bin/env python3
"""Extract per-episode action sequences + recorded states from a LeRobot dataset.

Produces an npz with arrays ordered by reset-state instance 0..7 (so both
datasets align to the same 8 reset states). Dataset A carries `instance_id`;
dataset B does not, so we infer the instance by matching each episode's frame-0
block_red1 xy to the 8 known reset-state red positions (all distinct).

Per episode (padded to Tmax, with a `lengths` array):
  joint_target (T,7), gripper_bin (T,), joint_position (T,7), ee_position (T,6),
  obj_red (T,7), obj_green (T,7).

    .venv/bin/python scripts/debug/extract_dataset.py <dataset_dir> <out.npz>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
CLASSIC_DIR = REPO / "scratchpad/red_green_2block/classic/red_on_green_2block"

# Canonical 8 reset-state red-block xy (env-local), index 0..7 <-> {i+1}.json.
RESET_RED_XY = np.array([
    json.loads((CLASSIC_DIR / f"{i + 1}.json").read_text())["block_positions"]["block_red1"][:2]
    for i in range(8)
])


def col(data, key):
    return key in data


def stack_col(data, key, idx):
    """Rows of a list-valued column for the given frame indices -> (n, dim)."""
    return np.asarray([np.asarray(data[key][i], dtype=np.float64) for i in idx])


def main() -> int:
    ds_dir = Path(sys.argv[1])
    out = Path(sys.argv[2])
    # read the success split plus the sibling failure split (so a failed episode
    # still contributes its recorded trajectory for a complete 8-instance set).
    pfs = sorted(ds_dir.glob("data/chunk-*/file-*.parquet"))
    fail = ds_dir.parent / "blocks_failure"
    if fail.exists():
        pfs += sorted(fail.glob("data/chunk-*/file-*.parquet"))

    RED = "observation.state.object_pose.block_red1"
    GREEN = "observation.state.object_pose.block_green1"
    JT = "action.joint_target"
    JP = "observation.state.joint_position"
    EE = "observation.state.ee_position"
    GB = "action.gripper_position"

    per_inst = {}
    n_eps = 0
    for pf in pfs:
        data = pq.read_table(pf).to_pydict()
        if RED not in data:
            print(f"ERROR: {pf} has no object_pose.block_red1 column")
            return 1
        ep = np.asarray(data["episode_index"]).reshape(-1)
        for e in sorted(set(ep.tolist())):
            n_eps += 1
            idx = np.flatnonzero(ep == e)
            red = stack_col(data, RED, idx)      # (T,7)
            red_xy0 = red[0, :2]
            if "instance_id" in data:
                inst = int(np.asarray(data["instance_id"][idx[0]]).reshape(-1)[0])
            else:
                inst = int(np.argmin(np.linalg.norm(RESET_RED_XY - red_xy0, axis=1)))
            if inst in per_inst:  # instance already seen (success wins over failure)
                continue
            per_inst[inst] = dict(
                jt=stack_col(data, JT, idx),
                gb=stack_col(data, GB, idx),  # (T,1)
                jp=stack_col(data, JP, idx),
                ee=stack_col(data, EE, idx),
                red=red,
                green=stack_col(data, GREEN, idx),
                T=len(idx),
                red_xy0=red_xy0,
            )

    insts = sorted(per_inst)
    print(f"{ds_dir.name} (+failure): {n_eps} episodes -> instances {insts}")
    if insts != list(range(8)):
        print(f"  WARNING: instances are not 0..7 exactly: {insts}")

    lengths = np.array([per_inst[i]["T"] for i in insts])
    Tmax = int(lengths.max())

    def pad(key):
        dim = np.atleast_2d(per_inst[insts[0]][key].reshape(per_inst[insts[0]]["T"], -1)).shape[1]
        arr = np.zeros((len(insts), Tmax, dim), dtype=np.float64)
        for k, i in enumerate(insts):
            v = per_inst[i][key].reshape(per_inst[i]["T"], -1)
            arr[k, : per_inst[i]["T"]] = v
            arr[k, per_inst[i]["T"]:] = v[-1]  # hold last
        return arr

    np.savez(
        out,
        instances=np.array(insts),
        lengths=lengths,
        joint_target=pad("jt"),
        gripper_bin=pad("gb")[..., 0],
        joint_position=pad("jp"),
        ee_position=pad("ee"),
        obj_red=pad("red"),
        obj_green=pad("green"),
        red_xy0=np.array([per_inst[i]["red_xy0"] for i in insts]),
    )
    print(f"  wrote {out}  Tmax={Tmax} lengths={lengths.tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
