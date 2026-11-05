#!/usr/bin/env python3
"""Translate ToyBlocks ``reset_states.pkl`` into TAMP blocks-classic instance JSONs.

Lets both datagen pipelines start from *bit-identical* initial states so a
dataset diff isolates pipeline differences rather than scene sampling.

The two representations:

  ToyBlocks   one ``reset_states.pkl`` holding a batched Isaac state dict
               (``rigid_object[name]["root_pose"]`` = env-local xyz+wxyz), plus
               ``profiles.json`` with per-profile block sizes / table height /
               friction, plus ``profile_ids`` mapping state -> profile.

  classic      one JSON per instance, ``1.json``, ``2.json``, ... containing
               ``block_positions`` / ``block_quaternions`` dicts plus the
               profile's ``cube_scale`` / ``table_height_delta`` /
               ``table_friction``.

Three conventions have to line up, and they are the reason this is a script and
not a copy:

1. **Indexing.** classic instance files are **1-based** (``1.json`` first) while
   ToyBlocks reset-state indices are 0-based, so instance *i* of ToyBlocks
   becomes ``{i+1}.json``.
2. **Table height.** classic applies ``z + table_height_delta`` at reset unless
   ``positions_include_table_height`` is set (see the sim2real template's
   ``thd`` handling). ToyBlocks poses are already the settled world-local z, so
   this writer sets the flag True and hands classic the z verbatim — no implicit
   shift. Both pipelines put the table at ``x=0.4888``, so the frames agree.
3. **Block size.** ToyBlocks carries per-block absolute sizes; classic derives
   one isotropic edge as ``base_cube_edge * cube_scale``. This script reports the
   ``--base_cube_edge`` / ``--cube_scale`` pair that reproduces the ToyBlocks
   size so env generation can be invoked to match, and refuses to guess when
   toyblocks's blocks are not all the same size.

Usage::

    python scripts/datagen/reset_states_to_classic.py \\
        --env-path outputs/r_on_red_5block \\
        --out-dir <classic task dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from toyblocks.utils.reset_states_io import load_reset_states  # noqa: E402

# classic's default unscaled cuboid edge; only used to report a matching pair.
CLASSIC_DEFAULT_BASE_EDGE = 0.02985
# Table root x both pipelines place the workspace table at, used as a sanity check.
EXPECTED_TABLE_X = 0.4888


def _block_names(states: dict) -> list[str]:
    """Rigid objects that are blocks (tables are scene furniture, not state)."""
    return [n for n in states["rigid_object"] if "table" not in n.lower()]


def _settled_block_z(dataset_root: Path, names: list[str]) -> np.ndarray:
    """First-frame block z of every episode in a recorded LeRobot dataset.

    Frame 0 of a recorded episode is the settled reset state, which is the
    height a block actually comes to rest at in that pipeline's scene — unlike
    the generated instance JSONs, which hold pre-settle spawn poses.
    """
    import pyarrow.parquet as pq

    files = sorted(dataset_root.glob("**/data/chunk-*/file-*.parquet"))
    out: list[float] = []
    for path in files:
        try:
            data = pq.read_table(path).to_pydict()
        except Exception:
            continue
        if "episode_index" not in data:
            continue
        # Prefer the requested blocks; otherwise fall back to whatever blocks the
        # reference happens to contain. The offset is a property of the scene's
        # table surface, not of which blocks sit on it, so a reference recorded
        # for a different task in the same scene is still valid.
        cols = [f"observation.state.object_pose.{n}" for n in names
                if f"observation.state.object_pose.{n}" in data]
        if not cols:
            cols = [c for c in data
                    if c.startswith("observation.state.object_pose.")
                    and "table" not in c.lower()]
        episodes = np.asarray(data["episode_index"]).reshape(-1)
        for episode in sorted(set(episodes.tolist())):
            first = int(np.flatnonzero(episodes == episode)[0])
            for col in cols:
                out.append(float(np.asarray(data[col][first])[2]))
    return np.asarray(out)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--env-path", required=True, type=Path,
                        help="toyblocks env package (holds reset_states.pkl + profiles.json)")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="classic task directory to write {i}.json into")
    parser.add_argument("--base-cube-edge", type=float, default=None,
                        help="Override the reported base edge (default: use the "
                             "toyblocks block size so cube_scale is exactly 1.0)")
    parser.add_argument("--z-offset", type=float, default=None,
                        help="Add this to every block z. The two scenes' table "
                             "surfaces sit at different heights, so a verbatim copy "
                             "leaves blocks floating or sunk into the table.")
    parser.add_argument("--settled-reference", type=Path, default=None,
                        help="A recorded dataset from the TARGET pipeline, used to "
                             "measure the z offset from *settled* block heights. "
                             "Do not calibrate against the target's generated "
                             "instance JSONs: those hold spawn poses, not resting "
                             "poses, and differ from the settled height by more than "
                             "a block half-height.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report the mapping without writing files")
    args = parser.parse_args()

    states, profile_ids, num_profiles = load_reset_states(args.env_path / "reset_states.pkl")
    profiles_path = args.env_path / "profiles.json"
    profiles = json.loads(profiles_path.read_text()) if profiles_path.exists() else None

    names = _block_names(states)
    n_states = states["rigid_object"][names[0]]["root_pose"].shape[0]
    print(f"toyblocks: {n_states} reset state(s), {len(names)} block(s), "
          f"num_profiles={num_profiles}")
    print(f"  blocks: {names}")

    # Sanity: the two pipelines must agree on where the table is, or the block
    # poses mean different things in each frame.
    table = next((n for n in states["rigid_object"] if "table" in n.lower()), None)
    if table is not None:
        table_x = float(states["rigid_object"][table]["root_pose"][0, 0])
        status = "OK" if abs(table_x - EXPECTED_TABLE_X) < 1e-3 else "MISMATCH"
        print(f"  table {table!r} x={table_x:.4f} vs classic default "
              f"{EXPECTED_TABLE_X:.4f} -> {status}")
        if status == "MISMATCH":
            print("  ERROR: table frames differ; block poses are not comparable.")
            return 1

    # Resolve per-profile physics. classic wants one isotropic scale per profile.
    sizes_seen: set[float] = set()
    profile_meta: dict[int, dict] = {}
    if profiles is not None:
        for prof in profiles["profiles"]:
            edges = {round(float(v[0]), 9) for v in prof["block_sizes"].values()}
            anisotropic = any(
                len({round(float(c), 9) for c in v}) > 1
                for v in prof["block_sizes"].values()
            )
            if anisotropic:
                print(f"  ERROR: profile {prof['profile_id']} has non-cubic blocks; "
                      "classic bakes a single isotropic edge and cannot represent them.")
                return 1
            if len(edges) > 1:
                print(f"  ERROR: profile {prof['profile_id']} mixes block sizes {sorted(edges)}; "
                      "classic bakes one edge for every block.")
                return 1
            edge = edges.pop()
            sizes_seen.add(edge)
            profile_meta[int(prof["profile_id"])] = {
                "edge": edge,
                "table_height": float(prof.get("table_height", 0.0)),
                "friction": float(prof.get("static_friction", 0.1)),
            }

    if len(sizes_seen) > 1:
        print(f"  NOTE: profiles use {len(sizes_seen)} distinct block edges {sorted(sizes_seen)}. "
              "classic bakes ONE edge per env folder, so these cannot all be honoured "
              "in a single classic task dir — generate one dir per size.")
    base_edge = args.base_cube_edge if args.base_cube_edge else (
        sorted(sizes_seen)[0] if sizes_seen else CLASSIC_DEFAULT_BASE_EDGE
    )

    print("\nclassic env-generation must be invoked with a matching block size:")
    for edge in sorted(sizes_seen) or [base_edge]:
        print(f"  edge {edge:.5f} m  ->  --base_cube_edge {base_edge:.6f} "
              f"--cube_scale_low {edge / base_edge:.6f} --cube_scale_high {edge / base_edge:.6f}")

    # Resting height differs between the two scenes even at the same nominal table
    # pose and block size, so preserve the *layout* (xy + orientation + resting on
    # the table) rather than the absolute z, which would float or sink the blocks.
    src_z = np.concatenate(
        [np.asarray(states["rigid_object"][n]["root_pose"])[:, 2] for n in names]
    )
    z_offset = args.z_offset
    if z_offset is not None:
        print(f"\nz offset (explicit): {z_offset:+.5f} m")
    elif args.settled_reference is not None:
        settled = _settled_block_z(args.settled_reference, names)
        if not settled.size:
            print(f"\nERROR: no block poses found under {args.settled_reference}")
            return 1
        z_offset = float(np.median(settled) - np.median(src_z))
        print(f"\nz offset from settled reference ({settled.size} pose(s)): {z_offset:+.5f} m")
        print(f"  target settled median z {np.median(settled):.5f} "
              f"(std {settled.std():.5f}) vs ToyBlocks {np.median(src_z):.5f}")
    else:
        z_offset = 0.0
        print("\nWARNING: no --z-offset or --settled-reference; copying z unchanged. "
              "If the two scenes' table surfaces differ, blocks will spawn floating "
              "or interpenetrating and will skid as physics resolves them.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i in range(n_states):
        pid = int(profile_ids[i]) if profile_ids is not None else 0
        meta = profile_meta.get(pid, {"edge": base_edge, "table_height": 0.0, "friction": 0.1})

        block_positions, block_quaternions = {}, {}
        for name in names:
            pose = np.asarray(states["rigid_object"][name]["root_pose"][i], dtype=np.float64)
            block_positions[name] = [float(pose[0]), float(pose[1]), float(pose[2]) + z_offset]
            block_quaternions[name] = [float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6])]

        env_data = {
            "block_positions": block_positions,
            "block_quaternions": block_quaternions,
            "cube_scale": meta["edge"] / base_edge,
            "table_height_delta": meta["table_height"],
            "table_friction": meta["friction"],
            "profile_id": pid,
            # ToyBlocks poses are already the settled z in the table's frame, so
            # classic must not add table_height_delta on top of them.
            "positions_include_table_height": True,
            "_source": {
                "pipeline": "toyblocks",
                "env_path": str(args.env_path),
                "reset_state_index": i,
                "z_offset_applied": z_offset,
            },
        }
        # classic instance files are 1-based.
        out = args.out_dir / f"{i + 1}.json"
        if not args.dry_run:
            out.write_text(json.dumps(env_data, indent=2) + "\n")
        written.append((i, out.name, pid))

    print(f"\n{'would write' if args.dry_run else 'wrote'} {len(written)} instance file(s) "
          f"to {args.out_dir}")
    for i, fname, pid in written[:4]:
        print(f"  reset state {i:2d} -> {fname:8s} (profile {pid})")
    if len(written) > 4:
        print(f"  ... and {len(written) - 4} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
