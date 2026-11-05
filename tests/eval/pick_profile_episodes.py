"""Pick one episode per distinct randomization profile from a datagen dataset.

``tests/eval/scene_load.sh`` uses this to spread its ``verify_dataset_loading.py``
cases across the ``(cube_scale, table_height_delta)`` profiles that codegen
sampled from the PDDL's ``;block_size_range`` / ``;table_height_range``
annotations, rather than verifying the same scene N times.

Prints one ``episode_index`` per line (lowest index of each group, groups ordered
by profile so the selection is stable for a given dataset), which is what makes
the output safe to read straight into a shell array.

Raises when fewer than ``--num-profiles`` distinct profiles were recorded. That
is the intended signal for two real failures, and neither should be papered over
by silently returning fewer episodes: the PDDL randomization annotations did not
take effect, or too few policies succeeded to span the spawn slots (only
successful episodes get recorded, so raise ``--max_policies`` in
``tests/tamp/datagen.sh`` rather than lowering ``--profiles``).

    python tests/eval/pick_profile_episodes.py \
        --dataset-path datasets/pnp_test/blocks_success --num-profiles 2
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import tyro

from toyblocks.utils.episode_dataset import CUBE_SCALE_DECIMALS, load_setups

# Table deltas are sampled in metres over a ~4 cm range, so a 0.1 mm grouping key
# separates every distinct sampled value without splitting one on float noise.
TABLE_HEIGHT_DECIMALS = 4


@dataclass
class Args:
    dataset_path: str
    """LeRobot dataset root (or directory of roots) to read episodes from."""
    num_profiles: int = 2
    """How many distinct profiles to select an episode for. Raises if fewer exist."""
    verbose: bool = False
    """Print the full profile -> episode grouping to stderr."""


def profile_key(setup: dict) -> tuple[float, float]:
    """Group key for one episode: its spawned block size and table offset.

    These are the two randomized quantities the eval side must reproduce from the
    dataset -- ``cube_scale`` decides which spawn slot the episode can run in, and
    ``table_height_delta`` decides the table baseline every recorded block Z is
    relative to. Friction is randomized too but is not part of the key: it is not
    observable in a static scene comparison.
    """
    cube_scale = setup.get("cube_scale")
    table_height_delta = setup.get("table_height_delta")
    if cube_scale is None or table_height_delta is None:
        raise ValueError(
            f"episode {setup.get('episode_index')} records "
            f"cube_scale={cube_scale!r} table_height_delta={table_height_delta!r}; "
            "both are required to identify a randomization profile. The dataset "
            "predates the randomization columns, or datagen ran without the "
            ";block_size_range / ;table_height_range PDDL annotations."
        )
    return (
        round(float(cube_scale), CUBE_SCALE_DECIMALS),
        round(float(table_height_delta), TABLE_HEIGHT_DECIMALS),
    )


def group_by_profile(setups: list[dict]) -> dict[tuple[float, float], list[int]]:
    """Map each distinct profile to the sorted episode indices recorded for it."""
    groups: dict[tuple[float, float], list[int]] = {}
    for setup in setups:
        episode_index = setup.get("episode_index")
        if episode_index is None:
            raise ValueError(f"setup {setup.get('episode_id')!r} has no episode_index")
        groups.setdefault(profile_key(setup), []).append(int(episode_index))
    return {key: sorted(indices) for key, indices in sorted(groups.items())}


def pick_episodes(setups: list[dict], num_profiles: int) -> list[int]:
    """Lowest episode index of each of the first ``num_profiles`` distinct profiles."""
    groups = group_by_profile(setups)
    if len(groups) < num_profiles:
        raise ValueError(
            f"dataset has only {len(groups)} distinct (cube_scale, table_height_delta) "
            f"profile(s) across {len(setups)} episode(s) but {num_profiles} were "
            f"requested: {list(groups)}. Either the PDDL's ;block_size_range / "
            ";table_height_range annotations did not reach codegen (check "
            "non_sim_state_randomizations.json in the generated task package), or too "
            "few policies succeeded to span the spawn slots -- raise --max_policies in "
            "tests/tamp/datagen.sh rather than lowering --profiles."
        )
    return [indices[0] for indices in list(groups.values())[:num_profiles]]


def main(args: Args) -> None:
    setups = load_setups(args.dataset_path)
    if not setups:
        raise ValueError(f"no episodes found at {args.dataset_path}")
    if args.verbose:
        for key, indices in group_by_profile(setups).items():
            print(
                f"cube_scale={key[0]} table_height_delta={key[1]:+.4f}: {indices}",
                file=sys.stderr,
            )
    for episode_index in pick_episodes(setups, args.num_profiles):
        print(episode_index)


if __name__ == "__main__":
    main(tyro.cli(Args))
