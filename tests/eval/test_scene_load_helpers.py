"""CPU tests for the pure logic behind ``tests/eval/scene_load.sh``.

Two pieces, neither of which any other test reaches:

- ``pick_profile_episodes``: the profile grouping and selection that decide
  which episodes ``scene_load.sh`` verifies. Its "fewer profiles than requested"
  raise is the shell script's entire error path -- it is what turns "the PDDL
  randomization annotations never reached codegen" into a failing CI step rather
  than a run that quietly verifies the same scene twice.
- The comparison primitives in ``blocksuite.eval.scene_verification`` that the
  GPU rollout golden is built out of. Their branches (a shape change, a key-set
  change, a missing golden section) only fire when something has already gone
  wrong on a GPU box, so they are exactly the code most likely to be broken when
  it finally runs.

Runs without Isaac Lab: ``scene_verification`` needs numpy + torch, and
``pick_profile_episodes`` needs tyro; ``pyarrow`` is only pulled in by
``load_setups``, which is not called here.
"""

from __future__ import annotations

import pytest

from blocksuite.eval.scene_verification import (
    SceneMismatchReport,
    compare_array,
    compare_key_set,
    malformed_golden,
)
from tests.eval.pick_profile_episodes import (
    group_by_profile,
    pick_episodes,
    profile_key,
)


def episode(episode_index: int, cube_scale: float, table_height_delta: float) -> dict:
    """The three fields profile selection reads off a recorded episode."""
    return {
        "episode_index": episode_index,
        "cube_scale": cube_scale,
        "table_height_delta": table_height_delta,
    }


def report() -> SceneMismatchReport:
    return SceneMismatchReport(dataset_path="<test>", episode_index=0)


# --------------------------------------------------------------------------
# profile_key / group_by_profile
# --------------------------------------------------------------------------


def test_profile_key_rounds_to_the_grouping_precision():
    """Two episodes spawned at one size must not split on float-repr noise.

    ``cube_scale`` is rounded to micrometers and ``table_height_delta`` to 0.1
    mm, both far below any sampled difference. A split here would inflate the
    profile count and let ``--num-profiles 2`` "pass" while verifying the same
    scene twice.
    """
    a = profile_key(episode(0, 0.0400000001, 0.0200000001))
    b = profile_key(episode(1, 0.0399999999, 0.0199999999))
    assert a == b == (0.04, 0.02)


def test_profile_key_requires_both_randomized_fields():
    """A dataset predating the randomization columns must fail, not group to one."""
    with pytest.raises(ValueError, match="table_height_delta"):
        profile_key({"episode_index": 3, "cube_scale": 0.04})


def test_group_by_profile_orders_groups_and_episodes():
    """Selection must be stable for a given dataset, whatever order rows arrive in.

    ``scene_load.sh`` files each case by its position in this output, so an
    unstable order would silently change which scene a given rank means.
    """
    groups = group_by_profile(
        [
            episode(7, 0.05, 0.01),
            episode(2, 0.03, 0.02),
            episode(5, 0.05, 0.01),
            episode(1, 0.03, 0.02),
        ]
    )
    assert list(groups) == [(0.03, 0.02), (0.05, 0.01)]
    assert list(groups.values()) == [[1, 2], [5, 7]]


def test_group_by_profile_rejects_a_setup_without_an_episode_index():
    with pytest.raises(ValueError, match="episode_index"):
        group_by_profile([{"cube_scale": 0.04, "table_height_delta": 0.0}])


# --------------------------------------------------------------------------
# pick_episodes
# --------------------------------------------------------------------------


def test_pick_episodes_takes_the_lowest_index_of_each_profile():
    picked = pick_episodes(
        [episode(9, 0.05, 0.01), episode(4, 0.03, 0.02), episode(6, 0.03, 0.02)], 2
    )
    assert picked == [4, 9]


def test_pick_episodes_raises_when_too_few_profiles_exist():
    """The shell script's only error path.

    Returning fewer episodes instead would make ``scene_load.sh`` pass having
    verified one profile when it asked for two -- and the two real causes
    (randomization annotations that never reached codegen, or too few successful
    policies to span the spawn slots) both need fixing upstream.
    """
    setups = [episode(0, 0.04, 0.0), episode(1, 0.04, 0.0)]
    with pytest.raises(ValueError, match="only 1 distinct"):
        pick_episodes(setups, 2)


def test_pick_episodes_ignores_profiles_beyond_the_requested_count():
    setups = [episode(i, 0.03 + 0.01 * i, 0.0) for i in range(4)]
    assert pick_episodes(setups, 2) == [0, 1]


# --------------------------------------------------------------------------
# compare_array
# --------------------------------------------------------------------------


def test_compare_array_passes_within_tolerance():
    r = report()
    compare_array("joints", [[0.0, 1.0]], [[0.0, 1.0005]], 1e-2, r)
    assert r.ok
    (diff,) = r.diffs
    assert diff.passed
    assert diff.diff == pytest.approx(5e-4)
    # margin is what flags a field about to start flaking while it still passes.
    assert diff.margin == pytest.approx(0.05)


def test_compare_array_fails_outside_tolerance():
    r = report()
    compare_array("joints", [[0.0, 1.0]], [[0.0, 1.1]], 1e-2, r)
    assert not r.ok
    assert r.diffs[0].margin == pytest.approx(10.0)


def test_compare_array_reports_a_shape_change_as_its_own_failure():
    """An element-wise number over mismatched shapes would be a fabrication."""
    r = report()
    compare_array("joints", [[0.0, 1.0]], [[0.0, 1.0, 2.0]], 1e-2, r)
    (diff,) = r.diffs
    assert not diff.passed and diff.hard
    assert diff.dataset_value == [1, 3]
    assert diff.sim_value == [1, 2]
    assert diff.diff is None
    assert "shape changed" in diff.note


def test_compare_array_on_empty_arrays_passes_without_a_nan():
    """np.max over an empty array raises; the guard must yield 0.0, not NaN."""
    r = report()
    compare_array("joints", [], [], 1e-2, r)
    assert r.ok
    assert r.diffs[0].diff == 0.0


# --------------------------------------------------------------------------
# compare_key_set / malformed_golden
# --------------------------------------------------------------------------


def test_compare_key_set_is_silent_when_the_keys_match():
    """No diff at all, so a matching key set does not pad the report."""
    r = report()
    compare_key_set("blocks", {"b": 1, "a": 2}, {"a": 0, "b": 0}, r)
    assert r.diffs == []


def test_compare_key_set_hard_fails_on_a_differing_key_set():
    r = report()
    compare_key_set("blocks", {"a": 1, "c": 1}, {"a": 0, "b": 0}, r)
    (diff,) = r.diffs
    assert not diff.passed and diff.hard
    assert diff.dataset_value == ["a", "b"]
    assert diff.sim_value == ["a", "c"]


def test_malformed_golden_carries_the_caller_s_remedy():
    """A golden that cannot be compared against is a finding about the golden.

    It must read as one rather than as a KeyError with no context, and the
    re-record instruction has to come from the caller: only it knows how its
    goldens are produced.
    """
    r = report()
    malformed_golden("joint_position", r, remedy="run X and commit the result.")
    (diff,) = r.diffs
    assert not diff.passed and diff.hard
    assert diff.name == "golden_malformed[joint_position]"
    assert "run X and commit the result." in diff.note
