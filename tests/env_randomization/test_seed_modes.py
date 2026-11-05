"""The three DR seeding modes, and the properties each one is chosen for.

`stream` advances one rng over the episode list, so an episode's draws depend on how many episodes
precede it -- the reason a fragment had to replay the whole prefix. `indexed` derives them from
(seed, episode_index). `hash` derives them from a hash of (seed, concern, episode_index), which is
the default: the identity that reaches `indexed` is assembled by ADDING a per-problem salt to a
32-bit seed and offsetting each concern by a small constant, so neighbouring problems and concerns
enter the mixer at neighbouring points.
"""
import numpy as np
import pytest

from blocksuite.domain_randomization.seeds import (episode_rng, episode_rng_hashed,
                                                   rng_for_episode, _stable_seed_salt)


def draws(rng, n=6):
    return rng.random(n)


def test_hash_is_deterministic_across_calls_and_processes():
    """Same identity, same stream -- md5, not hash(), so it survives a new interpreter."""
    assert (draws(episode_rng_hashed(1234, 7, "camera"))
            == draws(episode_rng_hashed(1234, 7, "camera"))).all()


def test_neighbouring_seeds_decorrelate():
    """The additive-salt failure mode: two problems one apart must not draw alike."""
    a, b = draws(episode_rng_hashed(1000, 3, "camera")), draws(episode_rng_hashed(1001, 3, "camera"))
    assert not np.isclose(a, b).any()


def test_concerns_decorrelate_at_the_same_seed_and_episode():
    """+777 exposure / +999 skybox / +4242 fill used to be the only thing separating concerns."""
    seen = [draws(episode_rng_hashed(500, 2, c)) for c in
            ("camera", "lights", "env_sphere", "materials:table", "")]
    for i in range(len(seen)):
        for j in range(i + 1, len(seen)):
            assert not np.isclose(seen[i], seen[j]).any(), (i, j)


def test_consecutive_episodes_decorrelate():
    a, b = draws(episode_rng_hashed(77, 10, "camera")), draws(episode_rng_hashed(77, 11, "camera"))
    assert not np.isclose(a, b).any()


def test_hash_is_prefix_free_like_indexed():
    """The property sharding depends on: an episode's draws never depend on its neighbours."""
    direct = draws(episode_rng_hashed(42, 300, "camera"))
    for _ in range(299):                                  # "render" everything before it first
        episode_rng_hashed(42, _, "camera").random(6)
    assert (draws(episode_rng_hashed(42, 300, "camera")) == direct).all()


def test_hash_differs_from_indexed():
    """Different values, hence the never-mix-modes-within-a-dataset rule."""
    assert not np.isclose(draws(episode_rng_hashed(9, 4, "camera")),
                          draws(episode_rng(9, 4))).any()


@pytest.mark.parametrize("mode,expected_same_as", [("hash", "hash"), ("indexed", "indexed"),
                                                   (True, "indexed"), (False, "indexed")])
def test_dispatcher_routes_each_mode(mode, expected_same_as):
    """rng_for_episode is what the samplers call; only "hash" may take the hashed path.

    A bool still means indexed, because that is what the samplers passed before the modes existed.
    """
    got = draws(rng_for_episode(mode, 55, 6, "camera"))
    want = draws(episode_rng_hashed(55, 6, "camera") if expected_same_as == "hash"
                 else episode_rng(55, 6))
    assert (got == want).all()


def test_problem_salt_is_stable_and_distinct():
    a, b = _stable_seed_salt("problem_declarative_1_x"), _stable_seed_salt("problem_declarative_2_y")
    assert a == _stable_seed_salt("problem_declarative_1_x") and a != b


def test_hash_draws_are_uniform_enough_to_randomize_with():
    """A seeding change is worthless if the stream it produces is skewed."""
    xs = np.concatenate([episode_rng_hashed(3, e, "camera").random(64) for e in range(64)])
    assert 0.47 < xs.mean() < 0.53, xs.mean()
    assert (np.histogram(xs, bins=10, range=(0, 1))[0] > len(xs) / 20).all()
