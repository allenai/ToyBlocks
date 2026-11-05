"""Deterministic per-problem / per-string seed salts (md5-based) shared across DR concerns.
Moved verbatim from rerender.py. Pure, no globals."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict

def _stable_seed_salt(s: str) -> int:
    """Deterministic 32-bit salt from a string (stable across processes; unlike hash())."""
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


def _problem_seed_salt(job: Dict[str, Any]) -> int:
    """Per-problem salt so the same episode_index draws different samples across problems."""
    return _stable_seed_salt(Path(job["out_dir"]).name)


def episode_rng(base_seed: int, episode_index: int):
    """Generator whose draws depend ONLY on (base_seed, episode_index).

    SHARDING REQUIRES THIS. The legacy samplers advance ONE stream over sorted(episode_indices), so
    an episode's values depend on how many episodes precede it in the list: render episodes 0..511
    and episode 300 draws one thing; render only 256..511 and the same episode draws another. That
    is why a fragment currently has to load the whole episode prefix just to walk the rng forward.

    Deriving per episode removes the coupling entirely -- any shard reproduces any episode exactly,
    with no prefix and no shared state. Opt-in (indexed seeding) so previously rendered datasets
    stay reproducible on the legacy path.
    """
    import numpy as np
    return np.random.default_rng(np.random.SeedSequence([int(base_seed) & 0xFFFFFFFF,
                                                         int(episode_index) & 0xFFFFFFFF]))


def wave_rng(base_seed: int, wave_key: int, salt: int = 0):
    """Generator for ONE wave's shared draws (dome/skybox/exposure/sun/fill), keyed by wave.

    `wave_key` must be globally stable, so callers pass the wave's FIRST episode index rather than a
    loop counter: waves are aligned to num_envs boundaries in the global episode order, so the same
    wave carries the same key whether it was rendered by a full run or by a shard starting there.
    """
    import numpy as np
    return np.random.default_rng(np.random.SeedSequence([int(base_seed) & 0xFFFFFFFF,
                                                         int(wave_key) & 0xFFFFFFFF,
                                                         int(salt) & 0xFFFFFFFF]))


def episode_rng_hashed(base_seed: int, episode_index: int, concern: str = ""):
    """Generator keyed by a HASH of (base_seed, concern, episode_index) -- the `hash` seeding mode.

    `indexed` seeding is already order-independent, but it builds its SeedSequence out of two
    32-bit words, and the identity that reaches it is assembled by ADDITION: the per-problem salt
    is added to a base seed, and each concern offsets that sum by a small constant (+777 exposure,
    +999 skybox, +4242 fill, +12345 lights, +31337 scenarios). Nearby problems and neighbouring
    concerns therefore enter the mixer at nearby points, and the sum is truncated to 32 bits, so
    two (problem, concern) pairs can land on the same stream.

    Hashing the identity instead gives every (problem-salted seed, concern, episode) its own
    128 bits of entropy with full avalanche: change any part and the whole stream changes. It is
    still a pure function of the identity, so it keeps the property sharding depends on -- any
    shard reproduces any episode with no prefix and no shared state.
    """
    import numpy as np
    h = hashlib.md5(f"{int(base_seed)}|{concern}|{int(episode_index)}".encode("utf-8")).digest()
    words = [int.from_bytes(h[i:i + 4], "little") for i in range(0, 16, 4)]
    return np.random.default_rng(np.random.SeedSequence(words))


def rng_for_episode(mode, base_seed: int, episode_index: int, concern: str = ""):
    """The episode generator for a seeding mode. `mode` is "hash" | "indexed" | anything falsy.

    Callers historically passed a bool for `indexed`; True still means indexed, so the samplers
    keep working while the string modes are threaded through.
    """
    if mode == "hash":
        return episode_rng_hashed(base_seed, episode_index, concern)
    return episode_rng(base_seed, episode_index)
