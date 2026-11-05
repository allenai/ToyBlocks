"""IO helpers for ``reset_states.pkl`` written by stage-1 task generation.

The pickle holds either the bare state dict (legacy layout) or a wrapper that
also carries the scene profile each reset state was sampled under::

    {
        "states": <state dict>,             # articulation / rigid_object trees
        "profile_ids": np.ndarray[int64],   # (num_reset_states,)
        "num_profiles": int,
        "profiles": list[dict],             # optional: values of each profile id
    }

Scene profiles (block sizes, table height, friction) are baked into the
generated env package at spawn time and indexed by env slot, so a plan may only
be executed in a slot whose baked profile matches the profile its reset state
was sampled under. Recording the assignment here keeps stage 2 and stage 3 from
re-deriving it as ``instance_id % num_profiles``, which silently breaks whenever
plans are consumed in an order other than the one they were generated in.

``profiles`` records what each profile id *is* (per-block sizes and friction,
the schema of a ``profiles.json`` entry), so a reset state can be matched to an
env by value with :func:`spawn_profiles_match` instead of trusting that profile
id ``k`` means the same thing in the env that loads it.

All consumers go through :func:`load_reset_states` so env packages generated
before the wrapper existed keep working unchanged.
"""

from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

STATES_KEY = "states"
PROFILE_IDS_KEY = "profile_ids"
NUM_PROFILES_KEY = "num_profiles"
PROFILES_KEY = "profiles"

# Profile fields an env slot fixes for every episode it runs: block sizes are
# baked at spawn and friction is written from the slot. ``table_height`` is not
# compared because a reset state carries its own table pose.
SPAWN_PROFILE_FRICTION_FIELDS = ("static_friction", "dynamic_friction", "restitution")
# Profile values are copied, not recomputed, so equal profiles agree to float
# precision; this only absorbs float32 <-> float64 round trips.
PROFILE_MATCH_ATOL = 1e-6


def pack_reset_states(
    states: dict[str, Any],
    profile_ids: np.ndarray,
    num_profiles: int,
    profiles: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the wrapper payload to pickle as ``reset_states.pkl``.

    Args:
        states: Reset state dict, leaves batched over the reset-state axis.
        profile_ids: Scene profile of each reset state, shape ``(num_reset_states,)``.
        num_profiles: Number of scene profiles baked into the env package.
        profiles: Values of every profile id referenced by ``profile_ids``, each
            a ``profiles.json`` entry (``profile_id``, ``block_sizes``, friction).

    Returns:
        Dict ready for ``pickle.dump``.
    """
    payload: dict[str, Any] = {
        STATES_KEY: states,
        PROFILE_IDS_KEY: np.asarray(profile_ids, dtype=np.int64),
        NUM_PROFILES_KEY: int(num_profiles),
    }
    if profiles is not None:
        payload[PROFILES_KEY] = [dict(p) for p in profiles]
    return payload


def unpack_reset_states(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray | None, int | None]:
    """Split an already-unpickled payload into states / profile ids / profile count.

    Legacy payloads (the bare state dict) yield ``(payload, None, None)``.

    Args:
        payload: Object read from ``reset_states.pkl``.

    Returns:
        ``(states, profile_ids, num_profiles)`` where the latter two are None for
        the legacy layout.
    """
    if isinstance(payload, dict) and STATES_KEY in payload:
        profile_ids = payload.get(PROFILE_IDS_KEY)
        if profile_ids is not None:
            profile_ids = np.asarray(profile_ids, dtype=np.int64)
        num_profiles = payload.get(NUM_PROFILES_KEY)
        return payload[STATES_KEY], profile_ids, num_profiles
    return payload, None, None


def unpack_reset_state_profiles(payload: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Recorded profile values of an unpickled payload, or None when not recorded."""
    if isinstance(payload, dict) and STATES_KEY in payload:
        profiles = payload.get(PROFILES_KEY)
        return None if profiles is None else [dict(p) for p in profiles]
    return None


def load_reset_states_with_profiles(
    path: str | Path,
) -> tuple[dict[str, Any], np.ndarray | None, int | None, list[dict[str, Any]] | None]:
    """Like :func:`load_reset_states`, plus the recorded profile values.

    Args:
        path: Path to the pickle.

    Returns:
        ``(states, profile_ids, num_profiles, profiles)``; ``profiles`` is None for
        packages that recorded only profile ids.
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)
    states, profile_ids, num_profiles = unpack_reset_states(payload)
    return states, profile_ids, num_profiles, unpack_reset_state_profiles(payload)


def spawn_profiles_match(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    atol: float = PROFILE_MATCH_ATOL,
) -> bool:
    """True when two scene profiles spawn the same blocks with the same friction.

    Compares per-block sizes (same block names, same 3-vector each) and the
    friction fields both profiles record; ``profile_id`` and ``table_height``
    are ignored (see :data:`SPAWN_PROFILE_FRICTION_FIELDS`).
    """
    sizes_a = a.get("block_sizes") or {}
    sizes_b = b.get("block_sizes") or {}
    if set(sizes_a) != set(sizes_b):
        return False
    for name, size in sizes_a.items():
        va = np.asarray(size, dtype=np.float64).reshape(-1)
        vb = np.asarray(sizes_b[name], dtype=np.float64).reshape(-1)
        if va.shape != vb.shape or not np.allclose(va, vb, rtol=0.0, atol=atol):
            return False
    for field in SPAWN_PROFILE_FRICTION_FIELDS:
        if field in a and field in b and abs(float(a[field]) - float(b[field])) > atol:
            return False
    return True


def reset_state_slot_compat(
    profile_ids: np.ndarray,
    recorded_profiles: Sequence[Mapping[str, Any]] | None,
    slot_profiles: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Which spawn slots may run each reset state.

    Reset states that record their profile values are compared value by value
    (:func:`spawn_profiles_match`) against every slot, so the match holds even
    when a profile id means something else in the loading env. Reset states that
    record only ids are taken to index ``slot_profiles`` directly (they were
    written by the same env package).

    Args:
        profile_ids: Recorded profile id of each reset state, ``(num_reset_states,)``.
        recorded_profiles: Recorded values per profile id, or None when not recorded.
        slot_profiles: Values of every spawn slot of the loading env, in slot order.

    Returns:
        Bool array ``(num_reset_states, num_slots)``.
    """
    ids = np.asarray(profile_ids, dtype=np.int64).reshape(-1)
    distinct = sorted(set(ids.tolist()))
    if recorded_profiles is not None:
        recorded = {int(p["profile_id"]): p for p in recorded_profiles}
        missing = [pid for pid in distinct if pid not in recorded]
        if missing:
            raise ValueError(
                f"reset states reference profile(s) {missing} but record values only for "
                f"{sorted(recorded)}"
            )
        rows = {
            pid: [spawn_profiles_match(recorded[pid], sp) for sp in slot_profiles]
            for pid in distinct
        }
    else:
        rows = {pid: [pid == s for s in range(len(slot_profiles))] for pid in distinct}
    if not ids.size:
        return np.zeros((0, len(slot_profiles)), dtype=bool)
    return np.array([rows[pid] for pid in ids.tolist()], dtype=bool).reshape(
        len(ids), len(slot_profiles)
    )


def sample_compatible_reset_states(compat: torch.Tensor, env_slots: torch.Tensor) -> torch.Tensor:
    """Draw one reset state per env, uniformly among those its slot may run.

    Args:
        compat: Bool ``(num_reset_states, num_slots)`` from :func:`reset_state_slot_compat`.
        env_slots: Spawn slot of each env being reset, ``(num_envs,)``.

    Returns:
        Reset-state index per env, on ``env_slots``' device.

    Raises:
        ValueError: when some slot has no compatible reset state.
    """
    picks = torch.empty(len(env_slots), device=env_slots.device, dtype=torch.long)
    for slot in env_slots.unique().tolist():
        rows = (env_slots == slot).nonzero().flatten()
        candidates = compat[:, slot].nonzero().flatten().to(env_slots.device)
        if len(candidates) == 0:
            raise ValueError(f"no reset state was sampled under the profile of spawn slot {slot}")
        picks[rows] = candidates[
            torch.randint(0, len(candidates), (len(rows),), device=env_slots.device)
        ]
    return picks


def describe_spawn_profile(profile: Mapping[str, Any]) -> str:
    """Short human-readable form of a profile for error messages."""
    sizes = profile.get("block_sizes") or {}
    distinct = sorted({tuple(round(float(v), 5) for v in s) for s in sizes.values()})
    friction = "/".join(
        f"{float(profile[f]):.3f}" for f in SPAWN_PROFILE_FRICTION_FIELDS if f in profile
    )
    return f"profile {profile.get('profile_id')}: block sizes {distinct}, friction {friction}"


def load_reset_states(
    path: str | Path,
) -> tuple[dict[str, Any], np.ndarray | None, int | None]:
    """Read ``reset_states.pkl`` in either layout.

    Args:
        path: Path to the pickle.

    Returns:
        ``(states, profile_ids, num_profiles)``; ``profile_ids`` and
        ``num_profiles`` are None for env packages generated before profile ids
        were recorded.
    """
    with open(path, "rb") as f:
        return unpack_reset_states(pickle.load(f))


def profile_id_for_instance(
    instance_id: int,
    profile_ids: np.ndarray | None,
    num_profiles: int | None,
) -> int:
    """Resolve the scene profile of one reset state / plan instance.

    Falls back to the legacy ``instance_id % num_profiles`` convention when the
    env package predates recorded profile ids, and to profile 0 when the profile
    count is unknown too.

    Args:
        instance_id: Reset-state index the plan was generated from.
        profile_ids: Recorded per-reset-state profiles, or None for legacy packages.
        num_profiles: Number of baked profiles, or None when unknown.

    Returns:
        Scene profile id.
    """
    if profile_ids is not None and 0 <= instance_id < len(profile_ids):
        return int(profile_ids[instance_id])
    if num_profiles:
        return instance_id % int(num_profiles)
    return 0
