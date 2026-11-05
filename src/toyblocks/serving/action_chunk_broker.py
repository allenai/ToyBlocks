"""Action chunk broker — dispenses one action at a time from a chunked policy.

This is a self-contained replacement for ``openpi_client.ActionChunkBroker``
so the ToyBlocks evaluation stack does not require the ``openpi`` package.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Dict

import numpy as np

from toyblocks.serving.base_policy import BasePolicy, walk_policy_chain



def _reset_envs_on(policy: Any, env_ids: Iterable[int]) -> None:
    """Forward a per-env reset to *policy*, or do nothing if it has no notion of one.

    Deliberately does **not** fall back to ``policy.reset()``: a global reset is
    exactly what a per-env reset exists to avoid, and calling it here would put
    back the cross-env coupling (one env's episode boundary clearing every other
    env's chunk).
    """
    reset_envs = getattr(policy, "reset_envs", None)
    if reset_envs is not None:
        reset_envs(env_ids)


class ActionChunkBroker(BasePolicy):
    """Wraps a chunked policy and returns one action per ``infer`` call.

    By default each env keeps its own phase: whenever any env's chunk runs out
    (or it resets), the whole batch is re-inferred and only that env takes its
    slice. Once episodes end at different steps, that is an inference on most
    steps. With ``sync=True`` every env shares one chunk boundary instead: the
    batch is inferred once per period (``action_horizon``, or the chunk length
    if shorter) and every env takes its slice. An env reset between boundaries
    has no chunk until the next one; it is listed in :attr:`waiting_envs`, is
    handed a placeholder row, and the caller must hold it in place until then.
    """

    def __init__(self, policy: BasePolicy, action_horizon: int, sync: bool = False) -> None:
        self._policy = policy
        self._action_horizon = action_horizon
        self._sync = sync
        # sync only: dispensed steps since the last boundary, and the boundary period.
        self._steps_since_boundary = 0
        self._period = action_horizon
        # sync only: zeros shaped like one env's row, dispensed to waiting envs,
        # and the envs the last dispense handed it to.
        self._placeholder_row: Dict[str, Any] | None = None
        self._waiting: list[int] = []
        # Per-env state, sized lazily from the first inference response.
        self._chunks: list[Dict[str, Any] | None] = []
        self._cur_step: list[int] = []
        self._shared: Dict[str, Any] = {}
        self._chunked_keys: list[str] = []
        # Last row dispensed per env, handed back to held envs as a placeholder.
        self._last_rows: list[Dict[str, Any] | None] = []
        self._held: set[int] = set()

    def hold_envs(self, env_ids: Iterable[int]) -> None:
        """Freeze ``env_ids`` for the following :meth:`infer` calls, replacing any previous hold.

        A held env neither advances its chunk nor triggers inference when its
        chunk is exhausted; it is handed its last dispensed row as a placeholder
        for the caller to override, and resumes at the chunk index it paused at
        once it is no longer held.
        """
        self._held = set(env_ids)

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """ 
        Vectorized policy inference. N envs have their own chunk buffer and phase into that chunk.
        Step through each env and advance the chunk buffer. Chunks that are exhausted are set to None; those chunks' results are processed using "absorb".  
        """
        if not self._chunks:
            self._absorb(self._policy.infer(obs), None)
        elif self._sync:
            if self._at_sync_boundary():
                self._absorb(self._policy.infer(obs), range(len(self._chunks)))
        else:
            stale = [
                i for i, chunk in enumerate(self._chunks)
                if chunk is None and i not in self._held
            ]
            if stale:
                # One batched call serves every env; only the envs that actually
                # need a chunk take rows from it, so the rest keep their phase.
                self._absorb(self._policy.infer(obs), stale)
        return self._dispense()

    def _absorb(self, results: Dict[str, Any], env_ids: Iterable[int] | None) -> None:
        """Store the per-env slices of *results* for ``env_ids`` (``None`` = all)."""
        per_env, shared, num_envs = _split_envs(results)
        self._shared = shared
        self._chunked_keys = list(per_env[0])
        if not self._chunks:
            self._chunks = [None] * num_envs
            self._cur_step = [0] * num_envs
            self._last_rows = [None] * num_envs
            env_ids = range(num_envs)
        elif num_envs != len(self._chunks):
            raise ValueError(
                f"policy returned a chunk for {num_envs} envs but this broker is "
                f"tracking {len(self._chunks)}; call reset() when num_envs changes."
            )
        for i in env_ids:
            self._chunks[i] = per_env[i]
            self._cur_step[i] = 0
        if self._sync:
            self._steps_since_boundary = 0
            self._period = min(self._action_horizon, self._chunk_length(per_env[0]))
            self._placeholder_row = {
                k: np.zeros_like(per_env[0][k][0]) for k in self._chunked_keys
            }

    def _at_sync_boundary(self) -> bool:
        """Whether the next :meth:`infer` re-infers the whole batch (``sync`` only).

        At the period, or earlier once no env has a chunk left to execute (every
        env reset since the last boundary), since waiting would only idle them all.
        """
        return self._steps_since_boundary >= self._period or all(
            chunk is None for chunk in self._chunks
        )

    @property
    def waiting_envs(self) -> list[int]:
        """Envs the last :meth:`infer` handed a placeholder row (``sync`` only; else empty).

        They were reset since the last boundary and have no chunk until the
        next one, so the caller holds them in place instead of stepping the
        placeholder.
        """
        return list(self._waiting)

    def _dispense(self) -> Dict[str, Any]:
        """Take each env's current step and advance (retiring exhausted chunks).

        Held envs get their last dispensed row and keep their phase.
        """
        rows: list[Dict[str, Any]] = []
        self._waiting = []
        for i, chunk in enumerate(self._chunks):
            last_row = self._last_rows[i]
            if i in self._held and last_row is not None:
                rows.append(last_row)
                continue
            if chunk is None and self._sync:
                rows.append(self._placeholder_row)
                self._waiting.append(i)
                continue
            if chunk is None:
                raise RuntimeError(
                    f"env {i} is held with no chunk and nothing dispensed yet; hold "
                    "only envs that have been served at least once since their reset."
                )
            row = {k: chunk[k][self._cur_step[i]] for k in self._chunked_keys}
            self._last_rows[i] = row
            rows.append(row)
            self._cur_step[i] += 1
            # Retire at the horizon or at the end of the chunk, whichever comes
            # first, so action_horizon > chunk size cannot IndexError.
            chunk_len = self._chunk_length(chunk)
            if self._cur_step[i] >= self._action_horizon or self._cur_step[i] >= chunk_len:
                self._chunks[i] = None
        self._steps_since_boundary += 1
        out: Dict[str, Any] = {
            k: np.stack([row[k] for row in rows], axis=0) for k in self._chunked_keys
        }
        out.update(self._shared)
        return out

    @staticmethod
    def _chunk_length(d: Dict[str, Any]) -> int:
        """Return the time-axis length of the first ndarray leaf."""
        for v in d.values():
            if isinstance(v, np.ndarray) and v.ndim >= 1:
                return v.shape[0]
            if isinstance(v, dict):
                n = ActionChunkBroker._chunk_length(v)
                if n > 0:
                    return n
        return 0

    @property
    def needs_inference(self) -> bool:
        """Whether the next :meth:`infer` will hit the wrapped policy.

        The chunk boundaries are the only steps whose observation reaches the
        server, so a diagnostic that wants to mark them has to ask rather than
        recompute ``step % action_horizon``: any :meth:`reset` (which the eval
        fires whenever *any* env finishes an episode) re-phases them.

        Mirrors the condition :meth:`infer` uses to decide whether to call the
        wrapped policy: no chunks yet (initial call or post-reset), or any
        per-env chunk of an env that is not held was retired to ``None`` by
        :meth:`_dispense` / :meth:`reset_envs`.
        """
        if self._sync:
            return (not self._chunks) or self._at_sync_boundary()
        return (not self._chunks) or any(
            chunk is None and i not in self._held for i, chunk in enumerate(self._chunks)
        )

    def needs_inference_for_env(self, env_index: int) -> bool:
        """Whether the next :meth:`infer` hands *env_index* the first row of a new chunk.

        Same contract as :meth:`PerEnvActionChunkBroker.needs_inference_for_env`:
        under ``sync`` every env takes a new chunk at the shared boundary, else
        only the envs whose chunk ran out (or were reset) and are not held.
        """
        if self._sync or not self._chunks:
            return self.needs_inference
        return self._chunks[env_index] is None and env_index not in self._held

    def reset(self) -> None:
        self._policy.reset()
        self._chunks = []
        self._cur_step = []
        self._shared = {}
        self._chunked_keys = []
        self._last_rows = []
        self._held = set()
        self._steps_since_boundary = 0
        self._period = self._action_horizon
        self._placeholder_row = None
        self._waiting = []

    def reset_envs(self, env_ids: Iterable[int]) -> None:
        """Drop the cached chunk for ``env_ids`` only, leaving other envs' phases alone.

        Call this at an episode boundary so the envs that just reset re-plan
        against their new scene on the next ``infer`` while the envs still
        mid-episode continue executing the chunk they are partway through.
        """
        env_ids = list(env_ids)
        _reset_envs_on(self._policy, env_ids)
        for i in env_ids:
            self._held.discard(i)
            if 0 <= i < len(self._chunks):
                self._chunks[i] = None
                self._cur_step[i] = 0
                self._last_rows[i] = None


def _split_envs(results: Dict[str, Any]) -> tuple[list[Dict[str, Any]], Dict[str, Any], int]:
    """Split a ``(T, B, ...)`` response into per-env ``(T, ...)`` chunks.

    ``B`` is read off the leaf with the most axes, which is the action chunk
    itself: a batched response always carries ``(T, B, action_dim)`` and any
    per-env aux is wider still, while non-chunked leaves (openpi's ``(B, D)``
    ``state``) have fewer. Leaves whose axis 1 does not match ``B`` are returned
    as shared pass-through rather than being sliced on a meaningless axis.
    """
    num_envs: int | None = None
    best_ndim = -1
    for v in results.values():
        if isinstance(v, np.ndarray) and v.ndim >= 2 and v.ndim > best_ndim:
            best_ndim = v.ndim
            num_envs = v.shape[1]
    if num_envs is None:
        raise ValueError(
            "ActionChunkBroker expects a batched (T, B, ...) response but none of "
            f"{sorted(results)} has 2+ axes. An unbatched (T, D) server needs "
            "PolicyClientAdapter(batched=False) + PerEnvActionChunkBroker "
            "(--unbatched)."
        )

    per_env: list[Dict[str, Any]] = [{} for _ in range(num_envs)]
    shared: Dict[str, Any] = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray) and v.ndim >= 2 and v.shape[1] == num_envs:
            for i in range(num_envs):
                per_env[i][k] = v[:, i, ...]
        else:
            shared[k] = v
    if not per_env[0]:
        raise ValueError(
            f"no response key in {sorted(results)} carries a (T, B={num_envs}, ...) "
            "chunk axis, so there is nothing to dispense per step."
        )
    return per_env, shared, num_envs


def _slice_step(d: Dict[str, Any], step: int) -> Dict[str, Any]:
    """Index into the chunk dimension of every ndarray leaf in *d*."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = v[step, ...]
        elif isinstance(v, dict):
            out[k] = _slice_step(v, step)
        else:
            out[k] = v
    return out


class PerEnvActionChunkBroker(BasePolicy):
    """Like :class:`ActionChunkBroker`, but keeps separate chunk buffers per env index.

    :class:`PolicyClientAdapter` with ``batched=False`` calls ``infer`` once per
    parallel environment each step. A single shared chunk buffer would hand env
    *i* the *i*-th row of env 0's chunk; this wrapper keys state by env index so
    each environment advances its own horizon independently.
    """

    def __init__(self, policy: BasePolicy, action_horizon: int) -> None:
        self._policy = policy
        self._action_horizon = action_horizon
        self._last_results: dict[int, Dict[str, Any] | None] = {}
        self._cur_step: dict[int, int] = {}
        self._last_dispensed: dict[int, Dict[str, Any]] = {}
        self._held: set[int] = set()

    def hold_envs(self, env_ids: Iterable[int]) -> None:
        """Same contract as :meth:`ActionChunkBroker.hold_envs`, via :meth:`infer_for_env`."""
        self._held = set(env_ids)

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        raise RuntimeError(
            "PerEnvActionChunkBroker is only used with PolicyClientAdapter "
            "unbatched mode, which calls infer_for_env(env_index, obs)."
        )

    def infer_for_env(self, env_index: int, obs: Dict[str, Any]) -> Dict[str, Any]:
        if env_index in self._held and env_index in self._last_dispensed:
            return self._last_dispensed[env_index]
        if self._last_results.get(env_index) is None:
            self._last_results[env_index] = self._policy.infer(obs)
            self._cur_step[env_index] = 0

        results = _slice_step(self._last_results[env_index], self._cur_step[env_index])
        self._cur_step[env_index] += 1

        chunk_len = ActionChunkBroker._chunk_length(self._last_results[env_index])
        if self._cur_step[env_index] >= self._action_horizon or self._cur_step[env_index] >= chunk_len:
            self._last_results[env_index] = None

        self._last_dispensed[env_index] = results
        return results

    def needs_inference_for_env(self, env_index: int) -> bool:
        """Whether the next :meth:`infer_for_env` for *env_index* hits the policy.

        Per-env counterpart of :attr:`ActionChunkBroker.needs_inference`; each env
        advances its own horizon here, so the boundaries differ between envs.
        """
        if env_index in self._held and env_index in self._last_dispensed:
            return False
        return self._last_results.get(env_index) is None

    def reset(self) -> None:
        if hasattr(self._policy, "reset"):
            self._policy.reset()
        self._last_results.clear()
        self._cur_step.clear()
        self._last_dispensed.clear()
        self._held.clear()

    def reset_envs(self, env_ids: Iterable[int]) -> None:
        """Drop the cached chunk for ``env_ids`` only.

        ``reset()`` clears every env's buffer, so calling it at one env's episode
        boundary truncates the chunks all the other envs are partway through.
        """
        env_ids = list(env_ids)
        _reset_envs_on(self._policy, env_ids)
        for i in env_ids:
            self._last_results.pop(i, None)
            self._cur_step.pop(i, None)
            self._last_dispensed.pop(i, None)
            self._held.discard(i)


def find_action_chunk_broker(policy: Any) -> ActionChunkBroker | PerEnvActionChunkBroker | None:
    """The chunk broker inside a wrapped policy, if any.

    A diagnostic that wants to know whether *this* step is an inference has to
    reach through the client wrappers (see :func:`walk_policy_chain`), mirroring
    ``find_obs_transform``.
    """
    for node in walk_policy_chain(policy):
        if isinstance(node, (ActionChunkBroker, PerEnvActionChunkBroker)):
            return node
    return None
