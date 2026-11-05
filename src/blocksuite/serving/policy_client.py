"""Adapter that bridges batched torch-tensor env observations to a per-env
numpy-dict policy server (e.g. a WebsocketClientPolicy).

Usage::

    from blocksuite.serving import WebsocketClientPolicy
    from blocksuite.serving.policy_client import PolicyClientAdapter

    ws_policy = WebsocketClientPolicy(host="localhost", port=8000)
    adapter = PolicyClientAdapter(ws_policy, action_key="actions", device="cuda:0")
    action_tensor = adapter.act(obs_from_env)
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from blocksuite.serving.action_chunk_broker import (
    ActionChunkBroker,
    PerEnvActionChunkBroker,
    find_action_chunk_broker,
)
from blocksuite.utils.common_utils import to_numpy


def _cuda_sync_if_available() -> None:
    """``torch.cuda.synchronize()`` when a CUDA device is in use."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class PolicyClientAdapter:
    """Wraps any policy that implements ``infer(obs: dict) -> dict`` so it can
    be driven by a batched Isaac Lab vectorised environment.

    Two inference modes are supported:

    - **batched** (default): the entire ``(num_envs, ...)`` observation is
      converted to numpy and forwarded in a single ``policy.infer`` call.
      The policy server is expected to return actions with a matching leading
      batch dimension.  This is the fastest path (one network round-trip per
      step).
    - **unbatched**: the observation is split per environment and
      ``policy.infer`` is called once for each env.  Use this when the policy
      server only accepts single (unbatched) observations (e.g. the
      single-agent Molmo2 HL + LL server).
    """

    def __init__(
        self,
        policy: Any,
        action_key: str = "actions",
        device: str | torch.device = "cuda:0",
        batched: bool = True,
    ) -> None:
        """
        Args:
            policy: Object with ``infer(obs: Dict) -> Dict`` and optional
                ``reset()`` methods (matches the ``BasePolicy`` ABC).
            action_key: Key used to look up the action array in the dict
                returned by ``policy.infer``.
            device: Torch device to place the output action tensor on.
            batched: If ``True`` (default), send the full batched observation
                in a single ``infer`` call.  If ``False``, unbatch and call
                ``infer`` once per environment.
        """
        self.policy = policy
        self.action_key = action_key
        self.device = torch.device(device)
        self.batched = batched
        # Auxiliary numpy fields from the most recent ``act`` call (every key
        # in the policy response other than ``action_key``). Useful when the
        # policy returns side-channel info (e.g. ``action_pd_mode``) that the
        # caller wants to read alongside the action tensor.
        self.last_aux: dict[str, np.ndarray] = {}
        # None when the last act() copied no observation (the chunk broker
        # served that step from its cached chunk).
        self.last_obs_to_cpu_ms: float | None = 0.0
        self.last_action_to_gpu_ms: float = 0.0
        self._chunk_broker: ActionChunkBroker | PerEnvActionChunkBroker | None = (
            find_action_chunk_broker(policy)
        )
        self._next_episode_seed: int | None = None
        self._input_trace: dict[str, str | int] | None = None
        self._pending_first_batch_dir: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def act(self, obs: torch.Tensor | dict) -> torch.Tensor:
        """Run inference on a *batched* observation and return a batched action.

        Args:
            obs: Either a flat tensor of shape ``(num_envs, obs_dim)`` or a
                (possibly nested) dict whose leaf values are tensors with a
                leading ``num_envs`` dimension.

        Returns:
            Tensor of shape ``(num_envs, action_dim)`` on ``self.device``.
        """
        if self.batched:
            return self._act_batched(obs)
        return self._act_unbatched(obs)

    @property
    def transfer_timings(self) -> dict[str, float | None]:
        return {
            "obs_to_cpu_ms": self.last_obs_to_cpu_ms,
            "action_to_gpu_ms": self.last_action_to_gpu_ms,
        }

    def set_episode_seed(self, seed: int) -> None:
        """Schedule a per-episode re-seed on the policy server.

        The seed is injected as ``__episode_seed__`` into the next observation
        dict sent to the server, which pops it and re-seeds before inference.
        """
        self._next_episode_seed = seed

    def set_policy_input_trace(
        self,
        out_dir: str | None,
        env_idx: int = 0,
        hl_dir: str | None = None,
    ) -> None:
        """Ask the server to dump what it hands to the model, until cleared.

        Injected as ``__ll_input_trace__`` on every subsequent request (unlike
        the one-shot seed): only the calls that survive the action-chunk broker
        reach the server, and that set is exactly the handovers worth dumping.
        ``out_dir`` is the LL panel directory; ``hl_dir`` (when set) is the
        sibling Molmo2 HL panel directory. Both are written by the *server*
        process, so they are only readable by the caller when the two share a
        filesystem. ``None`` clears the request.
        """
        if out_dir is None:
            self._input_trace = None
        else:
            self._input_trace = {"dir": str(out_dir), "env_idx": int(env_idx)}
            if hl_dir is not None:
                self._input_trace["hl_dir"] = str(hl_dir)

    def reset(self) -> None:
        """Forward a reset signal to the underlying policy (clears internal
        state such as action-chunk buffers)."""
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        self.last_aux = {}

    def reset_envs(self, env_ids: Any) -> None:
        """Forward a reset for ``env_ids`` only, leaving other envs' state intact.

        Use this at an episode boundary in a vectorised eval: :meth:`reset` drops
        the action-chunk buffer for *every* env, so one env finishing would
        truncate the chunks the others are partway through. Policies with no
        per-env reset fall back to the global one, matching the old behaviour.
        """
        reset_envs = getattr(self.policy, "reset_envs", None)
        if reset_envs is None:
            self.reset()
            return
        reset_envs(env_ids)

    # ------------------------------------------------------------------
    # Inference strategies
    # ------------------------------------------------------------------

    def _act_batched(self, obs: torch.Tensor | dict) -> torch.Tensor:
        """Single ``infer`` call with the full batched observation."""
        broker = self._chunk_broker
        if isinstance(broker, ActionChunkBroker) and not broker.needs_inference:
            # Served from the cached chunk: the broker reads no observation, so
            # skip the device->host copy of every camera. A pending episode seed
            # or first-batch dump rides on the next request that reaches the server.
            self.last_obs_to_cpu_ms = None
            response = self.policy.infer({})
            self._capture_aux(response)
            return self._action_to_device(response[self.action_key])

        _cuda_sync_if_available()
        t0 = time.perf_counter()
        obs_numpy = to_numpy(obs)
        _cuda_sync_if_available()
        self.last_obs_to_cpu_ms = (time.perf_counter() - t0) * 1000.0

        if self._next_episode_seed is not None and isinstance(obs_numpy, dict):
            obs_numpy["__episode_seed__"] = np.array(self._next_episode_seed, dtype=np.int64)
            self._next_episode_seed = None

        if self._input_trace is not None and isinstance(obs_numpy, dict):
            obs_numpy["__ll_input_trace__"] = dict(self._input_trace)
            
        if self._pending_first_batch_dir is not None and isinstance(obs_numpy, dict):
            obs_numpy["__save_first_batch_dir__"] = np.frombuffer(
                self._pending_first_batch_dir.encode("utf-8"), dtype=np.uint8
            ).copy()
            self._pending_first_batch_dir = None

        response = self.policy.infer(obs_numpy)
        self._capture_aux(response)
        return self._action_to_device(response[self.action_key])

    def _action_to_device(self, action: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)
        t1 = time.perf_counter()
        result = action.to(self.device)
        _cuda_sync_if_available()
        self.last_action_to_gpu_ms = (time.perf_counter() - t1) * 1000.0
        return result

    def _act_unbatched(self, obs: torch.Tensor | dict) -> torch.Tensor:
        """One ``infer`` call per environment, then stack results."""
        broker = self._chunk_broker
        num_envs = obs.shape[0] if isinstance(obs, torch.Tensor) else _infer_batch_size(obs)
        if isinstance(broker, PerEnvActionChunkBroker):
            # Only envs whose chunk ran out reach the server; the rest are served
            # from their cached chunk and get no copy of their observation.
            needed = [i for i in range(num_envs) if broker.needs_inference_for_env(i)]
        else:
            needed = list(range(num_envs))
        per_env_obs: list[dict[str, Any] | np.ndarray] = [{} for _ in range(num_envs)]
        if needed:
            _cuda_sync_if_available()
            t0 = time.perf_counter()
            if len(needed) == num_envs:
                per_env_obs = self._unbatch_obs(obs)
            else:
                for i, single_obs in zip(needed, self._unbatch_obs(_select_envs(obs, needed))):
                    per_env_obs[i] = single_obs
            _cuda_sync_if_available()
            self.last_obs_to_cpu_ms = (time.perf_counter() - t0) * 1000.0
        else:
            self.last_obs_to_cpu_ms = None

        if self._next_episode_seed is not None and needed:
            for single_obs in per_env_obs:
                if isinstance(single_obs, dict):
                    single_obs["__episode_seed__"] = np.array(
                        self._next_episode_seed, dtype=np.int64
                    )
            self._next_episode_seed = None

        if self._input_trace is not None:
            # One request per env here, each already B == 1, so the followed env
            # is selected client-side and the server slices batch element 0.
            traced = int(self._input_trace["env_idx"])
            if traced < len(per_env_obs) and isinstance(per_env_obs[traced], dict):
                payload: dict[str, str | int] = {
                    "dir": self._input_trace["dir"],
                    "env_idx": 0,
                }
                if "hl_dir" in self._input_trace:
                    payload["hl_dir"] = self._input_trace["hl_dir"]
                per_env_obs[traced]["__ll_input_trace__"] = payload
                
        if self._pending_first_batch_dir is not None and needed:
            first_obs = per_env_obs[needed[0]]
            if isinstance(first_obs, dict):
                first_obs["__save_first_batch_dir__"] = np.frombuffer(
                    self._pending_first_batch_dir.encode("utf-8"), dtype=np.uint8
                ).copy()
            self._pending_first_batch_dir = None

        actions = []
        per_env_aux: list[dict[str, np.ndarray]] = []
        infer_for_env = getattr(self.policy, "infer_for_env", None)
        for i, single_obs in enumerate(per_env_obs):
            if infer_for_env is not None:
                response = infer_for_env(i, single_obs)
            else:
                response = self.policy.infer(single_obs)
            per_env_aux.append(
                {
                    k: v
                    for k, v in response.items()
                    if k != self.action_key and isinstance(v, np.ndarray)
                }
            )
            action = response[self.action_key]
            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action)
            actions.append(action)
        # Stack per-env aux ndarrays back into a leading-batch layout so
        # callers see the same shape as in the batched path.
        self.last_aux = {}
        if per_env_aux:
            common_keys = set(per_env_aux[0].keys())
            for aux in per_env_aux[1:]:
                common_keys &= set(aux.keys())
            for k in common_keys:
                self.last_aux[k] = np.stack([aux[k] for aux in per_env_aux], axis=0)

        t1 = time.perf_counter()
        result = torch.stack(actions, dim=0).to(self.device)
        _cuda_sync_if_available()
        self.last_action_to_gpu_ms = (time.perf_counter() - t1) * 1000.0
        return result

    def _capture_aux(self, response: dict[str, Any]) -> None:
        """Store every ndarray response key other than ``action_key`` on
        ``self.last_aux`` so callers can read side-channel outputs."""
        self.last_aux = {
            k: v for k, v in response.items() if k != self.action_key and isinstance(v, np.ndarray)
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _unbatch_obs(obs: torch.Tensor | dict) -> list[dict[str, Any] | np.ndarray]:
        """Split a batched observation into a list of per-env numpy observations."""
        if isinstance(obs, torch.Tensor):
            np_obs = to_numpy(obs)
            return [np_obs[i] for i in range(np_obs.shape[0])]

        num_envs = _infer_batch_size(obs)
        out: list[dict] = [{} for _ in range(num_envs)]
        _split_dict(obs, out)
        return out


def _select_envs(obs: torch.Tensor | dict, env_ids: list[int]) -> torch.Tensor | dict:
    """``obs`` restricted to the rows ``env_ids`` (leaf tensors / arrays indexed on dim 0)."""
    if isinstance(obs, (torch.Tensor, np.ndarray)):
        return obs[env_ids]
    out: dict[str, Any] = {}
    for key, value in obs.items():
        if isinstance(value, (dict, torch.Tensor, np.ndarray)):
            out[key] = _select_envs(value, env_ids)
        else:
            out[key] = value
    return out


def _infer_batch_size(d: dict) -> int:
    """Walk a nested dict to find the first tensor/array and return its batch dim."""
    for v in d.values():
        if isinstance(v, (torch.Tensor, np.ndarray)):
            return v.shape[0]
        if isinstance(v, dict):
            return _infer_batch_size(v)
    raise ValueError("Cannot infer batch size: no tensors found in observation dict")


def _split_dict(src: dict, dst_list: list[dict]) -> None:
    """Recursively index into each leaf tensor at position *i* and store the
    numpy result in ``dst_list[i]``."""
    for key, value in src.items():
        if isinstance(value, dict):
            for d in dst_list:
                d[key] = {}
            sub_dsts = [d[key] for d in dst_list]
            _split_dict(value, sub_dsts)
        elif isinstance(value, (torch.Tensor, np.ndarray)):
            np_val = to_numpy(value) if isinstance(value, torch.Tensor) else value
            for i, d in enumerate(dst_list):
                d[key] = np_val[i]
        else:
            for d in dst_list:
                d[key] = value
