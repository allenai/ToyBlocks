"""Abstract base class for policies served over the network.

Any model that should be deployable via :class:`PolicyServer` (or the openpi
``WebsocketPolicyServer``) only needs to subclass :class:`BasePolicy` and
implement :meth:`infer`.

The interface is intentionally identical to ``openpi_client.base_policy.BasePolicy``
so that either can be used interchangeably on the server or client side.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Iterator


class BasePolicy(abc.ABC):
    """Minimal contract that every servable policy must satisfy.

    Subclasses implement :meth:`infer` to map an observation dict to an action
    dict.  The observation and action dicts use string keys and NumPy array (or
    scalar) values — the transport layer handles serialisation.

    Example::

        class MyPolicy(BasePolicy):
            def __init__(self, model):
                self._model = model

            def infer(self, obs: dict) -> dict:
                action = self._model.predict(obs["observation/image"])
                return {"actions": action}
    """

    @abc.abstractmethod
    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single inference step.

        Args:
            obs: Observation dict.  Keys and value shapes depend on the
                upstream observation transform; at minimum expect NumPy arrays.

        Returns:
            Action dict.  Must contain at least an ``"actions"`` key whose
            value is a NumPy array of shape ``(action_dim,)`` (unbatched),
            ``(num_envs, action_dim)`` (batched), or
            ``(action_horizon, action_dim)`` (chunked).
        """

    def reset(self) -> None:
        """Reset internal state (e.g. action-chunk buffers, history).

        Called by the evaluation harness whenever episodes terminate.
        The default implementation is a no-op.
        """

    @property
    def exclusive(self) -> bool:
        """True when this policy may serve only one client at a time.

        Override in policies that carry per-episode state across :meth:`infer`
        calls (HL latency buffers, episode ids, action-chunk history): the server
        holds a single policy instance, so two concurrent clients would interleave
        their episode timelines into one set of buffers and quietly condition each
        run on the other's state. :class:`PolicyServer` rejects the second
        concurrent connection instead. Stateless or batched-over-envs policies can
        leave this False and be shared by every eval worker.
        """
        return False

    @property
    def metadata(self) -> Dict[str, Any]:
        """Optional metadata sent to the client on connection.

        Override to advertise model name, action space info, expected
        observation keys, etc.  The default is an empty dict.
        """
        return {}


def walk_policy_chain(policy: Any) -> Iterator[Any]:
    """Yield every link of a wrapped policy, outermost first.

    The eval stacks ``PolicyClientAdapter -> ActionChunkBroker ->
    ObsTransformPolicy -> WebsocketClientPolicy``, so anything that needs one
    specific link (the obs transform, the chunk broker) has to reach through the
    wrappers rather than build a second one, which would drift from the
    negotiated stack the moment a server-advertised setting changes. Wrappers
    expose their inner policy as either ``policy`` or ``_policy``; the id set
    bounds the walk if a chain is ever made cyclic.
    """
    seen: set[int] = set()
    while policy is not None and id(policy) not in seen:
        seen.add(id(policy))
        yield policy
        policy = getattr(policy, "policy", None) or getattr(policy, "_policy", None)
