"""Remote vlagents client helpers (shared-memory fallback, transport warnings)."""

from __future__ import annotations

import logging
import time

from vlagents.client import RemoteAgent
from vlagents.policies.interface import Act, Obs

logger = logging.getLogger(__name__)


def _looks_like_shm_permission_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "permission denied" in message and (
        "/psm_" in message or "shared_memory" in message or "shm_" in message
    )


def _looks_like_not_initialized_error(exc: BaseException) -> bool:
    return "not initialized" in str(exc).lower()


def ensure_server_ready(
    remote_agent: RemoteAgent,
    *,
    timeout_s: float = 600.0,
    poll_s: float = 0.5,
) -> None:
    """Block until the remote vlagents server finishes heavy model initialization."""
    remote_agent.ensure_connected()
    try:
        if remote_agent.is_initialized():
            return
    except Exception:
        logger.warning(
            "lost connection while checking vlagents server init; reconnecting",
            exc_info=True,
        )
        remote_agent.reconnect()

    logger.info(
        "waiting for vlagents server to finish loading model (%s:%s, model=%s)",
        remote_agent.host,
        remote_agent.port,
        remote_agent.model,
    )
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if remote_agent.is_initialized():
                logger.info("vlagents server ready")
                return
        except Exception:
            logger.warning(
                "lost connection while waiting for vlagents server init; reconnecting",
                exc_info=True,
            )
            remote_agent.reconnect()
        time.sleep(poll_s)
    raise TimeoutError(
        f"vlagents server not initialized after {timeout_s:.0f}s "
        f"({remote_agent.host}:{remote_agent.port}, model={remote_agent.model})"
    )


def remote_checkpoint_path(remote_agent: RemoteAgent) -> str:
    """Checkpoint the vlagents server loaded, or ``""`` when it cannot be read.

    Served by ``vlagents_server``'s ``exposed_checkpoint_path``. Returns ``""``
    for a server started without it, or one that is unreachable. Never raises
    and never reconnects, so a GUI poll cannot disturb the inference
    connection.
    """
    try:
        conn = remote_agent.c
        if conn is None:
            return ""
        return str(conn.root.checkpoint_path() or "")
    except Exception:
        logger.debug("remote checkpoint path unavailable", exc_info=True)
        return ""


def remote_agent_act(remote_agent: RemoteAgent, obs: Obs) -> Act:
    """Call ``RemoteAgent.act`` with a JPEG fallback when shared memory is inaccessible."""
    ensure_server_ready(remote_agent)
    try:
        return remote_agent.act(obs)
    except Exception as exc:
        if remote_agent.on_same_machine and _looks_like_shm_permission_error(exc):
            logger.warning(
                "Shared-memory camera transfer failed (%s). Client and server must run "
                "as the same UNIX user for on_same_machine=true. Retrying with JPEG encoding.",
                exc,
            )
            remote_agent.reconnect(on_same_machine=False, jpeg_encoding=True)
            ensure_server_ready(remote_agent)
            return remote_agent.act(obs)
        if _looks_like_not_initialized_error(exc):
            logger.info("vlagents server still initializing; waiting before retry")
            ensure_server_ready(remote_agent)
            return remote_agent.act(obs)
        raise
