"""WebSocket client that implements the :class:`BasePolicy` interface.

Connects to a :class:`PolicyServer` (or openpi ``WebsocketPolicyServer``)
and forwards ``infer`` calls over the wire.  This is a self-contained
replacement for ``openpi_client.WebsocketClientPolicy`` so that the
blocksuite evaluation stack does not require the ``openpi`` package.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

import websockets.sync.client

from blocksuite.serving.base_policy import BasePolicy
from blocksuite.serving import msgpack_numpy
from blocksuite.serving.policy_server import POLICY_WS_PING_INTERVAL, POLICY_WS_PING_TIMEOUT

logger = logging.getLogger(__name__)


class WebsocketClientPolicy(BasePolicy):
    """Talks to a remote policy server over WebSocket + msgpack.

    Uses the same keepalive defaults as :class:`~blocksuite.serving.policy_server.PolicyServer`
    (see ``POLICY_WS_PING_INTERVAL`` / ``POLICY_WS_PING_TIMEOUT``). Pass ``ping_interval=None``
    to disable keepalive pings.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        *,
        ping_interval: Optional[float] = POLICY_WS_PING_INTERVAL,
        ping_timeout: Optional[float] = POLICY_WS_PING_TIMEOUT,
    ) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict[str, Any]:
        return self._server_metadata

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logger.info("Waiting for server at %s ...", self._uri)
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    ping_interval=self._ping_interval,
                    ping_timeout=self._ping_timeout,
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logger.info("Still waiting for server...")
                time.sleep(5)

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def reset(self) -> None:
        pass
