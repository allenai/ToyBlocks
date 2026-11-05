"""WebSocket policy server — wire-compatible with the openpi client.

Wraps any :class:`BasePolicy` and serves it over a WebSocket using msgpack
with NumPy array support.  The protocol is identical to
``openpi.serving.WebsocketPolicyServer``, so existing clients
(``openpi_client.WebsocketClientPolicy`` or the ToyBlocks
``WebsocketClientPolicy``) work without modification.

Usage::

    from toyblocks.serving import BasePolicy, PolicyServer

    class MyPolicy(BasePolicy):
        def infer(self, obs):
            return {"actions": np.zeros(7)}

    server = PolicyServer(MyPolicy(), port=8000)
    server.serve_forever()
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import http
import logging
import time
import traceback

import websockets.asyncio.server as _server
import websockets.frames

from toyblocks.serving import msgpack_numpy
from toyblocks.serving.base_policy import BasePolicy

logger = logging.getLogger(__name__)

# WebSocket keepalive defaults for policy RPC (batched infer + Isaac can stall for minutes).
# Library default (20s / 20s) often closes with 1011 under load; shared with WebsocketClientPolicy.
POLICY_WS_PING_INTERVAL: float | None = 60.0    # in seconds
POLICY_WS_PING_TIMEOUT: float | None = 180.0


class PolicyServer:
    """Serves a :class:`BasePolicy` over WebSocket + msgpack.

    Wire protocol
    -------------
    1. On connection the server sends one msgpack frame: ``policy.metadata``.
    2. The client then enters a request/response loop: it sends a msgpack-encoded
       observation dict, and the server replies with a msgpack-encoded action dict.
    3. ``GET /healthz`` returns HTTP 200 ``OK`` for liveness probes.

    Keepalive defaults (``POLICY_WS_PING_INTERVAL`` / ``POLICY_WS_PING_TIMEOUT``) are
    tuned for long model forwards and multi-client queues. Pass ``ping_interval=None``
    to disable pings entirely.
    """

    def __init__(
        self,
        policy: BasePolicy,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
        *,
        ping_interval: float | None = POLICY_WS_PING_INTERVAL,
        ping_timeout: float | None = POLICY_WS_PING_TIMEOUT,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata if metadata is not None else policy.metadata
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        logging.getLogger("websockets.server").setLevel(logging.INFO)
        # ``infer`` is synchronous (JAX / PyTorch). Running it on the asyncio loop
        # blocks handshakes and I/O for every other client; offload to one worker
        # so the event loop stays responsive while keeping GPU work serialized.
        self._infer_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="policy_infer",
        )
        # Set while an exclusive policy is serving a client; see BasePolicy.exclusive.
        self._active_client: str | None = None

    def serve_forever(self) -> None:
        """Block and serve until the process is killed."""
        asyncio.run(self._run())

    async def _run(self) -> None:
        try:
            async with _server.serve(
                self._handler,
                self._host,
                self._port,
                compression=None,
                max_size=None,
                process_request=_health_check,
                ping_interval=self._ping_interval,
                ping_timeout=self._ping_timeout,
            ) as server:
                logger.info("PolicyServer listening on %s:%s", self._host, self._port)
                await server.serve_forever()
        finally:
            self._infer_executor.shutdown(wait=True)

    async def _handler(self, websocket: _server.ServerConnection) -> None:
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()

        client = str(websocket.remote_address)
        if self._policy.exclusive and self._active_client is not None:
            # Sharing a stateful single-agent policy between two eval workers
            # silently mixes their episode timelines (see BasePolicy.exclusive),
            # so refuse rather than serve both badly.
            msg = (
                f"{self._policy.__class__.__name__} serves one client at a time and is "
                f"already serving {self._active_client}; {client} was rejected. Run one "
                "eval worker per server, or serve a non-exclusive policy."
            )
            logger.error(msg)
            await websocket.send(msg)
            await websocket.close(
                code=websockets.frames.CloseCode.TRY_AGAIN_LATER,
                reason="Policy already serving another client.",
            )
            return
        if self._policy.exclusive:
            self._active_client = client

        try:
            await websocket.send(packer.pack(self._metadata))

            prev_total_time: float | None = None
            while True:
                try:
                    start_time = time.monotonic()
                    obs = msgpack_numpy.unpackb(await websocket.recv())

                    loop = asyncio.get_running_loop()

                    def _infer_and_time() -> tuple[dict, float]:
                        t0 = time.monotonic()
                        out = self._policy.infer(obs)
                        return out, (time.monotonic() - t0) * 1000

                    action, infer_ms = await loop.run_in_executor(
                        self._infer_executor,
                        _infer_and_time,
                    )

                    action["server_timing"] = {"infer_ms": infer_ms}
                    if prev_total_time is not None:
                        action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                    await websocket.send(packer.pack(action))
                    prev_total_time = time.monotonic() - start_time

                except websockets.ConnectionClosed:
                    logger.info("Connection from %s closed", websocket.remote_address)
                    break
                except Exception:
                    await websocket.send(traceback.format_exc())
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error. Traceback included in previous frame.",
                    )
                    raise
        finally:
            # Also runs on the error path above: a client that died mid-episode
            # would otherwise leave its aged HL buffer for the next connection.
            if self._policy.exclusive:
                self._active_client = None
            self._policy.reset()


def _health_check(
    connection: _server.ServerConnection,
    request: _server.Request,
) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
