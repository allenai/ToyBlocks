"""Standalone HTTP queue server for episode setups.

Serves a fixed list of per-episode setups (see ``docs/source/user_guide/workflows/evals/launch_eval.md`` and
:mod:`blocksuite.utils.episode_queue`) one at a time. Multiple
``scripts/evaluate.py`` clients can poll the same server to consume a shared
queue without duplicating episodes. Input is a LeRobot dataset directory;
setups come from :func:`blocksuite.utils.episode_dataset.load_setups`.

Endpoints:
    GET /next          -> {"setup": <dict|null>}  (null when empty)
    GET /next?slot=N&slot_scales=s0,s1,...
                       -> next setup whose recorded cube_scale maps to slot N
                          (exact per-episode block-size matching; see below).
                          ``slot_scales`` is the client's baked slot ordering and
                          must equal this server's.
    GET /size          -> {"remaining": <int>, "total": <int>,
                           "expert_seconds_max": <float>}
    GET /healthz       -> {"ok": true}

``expert_seconds_max`` is the duration of the longest expert demonstration in
the queue (the max recorded ``expert_episode_length`` over the dataset's fps).
The eval pipeline reads it to size the global ``episode_length_s`` so a
per-episode ``--expert_timeout_scale`` budget is always the binding timeout.

Cube-scale slot routing: block scale is non-sim-state (baked at spawn), so the
generated env spawns one env per distinct recorded ``cube_scale`` (slot ``i`` =
the i-th value in :func:`~blocksuite.utils.episode_dataset.distinct_cube_scales`).
When the loaded setups carry ``cube_scale``, this server groups them by that slot
and serves env ``i`` (which passes the slot it was spawned at) only episodes whose
size matches its spawned blocks. Codegen and this server derive the identical slot
ordering from the SAME episodes, so they line up; the client sends its baked
ordering as ``?slot_scales=`` on every fetch and any difference is a hard error
here, since the two being built from different ``--episode-indices`` would make
slot ``i`` mean a different block size on each side.
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from blocksuite.utils.episode_dataset import (
    CUBE_SCALE_DECIMALS,
    dataset_fps,
    distinct_cube_scales,
    load_setups,
    parse_episode_indices_spec,
)

logger = logging.getLogger("episode_queue")


class QueueHandler(BaseHTTPRequestHandler):
    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/next":
            query = parse_qs(parsed.query)
            slot_vals = query.get("slot")
            slot = int(slot_vals[0]) if slot_vals else None
            scales_vals = query.get("slot_scales")
            client_slot_scales = (
                [float(x) for x in scales_vals[0].split(",") if x]
                if scales_vals
                else None
            )
            self._send_json(
                200, {"setup": self.server.get_next(slot, client_slot_scales)}
            )
        elif parsed.path == "/size":
            self._send_json(200, self.server.size_status())
        elif parsed.path == "/healthz":
            self._send_json(200, {"ok": True})
        else:
            self._send_json(404, {"error": f"unknown path {self.path!r}"})

    def log_message(self, format: str, *args) -> None:
        logger.debug("%s - %s", self.address_string(), format % args)


class QueueServer(HTTPServer):
    def __init__(
        self,
        server_address,
        RequestHandlerClass,
        setups: list[dict],
        *,
        expert_seconds_max: float,
    ):
        super().__init__(server_address, RequestHandlerClass)
        self.total = len(setups)
        # Reported on /size; computed once by main() while it still has the
        # dataset path (fps lives in meta/info.json, not on the setups).
        self.expert_seconds_max = float(expert_seconds_max)
        self.lock = threading.Lock()

        # Cube-scale slot routing is all-or-nothing: block size is non-sim-state,
        # so the client asks for its env's spawn slot and can only run an episode
        # in the env spawned at its size. A partially-populated cube_scale column
        # would drop the whole queue to FIFO while the client keeps sending slot
        # hints, landing episodes on wrong-sized blocks with no error.
        self.slot_values: list[float] | None = None
        self.slots: dict[float, deque[dict]] = {}
        self.setups: deque[dict] = deque()
        with_scale = [s for s in setups if s.get("cube_scale") is not None]
        if with_scale and len(with_scale) != len(setups):
            missing = [
                s.get("episode_id", s.get("episode_index"))
                for s in setups
                if s.get("cube_scale") is None
            ]
            raise ValueError(
                f"{len(missing)} of {len(setups)} episodes record no cube_scale while "
                f"the rest do (e.g. {missing[:5]}). Serving these FIFO while the eval "
                "client requests size-matched slots would run episodes on wrong-sized "
                "blocks. Restrict --episode-indices to episodes that all record a "
                "cube_scale, or regenerate the dataset so every episode does."
            )
        # Longest expert demonstration first, so the episodes most likely to run
        # longest (their timeout scales with the demo) start earliest and a slot's
        # last stragglers are short ones. Stable, so equal lengths keep their order.
        setups = sorted(setups, key=lambda s: -int(s["expert_episode_length"]))
        if setups and with_scale:
            self.slot_values = distinct_cube_scales(s["cube_scale"] for s in setups)
            self.slots = {v: deque() for v in self.slot_values}
            for s in setups:
                self.slots[round(float(s["cube_scale"]), CUBE_SCALE_DECIMALS)].append(s)
            logger.info(
                "queue ready with %d setups across %d cube-scale slot(s): %s",
                self.total, len(self.slot_values),
                [round(v, 4) for v in self.slot_values],
            )
        else:
            self.setups = deque(setups)
            logger.info("queue ready with %d setups (no cube-scale routing)", self.total)

    def get_next(
        self, slot: int | None = None, client_slot_scales: list[float] | None = None
    ):
        with self.lock:
            if self.slot_values is None:
                if client_slot_scales:
                    raise ValueError(
                        f"the eval task bakes {len(client_slot_scales)} cube-scale spawn "
                        f"slot(s) ({[round(v, 4) for v in client_slot_scales]}) but this "
                        "queue's episodes record no cube_scale, so it cannot serve "
                        "size-matched episodes. Start the queue over the same "
                        "--episode-indices the task was generated from."
                    )
                return self.setups.popleft() if self.setups else None
            if not client_slot_scales:
                raise ValueError(
                    "this queue routes by cube-scale slot "
                    f"({[round(v, 4) for v in self.slot_values]}) but the eval task bakes "
                    "no cube-scale spawn slots, so its envs are not spawned at the "
                    "recorded block sizes. Regenerate the task with "
                    "generate_task_from_dataset.py over the same --episode-indices."
                )
            # Compare the slot ordering itself, not just its length: both sides derive
            # it from distinct_cube_scales over their own episode set, so two equal-sized
            # episode sets with different scales would otherwise pass and silently map
            # slot i to a different block size on each side.
            client_rounded = [round(v, CUBE_SCALE_DECIMALS) for v in client_slot_scales]
            queue_rounded = [round(v, CUBE_SCALE_DECIMALS) for v in self.slot_values]
            if client_rounded != queue_rounded:
                raise ValueError(
                    "cube-scale slot ordering differs between the eval task "
                    f"({[round(v, 4) for v in client_rounded]}) and this queue "
                    f"({[round(v, 4) for v in queue_rounded]}), so slot indices refer to "
                    "different block sizes. Pass the same --episode-indices to "
                    "generate_task_from_dataset.py and serve_episode_queue.py."
                )
            # Slot routing: serve only the requested slot's queue. A caller with no
            # slot hint cannot be served safely — handing it any non-empty slot
            # would spawn the episode's blocks at the wrong size.
            if slot is None:
                raise ValueError(
                    "this queue routes by cube-scale slot "
                    f"({[round(v, 4) for v in self.slot_values]}) but /next was called "
                    "without a ?slot= hint. Serving an arbitrary slot would run the "
                    "episode at the wrong block size."
                )
            if not 0 <= slot < len(self.slot_values):
                raise ValueError(
                    f"?slot={slot} is not one of this queue's {len(self.slot_values)} "
                    "cube-scale slots"
                )
            dq = self.slots[self.slot_values[slot]]
            return dq.popleft() if dq else None

    def size_status(self) -> dict:
        with self.lock:
            if self.slot_values is None:
                remaining = len(self.setups)
                slot_queues: list[deque[dict]] = []
            else:
                slot_queues = [self.slots[v] for v in self.slot_values]
                remaining = sum(len(dq) for dq in slot_queues)
            return {
                "remaining": remaining,
                "total": self.total,
                "expert_seconds_max": self.expert_seconds_max,
                # Per cube-scale slot, in slot order (empty without slot routing):
                # queued episodes and their summed expert frames, from which the
                # eval client sizes each slot's env count.
                "slot_counts": [len(dq) for dq in slot_queues],
                "slot_expert_frames": [
                    sum(int(s["expert_episode_length"]) for s in dq) for dq in slot_queues
                ],
            }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to a LeRobot dataset directory (or a directory whose "
        "immediate children are themselves LeRobot datasets).",
    )
    parser.add_argument(
        "--episode-indices",
        default=None,
        help="Filter to specific episode_index values, e.g. '0,3,5,10-19' or "
        "'@PATH'. Applied before --limit; raises on missing indices.",
    )
    parser.add_argument(
        "--allow-duplicate-episodes",
        action="store_true",
        default=False,
        help="Allow duplicate entries in --episode-indices (e.g. '0,0,0' "
        "produces 3 queue items for episode 0). Useful for running the "
        "same episode multiple times across parallel envs.",
    )
    parser.add_argument(
        "--init-after-hl-action",
        default=None,
        help="Start each episode at the first frame after every instance of "
        "any of these high-level actions (comma or '|', e.g. 'flip,rotate'). "
        "'flip' matches both 'flip' and 'reorient'. Episodes with no match "
        "start at frame 0; episodes whose last match is the last frame are "
        "skipped. Empty/omitted starts every episode at frame 0.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Stop after this many setups (smoke tests)."
    )
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    episode_indices = (
        parse_episode_indices_spec(
            args.episode_indices,
            allow_duplicates=args.allow_duplicate_episodes,
        )
        if args.episode_indices
        else None
    )
    init_after = (args.init_after_hl_action or "").strip() or None
    setups = load_setups(
        args.dataset_path,
        limit=args.limit,
        episode_indices=episode_indices,
        allow_duplicate_episodes=args.allow_duplicate_episodes,
        init_after_hl_action=init_after,
    )
    with_expert_videos = sum(1 for setup in setups if setup.get("expert_videos"))
    without_expert_videos = len(setups) - with_expert_videos
    logger.info(
        "loaded %d setups from %s (episode_indices=%s, limit=%s, "
        "init_after_hl_action=%s, with_expert_videos=%d, without_expert_videos=%d)",
        len(setups),
        args.dataset_path,
        f"{len(episode_indices)} indices" if episode_indices else "all",
        args.limit,
        init_after or "off",
        with_expert_videos,
        without_expert_videos,
    )
    if without_expert_videos:
        logger.warning(
            "%d/%d setups have no expert_videos; their W&B rows will not include "
            "expert_first_frame unless this is fixed upstream.",
            without_expert_videos,
            len(setups),
        )

    fps = dataset_fps(args.dataset_path)
    expert_seconds_max = max(s["expert_episode_length"] for s in setups) / fps
    logger.info(
        "longest expert demonstration in the queue: %.2fs (%d frames at %.3g fps)",
        expert_seconds_max,
        max(s["expert_episode_length"] for s in setups),
        fps,
    )

    server = QueueServer(
        (args.bind, args.port),
        QueueHandler,
        setups,
        expert_seconds_max=expert_seconds_max,
    )
    logger.info("serving queue on %s:%d", args.bind, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
