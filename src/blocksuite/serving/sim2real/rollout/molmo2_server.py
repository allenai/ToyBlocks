"""Molmo2 vision-language policy server and client.

The model, the question and the answer parsing are :mod:`blocksuite.serving.molmo2_hl`, the same code
the sim eval serves, run at batch size 1. What the checkpoint was trained to see -- front-only or
front + gripper, the front crop, the frame size, the preprocessor style and the question format -- is
read from its training config. The client's ``molmo_*`` settings are only cross-checked against it.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import socket
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np
from vlagents import register_agent
from vlagents.__main__ import start_server
from vlagents.client import RemoteAgent
from vlagents.policies.interface import Act, Agent, Obs, SingleAct, SingleObs

from blocksuite.serving.molmo2_hl import HLContract, Molmo2HLPolicy, to_hwc_uint8
from blocksuite.serving.sim2real.rollout.remote_agent_utils import ensure_server_ready, remote_agent_act

logger = logging.getLogger(__name__)

AGENT_NAME = "molmo2"
DEFAULT_PORT = 8082
DEFAULT_CHECKPOINT = ""
LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "::1")

# ``<u, v>`` in an HL string; the y of an html-v2 point can go negative after undoing point_y_offset_px.
_UV_RE = re.compile(r"<\s*(-?\d+),\s*(-?\d+)\s*>")
# The html-v2 instruction the server appends; an operator prompt must not carry it as well.
_HTML_V2_INSTRUCTION = "predict action."


def _parse_target_xy(action_str: str) -> np.ndarray | None:
    """Parse the first ``<u,v>`` from a Molmo2 action string, or return None."""
    m = _UV_RE.search(action_str)
    if m:
        return np.array([float(m.group(1)), float(m.group(2))], dtype=np.float32)
    return None


def _strip_coords(action_str: str) -> str:
    """Strip ``<u,v>`` coordinate tokens from a Molmo2 string, leaving just the action words."""
    return _UV_RE.sub("", action_str).replace("  ", " ").strip()


def parse_molmo_answer(raw: str, contract: HLContract) -> tuple[str, np.ndarray | None, str]:
    """Decode a Molmo2 answer into ``(language, target_xy, prev_action_str)``.

    ``contract.parse_output`` handles both answer formats: an html-v2 point (0-1000 of the crop the
    model saw) comes back as pixels of the full ``contract.image_size`` frame, and a legacy
    ``pick <812, 705>`` is already in those pixels. ``prev_action_str`` is what the next question
    is seeded with. ``language`` is ``"done"`` when the answer ends the episode. An html-v2 answer
    with no target is passed through as text with no point.
    """
    try:
        hl_str, prev_action = contract.parse_output(raw)
    except ValueError:
        logger.warning("molmo2 answer has no html-v2 target and action: %r", raw)
        hl_str, prev_action = raw, raw
    hl_str = hl_str.strip()
    # ``done`` terminates the episode in both formats and carries no point.
    if hl_str.endswith("done"):
        return "done", None, "done"
    return _strip_coords(hl_str), _parse_target_xy(hl_str), prev_action.strip()


def _settings_mismatches(contract: HLContract, info: dict[str, Any], front: np.ndarray) -> list[str]:
    """Where the client's ``molmo_*`` settings (sent in ``info``) disagree with the checkpoint config."""
    problems: list[str] = []
    want_w, want_h = (int(v) for v in contract.image_size)
    got_h, got_w = front.shape[:2]
    if (got_w, got_h) != (want_w, want_h):
        problems.append(f"front_rgb is {got_w}x{got_h}, but the checkpoint was trained on {want_w}x{want_h} frames")
    if "image_size" in info and tuple(int(v) for v in info["image_size"]) != (want_w, want_h):
        problems.append(f"molmo_input_size={tuple(info['image_size'])}, checkpoint image_size={(want_w, want_h)}")
    if "front_crop" in info:
        sent_crop = tuple(int(v) for v in info["front_crop"]) if info["front_crop"] is not None else None
        if sent_crop != contract.front_crop:
            problems.append(f"molmo_front_crop={sent_crop}, checkpoint front_crop={contract.front_crop}")
    if "front_view_only" in info and bool(info["front_view_only"]) != contract.front_view_only:
        problems.append(
            f"molmo_front_view_only={bool(info['front_view_only'])}, "
            f"checkpoint front_view_only={contract.front_view_only}"
        )
    return problems


class Molmo2Agent(Agent):
    """vlagents Agent serving :class:`Molmo2HLPolicy` at batch size 1, one rolling previous action."""

    def __init__(
        self,
        default_checkpoint_path: str = DEFAULT_CHECKPOINT,
        device: str = "cuda:0",
        max_steps: int = 128,
        **kwargs: Any,
    ) -> None:
        super().__init__(default_checkpoint_path=default_checkpoint_path, **kwargs)
        if not self.default_checkpoint_path:
            raise ValueError(
                "molmo2 server needs --checkpoint pointing at a Molmo2 HL checkpoint dir"
            )
        self.device = device
        self.max_steps = int(max_steps)
        self.policy: Molmo2HLPolicy | None = None
        self._prev_action_str: str = ""

    def initialize(self) -> None:
        """Heavy init (runs on AgentService's background thread): load the Molmo2 model."""
        self.policy = Molmo2HLPolicy(
            self.default_checkpoint_path,
            device=self.device,
            max_new_tokens=self.max_steps,
            max_batch=1,
        )

    def act(self, obs: Obs) -> Act:
        super().act(obs)
        assert self.policy is not None, "Molmo2Agent not initialized"
        contract = self.policy.contract
        acts: dict[str, SingleAct] = {}
        for cam_name, single_obs in obs.obs.items():
            cams = single_obs.cameras or {}
            img_front_raw = cams.get("front_rgb")
            img_gripper_raw = cams.get("gripper_rgb")
            if img_front_raw is None:
                raise ValueError(f"molmo2 obs for {cam_name!r} requires front_rgb (got keys={list(cams)})")
            info_in = single_obs.info or {}
            user_prompt = str(info_in.get("prompt") or obs.language_instruction or "").strip()
            if not user_prompt:
                raise ValueError(
                    "molmo2 obs.info has no 'prompt' and Obs.language_instruction is empty; "
                    "the HL model needs a natural-language goal to condition on."
                )
            if contract.html_v2_variant and user_prompt.lower().endswith(_HTML_V2_INSTRUCTION):
                raise ValueError(
                    f"molmo2 prompt {user_prompt!r} already ends in the html-v2 instruction; type only the "
                    "goal (e.g. 'stack Q Y C'), the server appends the instruction the checkpoint was trained on."
                )
            if info_in.get("reset_prev_action"):
                self._prev_action_str = ""

            front = to_hwc_uint8(img_front_raw)
            problems = _settings_mismatches(contract, info_in, front)
            if problems:
                raise ValueError(
                    "molmo2 client settings disagree with the checkpoint's training config "
                    f"({self.default_checkpoint_path}): " + "; ".join(problems)
                )
            # Crops the front frame and picks front-only vs front + gripper, as training did.
            images = self.policy.images_for(front, img_gripper_raw)

            # ``use_prev_action`` (default True) is a per-call opt-out: when
            # False the previous-action prefix is dropped from this call's
            # question, but ``_prev_action_str`` is still updated below so the
            # rolling history remains coherent if the caller re-enables it.
            use_prev_action = bool(info_in.get("use_prev_action", True))
            prev_action_str = self._prev_action_str
            # Mirror only what was actually fed to the model this call so the
            # GUI's "prev_action_str" display matches ``model_prompt``.
            prev_action_used = prev_action_str if use_prev_action else ""
            model_prompt = contract.question(user_prompt, prev_action_used)

            t0 = time.perf_counter()
            raw = self.policy.generate([images], [model_prompt])[0]
            infer_ms = (time.perf_counter() - t0) * 1000.0

            language, target_xy, prev_action_next = parse_molmo_answer(raw, contract)
            logger.info(
                "molmo2: question=%r n_images=%d front_shape=%s -> raw=%r (language=%r, target_xy=%s, %.1fms)",
                model_prompt,
                len(images),
                tuple(images[0].shape),
                raw,
                language,
                target_xy.tolist() if target_xy is not None else None,
                infer_ms,
            )
            self._prev_action_str = prev_action_next

            acts[cam_name] = SingleAct(
                action=np.zeros(1, dtype=np.float32),
                info={
                    "raw": raw,
                    "language": language,
                    "target_xy": (
                        target_xy
                        if target_xy is not None
                        else np.array([np.nan, np.nan], dtype=np.float32)
                    ),
                    "prompt": user_prompt,
                    "model_prompt": model_prompt,
                    "prev_action_str": prev_action_used,
                    "infer_ms": infer_ms,
                },
            )
        return Act(acts=[acts])


register_agent(AGENT_NAME, Molmo2Agent)


def _start_parent_watchdog(parent_pid: int, poll_s: float = 1.0) -> None:
    """Exit this process once the launching client (parent) is gone."""

    def _watch() -> None:
        while True:
            if os.getppid() != parent_pid:
                logging.info("molmo2 server: parent %d is gone, exiting", parent_pid)
                os._exit(0)
            time.sleep(poll_s)

    threading.Thread(target=_watch, name="parent-watchdog", daemon=True).start()


def _is_local_host(host: str) -> bool:
    return host in LOCAL_HOSTS or host == socket.gethostname()


def _port_open(host: str, port: int, timeout_s: float = 0.5) -> bool:
    try:
        with socket.create_connection((host if host != "0.0.0.0" else "127.0.0.1", port), timeout=timeout_s):
            return True
    except OSError:
        return False


class Molmo2Client:
    """Synchronous client: one ``front_rgb`` frame + prompt in, ``(language, target_xy)`` out."""

    def __init__(self, host: str = "localhost", port: int = DEFAULT_PORT, ready_timeout_s: float = 600.0) -> None:
        self.host = host
        self.port = int(port)
        self.remote_agent = self._connect_with_retry(ready_timeout_s)
        ensure_server_ready(self.remote_agent, timeout_s=ready_timeout_s)
        logger.info("molmo2 client ready (%s:%d, shared memory=%s)", self.host, self.port, self.remote_agent.on_same_machine)

    @classmethod
    def from_config(cls, cfg: Any) -> Molmo2Client:
        return cls(cfg.molmo_host, cfg.molmo_port)

    def _connect_with_retry(self, timeout_s: float) -> RemoteAgent:
        deadline = time.monotonic() + timeout_s
        last_exc: Exception | None = None
        while time.monotonic() < deadline:
            try:
                return RemoteAgent(self.host, self.port, AGENT_NAME, on_same_machine=_is_local_host(self.host), jpeg_encoding=False, image_size=None)
            except (ConnectionRefusedError, EOFError, OSError) as exc:
                last_exc = exc
                time.sleep(0.5)
        raise TimeoutError(
            f"molmo2 server not reachable at {self.host}:{self.port} after {timeout_s:.0f}s"
        ) from last_exc

    def build_obs(
        self,
        front_rgb: np.ndarray,
        prompt: str,
        *,
        gripper_rgb: np.ndarray | None = None,
        reset_prev_action: bool = False,
        use_prev_action: bool = True,
        image_size: tuple[int, int] | None = None,
        front_crop: tuple[int, int, int, int] | None = None,
        front_view_only: bool | None = None,
    ) -> Obs:
        info: dict[str, Any] = {"prompt": str(prompt)}
        if reset_prev_action:
            info["reset_prev_action"] = True
        # The client's molmo_* settings, sent so the server can check them
        # against the checkpoint's training config (which it crops by). The
        # crop is sent even when None, since "no crop" is a setting too.
        if image_size is not None:
            info["image_size"] = (int(image_size[0]), int(image_size[1]))
            info["front_crop"] = tuple(int(v) for v in front_crop) if front_crop is not None else None
        if front_view_only is not None:
            info["front_view_only"] = bool(front_view_only)
        # Only send the flag when it differs from the server default so older
        # servers (that ignore it) keep their prior behaviour.
        if not use_prev_action:
            info["use_prev_action"] = False
        cameras: dict[str, np.ndarray] = {
            "front_rgb": np.ascontiguousarray(front_rgb),
        }
        # Front-only checkpoints: omit gripper_rgb so the server feeds a
        # single-image prompt matching training.
        if gripper_rgb is not None:
            cameras["gripper_rgb"] = np.ascontiguousarray(gripper_rgb)
        return Obs(
            obs={
                "molmo2": SingleObs(cameras=cameras, info=info),
            },
            language_instruction=str(prompt),
        )

    @staticmethod
    def unpack(act: Act) -> dict[str, Any]:
        if not act.acts:
            return {}
        single_act = next(iter(act.acts[0].values()))
        info = dict(single_act.info or {})
        target = info.get("target_xy")
        if target is not None:
            target_arr = np.asarray(target, dtype=np.float32).reshape(-1)
            if target_arr.size >= 2 and not np.any(np.isnan(target_arr[:2])):
                info["target_xy"] = target_arr[:2]
            else:
                info["target_xy"] = None
        return info

    def infer(
        self,
        front_rgb: np.ndarray,
        prompt: str,
        *,
        gripper_rgb: np.ndarray | None = None,
        reset_prev_action: bool = False,
        use_prev_action: bool = True,
        image_size: tuple[int, int] | None = None,
        front_crop: tuple[int, int, int, int] | None = None,
        front_view_only: bool | None = None,
    ) -> dict[str, Any]:
        obs = self.build_obs(
            front_rgb,
            prompt,
            gripper_rgb=gripper_rgb,
            reset_prev_action=reset_prev_action,
            use_prev_action=use_prev_action,
            image_size=image_size,
            front_crop=front_crop,
            front_view_only=front_view_only,
        )
        return self.unpack(remote_agent_act(self.remote_agent, obs))

    def close(self) -> None:
        try:
            self.remote_agent.close()
        except Exception:
            logger.exception("failed to close molmo2 client")


def launch_molmo2_server(cfg: Any, *, port: int | None = None, extra_args: tuple[str, ...] | list[str] = (), parent_pid: int | None = None) -> subprocess.Popen | None:
    host = cfg.molmo_host
    port = int(port if port is not None else cfg.molmo_port)
    checkpoint = getattr(cfg, "molmo_checkpoint", "") or ""
    if not checkpoint:
        raise ValueError("use_molmo=true but molmo_checkpoint is empty in the config; set it to the Molmo2 HL checkpoint directory (vlagent_rcs_client.json).")
    if not _is_local_host(host):
        logger.info("molmo_host=%s is remote; not launching a local molmo2 server", host)
        return None
    if _port_open(host, port):
        logger.info("molmo2 server already listening on %s:%d; not launching another", host, port)
        return None
    cmd = [sys.executable, "-m", "blocksuite.serving.sim2real.rollout.molmo2_server", "--host", host, "--port", str(port), "--checkpoint", str(checkpoint), "--parent-pid", str(parent_pid if parent_pid is not None else os.getpid())]
    cmd.extend(extra_args)
    proc = subprocess.Popen(cmd)
    logger.info("launched molmo2 server (pid %d): %s", proc.pid, " ".join(cmd))
    return proc


def stop_molmo2_server(proc: subprocess.Popen | None, timeout_s: float = 10.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=timeout_s)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--checkpoint", required=True, help="Path to the Molmo2 HL checkpoint directory (contains config.yaml).")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, default=128, help="Molmo2 generate() max token budget per call. A legacy 'pick <u, v>' answer fits in ~20 tokens, but an html-v2 answer carries the whole 'now: ... next: ... Do: ... <points .../>' commentary and gets truncated below ~80.")
    parser.add_argument("--parent-pid", type=int, default=0, help="exit when this process disappears (0 = never)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("serving %s on %s:%d (checkpoint=%s)", AGENT_NAME, args.host, args.port, args.checkpoint)

    if args.parent_pid > 0:
        _start_parent_watchdog(args.parent_pid)

    kwargs = dict(default_checkpoint_path=args.checkpoint, device=args.device, max_steps=args.max_steps)
    start_server(AGENT_NAME, json.dumps(kwargs), port=args.port, host=args.host)


if __name__ == "__main__":
    main()
