"""TAM (Torque Adaptation Module) helpers for the vlagent RCS client.

Ported from ``examples/inference/franka_tam.py``. TAM adds a learned residual
torque on top of the Franka PD controller. Two pieces run outside the C++
controller and live here:

- the **history encoder** (JAX, ``simadaptor``): consumes the 1 kHz
  ``(t, q, qd, tau, gravity)`` history ring exported by the controller and
  emits a latent at ~5 Hz which is pushed back with ``set_tam_latent``;
- the **adaptor MLP** export: packed once into the controller through
  ``set_tam_mlp_weight``.

The controller applies zero residual until both the MLP weights and the first
latent have arrived, then ramps the residual in over 1 s.

Everything here is imported lazily (``simadaptor`` / JAX are only needed when
``tam`` is enabled in ``vlagent_rcs_client.json``).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from rcs._core.common import RobotPlatform
from rcs.utils import SimpleFrameRate

from toyblocks.serving.sim2real.rollout.hardware_config import HOME_POSE, ROBOT_INSTANCE

if TYPE_CHECKING:
    import gymnasium as gym
    from rcs_fr3._core.hw import Franka

    from toyblocks.serving.sim2real.rollout.inference_config import InferenceConfig

# The TAM history encoder (JAX) shares the GPU with everything else on this
# machine (ffstereo, molmo2, ...); never let JAX preallocate ~75% of device
# memory. Must be set before JAX initializes its backend (first use inside
# ``init_tam_runtime``). ``vlagent_rcs_client`` sets it as well, before any
# import, which is the reliable spot; this is a fallback for other importers.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

logger = logging.getLogger(__name__)

TAM_HISTORY_SIZE = 4000  # mirrors the C++ TAM_HISTORY_SIZE ring
TAM_LATENT_RATE_HZ = 5  # latent update rate of the history encoder


def tam_ideal_model_xml() -> str:
    """The ideal-model MJCF used by BOTH the TAM history encoder and the
    controller's ideal-model gravity compensation
    (``FrankaConfig.tam_ideal_model_path``). They must be the same model, so
    both sides resolve it through this one function."""
    from simadaptor.assets import default_panda_xml

    return str(default_panda_xml())


def tam_supported() -> bool:
    """TAM lives in the Franka hardware controller; there is no sim path."""
    return ROBOT_INSTANCE == RobotPlatform.HARDWARE


def hardware_robot(env: gym.Env) -> Franka:
    """The ``right`` Franka hardware handle behind the wrapper stack."""
    return env.get_wrapper_attr("envs")["right"].get_wrapper_attr("robot")


def init_tam_runtime() -> Any:
    """Load the TAM checkpoint: the streaming history encoder runs in this
    process (JAX; a GPU is strongly recommended) and the adaptor MLP is
    exported once into the C++ controller (see ``run_history_encoder``).

    The default checkpoint is downloaded once into ``~/.cache/simadaptor``;
    the ideal-model MJCF is installed with the ``torque-adaptation-module``
    package."""
    from simadaptor.deploy.history_runtime import RealTimeHistoryAdaptor

    # from_checkpoint enforces an applied-torque checkpoint: this integration
    # records a single commanded-torque stream.
    runtime = RealTimeHistoryAdaptor.from_checkpoint(xml_path=tam_ideal_model_xml())
    logger.info(
        "TAM ready: adaptor binary %d bytes, applied-torque mode",
        len(runtime.adaptor_weight_bytes()),
    )
    return runtime


def check_ideal_model_alignment(robot: Franka, runtime: Any) -> None:
    """Refuse to run if the controller's ideal-model gravity uses a different
    MJCF than the history encoder: the adaptor's torque space, the encoder's
    ideal model and the gravity feedforward must agree."""
    controller_xml = str(robot.get_config().tam_ideal_model_path)
    encoder_xml = str(runtime.inf.xml_path)
    if not controller_xml:
        raise RuntimeError(
            "TAM is enabled but FrankaConfig.tam_ideal_model_path is empty; the "
            "controller would use the robot's gravity model while the encoder "
            f"uses {encoder_xml}"
        )
    if os.path.realpath(controller_xml) != os.path.realpath(encoder_xml):
        raise RuntimeError(
            "ideal-model mismatch: controller gravity uses "
            f"{controller_xml} but the history encoder uses {encoder_xml}"
        )


def warmup_history_encoder(runtime: Any, cfg: InferenceConfig) -> None:
    """Compile every JAX path the live encoder will hit, before the robot
    moves. Drives synthetic 1 kHz history through the real ``push_window``
    path (same growing-then-saturated ring the live loop feeds) until pushes
    are consistently under ``cfg.tam_warmup_target_ms``, then resets the
    stream so live data starts clean. Blocks until fast or the timeout
    elapses (a timeout only warns)."""
    if not cfg.tam_warmup or runtime is None:
        return
    rt = runtime
    dof = int(getattr(rt, "_dof", 7))
    ring = TAM_HISTORY_SIZE
    step = 1000 // TAM_LATENT_RATE_HZ  # 1 kHz over one encoder period
    dt = 1e-3
    base_q = np.zeros(dof, dtype=np.float32)
    try:
        base_q[:] = np.asarray(HOME_POSE, dtype=np.float32)[:dof]
    except Exception:
        pass
    phase = np.linspace(0.0, np.pi, dof, dtype=np.float32)

    ts = np.zeros((0,), dtype=np.float64)
    q = np.zeros((0, dof), dtype=np.float32)
    qd = np.zeros((0, dof), dtype=np.float32)
    tau = np.zeros((0, dof), dtype=np.float32)
    grav = np.zeros((0, dof), dtype=np.float32)

    logger.info(
        "TAM warmup: compiling encoder (target < %.0f ms/push)...",
        cfg.tam_warmup_target_ms,
    )
    t_cur = 0.0
    streak = 0
    n_push = 0
    last_ms = float("nan")
    t_start = time.time()
    while time.time() - t_start < cfg.tam_warmup_timeout_s:
        tnew = t_cur + dt * np.arange(1, step + 1, dtype=np.float64)
        t_cur = float(tnew[-1])
        w = 2.0 * np.pi * 0.2  # 0.2 Hz gentle motion
        qn = base_q[None, :] + 0.1 * np.sin(w * tnew[:, None] + phase[None, :]).astype(np.float32)
        qdn = (0.1 * w * np.cos(w * tnew[:, None] + phase[None, :])).astype(np.float32)
        taun = (2.0 * np.sin(w * tnew[:, None] + phase[None, :])).astype(np.float32)
        gravn = np.tile(
            np.array([0, -25, 0, 18, 0, 2, 0][:dof], dtype=np.float32), (step, 1)
        )
        # Append and keep only the most recent `ring` samples, like the C++
        # ring returned by get_tam_history().
        ts = np.concatenate([ts, tnew])[-ring:]
        q = np.concatenate([q, qn])[-ring:]
        qd = np.concatenate([qd, qdn])[-ring:]
        tau = np.concatenate([tau, taun])[-ring:]
        grav = np.concatenate([grav, gravn])[-ring:]

        t0 = time.perf_counter()
        latent = rt.push_window(ts, q, qd, tau, gravity=grav)
        # Force the async JAX result so the timing reflects real compute.
        if latent is not None:
            block = getattr(latent, "block_until_ready", None)
            if block is not None:
                block()
            else:
                np.asarray(latent)
        last_ms = (time.perf_counter() - t0) * 1e3
        n_push += 1

        # Only count a push toward the fast streak once the ring is full, a
        # token was actually emitted (latent is not None), and it was fast --
        # so warmup can't succeed on empty or not-yet-compiled pushes.
        if ts.shape[0] >= ring and latent is not None and last_ms <= cfg.tam_warmup_target_ms:
            streak += 1
        else:
            streak = 0
        if streak >= cfg.tam_warmup_streak:
            logger.info("TAM warmup done: %d pushes, last %.1f ms", n_push, last_ms)
            break
    else:
        logger.warning(
            "TAM warmup did not stabilize within %.0fs (last %.1f ms); "
            "expect stalls at the start of the run",
            cfg.tam_warmup_timeout_s,
            last_ms,
        )
    rt.reset()  # discard synthetic history before the live stream


def run_history_encoder(
    robot: Franka, runtime: Any, stop_event: threading.Event | None = None
) -> None:
    """Feed the 1 kHz controller history through the TAM history encoder
    (~5 Hz) and push the resulting latent back into the C++ controller.
    Returns when ``stop_event`` is set (runs forever without one).

    The C++ side applies zero residual until both the MLP weights and the
    first latent have arrived, then ramps the residual in over 1 s."""
    # The weight vector carries the packed adaptor binary, one byte per
    # float64 element (parsed and validated on the C++ side).
    robot.set_tam_mlp_weight(
        np.frombuffer(runtime.adaptor_weight_bytes(), dtype=np.uint8).astype(np.float64)
    )

    hist_encoder_framerate = SimpleFrameRate(TAM_LATENT_RATE_HZ)
    latents_sent = 0
    while stop_event is None or not stop_event.is_set():
        try:
            # The runtime handles everything stream-related internally:
            # overlapping polls are deduplicated by timestamp, short holes
            # (e.g. a controller restart during a gain switch) are bridged
            # with masked padding on its dense grid, and a backwards or
            # over-long gap restarts the stream.
            latent = runtime.push_history_samples(robot.get_tam_history())
            if latent is not None:
                latent_arr = np.asarray(latent, dtype=np.float64).reshape((-1,))
                robot.set_tam_latent(latent_arr)
                latents_sent += 1
                if latents_sent == 1:
                    logger.info("TAM: first latent sent, residual ramps in on the controller")
        except Exception:
            logger.exception("TAM history encoder step failed; retrying")
            time.sleep(0.5)
        hist_encoder_framerate()
    logger.info("TAM history encoder stopped after %d latents", latents_sent)


def launch_history_encoder_thread(
    robot: Franka, runtime: Any, stop_event: threading.Event
) -> threading.Thread:
    """Start ``run_history_encoder`` on a daemon thread."""
    thread = threading.Thread(
        target=run_history_encoder,
        args=(robot, runtime, stop_event),
        name="tam-history-encoder",
        daemon=True,
    )
    thread.start()
    return thread
