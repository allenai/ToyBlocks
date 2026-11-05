"""Serve a ManiFlow policy over WebSocket using :class:`PolicyServer`.

Loads the ManiFlow low-level model from the ``maniflow-tb`` submodule (the ``hvla``
package) and serves it with ``blocksuite.serving.{base_policy,policy_server}``, plus the
blocksuite obs/action plumbing: batched obs pass-through, ``list[str]`` ``task_name``
forwarding, and the ``(B, T, D) -> (T, B, D)`` action transpose for
:class:`ActionChunkBroker`.

High-level conditioning is selected via ``--hl``:
* ``oracle`` (default): LL-only batched :class:`ManiFlowInferAdapter`; the client
  injects ``oracle_task_name``/``target_heatmap`` (client-side oracle HL).
* ``model``: batched :class:`HLLLInferAdapter` that runs a Molmo2 HL model
  (one batched call over the envs due an update) to produce the language +
  target heatmap when the client does not inject its own, then the LL over the
  whole batch. Requires ``--hl_checkpoint``.

When bumping the maniflow-tb submodule, check its policy/model and normalization-stats
changes against this file.
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import socket
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import torch
import tyro

from blocksuite.serving.base_policy import BasePolicy
from blocksuite.serving.policy_server import PolicyServer

# hl_question_with_prev_action is re-exported: callers and tests import it from here.
from blocksuite.serving.molmo2_hl import Molmo2HLPolicy, hl_question_with_prev_action  # noqa: F401

if TYPE_CHECKING:
    # Annotation-only, and this module's ``from __future__ import annotations``
    # keeps it that way. Imported lazily for the same reason
    # ``maniflow_serving_utils`` defers hvla: the adapters below are plain
    # numpy/torch over an already-built policy object, so ``tests/serving``
    # covers them on a CPU box where the maniflow extra is not installed. The
    # real weights arrive through ``load_maniflow_policy_common``, which imports
    # hvla itself at call time.
    from hvla.model.modules.maniflow.flow_policy import ManiFlowMultiModalPolicy


from blocksuite.serving.maniflow_serving_utils import (
    DEFAULT_TARGET_SIGMA,
    PLACE_ON_TABLE_ACTION,
    _build_heatmap,
    _parse_target_xy,
    _pil_to_tensor,
    _strip_coords,
    client_advertised_obs_keys,
    format_task_name,
    inject_target_front_crop_from_native,
    inject_target_front_crop_rgb,
    ManiFlowNormalizationMixin,
    load_maniflow_policy_common,
    postprocess_actions,
    select_and_validate_obs,
    state_norm_metadata,
    summarize_model_inputs,
)


def _downsample_heatmap(hm: np.ndarray, size: int = 56) -> np.ndarray:
    """Stride-subsample a 2D heatmap to ``~size x size`` for a cheap viz aux.

    The client viz (record_episode) normalizes + resizes the map to the frame,
    so a coarse map is sufficient and keeps the msgpack round-trip small.
    """
    hm = np.asarray(hm, dtype=np.float32)
    h, w = hm.shape[-2], hm.shape[-1]
    step = max(1, min(h, w) // size)
    return np.ascontiguousarray(hm[::step, ::step])


def _maybe_render_hl_input_trace(
    input_trace: Any,
    *,
    front_rgb: Any,
    gripper_rgb: Any,
    question: str,
    prev_action: str,
    hl_output: str,
    sim_time_ms: Any,
    episode_id: Any,
) -> None:
    """Dump the exact Molmo2 question when the client asked for an HL trace.

    Called at the HL invocation, not the LL handover, so a K-gated run writes
    one panel per Molmo2 forward rather than one per action chunk.
    """
    if input_trace is None or "hl_dir" not in input_trace:
        return
    if sim_time_ms is None or episode_id is None:
        raise ValueError(
            "an HL policy-input trace was requested via '__ll_input_trace__' "
            "'hl_dir' but the request carried no 'hl_sim_time_ms' / "
            "'hl_episode_id', which are the keys each panel is named and "
            "ordered by. Only the blocksuite eval client stamps them; drop "
            "the trace request or use that client."
        )
    from blocksuite.serving.policy_input_trace import render_hl_input_panel

    env_idx = int(input_trace["env_idx"])
    flat_time = np.asarray(sim_time_ms).reshape(-1)
    flat_episode = np.asarray(episode_id).reshape(-1)
    png = render_hl_input_panel(
        front_rgb=front_rgb,
        gripper_rgb=gripper_rgb,
        question=question,
        prev_action=prev_action,
        hl_output=hl_output,
        out_dir=str(input_trace["hl_dir"]),
        env_idx=env_idx,
        sim_time_ms=float(flat_time[env_idx if flat_time.size > 1 else 0]),
        episode_id=int(flat_episode[env_idx if flat_episode.size > 1 else 0]),
    )
    logging.info("Wrote HL policy-input trace panel: %s", png)


def _maybe_render_input_trace(
    input_trace: Any,
    torch_obs: dict[str, Any],
    *,
    sim_time_ms: Any,
    episode_id: Any,
    raw_state: np.ndarray | None,
    dropped: list[str],
) -> None:
    """Dump the exact ``predict_action`` input when the client asked for a trace.

    Called after ``select_and_validate_obs`` and before the model, so the panel
    is the dict the model is about to receive. ``__ll_input_trace__`` rides on
    the observation (see ``PolicyClientAdapter.set_policy_input_trace``), so a
    request with no sim-time stamp means the client is not the eval harness and
    the rows would have no keys -- that fails rather than guessing an index.
    """
    if input_trace is None:
        return
    if sim_time_ms is None or episode_id is None:
        raise ValueError(
            "a policy-input trace was requested via '__ll_input_trace__' but the "
            "request carried no 'hl_sim_time_ms' / 'hl_episode_id', which are the "
            "keys each panel is named and ordered by. Only the blocksuite eval "
            "client stamps them; drop the trace request or use that client."
        )
    from blocksuite.serving.policy_input_trace import render_ll_input_panel

    env_idx = int(input_trace["env_idx"])
    # The stamps arrive per-env from the batched client and as a single value
    # from the unbatched one (B == 1), so read them at the env the panel slices.
    flat_time = np.asarray(sim_time_ms).reshape(-1)
    flat_episode = np.asarray(episode_id).reshape(-1)
    png = render_ll_input_panel(
        torch_obs,
        out_dir=str(input_trace["dir"]),
        env_idx=env_idx,
        sim_time_ms=float(flat_time[env_idx if flat_time.size > 1 else 0]),
        episode_id=int(flat_episode[env_idx if flat_episode.size > 1 else 0]),
        raw_state=raw_state,
        dropped=dropped,
        summary=summarize_model_inputs(torch_obs),
    )
    logging.info("Wrote LL policy-input trace panel: %s", png)


def _maniflow_server_metadata(
    *,
    action_mode: str,
    obs_shapes: dict[str, Any],
    model_input_spec: dict[str, Any],
    stats: dict[str, Any] | None,
) -> dict[str, Any]:
    """The metadata keys both ManiFlow serving adapters advertise.

    ``state_dim`` is read off the model input spec rather than a built encoder:
    it is the width this server validates ``agent_pos`` against in
    ``select_and_validate_obs``, and the client derives its entire state layout
    from it, so the two disagreeing is a silently wrong state vector rather than
    a shape error.
    """
    meta: dict[str, Any] = {
        "action_mode": action_mode,
        # The checkpoint's declared obs keys, so the client's obs transform can
        # configure optional modalities (e.g. depth) from the model that is
        # actually being served instead of a hand-set flag.
        "obs_keys": client_advertised_obs_keys(obs_shapes),
        # The text format the checkpoint was trained on, so whoever builds
        # `task_name` upstream (oracle HL controller, obs transform) matches it
        # instead of guessing. None for a non-language-conditioned model.
        "language_mode": model_input_spec.get("language_mode"),
        "state_dim": model_input_spec.get("state_dim"),
        # State bands, so the policy-input panels can show each raw dim against
        # the range the server normalizes it into.
        "state_norm": state_norm_metadata(stats),
        "use_target_front_crop": int(model_input_spec.get("use_target_front_crop", -1)),
        "action_dim8": model_input_spec.get("action_dim8", "pd_mode"),
    }
    pc_bounds = model_input_spec.get("per_camera_ptmap_bounds")
    if pc_bounds is not None:
        meta["per_camera_ptmap_bounds"] = pc_bounds
    return meta

def _save_first_batch_server(
    save_dir_str: str, torch_obs: dict[str, Any], action: torch.Tensor
) -> None:
    """Dump the model-ready batch (post-normalization, post-validation) to disk.

    Reuses training's ``save_first_batch_data`` so the output format is
    byte-for-byte identical to what training produces. The action tensor is the
    raw (normalized) model output before denormalization, matching training.
    """
    from accelerate import PartialState

    from hvla.training.trainer_utils.trainer_tools import save_first_batch_data

    # save_first_batch_data uses accelerate's logger which needs PartialState.
    if not PartialState._shared_state:
        PartialState()

    batch: dict[str, Any] = {
        "obs": torch_obs,
        "action": action,
        "lang": torch_obs.get("task_name", []),
    }

    # save_first_batch_data writes to <output_dir>/first_batch_data; pass the parent.
    output_dir = str(pathlib.Path(save_dir_str).parent)
    save_first_batch_data(output_dir, batch)
    logging.info("  [first_batch] saved to %s", save_dir_str)


class ManiFlowInferAdapter(BasePolicy, ManiFlowNormalizationMixin):
    """Adapts ManiFlow policy for PolicyServer.

    Args:
        action_mode: ``"relative"`` (default) – the model was trained on
            per-step joint deltas (``relative_joint_target``); the server
            converts to absolute targets via cumsum + current state.
            ``"absolute"`` – the model was trained on absolute joint targets
            (``joint_target``); the server emits them directly.
    """

    VALID_ACTION_MODES = ("relative", "absolute")

    def __init__(
        self,
        policy: ManiFlowMultiModalPolicy | None,
        stats: dict[str, Any] | None = None,
        action_mode: str = "relative",
        obs_video_dir: str | None = None,
        obs_video_fps: int = 10,
        replicate_noise: bool = False,
        model_input_spec: dict[str, Any] | None = None,
    ) -> None:
        if action_mode not in self.VALID_ACTION_MODES:
            raise ValueError(
                f"action_mode must be one of {self.VALID_ACTION_MODES}, got {action_mode!r}"
            )
        self._policy = policy
        # None (--skip_ll): metadata only; the client sends hl_only requests.
        if policy is not None:
            policy.eval()
            policy.to("cuda" if torch.cuda.is_available() else "cpu")
        self._obs_video_dir = obs_video_dir
        self._obs_video_fps = obs_video_fps
        self._obs_frames: list[np.ndarray] = []
        self._obs_video_counter = 0
        self._first_inference_logged = False
        # Model input contract derived from the (ptmap-stripped) shape_meta; used
        # to select + validate client obs right before predict_action.
        self._model_input_spec = model_input_spec or {}
        self._obs_shapes = self._model_input_spec.get("obs_shapes", {})
        self._conditioning_keys = self._model_input_spec.get("conditioning_keys", frozenset())
        self._use_target_front_crop = int(
            self._model_input_spec.get("use_target_front_crop", -1)
        )
        self._dropped_logged = False
        self._target_crop_logged = False
        self._replicate_noise = replicate_noise
        if replicate_noise:
            self._patch_replicate_noise()
        # _init_stats sets these; bind them first so `metadata` (read on every
        # client connect) works for a stats-less adapter instead of raising.
        self._stats = None
        self._action_mode = action_mode
        if stats:
            self._init_stats(stats, action_mode)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "policy_class": self._policy.__class__.__name__,
            "supports_batched_obs": True,
            "has_action_stats": self._stats is not None,
            **_maniflow_server_metadata(
                action_mode=self._action_mode,
                obs_shapes=self._obs_shapes,
                model_input_spec=self._model_input_spec,
                stats=self._stats,
            ),
        }

    def _patch_replicate_noise(self) -> None:
        """Replace ``conditional_sample`` so every sample in the batch starts
        from the same initial noise.  This makes identical observations produce
        identical actions regardless of batch position."""
        policy = self._policy
        original_sample_ode = policy.sample_ode

        def _replicated_conditional_sample(condition_data, **kwargs):
            B = condition_data.shape[0]
            single_shape = (1,) + condition_data.shape[1:]
            noise_one = torch.randn(
                size=single_shape,
                dtype=condition_data.dtype,
                device=condition_data.device,
            )
            noise = noise_one.expand(B, -1, -1).contiguous()
            ode_traj = original_sample_ode(
                x0=noise,
                N=policy.num_inference_steps,
                **kwargs,
            )
            return ode_traj[-1]

        policy.conditional_sample = _replicated_conditional_sample
        logging.info("replicate_noise enabled: patched policy.conditional_sample")

    def reset(self) -> None:
        self._flush_obs_video()

    def _capture_obs_frame(self, obs: dict[str, Any]) -> None:
        """Stash a side-by-side (front | gripper) uint8 frame for video."""
        if self._obs_video_dir is None:
            return
        front = obs.get("front_rgb")
        gripper = obs.get("gripper_rgb")
        if front is None and gripper is None:
            return
        panels: list[np.ndarray] = []
        for img in (front, gripper):
            if img is None:
                continue
            a = np.asarray(img)
            # (B, 1, C, H, W) or (1, C, H, W) or (C, H, W) float [0,1] → HWC uint8
            while a.ndim > 3:
                a = a[0]
            if a.shape[0] in (1, 3):  # CHW
                a = np.transpose(a, (1, 2, 0))
            a = np.clip(a * 255, 0, 255).astype(np.uint8)
            if a.shape[-1] == 1:
                a = np.repeat(a, 3, axis=-1)
            panels.append(a)
        if panels:
            self._obs_frames.append(np.concatenate(panels, axis=1))

    def _flush_obs_video(self) -> None:
        if not self._obs_frames or self._obs_video_dir is None:
            self._obs_frames = []
            return
        import imageio

        out_dir = pathlib.Path(self._obs_video_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"server_obs_{self._obs_video_counter:04d}.mp4"
        writer = imageio.get_writer(str(path), fps=self._obs_video_fps, codec="libx264")
        for frame in self._obs_frames:
            writer.append_data(frame)
        writer.close()
        logging.info("Saved %d-frame obs video to %s", len(self._obs_frames), path)
        self._obs_frames = []
        self._obs_video_counter += 1

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        ep_seed = obs.pop("__episode_seed__", None)
        if ep_seed is not None:
            seed_val = int(ep_seed)
            torch.manual_seed(seed_val)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed_val)
            logging.info("Per-episode re-seed: %d", seed_val)

        # Client requests a first-batch dump by injecting this key (uint8-encoded path).
        save_first_batch_raw = obs.pop("__save_first_batch_dir__", None)
        save_first_batch_dir: str | None = None
        if save_first_batch_raw is not None:
            save_first_batch_dir = bytes(np.asarray(save_first_batch_raw, dtype=np.uint8)).decode("utf-8")

        # HL-latency bookkeeping keys are forwarded by ManiflowObsTransform for
        # the ``--hl model`` server; this LL-only adapter has no HL buffer, so
        # take them out here rather than let the ndarray branch below coerce them
        # into tensors passed to predict_action. They are kept (not discarded)
        # because they are the row keys of the policy-input trace below.
        trace_sim_time_ms = obs.pop("hl_sim_time_ms", None)
        trace_episode_id = obs.pop("hl_episode_id", None)
        obs.pop("hl_env_index", None)
        input_trace = obs.pop("__ll_input_trace__", None)
        # Only the --hl model adapter crops from the native frame (before calling here).
        obs.pop("native_front_rgb", None)

        self._capture_obs_frame(obs)
        device = next(self._policy.parameters()).device
        dtype = next(self._policy.parameters()).dtype

        raw_state: np.ndarray | None = None
        torch_obs: dict[str, Any] = {}
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                if k == "agent_pos":
                    raw_state = v.copy()
                    if self._stats:
                        v = self._normalize_state(v)
                # Legacy unbatched demo clients send raw HWC uint8 images and
                # 1-D state; add a leading (1, 1) to reach the policy format.
                # Properly shaped batched inputs (e.g. from ManiflowObsTransform
                # which emits (B, 1, 3, H, W) float CHW and (B, 1, D) state, for
                # D in {8, 9, 19}) pass through untouched.
                if v.ndim == 3 and v.dtype == np.uint8 and v.shape[-1] == 3:
                    t = _pil_to_tensor(v).to(device=device, dtype=dtype)
                    t = t.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, H, W)
                elif v.ndim == 1:
                    t = torch.from_numpy(v).to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)  # (1, 1, D)
                else:
                    t = torch.from_numpy(v).to(device=device, dtype=dtype)
                torch_obs[k] = t
            elif torch.is_tensor(v):
                torch_obs[k] = v.to(device=device, dtype=dtype)
            elif isinstance(v, (list, tuple)):
                # Forward lists/tuples (e.g. task_name = [prompt] * B for
                # language-conditioned ManiFlow). dict_apply inside
                # predict_action has a list branch that leaves non-tensor
                # items untouched, so list[str] passes through cleanly.
                torch_obs[k] = list(v)
            elif k in self._conditioning_keys:
                # Dropping a conditioning key would run the model unconditioned and
                # report the result as a conditioned one. ManiflowObsTransform emits
                # task_name as list[str] of length B; a bare str (legacy client) is
                # ambiguous for B > 1, so demand the batched form instead of
                # broadcasting a guess.
                raise TypeError(
                    f"Conditioning key {k!r} arrived as {type(v).__name__} "
                    f"({v!r}), which predict_action cannot consume. Send it as a "
                    "list[str] of length B (one entry per env)."
                )
            # Everything else is dropped (metadata, raw strings): dict_apply would
            # index them as tensors and crash.

        # Client-side HL conditioning (--use_oracle): the eval
        # harness injects a per-env language prompt (task_name) and spatial goal
        # (target_heatmap) via ManiflowObsTransform. The passthrough above already
        # forwards them (list[str] and ndarray), but guarantee the heatmap is a
        # (B, 1, 3, 224, 224) float32 tensor so batched B>1 conditioning matches
        # the HLLLInferAdapter (--hl model) exactly rather than relying on
        # incidental dtypes.
        hm = torch_obs.get("target_heatmap")
        if hm is not None and torch.is_tensor(hm):
            torch_obs["target_heatmap"] = hm.to(device=device, dtype=torch.float32)

        # Exact per-inference dump of what the LL (ManiFlow) model is fed. The HL
        # side lives client-side for this server (the oracle/Molmo controller
        # injects task_name + target_heatmap via ManiflowObsTransform), so these
        # two keys ARE the high-level conditioning; everything else is the LL's
        # own proprio/vision. peaks= shows where each target_heatmap Gaussian
        # lands so a wrong spatial goal is visible.
        crop_from_heatmap = "target_front_crop_rgb" not in torch_obs
        inject_target_front_crop_rgb(torch_obs, self._use_target_front_crop)
        if self._use_target_front_crop >= 0 and crop_from_heatmap and not self._target_crop_logged:
            self._target_crop_logged = True
            logging.info(
                "Deriving target_front_crop_rgb on the server (bottom_pct=%d) from "
                "front_rgb + target_heatmap; pass --enable_crop on the eval client "
                "for training-parity crops from native front + target_xy.",
                self._use_target_front_crop,
            )

        logging.info(
            "LL (ManiFlow) inference inputs [action_mode=%s]:\n%s",
            self._action_mode,
            summarize_model_inputs(torch_obs),
        )

        # Server is authoritative: drop keys the model doesn't declare (e.g. the
        # zeroed ptmaps the client always sends) and hard-raise on a missing or
        # wrong-shape obs before it explodes deep inside the model.
        torch_obs, dropped = select_and_validate_obs(
            torch_obs, self._obs_shapes, self._conditioning_keys
        )
        if dropped and not self._dropped_logged:
            logging.info("Dropping obs keys not in model shape_meta: %s", dropped)
            self._dropped_logged = True

        _maybe_render_input_trace(
            input_trace,
            torch_obs,
            sim_time_ms=trace_sim_time_ms,
            episode_id=trace_episode_id,
            raw_state=raw_state,
            dropped=dropped,
        )

        # bf16 autocast, as the checkpoint was trained (bf16 mixed precision).
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
            result = self._policy.predict_action(torch_obs)
        result = {k: v.float() if torch.is_tensor(v) else v for k, v in result.items()}

        if save_first_batch_dir is not None:
            _save_first_batch_server(save_first_batch_dir, torch_obs, result["action"])

        actions = result["action"].detach().cpu().numpy()
        if self._stats:
            actions = self._denormalize_action(actions)
        actions = postprocess_actions(
            actions,
            self._action_mode,
            raw_state,
            self._model_input_spec.get("action_dim8", "pd_mode"),
        )

        if not self._first_inference_logged:
            self._first_inference_logged = True
            raw_range = (result["action"].min().item(), result["action"].max().item())
            logging.info(
                "First inference: raw shape=%s range=[%.4f, %.4f], "
                "post-denorm arm range=[%.4f, %.4f], "
                "gripper values=%s, %s values=%s, action_mode=%s",
                tuple(result["action"].shape),
                *raw_range,
                actions[..., :7].min(),
                actions[..., :7].max(),
                np.unique(actions[..., 7]).tolist(),
                # dim 8 is pd_mode or play_mode depending on the checkpoint.
                self._model_input_spec.get("action_dim8", "pd_mode"),
                np.unique(actions[..., 8]).tolist(),
                self._action_mode,
            )
            tn = torch_obs.get("task_name")
            hm = torch_obs.get("target_heatmap")
            logging.info(
                "First inference HL conditioning: task_name=%s target_heatmap=%s",
                tn if isinstance(tn, list) else type(tn).__name__,
                (tuple(hm.shape), str(hm.dtype)) if torch.is_tensor(hm) else None,
            )

        return {"actions": actions}


# Pre-registered HL-latency slots on the server-side adapter, indexed by the
# client's ``hl_env_index``. ``launch_eval.py`` rejects ``--policy
# molmo2_maniflow`` with ``--num-envs`` above this; the server does not learn
# ``num_envs`` at construction.
_HL_SERVER_MAX_ENVS = 128


def _per_row(value: Any, batch: int, dtype: Any) -> np.ndarray | None:
    """A per-request control stamp as one value per batch row (a single value broadcasts)."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=dtype).reshape(-1)
    if arr.size == 1 and batch > 1:
        arr = np.repeat(arr, batch)
    if arr.size != batch:
        raise ValueError(f"control stamp has {arr.size} values for a batch of {batch}")
    return arr


def _row_frame(frames: Any, row: int, key: str) -> np.ndarray:
    """Row ``row`` of a batched ``(B, H, W, 3)`` native camera frame (or the one unbatched frame)."""
    if torch.is_tensor(frames):
        frames = frames.detach().cpu().numpy()
    arr = np.asarray(frames)
    if arr.ndim == 3:
        if row != 0:
            raise ValueError(f"{key} is a single frame but row {row} was requested")
        return arr
    if arr.ndim == 4:
        return arr[row]
    raise ValueError(f"{key}: expected (B, H, W, 3) or (H, W, 3), got shape {arr.shape}")


class HLLLInferAdapter(BasePolicy):
    """Batched HL (Molmo2) + LL (ManiFlow) server adapter.

    One request carries any number of envs (``hl_env_index`` names them). The
    Molmo2 HL runs once, batched, over the envs due an HL update, and the LL
    runs once over the whole batch through the wrapped
    :class:`ManiFlowInferAdapter`, with the HL's ``task_name`` +
    ``target_heatmap`` injected -- exactly the path a client-side oracle HL
    takes, so the LL sees the same tensors either way.

    When the client injects its own HL conditioning (``target_heatmap`` in the
    obs, e.g. the eval harness ``--use_oracle`` path) the request goes straight
    to the LL and Molmo2 is skipped.

    The HL sees the client's native camera frames (``hl_front_rgb`` /
    ``hl_gripper_rgb``, ``(B, H, W, 3)`` uint8), not the LL's 224x224 squares:
    it was trained on full 640x360 frames and emits targets in that space. Which
    cameras, the front crop and the goal phrasing come from the HL checkpoint
    (:class:`~blocksuite.serving.molmo2_hl.HLContract`) and are advertised in
    :attr:`metadata`; the client builds each env's goal prompt (``hl_goal``) to
    that phrasing, and ``--hl_prompt`` overrides it.

    Each Molmo2 call feeds back the env's previous raw HL string as
    ``Previous action: {prev}. {goal}``, matching TAMP ``--use_prev_action_str``;
    the first call of an episode is the bare goal.

    By default every request re-runs the HL. A client that sends ``hl_refresh``
    (one bool per row) gets sim2real's ``async_molmo`` instead: the HL runs only
    for rows that ask for it (the LL just finished a skill) or have no HL output
    yet this episode, and the rest reuse their last output. The LL-facing ``task_name``
    is rebuilt from the HL output in the LL checkpoint's own ``language_mode``
    (see :meth:`_ll_task_name`).
    """

    def __init__(
        self,
        ll_adapter: ManiFlowInferAdapter | None = None,
        hl_policy: Molmo2HLPolicy | None = None,
        hl_prompt: str = "",
        orig_cam_w: int = 640,
        orig_cam_h: int = 360,
        target_sigma: float = DEFAULT_TARGET_SIGMA,
        hl_update_interval_ms: float = 0.0,
        hl_update_interval_jitter_ms: float = 0.0,
        estimated_hl_inference_time_ms: float = 0.0,
        seed: int | None = None,
        *,
        ll_policy: ManiFlowMultiModalPolicy | None = None,
        stats: dict[str, Any] | None = None,
        action_mode: str = "absolute",
        model_input_spec: dict[str, Any] | None = None,
    ) -> None:
        # Pre-batching constructor (``ll_policy`` + ``stats`` / ``action_mode`` /
        # ``model_input_spec``): wrap the raw LL policy in its adapter here.
        if ll_adapter is None:
            if ll_policy is None:
                raise TypeError("HLLLInferAdapter needs ll_adapter (or ll_policy)")
            ll_adapter = ManiFlowInferAdapter(
                ll_policy,
                stats=stats,
                action_mode=action_mode,
                model_input_spec=model_input_spec,
            )
        elif ll_policy is not None or stats is not None or model_input_spec is not None:
            raise TypeError(
                "pass either ll_adapter or ll_policy (+ stats / model_input_spec), not both"
            )
        if hl_policy is None:
            raise TypeError("HLLLInferAdapter needs hl_policy")
        self._ll = ll_adapter
        self._hl_policy = hl_policy
        # Optional manual override for the goal fed to the Molmo2 HL model.
        self._hl_prompt_override = hl_prompt or None
        spec = ll_adapter._model_input_spec
        # The LL's text contract: this adapter builds `task_name` from the HL
        # string, so it must format to the mode the LL was trained on.
        self._language_mode = spec.get("language_mode")
        # A checkpoint trained with use_previous_front_frame reads the M-aged
        # front frame the HL was invoked on as ``prev_front_rgb``.
        self._wants_prev_front_frame = "prev_front_rgb" in spec.get("obs_shapes", {})
        self._orig_cam_w = orig_cam_w
        self._orig_cam_h = orig_cam_h
        self._target_sigma = target_sigma

        # High-level latency (K = update interval, M = estimated HL inference
        # time), both in milliseconds, per env. infer() is called once per
        # action_horizon steps (chunk broker), so K/M here are quantized to
        # action_horizon * step_dt and a K below that is a no-op. See
        # blocksuite.serving.hl_latency.HLLatencyBuffer.
        from blocksuite.serving.hl_latency import HLLatencyBuffer

        self._hl_latency = (
            HLLatencyBuffer(
                update_interval_ms=hl_update_interval_ms,
                estimated_inference_time_ms=estimated_hl_inference_time_ms,
                num_envs=_HL_SERVER_MAX_ENVS,
                update_interval_jitter_ms=hl_update_interval_jitter_ms,
                seed=seed,
            )
            if (hl_update_interval_ms > 0 or estimated_hl_inference_time_ms > 0)
            else None
        )
        self._last_episode_id: dict[int, int] = {}
        # Last HL output of the current episode that carried a real instruction
        # (i.e. not "done"). Reused to condition the LL on the terminating step,
        # since the LL never saw "done" as an instruction during training (same
        # rationale as OracleHLController's repeat-last-instruction).
        self._last_lang: dict[int, str] = {}
        self._last_xy: dict[int, np.ndarray | None] = {}
        # Last raw Molmo2 output per env, fed back as ``Previous action:`` on
        # the next HL call. Separate from ``_last_lang`` (coords-stripped, LL-facing).
        self._prev_action_str: dict[int, str] = {}
        # What the last Molmo2 call actually received, for the debug video /
        # HL input panel between K-gated updates.
        self._hl_prev_provided: dict[int, str] = {}
        self._hl_question_sent: dict[int, str] = {}

    @property
    def exclusive(self) -> bool:
        """One client at a time: this server holds every env's HL state keyed by
        ``hl_env_index``, so a second client would interleave a different sim
        clock into the same per-env slots. The server rejects the second
        connection instead (see BasePolicy.exclusive).
        """
        return True

    @property
    def metadata(self) -> dict[str, Any]:
        # The shared keys describe the LL: on the client-injected HL path the
        # client owns `task_name`, so it needs the LL's text contract.
        meta = dict(self._ll.metadata)
        contract = getattr(self._hl_policy, "contract", None)
        meta.update(
            {
                "policy_class": "HLLLInferAdapter",
                "ll_policy_class": meta.get("policy_class"),
                "hl_policy_class": type(self._hl_policy).__name__,
                "supports_batched_obs": True,
                # The client sends the HL its native frames (hl_front_rgb /
                # hl_gripper_rgb) and the goal in the checkpoint's phrasing (hl_goal).
                "hl_native_frames": True,
                "hl_front_view_only": bool(getattr(contract, "front_view_only", False)),
                "hl_prompt_mode": getattr(contract, "prompt_mode", None),
                "hl_html_v2_variant": getattr(contract, "html_v2_variant", None),
            }
        )
        return meta

    def reset(self) -> None:
        """Drop all per-run HL state (the server calls this when a connection ends).

        The policy object outlives any single WebSocket connection, so without
        this the next client would inherit the previous one's aged buffer and be
        conditioned on another run's HL output.
        """
        if self._hl_latency is not None:
            self._hl_latency.reset_all()
        self._last_episode_id.clear()
        self._last_lang.clear()
        self._last_xy.clear()
        self._prev_action_str.clear()
        self._hl_prev_provided.clear()
        self._hl_question_sent.clear()
        self._ll.reset()

    def _start_episode(self, env_idx: int) -> None:
        """Forget ``env_idx``'s HL history: its cached instruction belongs to the finished episode."""
        if self._hl_latency is not None:
            self._hl_latency.reset(env_idx)
        self._last_lang.pop(env_idx, None)
        self._last_xy.pop(env_idx, None)
        self._prev_action_str.pop(env_idx, None)
        self._hl_prev_provided.pop(env_idx, None)
        self._hl_question_sent.pop(env_idx, None)

    def _resolve_hl_goal(self, obs: dict[str, Any], hl_goal: Any, row: int) -> str:
        """Goal prompt fed to the Molmo2 HL model for batch row ``row``.

        Prefers the ``--hl_prompt`` manual override, then the client's
        ``hl_goal`` (built in the checkpoint's phrasing, see
        ``HLContract.goal_prompt``), then the row's ``task_name``. Fails loudly
        when none is available rather than substituting a placeholder prompt.
        """
        if self._hl_prompt_override:
            return self._hl_prompt_override
        for source in (hl_goal, obs.get("task_name")):
            if source is None:
                continue
            values = [str(v) for v in np.asarray(source, dtype=object).reshape(-1)]
            lang = values[row] if len(values) > 1 else (values[0] if values else "")
            if lang:
                return lang
        raise ValueError(
            "HL model has no language goal: the client sent no 'hl_goal' / "
            "'task_name' and no --hl_prompt override was set. Pass --hl_prompt to "
            "force a goal, or ensure the eval client forwards the per-episode prompt."
        )

    def _ll_task_name(self, lang: str, target_xy: np.ndarray | None) -> str:
        """The LL-facing ``task_name`` for a Molmo2 HL output, in the LL's format.

        Molmo2 emits the action and its target together (``"pick <409, 210>"``),
        and this adapter splits them into a phrase plus a heatmap. That split is
        only right for a heatmap-conditioned LL: an ``action_goal`` checkpoint
        kept the target *in the language*, and a checkpoint whose
        ``target_injection_mode`` is unset never receives the heatmap at all, so
        stripping the coordinates would leave it with no spatial goal.

        A missing point is not rewritten here: ``format_task_name`` returns the
        bare phrase (sim2real ``format_language_goal``), used for coord-free
        ``place on table`` and the ``done``-with-nothing-cached path.
        """
        if self._language_mode is None:
            # Not language-conditioned: select_and_validate_obs drops task_name
            # before inference, and only the HL prompt resolver reads it.
            return lang
        return format_task_name(lang, target_xy, self._language_mode)

    def _cond_frame(self, obs: dict[str, Any], row: int) -> np.ndarray | None:
        """Row ``row``'s model-ready front frame, cached with a fresh HL output for ``prev_front_rgb``.

        Stored in the client's already-model-ready layout so replaying it later
        needs no re-preprocessing (and so the aged frame is bit-identical to the
        ``front_rgb`` the LL saw on that step).
        """
        if not self._wants_prev_front_frame:
            return None
        front = obs.get("front_rgb")
        if front is None:
            raise ValueError(
                "the served checkpoint consumes 'prev_front_rgb' but the client sent "
                "no 'front_rgb' to cache as the aged conditioning frame."
            )
        if torch.is_tensor(front):
            front = front.detach().cpu().numpy()
        return np.array(np.asarray(front)[row], dtype=np.float32, copy=True)

    def _run_hl(
        self,
        obs: dict[str, Any],
        rows: list[int],
        env_ids: np.ndarray,
        *,
        hl_goal: Any,
        hl_front: Any,
        hl_gripper: Any,
        sim_time_ms: np.ndarray | None,
        episode_id: np.ndarray | None,
        input_trace: Any,
    ) -> dict[int, str]:
        """One batched Molmo2 call for batch rows ``rows``; returns ``{row: raw HL string}``."""
        if not hasattr(self._hl_policy, "generate"):
            return self._run_single_call_hl(
                obs, rows, env_ids, hl_goal=hl_goal, sim_time_ms=sim_time_ms
            )
        if hl_front is None:
            raise ValueError(
                "the Molmo2 HL needs the native front frame ('hl_front_rgb', (B, H, W, 3)); "
                "the LL's 224x224 front_rgb is a squashed view it was never trained on. "
                "Use an eval client that reads 'hl_native_frames' from the server metadata."
            )
        contract = self._hl_policy.contract
        questions: list[str] = []
        images: list[list[np.ndarray]] = []
        for r in rows:
            env = int(env_ids[r])
            goal = self._resolve_hl_goal(obs, hl_goal, r)
            questions.append(contract.question(goal, self._prev_action_str.get(env, "")))
            gripper = None if hl_gripper is None else _row_frame(hl_gripper, r, "hl_gripper_rgb")
            images.append(self._hl_policy.images_for(_row_frame(hl_front, r, "hl_front_rgb"), gripper))
        raws = self._hl_policy.generate(images, questions)
        out: dict[int, str] = {}
        for r, raw, question, ims in zip(rows, raws, questions, images):
            env = int(env_ids[r])
            prev = self._prev_action_str.get(env, "")
            logging.info("HL (Molmo2) env %d: %r -> %r", env, question, raw)
            raw, self._prev_action_str[env] = contract.parse_output(raw)
            self._hl_prev_provided[env] = prev
            self._hl_question_sent[env] = question
            out[r] = raw
            if self._hl_latency is not None:
                self._hl_latency.commit(
                    env,
                    float(sim_time_ms[r]),  # type: ignore[index]
                    {"lang": _strip_coords(raw), "xy": _parse_target_xy(raw), "raw": raw},
                    self._cond_frame(obs, r),
                )
            if input_trace is not None and env == int(input_trace.get("env_idx", -1)):
                _maybe_render_hl_input_trace(
                    input_trace,
                    front_rgb=ims[0],
                    # A front-only checkpoint gets no gripper frame: drawn black.
                    gripper_rgb=ims[1] if len(ims) > 1 else np.zeros_like(ims[0]),
                    question=question,
                    prev_action=prev,
                    hl_output=raw,
                    sim_time_ms=sim_time_ms,
                    episode_id=episode_id,
                )
        return out

    def _run_single_call_hl(
        self,
        obs: dict[str, Any],
        rows: list[int],
        env_ids: np.ndarray,
        *,
        hl_goal: Any,
        sim_time_ms: np.ndarray | None,
    ) -> dict[int, str]:
        """:meth:`_run_hl` for an HL with only the pre-batching ``get_action(obs, lang)`` surface.

        One call per due row on the whole request ``obs`` (no native HL frames,
        contract, or input trace), with the ``Previous action:`` question.
        """
        out: dict[int, str] = {}
        for r in rows:
            env = int(env_ids[r])
            prev = self._prev_action_str.get(env, "")
            question = hl_question_with_prev_action(self._resolve_hl_goal(obs, hl_goal, r), prev)
            raw = str(self._hl_policy.get_action(obs, lang=question))  # type: ignore[attr-defined]
            logging.info("HL (single-call) env %d: %r -> %r", env, question, raw)
            self._prev_action_str[env] = raw
            self._hl_prev_provided[env] = prev
            self._hl_question_sent[env] = question
            out[r] = raw
            if self._hl_latency is not None:
                self._hl_latency.commit(
                    env,
                    float(sim_time_ms[r]),  # type: ignore[index]
                    {"lang": _strip_coords(raw), "xy": _parse_target_xy(raw), "raw": raw},
                    self._cond_frame(obs, r),
                )
        return out

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        # HL-only inputs never reach the LL.
        hl_refresh = obs.pop("hl_refresh", None)
        hl_only = bool(obs.pop("hl_only", False))
        hl_goal = obs.pop("hl_goal", None)
        hl_front = obs.pop("hl_front_rgb", None)
        hl_gripper = obs.pop("hl_gripper_rgb", None)

        # 1. The client may inject its own HL (--use_oracle): with a
        # target_heatmap present the LL runs on it directly and Molmo2 is skipped.
        # (task_name alone is always present as the resolver prompt, so
        # target_heatmap is the discriminator.)
        if obs.get("target_heatmap") is not None:
            return self._ll.infer(obs)

        state = obs.get("agent_pos")
        if state is None:
            raise ValueError("HL-LL infer requires agent_pos to determine the batch size")
        state = np.asarray(state)
        batch = int(state.shape[0]) if state.ndim >= 2 else 1
        # The unbatched client (PolicyClientAdapter(batched=False)) splits the env
        # axis off every control stamp, so a 0-d hl_env_index marks a one-env
        # request whose response must be a per-env (T, D) chunk.
        env_raw = obs.get("hl_env_index")
        unbatched = np.ndim(env_raw) == 0 if env_raw is not None else batch == 1
        env_ids = _per_row(env_raw, batch, np.int64)
        if env_ids is None:
            env_ids = np.arange(batch, dtype=np.int64)
        sim_time_ms = _per_row(obs.get("hl_sim_time_ms"), batch, np.float64)
        episode_id = _per_row(obs.get("hl_episode_id"), batch, np.int64)
        input_trace = obs.get("__ll_input_trace__")

        # A client reset() is a no-op over the wire, so an episode-id change is
        # the per-env HL reset signal.
        if episode_id is not None:
            for env, ep in zip(env_ids.tolist(), episode_id.tolist()):
                if self._last_episode_id.get(env) != ep:
                    self._start_episode(env)
                self._last_episode_id[env] = ep

        # 2. Molmo2, batched over the rows due an update: every row without HL
        # latency, else the rows whose K interval elapsed.
        if self._hl_latency is not None:
            # The whole premise of a latency run is the K/M model, so a missing
            # sim-time stamp must not silently degrade to fresh-every-step.
            if sim_time_ms is None:
                raise ValueError(
                    "HL latency is configured (--hl_update_interval_ms / "
                    "--estimated_hl_inference_time_ms) but the client sent no "
                    "'hl_sim_time_ms'. Without it the buffer cannot age anything and "
                    "the run would silently measure a fresh-every-step HL instead of "
                    "the latency model. Use an eval client that stamps sim time, or "
                    "drop the latency flags."
                )
            if hl_refresh is not None:
                raise ValueError(
                    "'hl_refresh' (async HL) and the HL latency model "
                    "(--hl_update_interval_ms / --estimated_hl_inference_time_ms) both "
                    "decide when the HL runs; use one."
                )
            due = [
                r for r in range(batch)
                if self._hl_latency.should_update(int(env_ids[r]), float(sim_time_ms[r]))
            ]
        elif hl_refresh is not None:
            refresh = _per_row(hl_refresh, batch, bool)
            due = [
                r for r in range(batch)
                if refresh[r] or int(env_ids[r]) not in self._prev_action_str  # type: ignore[index]
            ]
        else:
            due = list(range(batch))
        # An empty goal marks an env the client's queue ran out for (unscored filler):
        # no Molmo2 call, and the client holds it still.
        goals = [] if hl_goal is None else _per_row(hl_goal, batch, object)
        filler = {r for r, g in enumerate(goals) if not g}
        due = [r for r in due if r not in filler]
        fresh = (
            self._run_hl(
                obs, due, env_ids,
                hl_goal=hl_goal, hl_front=hl_front, hl_gripper=hl_gripper,
                sim_time_ms=sim_time_ms, episode_id=episode_id, input_trace=input_trace,
            )
            if due
            else {}
        )

        # 3. Per-row LL conditioning: the fresh output, or the M-aged one.
        langs: list[str] = []
        xys: list[np.ndarray | None] = []
        raws: list[str] = []
        dones: list[bool] = []
        latent_frames: list[np.ndarray | None] = []
        for r in range(batch):
            env = int(env_ids[r])
            if r in filler:
                langs.append("")
                xys.append(None)
                raws.append("")
                dones.append(False)
                latent_frames.append(None)
                continue
            if self._hl_latency is not None:
                payload, frame = self._hl_latency.select(env, float(sim_time_ms[r]))  # type: ignore[index]
                if payload is None:
                    raise RuntimeError(
                        f"HL latency buffer for env {env} is empty; expected a commit "
                        "when should_update fires on the episode's first call."
                    )
                lang, xy, raw = payload["lang"], payload["xy"], payload["raw"]
                latent_frames.append(frame)
            else:
                # Async HL: a row not due reuses its last output this episode.
                raw = fresh[r] if r in fresh else self._hl_policy.contract.parse_output(self._prev_action_str[env])[0]
                lang, xy = _strip_coords(raw), _parse_target_xy(raw)
            # "done" is the HL model's episode-termination token. Exact-token
            # check, matching the oracle's `lang == "done"` in eval/oracle/controller.py.
            done = lang.strip().lower() == "done"
            if done:
                # Condition the LL on the episode's last real instruction so the
                # final chunk stays in-distribution; the client ends the episode
                # from hl_done, so at most one step of it runs.
                if env in self._last_lang:
                    lang, xy = self._last_lang[env], self._last_xy.get(env)
                else:
                    logging.warning(
                        "Molmo2 HL emitted %r on the first HL call of env %d's episode, "
                        "so there is no prior instruction to repeat; conditioning on a "
                        "zero heatmap. hl_done ends the episode on this step.",
                        raw, env,
                    )
            elif xy is not None or lang.strip().lower() == PLACE_ON_TABLE_ACTION:
                # Cache every accepted HL instruction, including coord-free
                # ``place on table`` (sim2real no-point contract), so a later
                # ``done`` can repeat it.
                self._last_lang[env] = lang
                self._last_xy[env] = xy
            else:
                # A malformed generation (e.g. 'pick <25 “217, 194>'): the HL is a
                # sampled model, so one bad string must not end a whole eval. Keep
                # the env on its last valid instruction, as for "done"; with none
                # yet this episode the LL gets a zero heatmap ("no spatial goal").
                # ``place on table`` is the only no-point action on the sim2real
                # contract, so anything else without coordinates lands here.
                logging.warning(
                    "Molmo2 HL emitted %r for env %d, which carries no parseable "
                    "'<u,v>' target; %s.",
                    raw, env,
                    "repeating its last instruction" if env in self._last_lang
                    else "no earlier instruction this episode, so a zero heatmap",
                )
                if env in self._last_lang:
                    lang, xy = self._last_lang[env], self._last_xy.get(env)
            langs.append(lang)
            xys.append(xy)
            raws.append(raw)
            dones.append(done)

        front = obs.get("front_rgb")
        seq = int(np.asarray(front).shape[1]) if front is not None and np.ndim(front) == 5 else 1
        heatmaps = np.stack(
            [
                _build_heatmap(xy, self._orig_cam_w, self._orig_cam_h, self._target_sigma)
                if xy is not None
                else np.zeros((3, 224, 224), dtype=np.float32)
                for xy in xys
            ]
        ).astype(np.float32)
        obs["target_heatmap"] = np.repeat(heatmaps[:, None], seq, axis=1)  # (B, S, 3, 224, 224)
        obs["task_name"] = [self._ll_task_name(lang, xy) for lang, xy in zip(langs, xys)]
        # Crop-trained LL with a client that sent the native front frame: crop it
        # around each row's HL target at native resolution (a row with no target
        # gets a zero crop). Without it the LL adapter falls back to the
        # front_rgb + target_heatmap crop.
        native_front = obs.pop("native_front_rgb", None)
        if (
            native_front is not None
            and self._ll._use_target_front_crop >= 0
            and "target_front_crop_rgb" not in obs
        ):
            xy_rows = np.array(
                [np.full(2, np.nan) if xy is None else np.asarray(xy).reshape(2) for xy in xys],
                dtype=np.float32,
            )
            crop_obs: dict[str, Any] = {}
            inject_target_front_crop_from_native(
                crop_obs, self._ll._use_target_front_crop, native_front, xy_rows
            )
            obs["target_front_crop_rgb"] = crop_obs["target_front_crop_rgb"].numpy()
        # This adapter owns the aging on the Molmo2 HL path, so the aged frame it
        # buffered wins over anything the client sent.
        if self._wants_prev_front_frame:
            if self._hl_latency is None or any(f is None for f in latent_frames):
                raise ValueError(
                    "the served checkpoint consumes 'prev_front_rgb', the M-aged "
                    "front frame the HL was invoked on, but no HL-latency buffer is "
                    "running so no frame was ever aged. Serve with "
                    "--hl_update_interval_ms / --estimated_hl_inference_time_ms set "
                    "to the staleness the checkpoint was trained for."
                )
            obs["prev_front_rgb"] = np.stack(latent_frames).astype(np.float32)  # type: ignore[arg-type]

        # 4. One LL call for the whole batch; (T, B, D) actions. An hl_only request
        # (--ll_oracle runs the commands client-side) gets a one-step zero chunk.
        if hl_only:
            response = {"actions": np.zeros((1, batch, 8), dtype=np.float32)}
        else:
            response = self._ll.infer(obs)
        actions = response["actions"]
        steps = int(actions.shape[0])

        def tiled(rows: list[Any], dtype: Any = None) -> np.ndarray:
            # Every ndarray aux leaf is (T, B, ...): the client's chunk broker
            # slices each one per step and per env, like the actions.
            arr = np.asarray(rows, dtype=dtype)
            return np.repeat(arr[None], steps, axis=0)

        model_hms = [_downsample_heatmap(hm[0]) for hm in heatmaps]
        # Model HL overlay aux: the server-produced HL (heatmap + full
        # language-with-coords string), so the eval client can burn it onto the
        # front-camera overlay video, plus hl_done, the client's termination
        # signal, and the exact previous action / question last handed to Molmo2.
        response["hl_model_heatmap"] = tiled(model_hms, np.float32)
        response["hl_model_text"] = tiled([str(r) for r in raws])
        response["hl_done"] = tiled(dones, bool)
        response["hl_prev_action"] = tiled(
            [str(self._hl_prev_provided.get(int(e), "")) for e in env_ids]
        )
        response["hl_input_prompt"] = tiled(
            [str(self._hl_question_sent.get(int(e), "")) for e in env_ids]
        )
        if self._hl_latency is not None:
            # Side-by-side "real-time vs LL-view" viz aux. A row with no update
            # this call shows its aged latent for both panels.
            fresh_hms, fresh_texts = [], []
            for r in range(batch):
                if r in fresh:
                    xy = _parse_target_xy(fresh[r])
                    hm = (
                        _build_heatmap(xy, self._orig_cam_w, self._orig_cam_h, self._target_sigma)[0]
                        if xy is not None
                        else np.zeros((224, 224), dtype=np.float32)
                    )
                    fresh_hms.append(_downsample_heatmap(hm))
                    fresh_texts.append(_strip_coords(fresh[r]))
                else:
                    fresh_hms.append(model_hms[r])
                    fresh_texts.append(langs[r])
            response["hl_fresh_heatmap"] = tiled(fresh_hms, np.float32)
            response["hl_latent_heatmap"] = tiled(model_hms, np.float32)
            response["hl_fresh_text"] = tiled([str(t) for t in fresh_texts])
            response["hl_latent_text"] = tiled([str(t) for t in langs])

        if unbatched:
            # The unbatched client (PerEnvActionChunkBroker) slices T then stacks
            # the per-env rows itself, so it takes a (T, ...) chunk per env.
            response = {
                k: v[:, 0] if isinstance(v, np.ndarray) and v.ndim >= 2 else v
                for k, v in response.items()
            }
        return response


@dataclasses.dataclass
class Args:
    """Arguments for the serve_maniflow_policy entrypoint."""

    checkpoint: str
    config: str | None = None
    train_config: str | None = None
    use_ema: bool = False
    """Serve the EMA weights from the sibling ``*_training_state.pt`` instead of
    the online weights in ``--checkpoint``. Missing file / missing EMA entry is a
    hard error, not a fall back to the online weights."""
    stats_path: str = ""
    port: int = 8000
    seed: int | None = None
    """Random seed for reproducible action sampling. None disables fixed seeding."""
    deterministic_cuda: bool = False
    """Force deterministic CUDA operations (cudnn.deterministic + no benchmark)."""
    replicate_noise: bool = False
    """When True, every sample in a batched inference call starts from the
    same initial flow-matching noise.  This ensures identical observations
    produce identical actions regardless of batch position."""
    action_mode: str | None = None
    """Action mode override. When ``None`` (default), read from the checkpoint's
    ``config.yaml`` (``datasets.vla_data.action_mode``), which must be present --
    a config that omits it is a hard error rather than an assumed ``relative``,
    since the wrong convention mis-denormalizes every action. Pass ``relative``
    or ``absolute`` to override the config."""
    obs_video_dir: str | None = None
    """Directory to save server-side observation videos (side-by-side
    front + gripper).  One video per client connection.  Disabled by default."""
    obs_video_fps: int = 10
    """Frame rate for server-side observation videos."""
    hl: str = "oracle"
    """High-level conditioning source. ``oracle`` (default) runs the LL-only
    batched :class:`ManiFlowInferAdapter` and expects the client to inject
    ``oracle_task_name``/``target_heatmap`` (client-side oracle HL). ``model``
    runs the batched :class:`HLLLInferAdapter` with a Molmo2 HL model when no
    client heatmap is present."""
    skip_ll: bool = False
    """Don't load the ManiFlow model, only its config (``--hl model`` with an
    ``--ll_oracle`` client, which sends only ``hl_only`` requests)."""
    hl_checkpoint: str | None = None
    """Molmo2 HL checkpoint. Required when ``--hl model``; ignored otherwise."""
    hl_max_new_tokens: int = 256
    """Molmo2 decode budget per HL call (``--hl model`` only)."""
    hl_prev_action: bool = True
    """Feed each env's last HL action back into the next Molmo2 question (``--hl model`` only)."""
    hl_html_v2_variant: Literal["plan_stack", "plan_all", "plan", "stack", "all", "target"] | None = None
    """html-v2 question variant (``--hl model`` only): the steps asked before ``point to target.
    predict action.`` None reads it off the HL checkpoint config (``plan_prob`` / ``stack_point_prob`` /
    ``point_blocks``)."""
    hl_prompt: str = ""
    """Optional manual override for the goal fed to the Molmo2 HL model (``--hl
    model`` only). Empty (default) uses the per-episode goal the client sends as
    ``hl_goal``, phrased the way the HL checkpoint was trained (``goal.hl_prompt``
    when recorded, else e.g. ``Stack A B C`` from ``pddl_goal``); set it to force
    a fixed goal (e.g. ``"Stack A B"``)."""
    orig_cam_w: int = 640
    """Original front-camera width for HL target-coordinate scaling (``--hl model``)."""
    orig_cam_h: int = 360
    """Original front-camera height for HL target-coordinate scaling (``--hl model``)."""
    target_sigma: float = DEFAULT_TARGET_SIGMA
    """Gaussian sigma (in 14px feature-grid units) for the HL target heatmap (``--hl model``)."""
    hl_update_interval_ms: float = 0.0
    """K (ms): HL invocation interval (``--hl model`` only); see
    ``blocksuite.serving.hl_latency``. ``0`` = every infer."""
    hl_update_interval_jitter_ms: float = 0.0
    """Resampled ``U(-jitter, +jitter)`` ms added to each HL interval
    (``--hl model`` only). ``0`` = fixed K."""
    estimated_hl_inference_time_ms: float = 0.0
    """M (ms): assumed HL inference time (``--hl model`` only); see
    ``blocksuite.serving.hl_latency``. ``0`` = latest.

    A checkpoint trained with ``use_previous_front_frame`` also reads the M-aged
    front frame the HL was invoked on as ``prev_front_rgb``. Whether it does is
    read off the checkpoint rather than configured here, but it does mean such a
    checkpoint requires a non-zero M (or K): with an instantaneous HL there is no
    aged frame to hand it, and serving one raises."""


def load_safetensors_policy(args: Args, num_inference_steps: int | None = None) -> BasePolicy:
    """``num_inference_steps`` overrides the checkpoint's flow integration steps."""
    policy, stats, effective_action_mode, model_input_spec = load_maniflow_policy_common(
        checkpoint=args.checkpoint,
        stats_path=args.stats_path,
        config=args.config,
        train_config=args.train_config,
        action_mode_override=args.action_mode,
        use_ema=args.use_ema,
        build_policy=not args.skip_ll,
        num_inference_steps=num_inference_steps,
    )

    return ManiFlowInferAdapter(
        policy,
        stats=stats,
        action_mode=effective_action_mode,
        obs_video_dir=args.obs_video_dir,
        obs_video_fps=args.obs_video_fps,
        model_input_spec=model_input_spec,
        replicate_noise=args.replicate_noise,
    )


def load_hl_ll_policy(args: Args) -> BasePolicy:
    ll_adapter = load_safetensors_policy(args)
    hl_policy = Molmo2HLPolicy(
        args.hl_checkpoint,
        max_new_tokens=args.hl_max_new_tokens,
        use_prev_action=args.hl_prev_action,
        html_v2_variant=args.hl_html_v2_variant,
    )
    return HLLLInferAdapter(
        ll_adapter=cast(ManiFlowInferAdapter, ll_adapter),
        hl_policy=hl_policy,
        hl_prompt=args.hl_prompt,
        orig_cam_w=args.orig_cam_w,
        orig_cam_h=args.orig_cam_h,
        target_sigma=args.target_sigma,
        hl_update_interval_ms=args.hl_update_interval_ms,
        hl_update_interval_jitter_ms=args.hl_update_interval_jitter_ms,
        estimated_hl_inference_time_ms=args.estimated_hl_inference_time_ms,
        seed=args.seed,
    )


def main(args: Args) -> None:
    if args.seed is not None:
        import random

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        logging.info("Policy server seeded with seed=%d", args.seed)
    if args.deterministic_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logging.info("CUDA deterministic mode enabled")

    assert args.checkpoint.endswith(".safetensors")
    if args.hl == "oracle":
        policy = load_safetensors_policy(args)
    elif args.hl == "model":
        if not args.hl_checkpoint:
            raise ValueError("--hl model requires --hl_checkpoint (Molmo2 HL checkpoint)")
        policy = load_hl_ll_policy(args)
    else:
        raise ValueError(f"--hl must be 'oracle' or 'model', got {args.hl!r}")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating PolicyServer (host: %s, ip: %s)", hostname, local_ip)

    meta = policy.metadata
    # Includes state_dim / action_mode, which the client's obs transform configures
    # itself from -- the one place a layout disagreement is visible before inference.
    logging.info("Server metadata: %s", meta)
    server = PolicyServer(
        policy=cast(BasePolicy, policy),
        host="0.0.0.0",
        port=args.port,
        metadata=meta,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
