"""Observation transform for the ManiFlow Franka policy server.

The ManiFlow Franka policy expects (per ``train_maniflow_multimodal.yaml``):

    - ``front_rgb``:     (B, 1, 3, 224, 224) float32 in [0, 1] (CHW)
    - ``gripper_rgb``:   (B, 1, 3, 224, 224) float32 in [0, 1]
    - ``front_ptmap``:   (B, 1, 3, 224, 224) float32
    - ``gripper_ptmap``: (B, 1, 3, 224, 224) float32
    - ``agent_pos``:     (B, 1, D) float32

    D depends on the checkpoint and is enumerated in :data:`STATE_DIM_LAYOUTS`:
    8 (7 arm + 1 gripper), 9 (+ 1 pd_mode), or the extended 19 (7 arm + 7 jvel +
    3 ee_xyz + 1 gripper + 1 pd_mode) the ``franka_multimodal`` dataloader
    trains on.

State normalization (and action denormalization) is done by
:mod:`serve_maniflow_policy` on the server side, so this transform forwards
**raw** values as ``agent_pos`` and the server normalizes them to match
training.

Depth-conditioned checkpoints back-project on the GPU inside the model's
``obs_encoder`` instead of consuming point maps, so with ``depth_keys`` this
transform emits the raw ingredients for that instead of the point maps:

    - ``front_depth`` / ``gripper_depth``: (B, 1, 1, 224, 224) float32 depth in
      **meters** (not normalized), nearest-resized, with invalid samples zeroed.
      Only the keys the checkpoint declares are emitted, so a single-depth-camera
      checkpoint gets one.
    - ``camera_intrinsics``: (B, 1, N_cams, 4) ``[fx, fy, cx, cy]`` rescaled from
      the native camera resolution to 224x224. Rows follow ``sorted(depth_keys)``
      (front, then gripper) because the encoder stacks depth in that order and
      reshapes the intrinsics positionally.

A checkpoint trained with a previous front frame additionally reads
``prev_front_rgb`` (same layout as ``front_rgb``); it is sourced from the aged
HL conditioning frame the eval harness injects as ``hl_conditioning_image``.

The policy outputs 9-dim actions ``(7 arm, 1 gripper, 1 pd_mode)``.  The
:meth:`action_postprocess` hook passes all 9 dims through — the caller
(e.g. ``evaluate.py``) is responsible for extracting ``pd_mode`` to switch
actuator gains and slicing to 8 dims before ``env.step()``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

from toyblocks.serving.maniflow_serving_utils import (
    MANIFLOW_IMAGE_SIZE,
    TRAIN_MAX_DEPTH_M,
    crop_centered_on_target,
)
from toyblocks.serving.obs_transforms import BaseObsTransform, _depth_as_nhw
from toyblocks.serving.obs_transforms.depth_ptmap_bounds import (
    DEFAULT_FRANKA_PER_CAMERA_PTMAP_BOUNDS,
    mask_depth_m_by_z_bounds,
    z_bounds_for_depth_key,
)

PromptSource = str | Callable[[], str | None] | None
PerEnvPromptSource = Callable[[int], str] | None

# The ``agent_pos`` widths a ManiFlow checkpoint can declare, as the optional
# blocks each width adds to the always-present 7 arm joints + 1 gripper scalar.
# :meth:`ManiflowObsTransform.transform` assembles the state as
# ``[arm, joint_vel?, ee_xyz?, gripper, pd_mode?]``, so a width fixes the flags:
#   8  -- 7 arm + 1 gripper
#   9  -- 7 arm + 1 gripper + 1 pd_mode
#   19 -- 7 arm + 7 joint_vel + 3 ee_xyz + 1 gripper + 1 pd_mode
_SUPPORTED_AUGMENT_DEPTH = frozenset({"ejbl"})
HVLA_CAMERA_FOR_DEPTH_KEY: dict[str, str] = {
    "front_depth": "front_camera",
    "gripper_depth": "gripper_camera",
}

STATE_DIM_LAYOUTS: dict[int, dict[str, bool]] = {
    8: {"include_joint_vel": False, "include_ee_pos": False, "include_pd_mode": False},
    9: {"include_joint_vel": False, "include_ee_pos": False, "include_pd_mode": True},
    19: {"include_joint_vel": True, "include_ee_pos": True, "include_pd_mode": True},
}

# The layout assumed when the policy server advertises no ``state_dim``: the
# extended one the ``franka_multimodal`` dataloader trains on.
DEFAULT_STATE_DIM = 19


def state_dim_include_flags(state_dim: int | None) -> dict[str, bool]:
    """The ``include_*`` flags a checkpoint's ``state_dim`` implies.

    ``None`` means the server advertised no width, which resolves to
    :data:`DEFAULT_STATE_DIM`. An unlisted width raises: these flags are the only
    thing deciding which blocks go into ``agent_pos``, so guessing them feeds the
    model a state it was not trained on -- a plausible vector of the wrong
    columns, not a crash.
    """
    layout = STATE_DIM_LAYOUTS.get(DEFAULT_STATE_DIM if state_dim is None else state_dim)
    if layout is None:
        raise ValueError(
            f"state_dim must be one of {sorted(STATE_DIM_LAYOUTS)} (or None for "
            f"{DEFAULT_STATE_DIM}), got {state_dim}"
        )
    return dict(layout)


def transform_kwargs_from_server_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """The transform kwargs a served checkpoint's advertised metadata implies.

    Which modalities the LL consumes is a property of the checkpoint, not of the
    eval invocation, so the client reads them off the server's metadata instead of
    flags that can disagree with it. Lives here rather than inline in
    ``_make_policy_client`` so the negotiation can be tested against a real
    adapter's ``metadata`` on CPU
    (``tests/serving/test_infer_adapter.py::test_metadata_rederives_the_client_transform_layout``);
    that test is what makes the server's advertisement *sufficient* to rebuild the
    client's input layout rather than merely consistent with a hand-written copy.

    Keys the metadata does not carry are left out entirely, so the transform's own
    defaults apply.
    """
    kwargs: dict[str, Any] = {}
    if meta.get("state_dim") is not None:
        kwargs["state_dim"] = int(meta["state_dim"])
    if meta.get("obs_keys") is not None:
        obs_keys = [str(k) for k in meta["obs_keys"]]
        # Forwarded verbatim rather than reduced to a bool: a checkpoint may
        # declare one depth camera, and the transform sizes
        # ``camera_intrinsics`` from this list.
        kwargs["depth_keys"] = sorted(k for k in obs_keys if k.endswith("_depth"))
        # The zero-filled ptmap stand-in is only an input for a checkpoint that
        # declares the keys; sending it to one that does not is ~1MB per camera
        # per env per step that the server drops on arrival.
        kwargs["include_ptmap"] = any(k.endswith("_ptmap") for k in obs_keys)
        kwargs["include_prev_front_frame"] = "prev_front_rgb" in obs_keys
    bottom_pct = int(meta.get("use_target_front_crop", -1))
    if bottom_pct >= 0:
        kwargs["target_front_crop_bottom_pct"] = bottom_pct
        kwargs["enable_target_front_crop"] = True
    # A Molmo2 HL server (--hl model) reads native camera frames, front-only for
    # a checkpoint trained without the gripper view.
    if meta.get("hl_native_frames"):
        kwargs["include_hl_frames"] = True
        kwargs["hl_front_view_only"] = bool(meta.get("hl_front_view_only", False))
    pc_bounds = meta.get("per_camera_ptmap_bounds")
    if pc_bounds is not None:
        kwargs["per_camera_ptmap_bounds"] = pc_bounds
    return kwargs


def _native_rgb_uint8(img: np.ndarray) -> np.ndarray:
    """``(B, H, W, 3)`` uint8 copy of a native camera frame batch (RGBA / float [0, 1] accepted)."""
    rgb = np.asarray(img)[..., :3]
    if rgb.dtype != np.uint8:
        scale = 255.0 if rgb.size and float(rgb.max()) <= 1.01 else 1.0
        rgb = np.clip(rgb.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb)


def _hwc_uint8_to_chw_float(img: np.ndarray) -> np.ndarray:
    """(N, H, W, 3) uint8 → (N, 3, H, W) float32 in [0, 1]."""
    if img.ndim == 3:  # unbatched single image
        img = img[None, ...]
    if img.ndim != 4 or img.shape[-1] != 3:
        raise ValueError(f"Expected HWC uint8 image; got shape {img.shape}")
    f = img.astype(np.float32, copy=False) * np.float32(1.0 / 255.0)
    return np.transpose(f, (0, 3, 1, 2)).copy()


def _rgb_nhwc_to_gray_uint8(rgb: np.ndarray) -> np.ndarray:
    """(N, H, W, 3) uint8 RGB → (N, H, W) uint8 luminance (PIL ``L`` per frame)."""
    from PIL import Image

    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected NHWC RGB uint8; got shape {rgb.shape}")
    out = np.empty(rgb.shape[:3], dtype=np.uint8)
    for i in range(rgb.shape[0]):
        out[i] = np.array(Image.fromarray(rgb[i]).convert("L"), dtype=np.uint8)
    return out


def _center_crop_box(h: int, w: int, ratio: float) -> tuple[int, int, int]:
    """Center square crop window on an ``H x W`` grid (same math as RGB crop)."""
    crop = int(h * ratio)
    top = int(round((h - crop) / 2.0))
    left = int(round((w - crop) / 2.0))
    return top, left, crop


def _center_crop_resize(chw: np.ndarray, ratio: float) -> np.ndarray:
    """(N, 3, H, W) float32 → the central ``int(H * ratio)`` square, resized back to (H, W).

    The deterministic counterpart of the ManiFlow obs encoder's training
    augmentation, ``T.RandomCrop(int(img_size * random_crop_ratio))`` followed by
    ``T.Resize(img_size, antialias=True)``: same crop size and resize, with the
    offset torchvision's ``CenterCrop`` would pick instead of a random one.
    """
    import torch
    import torch.nn.functional as F

    _, _, h, w = chw.shape
    top, left, crop = _center_crop_box(h, w, ratio)
    t = torch.from_numpy(chw[:, :, top : top + crop, left : left + crop])
    t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False, antialias=True)
    return t.numpy().copy()


def _center_crop_resize_nearest_nhw(nhw: np.ndarray, ratio: float) -> np.ndarray:
    """(N, H, W) float32 → same crop box as RGB, resized back with nearest."""
    import torch
    import torch.nn.functional as F

    n, h, w = nhw.shape
    top, left, crop = _center_crop_box(h, w, ratio)
    t = torch.from_numpy(nhw[:, top : top + crop, left : left + crop])
    t = F.interpolate(t[:, None, ...], size=(h, w), mode="nearest").squeeze(1)
    return t.numpy().copy()


def _center_crop_resize_nearest_spatial(arr: np.ndarray, ratio: float) -> np.ndarray:
    """Apply :func:`_center_crop_resize_nearest_nhw` on the trailing H, W axes."""
    if arr.ndim < 3:
        raise ValueError(f"expected at least 3 dims (…, H, W); got {arr.shape}")
    *lead, h, w = arr.shape
    flat = arr.reshape(-1, h, w)
    warped = _center_crop_resize_nearest_nhw(flat, ratio)
    return warped.reshape(*lead, h, w)


def _default_per_camera_ptmap_bounds() -> dict[str, dict[str, list[float]]]:
    """Per-camera pointmap bounds for Z-masking eval depth (HVLA training defaults)."""
    return dict(DEFAULT_FRANKA_PER_CAMERA_PTMAP_BOUNDS)


class ManiflowObsTransform(BaseObsTransform):
    """Maps ToyBlocks observations to ManiFlow Franka multimodal inputs.

    Args:
        external_cam: ToyBlocks camera name feeding ``front_rgb``.
        wrist_cam: ToyBlocks camera name feeding ``gripper_rgb``.
        arm_joints: Number of leading joints treated as the arm (default 7 for Franka).
        gripper_joint_index: Index into ``joint_pos`` for the gripper scalar
            (default ``arm_joints``, i.e. the first non-arm joint).
        image_size: Target ``(H, W)`` for RGB inputs (224x224 for ManiFlow).
        image_interpolate_device: Optional override for the torch device used for
            batched image resize; see :meth:`BaseObsTransform.process_image`.
        resize_antialias: Antialias the bilinear RGB downscale (default ``True``).
            hvla's ``_pil_to_tensor`` resizes training frames with PIL
            ``Image.BILINEAR``, which always antialiases; plain
            ``F.interpolate(mode="bilinear")`` does not, and on the 640x360 ->
            224x224 Isaac frames that leaves a max per-pixel gap of ~60 levels
            vs PIL, against ~1 level with antialiasing on.
        center_crop_ratio: After the resize to ``image_size``, crop RGB, depth,
            and ``target_heatmap`` to the same central ``int(H * ratio)`` square
            and resize back (RGB bilinear+antialias; depth and heatmap nearest),
            matching a checkpoint trained with ``random_crop_ratio=ratio``.
            ``camera_intrinsics`` are not crop-adjusted (training uses nominal K
            at the model grid). ``None`` (default) feeds the full frame.
        include_ptmap: Emit ``front_ptmap`` / ``gripper_ptmap`` (zeros) alongside
            RGB. ``_make_policy_client`` sets this from whether the server's
            advertised ``obs_keys`` declare the ptmap inputs, so a pure-RGB
            checkpoint is not sent a stand-in it would only drop. Ignored when
            ``depth_keys`` is non-empty, since real depth supersedes the
            zero-filled stand-in.
        depth_keys: The model's depth input names (``front_depth`` and/or
            ``gripper_depth``); empty means the checkpoint reads no depth. Each
            is emitted as metric meters alongside a matching
            ``camera_intrinsics`` row. ``_make_policy_client`` sets this from the
            server's advertised ``obs_keys``, which is also what makes the env
            render depth at all.
        depth_max_m: Depth samples beyond this (and negative / non-finite ones)
            are zeroed as invalid, matching the training decode's
            ``MAX_DEPTH_M`` cutoff.
        augment_depth: Live depth noise profile before the depth is sent to the
            policy (``ejbl`` only). Matches HVLA training
            ``datasets.vla_data.augment_depth``; off by default.
        depth_aug_seed: Optional RNG seed for ``augment_depth`` (tests only).
            When unset, noise is stochastic each step like training.
        per_camera_ptmap_bounds: Optional override of vendored Franka ptmap bounds
            (``framework.per_camera_ptmap_bounds`` from the served checkpoint).
            When omitted, :data:`DEFAULT_FRANKA_PER_CAMERA_PTMAP_BOUNDS` applies.
        include_prev_front_frame: Emit the injected ``hl_conditioning_image`` as
            ``prev_front_rgb``, for a checkpoint trained with
            ``use_previous_front_frame``. The eval harness also publishes that key
            for the HL-latency video, so this is what separates "render it" from
            "feed it to the policy". ``_make_policy_client`` sets it from the
            server's advertised ``obs_keys``.
        include_hl_frames: Also emit the native ``external_cam`` / ``wrist_cam``
            frames as ``hl_front_rgb`` / ``hl_gripper_rgb`` ``(B, H, W, 3)`` uint8
            for a Molmo2 HL server, which was trained on full frames rather than
            the 224x224 LL inputs; ``hl_front_view_only`` drops the gripper one.
            Set from the server's ``hl_native_frames`` metadata.
        hl_front_view_only: See ``include_hl_frames``.
        include_joint_vel: Append 7-dim joint velocity to the state vector.
            Required for checkpoints trained with ``state_dim=19``.
        include_ee_pos: Append 3-dim end-effector XYZ position to the state
            vector.  Required for ``state_dim=19`` checkpoints.
        include_pd_mode: Append 1-dim pd_mode to the state vector.  Training
            reads it from the current row's ``action.pd_mode``, i.e. the gain
            profile in effect when the frame was captured, so the eval loop
            overwrites it every control step via :meth:`set_pd_mode` with
            :meth:`~toyblocks.eval.pd_mode_controller.PDModeController.current_pd_mode`.
            Without that sync it falls back to the first step of the last
            action chunk (one chunk stale) and to 1.0 before the first chunk
            (training episodes always start in stiff mode).  Required for
            ``state_dim`` 9 and 19 checkpoints.
        state_dim: The served checkpoint's declared ``agent_pos`` width, from
            policy server metadata. Purely a cross-check here -- the three
            ``include_*`` flags above are what assemble the state -- so a width
            that disagrees with them raises instead of reaching the model. See
            :data:`STATE_DIM_LAYOUTS`; subclasses may default the flags from it.
        prompt: Optional language instruction forwarded as ``task_name`` (a
            ``list[str]`` of length ``B``) so ``language_conditioned=True``
            ManiFlow checkpoints can read it. With ``language_conditioned=False``
            the key is a harmless no-op on the server. ``None`` (default) emits
            no ``task_name`` at all. May also be a zero-arg callable returning
            the current prompt (e.g. a bound
            :class:`~toyblocks.utils.prompt_resolver.PromptResolver` method)
            for per-episode prompts.
        per_env_prompt: Optional ``(env_index) -> str`` used instead of
            ``prompt`` so each env gets its own ``task_name``.
        pass_native_front_for_server_crop: With ``enable_target_front_crop``,
            also send the native front frame as ``native_front_rgb``
            ``(B, H, W, 3)`` uint8 so an ``--hl model`` server can build
            ``target_front_crop_rgb`` from it once its HL emits the target, and
            defer the client-side crop until ``target_xy`` / ``target_heatmap``
            is on the obs instead of requiring one.
    """

    #: Model depth input name -> the attribute naming the ToyBlocks camera it
    #: comes from. Also the set of depth keys this transform can serve. Public
    #: because the eval's policy-input verification and debug panel both resolve
    #: a depth key back to its camera through it.
    DEPTH_KEY_CAMERAS = {"front_depth": "external_cam", "gripper_depth": "wrist_cam"}

    def __init__(
        self,
        external_cam: str = "external_camera",
        wrist_cam: str = "wrist_camera",
        arm_joints: int = 7,
        gripper_joint_index: int | None = None,
        image_size: tuple[int, int] = (224, 224),
        image_interpolate_device: str | None = None,
        resize_antialias: bool = True,
        center_crop_ratio: float | None = None,
        include_ptmap: bool = True,
        depth_keys: Sequence[str] = (),
        depth_max_m: float = TRAIN_MAX_DEPTH_M,
        augment_depth: str | None = None,
        depth_aug_seed: int | None = None,
        per_camera_ptmap_bounds: dict[str, dict[str, list[float]]] | None = None,
        include_prev_front_frame: bool = False,
        include_hl_frames: bool = False,
        hl_front_view_only: bool = False,
        prompt: PromptSource = None,
        per_env_prompt: PerEnvPromptSource = None,
        include_joint_vel: bool = False,
        include_ee_pos: bool = False,
        include_pd_mode: bool = False,
        state_dim: int | None = None,
        enable_target_front_crop: bool = False,
        target_front_crop_bottom_pct: int = -1,
        orig_cam_w: float = 640.0,
        orig_cam_h: float = 360.0,
        pass_native_front_for_server_crop: bool = False,
    ) -> None:
        self.external_cam = external_cam
        self.wrist_cam = wrist_cam
        self.arm_joints = arm_joints
        self.gripper_joint_index = (
            gripper_joint_index if gripper_joint_index is not None else arm_joints
        )
        self.image_size = image_size
        self._resize_antialias = resize_antialias
        if center_crop_ratio is not None and not 0.0 < center_crop_ratio <= 1.0:
            raise ValueError(f"center_crop_ratio must be in (0, 1], got {center_crop_ratio}")
        self.center_crop_ratio = center_crop_ratio
        self.include_ptmap = include_ptmap
        # sorted(): the model stacks depth alphabetically and reshapes
        # ``camera_intrinsics`` positionally, so the emission order is fixed.
        self.depth_keys = sorted(depth_keys)
        unknown = [k for k in self.depth_keys if k not in self.DEPTH_KEY_CAMERAS]
        if unknown:
            raise ValueError(
                f"unrecognised depth keys {unknown}; this transform can source "
                f"{sorted(self.DEPTH_KEY_CAMERAS)} from its external_cam / "
                "wrist_cam."
            )
        self.depth_max_m = depth_max_m
        if augment_depth is not None:
            name = str(augment_depth).strip().lower()
            if name not in _SUPPORTED_AUGMENT_DEPTH:
                raise ValueError(
                    f"augment_depth={augment_depth!r} not supported; "
                    f"allowed: {sorted(_SUPPORTED_AUGMENT_DEPTH)}"
                )
            self.augment_depth = name
            from toyblocks.serving.maniflow_serving_utils import require_depth_aug_supported

            require_depth_aug_supported(name)
        else:
            self.augment_depth = None
        self._depth_aug_rng = (
            np.random.default_rng(depth_aug_seed)
            if self.augment_depth is not None
            else np.random.default_rng(0)
        )
        if per_camera_ptmap_bounds is not None:
            self._per_camera_ptmap_bounds: dict[str, dict[str, list[float]]] | None = {
                str(k): {
                    "min": list(v["min"]),
                    "max": list(v["max"]),
                }
                for k, v in per_camera_ptmap_bounds.items()
            }
        else:
            self._per_camera_ptmap_bounds = None
        self.include_prev_front_frame = include_prev_front_frame
        self.include_hl_frames = include_hl_frames
        self.hl_front_view_only = hl_front_view_only
        self.include_joint_vel = include_joint_vel
        self.include_ee_pos = include_ee_pos
        self.include_pd_mode = include_pd_mode
        if enable_target_front_crop and target_front_crop_bottom_pct < 0:
            raise ValueError(
                "enable_target_front_crop requires target_front_crop_bottom_pct >= 0 "
                f"(from the server's use_target_front_crop), got {target_front_crop_bottom_pct}"
            )
        self.enable_target_front_crop = enable_target_front_crop
        self.target_front_crop_bottom_pct = int(target_front_crop_bottom_pct)
        self.orig_cam_w = float(orig_cam_w)
        self.orig_cam_h = float(orig_cam_h)
        self.pass_native_front_for_server_crop = bool(pass_native_front_for_server_crop)
        self._logged_target_xy_heatmap_fallback = False
        # A declared state_dim is the checkpoint's own contract, so cross-check
        # the flags against it at construction, where the disagreement can be
        # named, instead of letting it surface at the first inference as an
        # agent_pos shape mismatch that says nothing about which block is missing.
        if state_dim is not None:
            expected_flags = state_dim_include_flags(state_dim)
            actual_flags = {
                "include_joint_vel": include_joint_vel,
                "include_ee_pos": include_ee_pos,
                "include_pd_mode": include_pd_mode,
            }
            if actual_flags != expected_flags:
                raise ValueError(
                    f"state_dim={state_dim} requires {expected_flags}, got {actual_flags}"
                )
        self._prompt_source: PromptSource = prompt
        self._per_env_prompt_source: PerEnvPromptSource = per_env_prompt
        self._image_interpolate_device = image_interpolate_device
        self._pd_mode: dict[int, float] = {}
        self._pd_mode_default: float = 1.0
        self._request_env_indices: list[int] | None = None

    def reset(self) -> None:
        """Reset per-episode state (called by :class:`ObsTransformPolicy`).

        Falls back to ``1.0`` because training episodes always start with the
        stiff profile applied by TAMP. The eval loop should call
        :meth:`set_pd_mode` right after this with the actual per-env pd_mode
        pulled from :class:`~toyblocks.eval.pd_mode_controller.PDModeController`
        so the ``agent_pos`` column reflects the physical gains rather than a
        hardcoded assumption.
        """
        self._pd_mode.clear()
        self._pd_mode_default = 1.0
        self._request_env_indices = None

    def reset_envs(self, env_ids: Any) -> None:
        """Drop pd_mode state for envs that finished an episode."""
        for env_idx in _normalize_env_ids(env_ids):
            self._pd_mode.pop(int(env_idx), None)

    def set_pd_mode(self, values: Any) -> None:
        """Overwrite the pd_mode column used in the next ``agent_pos``.

        Accepts a scalar or a per-env array/tensor (torch or numpy). A scalar
        sets the default and clears per-env overrides so every env uses it.
        The eval loop calls this after every ``PDModeController`` update (and
        after env resets) so the column reports the actuator gains actually in
        sim, which is what training's ``action.pd_mode`` state records. That
        takes precedence over the stale value :meth:`action_postprocess`
        stores and the hardcoded ``1.0`` default that :meth:`reset` falls back
        to.
        """
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim == 0:
            self._pd_mode_default = float(arr.item())
            self._pd_mode.clear()
        else:
            flat = arr.reshape(-1)
            for env_idx, value in enumerate(flat):
                self._pd_mode[int(env_idx)] = float(value)

    @property
    def prompt(self) -> str | None:
        """Resolve the current prompt (callable is invoked on each access)."""
        src = self._prompt_source
        if src is None:
            return None
        return src() if callable(src) else src

    def _env_indices_from_obs(self, obs: dict[str, Any], batch_size: int) -> list[int]:
        raw = obs.get("hl_env_index")
        if raw is None:
            return list(range(batch_size))
        arr = np.asarray(raw, dtype=np.int64).reshape(-1)
        if arr.shape[0] != batch_size:
            raise ValueError(
                f"hl_env_index has length {arr.shape[0]} but this observation "
                f"has batch size {batch_size}."
            )
        return [int(x) for x in arr]

    def _pd_mode_column(self, env_indices: list[int]) -> np.ndarray:
        """The ``(B, 1)`` pd_mode column for the request's env indices."""
        return np.asarray(
            [self._pd_mode.get(env_idx, self._pd_mode_default) for env_idx in env_indices],
            dtype=np.float32,
        ).reshape(-1, 1)

    def _zero_invalid_depth(self, depth_nhw: np.ndarray) -> np.ndarray:
        """Mark out-of-range depth samples invalid (0), matching :meth:`process_depth`."""
        out = np.asarray(depth_nhw, dtype=np.float32).copy()
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out[(out < 0.0) | (out > self.depth_max_m)] = 0.0
        return out

    def _per_camera_ptmap_bounds_for_masking(self) -> dict[str, dict[str, list[float]]]:
        if not self.depth_keys:
            raise RuntimeError(
                "ptmap Z depth bounds are only loaded for depth-conditioned checkpoints "
                f"(depth_keys non-empty); got {self.depth_keys!r}"
            )
        if self._per_camera_ptmap_bounds is None:
            self._per_camera_ptmap_bounds = _default_per_camera_ptmap_bounds()
        return self._per_camera_ptmap_bounds

    def _mask_depth_ptmap_z_nhw(self, depth_nhw: np.ndarray, depth_key: str) -> np.ndarray:
        """Zero depths outside the per-camera pointmap Z bounds (training parity)."""
        bounds = self._per_camera_ptmap_bounds_for_masking()
        z_min, z_max = z_bounds_for_depth_key(depth_key, bounds)
        d = np.asarray(depth_nhw, dtype=np.float32)
        if d.ndim == 2:
            return mask_depth_m_by_z_bounds(d, z_min, z_max)
        out = np.empty_like(d)
        for i in range(d.shape[0]):
            out[i] = mask_depth_m_by_z_bounds(d[i], z_min, z_max)
        return out

    def _apply_ejbl_depth_nhw(
        self,
        depth_nhw: np.ndarray,
        obs: dict[str, Any],
        cam: str,
        depth_key: str,
    ) -> np.ndarray:
        """EJBL noise on batched (N, H, W) depth at the policy grid (training parity)."""
        from toyblocks.serving.maniflow_serving_utils import require_depth_aug_supported
        from hvla.training.trainer_utils.depth_aug.disparity_common import baseline_for_camera
        from hvla.training.trainer_utils.depth_aug.elastic_disparity_depth import (
            ELASTIC_DEPTH_AUG_PRESET,
        )
        from hvla.training.trainer_utils.depth_aug.elastic_jbl_depth import add_ejbl_depth_noise
        from hvla.training.trainer_utils.depth_bake_profiles import resolve_profile

        require_depth_aug_supported("ejbl")
        hvla_cam = HVLA_CAMERA_FOR_DEPTH_KEY[depth_key]
        profile = resolve_profile("ejbl")
        if profile is None:
            require_depth_aug_supported("ejbl")

        rgb = self.process_image(
            self.get_image(obs, cam),
            self.image_size,
            interpolate_device=self._image_interpolate_device,
            antialias=self._resize_antialias,
        )
        gray = _rgb_nhwc_to_gray_uint8(rgb)

        native_w = _depth_as_nhw(self.get_depth(obs, cam)).shape[-1]
        target_w = self.image_size[1]
        sx = target_w / float(native_w)
        K = self.get_intrinsics(obs, cam)
        fx_batch = K[:, 0, 0].astype(np.float64) * sx
        baseline_m = baseline_for_camera(hvla_cam)

        out = depth_nhw.copy()
        n = out.shape[0]
        for i in range(n):
            out[i] = add_ejbl_depth_noise(
                out[i],
                float(fx_batch[i]),
                baseline_m,
                gray=gray[i],
                rng=self._depth_aug_rng,
                elastic_preset=profile.elastic_preset or ELASTIC_DEPTH_AUG_PRESET,
                noise_kwargs=profile.noise_kwargs or {},
                random_ideal_mix=profile.random_ideal_mix,
                ideal_mix_min=profile.ideal_mix_min,
                ideal_mix_max=profile.ideal_mix_max,
            )
        return self._zero_invalid_depth(out)

    def _depth_inputs(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        """The declared ``*_depth`` keys + ``camera_intrinsics`` for GPU back-projection.

        The model stacks depth by ``sorted(depth_keys)`` and reshapes
        ``camera_intrinsics`` to ``(B*N_cams, 4)`` positionally, so the intrinsics
        rows are emitted in that same alphabetical order, and only for the keys
        the checkpoint declares (a checkpoint with one depth camera expects a
        single row).
        """
        out: dict[str, np.ndarray] = {}
        intrinsics_rows: list[np.ndarray] = []
        target_h, target_w = self.image_size
        for depth_key in self.depth_keys:
            cam = getattr(self, self.DEPTH_KEY_CAMERAS[depth_key])
            raw = _depth_as_nhw(self.get_depth(obs, cam))  # (N, H, W) meters
            native_h, native_w = raw.shape[-2:]
            resized = self.process_depth(
                raw,
                self.image_size,
                max_depth_m=self.depth_max_m,
                interpolate_device=self._image_interpolate_device,
            )
            if self.augment_depth == "ejbl":
                resized = self._apply_ejbl_depth_nhw(resized, obs, cam, depth_key)
            resized = self._mask_depth_ptmap_z_nhw(resized, depth_key)
            if self.center_crop_ratio is not None:
                resized = _center_crop_resize_nearest_nhw(resized, self.center_crop_ratio)
            out[depth_key] = resized[:, None, None, ...]  # (B, 1, 1, H, W)

            # Native K per camera (front vs gripper): episode ingest, calibrated
            # preset, or default sim render on the obs — never crop-adjusted.
            # Squashing a 640x360 frame into a square grid distorts the aspect
            # ratio; training rescales fx/fy independently the same way, so this
            # replicates that rather than correcting it.
            K = self.get_intrinsics(obs, cam)  # (N, 3, 3)
            sx = target_w / float(native_w)
            sy = target_h / float(native_h)
            intrinsics_rows.append(
                np.stack(
                    [K[:, 0, 0] * sx, K[:, 1, 1] * sy, K[:, 0, 2] * sx, K[:, 1, 2] * sy],
                    axis=-1,
                ).astype(np.float32)  # (N, 4)
            )
        out["camera_intrinsics"] = np.stack(intrinsics_rows, axis=1)[
            :, None, ...
        ]  # (B, 1, N_cams, 4)
        return out

    def _rgb_input(self, img: np.ndarray) -> np.ndarray:
        """Resized (N, H, W, 3) uint8 → (N, 1, 3, H, W) float model input."""
        chw = _hwc_uint8_to_chw_float(img)
        if self.center_crop_ratio is not None:
            chw = _center_crop_resize(chw, self.center_crop_ratio)
        return chw[:, None, ...]

    @staticmethod
    def _heatmap_peak_xy(hm: np.ndarray) -> tuple[float, float] | None:
        """Argmax (x, y) on channel 0 of a single heatmap."""
        arr = np.asarray(hm, dtype=np.float32)
        if arr.ndim == 5:
            ch = arr[0, 0, 0]
        elif arr.ndim == 4 and arr.shape[0] == 3:
            ch = arr[0]
        elif arr.ndim == 4:
            ch = arr[0, 0]
        elif arr.ndim == 3:
            ch = arr[0]
        else:
            raise ValueError(f"unexpected target_heatmap shape {arr.shape}")
        peak_val = float(ch.max())
        if peak_val <= 0.0:
            return None
        idx = int(ch.argmax())
        y, x = divmod(idx, ch.shape[1])
        return float(x), float(y)

    def _native_target_xy_batch(
        self,
        obs: dict[str, Any],
        batch_size: int,
        env_indices: list[int],
    ) -> np.ndarray:
        """(B, 2) pixel targets in native front-camera resolution."""
        raw = obs.get("target_xy")
        if raw is not None:
            arr = np.asarray(raw, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, 2)
            if arr.shape[0] == batch_size:
                return arr
            if all(0 <= i < arr.shape[0] for i in env_indices):
                return np.stack([arr[i] for i in env_indices], axis=0)
            raise ValueError(
                f"target_xy has shape {arr.shape} but batch size is {batch_size}"
            )

        hm = obs.get("target_heatmap")
        if hm is None:
            raise ValueError(
                "enable_target_front_crop requires obs['target_xy'] (native pixels) "
                "or target_heatmap for peak fallback"
            )
        if not self._logged_target_xy_heatmap_fallback:
            print(
                "[WARN]: ManiflowObsTransform: target_front_crop has no obs['target_xy']; "
                "deriving native centers from target_heatmap peaks (not identical to "
                "training goal.position).",
                flush=True,
            )
            self._logged_target_xy_heatmap_fallback = True
        hm_np = np.asarray(hm, dtype=np.float32)
        if hm_np.ndim == 5 and hm_np.shape[0] == batch_size:
            planes = [hm_np[i] for i in range(batch_size)]
        elif hm_np.ndim == 4 and hm_np.shape[0] == batch_size:
            planes = [hm_np[i] for i in range(batch_size)]
        else:
            raise ValueError(
                f"target_heatmap batch {hm_np.shape} does not match batch_size {batch_size}"
            )
        out = np.full((batch_size, 2), np.nan, dtype=np.float32)
        scale_x = self.orig_cam_w / float(MANIFLOW_IMAGE_SIZE)
        scale_y = self.orig_cam_h / float(MANIFLOW_IMAGE_SIZE)
        for i, plane in enumerate(planes):
            peak = self._heatmap_peak_xy(plane)
            if peak is not None:
                out[i, 0] = peak[0] * scale_x
                out[i, 1] = peak[1] * scale_y
        return out

    @staticmethod
    def _spatial_target_present(obs: dict[str, Any]) -> bool:
        """Whether ``obs`` carries a ``target_xy`` / ``target_heatmap`` to crop around."""
        raw = obs.get("target_xy")
        if raw is not None and np.asarray(raw).size >= 2:
            return True
        hm = obs.get("target_heatmap")
        return hm is not None and np.asarray(hm).size > 0

    def _target_front_crop_rgb_batch(
        self,
        native_front: np.ndarray,
        obs: dict[str, Any],
        batch_size: int,
        env_indices: list[int],
    ) -> np.ndarray:
        """(B, 1, 3, H, W) float32 target-centered front crops."""
        if native_front.ndim == 3:
            native_front = native_front[None, ...]
        xy_batch = self._native_target_xy_batch(obs, batch_size, env_indices)
        crop_size = int(self.image_size[0])
        bottom_pct = self.target_front_crop_bottom_pct
        crops: list[np.ndarray] = []
        for i in range(batch_size):
            img = native_front[i]
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            cx, cy = float(xy_batch[i, 0]), float(xy_batch[i, 1])
            if not np.isfinite(cx) or not np.isfinite(cy):
                crop = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
            else:
                crop = crop_centered_on_target(
                    img, cx, cy, crop_size, bottom_pct
                )
            crops.append(crop)
        stacked = np.stack(crops, axis=0)
        return self._rgb_input(stacked)

    # ------------------------------------------------------------------
    # BaseObsTransform API
    # ------------------------------------------------------------------

    def transform(self, obs: dict[str, Any]) -> dict[str, Any]:
        joint_pos = np.asarray(self.get_joint_pos(obs))
        if joint_pos.ndim == 1:
            joint_pos = joint_pos[None, :]
        elif joint_pos.ndim != 2:
            raise ValueError(f"Expected joint_pos of shape (n,) or (B, n); got {joint_pos.shape}")
        required = max(self.arm_joints, self.gripper_joint_index + 1)
        if joint_pos.shape[-1] < required:
            raise ValueError(
                f"joint_pos has only {joint_pos.shape[-1]} joints but requires "
                f"{required} (arm={self.arm_joints}, gripper_idx={self.gripper_joint_index})."
            )
        B = joint_pos.shape[0]
        env_indices = self._env_indices_from_obs(obs, B)
        self._request_env_indices = env_indices

        arm = joint_pos[:, : self.arm_joints].astype(np.float32)
        gripper = joint_pos[:, self.gripper_joint_index : self.gripper_joint_index + 1].astype(
            np.float32
        )

        state_parts: list[np.ndarray] = [arm]

        if self.include_joint_vel:
            jvel = np.asarray(obs["agent"]["joint_vel"])
            if jvel.ndim == 1:
                jvel = jvel[None, :]
            elif jvel.ndim == 3: # (B, 1, D)
                jvel = jvel[:, 0, :]
            state_parts.append(jvel[:, : self.arm_joints].astype(np.float32))

        if self.include_ee_pos:
            ee = np.asarray(obs["agent"]["ee_pos"])
            if ee.ndim == 1:
                ee = ee[None, :]
            elif ee.ndim == 3:  # (B, 1, D)
                ee = ee[:, 0, :]
            state_parts.append(ee[:, :3].astype(np.float32))

        state_parts.append(gripper)

        if self.include_pd_mode:
            state_parts.append(self._pd_mode_column(env_indices))

        agent_pos = np.concatenate(state_parts, axis=-1)[:, None, :]

        # Once per run: the assembled agent_pos width and which optional blocks are
        # in it. This is the layout the checkpoint's state_dim has to agree with, so
        # it is worth having in the log of every eval.
        if not hasattr(self, "_logged_state_dim"):
            print(
                f"[INFO]: ManiflowObsTransform agent_pos={agent_pos.shape} "
                f"include_joint_vel={self.include_joint_vel} "
                f"include_ee_pos={self.include_ee_pos} "
                f"include_pd_mode={self.include_pd_mode}",
                flush=True,
            )
            self._logged_state_dim = True

        native_front = self.get_image(obs, self.external_cam)
        ext = self.process_image(
            native_front,
            self.image_size,
            interpolate_device=self._image_interpolate_device,
            antialias=self._resize_antialias,
        )
        wrist = self.process_image(
            self.get_image(obs, self.wrist_cam),
            self.image_size,
            interpolate_device=self._image_interpolate_device,
            antialias=self._resize_antialias,
        )
        front_rgb = self._rgb_input(ext)  # (B, 1, 3, H, W)
        gripper_rgb = self._rgb_input(wrist)

        out: dict[str, Any] = {
            "front_rgb": front_rgb,
            "gripper_rgb": gripper_rgb,
            "agent_pos": agent_pos,
        }
        if self.enable_target_front_crop:
            if self.pass_native_front_for_server_crop:
                # The --hl model server crops this around its HL target.
                out["native_front_rgb"] = _native_rgb_uint8(native_front)
            if not self.pass_native_front_for_server_crop or self._spatial_target_present(obs):
                out["target_front_crop_rgb"] = self._target_front_crop_rgb_batch(
                    native_front, obs, B, env_indices
                )
        if self.depth_keys:
            out.update(self._depth_inputs(obs))
        elif self.include_ptmap:
            zeros = np.zeros((B, 1, 3, self.image_size[0], self.image_size[1]), dtype=np.float32)
            out["front_ptmap"] = zeros
            out["gripper_ptmap"] = zeros.copy()
        if self._per_env_prompt_source is not None:
            out["task_name"] = [self._per_env_prompt_source(env_idx) for env_idx in env_indices]
        else:
            prompt = self.prompt
            if prompt is not None:
                # ManiFlow expects task_name as a list[str] of length B.
                # dict_apply inside predict_action has a list escape hatch that
                # leaves non-tensor items untouched (flow_policy.py line 33-34).
                out["task_name"] = [prompt] * B

        # Client-side oracle HL conditioning (see toyblocks.eval.oracle): when
        # the eval harness injects these keys into the obs dict, forward them.
        # ``oracle_task_name`` (per-env language) overrides the resolver prompt;
        # ``target_heatmap`` is the spatial goal the target-conditioned ManiFlow
        # checkpoint reads (a no-op key otherwise). ``to_numpy`` turns the
        # injected list[str] into a numpy string array; length ``B`` is the
        # batch, otherwise index by ``hl_env_index``.
        otn = obs.get("oracle_task_name")
        if otn is not None:
            names = np.asarray(otn).reshape(-1)
            if names.shape[0] == B:
                out["task_name"] = [str(x) for x in names]
            elif all(0 <= env_idx < names.shape[0] for env_idx in env_indices):
                out["task_name"] = [str(names[env_idx]) for env_idx in env_indices]
            else:
                raise ValueError(
                    f"oracle_task_name has length {names.shape[0]} but this "
                    f"observation has batch size {B} and env indices {env_indices}."
                )
        hm = obs.get("target_heatmap")
        if hm is not None:
            hm_arr = np.asarray(hm, dtype=np.float32)
            if self.center_crop_ratio is not None:
                hm_arr = _center_crop_resize_nearest_spatial(hm_arr, self.center_crop_ratio)
            out["target_heatmap"] = hm_arr

        # The aged HL conditioning frame is what a ``use_previous_front_frame``
        # checkpoint was trained on as ``prev_front_rgb``: training paired the
        # current frame with one sampled n_action_steps*U{1..4} steps back and
        # took the language/target from that same stale row, which is exactly
        # what the HL-latency buffer reproduces. Dropped by
        # ``select_and_validate_obs`` for a checkpoint that does not declare it.
        cond = obs.get("hl_conditioning_image") if self.include_prev_front_frame else None
        if cond is not None:
            out["prev_front_rgb"] = self._rgb_input(
                self.process_image(
                    np.asarray(cond),
                    self.image_size,
                    interpolate_device=self._image_interpolate_device,
                    antialias=self._resize_antialias,
                )
            )

        # Molmo2 HL inputs (``--hl model`` server): native frames and each env's
        # goal prompt (``hl_goal``, injected by the eval harness in the HL
        # checkpoint's phrasing; full length num_envs, so index by hl_env_index
        # when this request carries fewer envs).
        if self.include_hl_frames:
            out["hl_front_rgb"] = _native_rgb_uint8(native_front)
            if not self.hl_front_view_only:
                out["hl_gripper_rgb"] = _native_rgb_uint8(self.get_image(obs, self.wrist_cam))
        hl_goal = obs.get("hl_goal")
        if hl_goal is not None:
            goals = np.asarray(hl_goal).reshape(-1)
            if goals.shape[0] != B:
                goals = goals[env_indices]
            out["hl_goal"] = [str(g) for g in goals]
        hl_refresh = obs.get("hl_refresh")
        if hl_refresh is not None:
            refresh = np.asarray(hl_refresh, dtype=bool).reshape(-1)
            out["hl_refresh"] = refresh if refresh.shape[0] == B else refresh[env_indices]
        # --ll_oracle: Molmo2 only; the client runs the commands, so the server skips the LL.
        if obs.get("hl_only") is not None:
            out["hl_only"] = True

        # HL-latency bookkeeping (see toyblocks.serving.hl_latency): forward the
        # per-step sim-time stamp + episode id so the server-side buffer (the
        # ``--hl model`` HLLLInferAdapter) can gate its update interval and reset
        # on episode boundaries. The LL-only ManiFlowInferAdapter (``--hl oracle``)
        # pops these keys, so forwarding them is a no-op there.
        sim_time_ms = obs.get("hl_sim_time_ms")
        if sim_time_ms is not None:
            out["hl_sim_time_ms"] = np.asarray(sim_time_ms, dtype=np.float32)
        episode_id = obs.get("hl_episode_id")
        if episode_id is not None:
            out["hl_episode_id"] = np.asarray(episode_id, dtype=np.int64)
        env_index = obs.get("hl_env_index")
        if env_index is not None:
            out["hl_env_index"] = np.asarray(env_index, dtype=np.int64)
        return out

    # ------------------------------------------------------------------
    # Action postprocess: keep all 9 dims (7 arm + 1 gripper + 1 pd_mode).
    # The caller (evaluate.py) is responsible for extracting pd_mode and
    # passing only the first 8 dims to env.step(). The pd_mode stored here
    # is only a fallback for callers without a PDModeController: it is the
    # chunk's first step, one chunk stale by the next observation, and it is
    # written for every env even when only some of them take this chunk.
    # The eval loop overwrites it every step via set_pd_mode().
    # ------------------------------------------------------------------

    def action_postprocess(self, result: dict[str, Any]) -> dict[str, Any]:
        # Update internal pd_mode state for the next observation
        if "actions" in result and self.include_pd_mode:
            actions = result["actions"]
            env_indices = self._request_env_indices
            # actions shape: (T, B, D) or (B, T, D) depending on server/client
            # Server returns (T, B, 9). We want the first time step's pd_mode.
            if actions.ndim == 3 and actions.shape[-1] >= 9:
                pd_vals = actions[0, :, 8]
                target_indices = env_indices if env_indices is not None else list(
                    range(actions.shape[1])
                )
                for slot, env_idx in enumerate(target_indices):
                    self._pd_mode[int(env_idx)] = float(pd_vals[slot])
            elif actions.ndim == 2 and actions.shape[-1] >= 9:
                # HLLLInferAdapter drops the singleton batch axis and returns a
                # per-env (T, D) chunk, so the leading axis is time, not batch.
                env_idx = (env_indices or [0])[0]
                self._pd_mode[int(env_idx)] = float(actions[0, 8])
        return result


def _normalize_env_ids(env_ids: Any) -> list[int]:
    if isinstance(env_ids, torch.Tensor):
        env_ids = env_ids.detach().cpu().numpy()
    arr = np.asarray(env_ids).reshape(-1)
    return [int(x) for x in arr]
