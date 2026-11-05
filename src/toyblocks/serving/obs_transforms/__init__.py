"""Pluggable observation transforms for mapping ToyBlocks env observations to
the format expected by different policy servers.

To add support for a new policy, create a subclass of :class:`BaseObsTransform`
in a new file under this package, override :meth:`transform`, and register it in
:data:`OBS_TRANSFORM_REGISTRY`.
"""

from __future__ import annotations

import abc
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from toyblocks.serving.base_policy import walk_policy_chain


def _resize_nhwc_batched_torch(
    image_nhwc: np.ndarray,
    size_hw: tuple[int, int],
    device: torch.device,
    antialias: bool = False,
) -> np.ndarray:
    """Batched HWC resize via ``torch.nn.functional.interpolate`` (GPU or CPU).

    Expects ``image_nhwc`` of shape ``(N, H, W, C)`` with ``C`` in ``{3, 4}``,
    uint8 or float. Returns uint8 ``(N, size_hw[0], size_hw[1], 3)`` on CPU
    (numpy), contiguous for msgpack / policy servers.

    ``antialias=True`` makes the bilinear downscale match PIL's
    ``Image.BILINEAR`` (which always antialiases) instead of plain point-sampled
    bilinear; use it when the training pipeline resized with PIL.
    """
    if image_nhwc.ndim != 4:
        raise ValueError(f"Batched torch resize expects (N,H,W,C); got {image_nhwc.shape}")
    arr = np.ascontiguousarray(image_nhwc)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    t = torch.from_numpy(arr)
    t = t.to(device, non_blocking=device.type == "cuda")
    # NCHW, float in [0, 1] for interpolate
    t = t.permute(0, 3, 1, 2).contiguous()
    if t.dtype == torch.uint8:
        t = t.float().mul_(1.0 / 255.0)
    elif torch.is_floating_point(t):
        t = t.float().clamp_(0.0, 1.0)
    else:
        t = t.float()
    t = F.interpolate(
        t, size=size_hw, mode="bilinear", align_corners=False, antialias=antialias
    )
    # round_() before the cast: ``.to(torch.uint8)`` truncates toward zero, which
    # is a systematic half-level dark bias on every frame served to a policy. It
    # also makes the output kernel-sensitive -- the frame round-trips uint8 ->
    # x1/255 -> bilinear -> x255, so wherever the blend degenerates to a single
    # source pixel (the borders under align_corners=False) the value lands at
    # ``v +/- 1 ULP`` and truncation turns ``v - eps`` into ``v - 1``, differently
    # on different platforms.
    t = t.mul_(255.0).round_().clamp_(0.0, 255.0).to(torch.uint8)
    t = t.permute(0, 2, 3, 1).contiguous()
    return t.cpu().numpy()


def _depth_as_nhw(depth: np.ndarray) -> np.ndarray:
    """``(H,W)`` / ``(H,W,1)`` / ``(N,H,W)`` / ``(N,H,W,1)`` depth -> ``(N,H,W)`` float32."""
    d = np.asarray(depth)
    if d.ndim >= 3 and d.shape[-1] == 1:
        d = d[..., 0]
    if d.ndim == 2:
        d = d[None, ...]
    if d.ndim != 3:
        raise ValueError(
            f"Expected depth of shape (H,W[,1]) or (N,H,W[,1]); got {np.asarray(depth).shape}"
        )
    return d.astype(np.float32, copy=False)


def _resolve_interpolate_device(interpolate_device: str | torch.device | None) -> torch.device:
    """Device for batched ``interpolate``; explicit *interpolate_device* wins."""
    if interpolate_device is not None:
        return torch.device(interpolate_device)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseObsTransform(abc.ABC):
    """Abstract base for observation transforms.

    Subclasses only need to override :meth:`transform`.  The static helper
    methods provide common operations (image processing, joint extraction)
    that any subclass can reuse.
    """

    def __call__(self, obs: dict[str, Any]) -> dict[str, Any]:
        return self.transform(obs)

    @abc.abstractmethod
    def transform(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Map ToyBlocks numpy observation dict(s) to the policy server format.

        Args:
            obs: Numpy observation dict from ``to_numpy`` conversion. With
                batched vectorised envs, leaves include a leading batch dimension
                ``N`` (e.g. ``rgb`` shape ``(N, H, W, C)``, ``joint_pos`` ``(N, n)``).
                Unbatched leaves look like::

                    {"agent": {"joint_pos": (n_joints,), ...},
                     "images": {"cam_name": {"rgb": (H, W, C)}}}

        Returns:
            Dict in the format expected by the policy server's ``infer``.
        """

    # ------------------------------------------------------------------
    # Reusable helpers
    # ------------------------------------------------------------------

    @staticmethod
    def process_image(
        image: np.ndarray,
        size: tuple[int, int] = (224, 224),
        *,
        interpolate_device: str | torch.device | None = None,
        antialias: bool = False,
    ) -> np.ndarray:
        """Convert a raw camera image to a policy-ready format.

        Handles RGBA -> RGB, float -> uint8, and resizing in one call.

        Args:
            image: Array of shape ``(H, W, 3|4)`` or batched ``(N, H, W, 3|4)``
                in uint8 or float32 (e.g. vectorised Isaac Lab observations).
            size: Target ``(height, width)``.
            interpolate_device: Optional override for the torch device used when
                resizing with :func:`torch.nn.functional.interpolate` (**bilinear**;
                point-sampled by default, antialiased with ``antialias=True``).
                Batched ``(N, H, W, C)`` always uses this path; default device is
                ``cuda:0`` if CUDA is available, otherwise ``cpu``. For
                **unbatched** ``(H, W, C)``, PIL + LANCZOS is used unless this
                argument is set, in which case a single batch is resized on the
                given device.
            antialias: Antialias the torch bilinear downscale so it matches PIL
                ``Image.BILINEAR`` (the hvla training resize). Only affects the
                torch paths above; the unbatched PIL LANCZOS path is unchanged.
                Default ``False`` keeps the droid / Pi inputs byte-identical.

        Returns:
            ``uint8`` array of shape ``(size[0], size[1], 3)`` or, when batched,
            ``(N, size[0], size[1], 3)``.
        """
        image = np.asarray(image)

        if image.ndim == 4:
            dev = _resolve_interpolate_device(interpolate_device)
            return _resize_nhwc_batched_torch(image, size, dev, antialias=antialias)
        if image.ndim != 3:
            raise ValueError(
                f"Expected image of shape (H, W, C) or (N, H, W, C); got {image.shape}"
            )

        dev: torch.device | None = None
        if interpolate_device is not None:
            dev = torch.device(interpolate_device)

        if dev is not None:
            batch = image[np.newaxis, ...]
            out = _resize_nhwc_batched_torch(batch, size, dev, antialias=antialias)
            return out[0]

        if image.shape[-1] == 4:
            image = image[..., :3]

        if np.issubdtype(image.dtype, np.floating):
            image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)

        if (image.shape[0], image.shape[1]) != size:
            pil_img = Image.fromarray(image)
            pil_img = pil_img.resize((size[1], size[0]), Image.LANCZOS)
            image = np.asarray(pil_img)

        return image

    @staticmethod
    def process_depth(
        depth: np.ndarray,
        size: tuple[int, int] = (224, 224),
        *,
        max_depth_m: float | None = None,
        interpolate_device: str | torch.device | None = None,
    ) -> np.ndarray:
        """Resize raw metric depth to the policy grid, preserving meters.

        Deliberately **not** :meth:`process_image`: bilinear interpolation
        invents depths across object silhouettes and the uint8 round-trip
        destroys the metric scale, so this uses nearest-neighbour on float32
        and never rescales the values.

        Args:
            depth: ``(H, W[, 1])`` or batched ``(N, H, W[, 1])`` float depth in
                **meters** (Isaac Lab's ``distance_to_image_plane``).
            size: Target ``(height, width)``.
            max_depth_m: When set, depths above it (and any negative or
                non-finite ones) are marked invalid by setting them to ``0.0``,
                matching the dataloader's decode of the training videos.
            interpolate_device: Torch device for the resize; defaults to
                ``cuda:0`` when available, otherwise ``cpu``.

        Returns:
            ``(N, size[0], size[1])`` float32 depth in meters.
        """
        d = _depth_as_nhw(depth)
        dev = _resolve_interpolate_device(interpolate_device)
        t = torch.from_numpy(np.ascontiguousarray(d)).to(dev).unsqueeze(1)  # (N,1,H,W)
        t = F.interpolate(t, size=size, mode="nearest")
        out = t.squeeze(1).cpu().numpy()
        if max_depth_m is not None:
            # Background pixels come back as +inf under
            # ``depth_clipping_behavior="none"``; the training decode maps every
            # out-of-range sample to the same "no measurement" sentinel of 0.
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            out[(out < 0.0) | (out > max_depth_m)] = 0.0
        return out

    @staticmethod
    def get_image(obs: dict[str, Any], camera_name: str) -> np.ndarray:
        """Extract a raw image array from the ToyBlocks observation dict.

        Expects the standard ``obs["images"][camera_name]["rgb"]`` layout.
        """
        return np.asarray(obs["images"][camera_name]["rgb"])

    @staticmethod
    def get_depth(obs: dict[str, Any], camera_name: str) -> np.ndarray:
        """Raw metric depth for one camera (``obs["images"][cam]["depth"]``).

        Present only when the env ran with a ``depth`` component in its
        ``obs_mode``; the ``KeyError`` names the missing piece.
        """
        cam = obs["images"][camera_name]
        if "depth" not in cam:
            raise KeyError(
                f"camera {camera_name!r} has no 'depth' entry (keys: {sorted(cam)}); "
                "run the env with 'depth' in --obs_mode (e.g. 'rgb+state+depth')."
            )
        return np.asarray(cam["depth"])

    @staticmethod
    def get_intrinsics(obs: dict[str, Any], camera_name: str) -> np.ndarray:
        """``(N, 3, 3)`` native-resolution pinhole intrinsics for one camera."""
        cam = obs["images"][camera_name]
        if "intrinsics" not in cam:
            raise KeyError(
                f"camera {camera_name!r} has no 'intrinsics' entry (keys: {sorted(cam)}); "
                "run the env with 'depth' in --obs_mode (e.g. 'rgb+state+depth')."
            )
        K = np.asarray(cam["intrinsics"], dtype=np.float32)
        if K.shape[-2:] != (3, 3):
            raise ValueError(f"Expected (3,3) or (N,3,3) intrinsics; got {K.shape}")
        return K.reshape(-1, 3, 3)

    @staticmethod
    def get_joint_pos(obs: dict[str, Any]) -> np.ndarray:
        """Extract joint positions from the ToyBlocks observation dict."""
        return np.asarray(obs["agent"]["joint_pos"])

    def action_postprocess(self, result: dict[str, Any]) -> dict[str, Any]:
        """Optional hook: transform the server's response dict before returning to the client.

        Default is identity. Subclasses override this to, e.g., slice policy
        actions to match a specific env action space. :class:`ObsTransformPolicy`
        calls this on every ``infer`` result.
        """
        return result


# ---------------------------------------------------------------------------
# Policy wrapper
# ---------------------------------------------------------------------------


class ObsTransformPolicy:
    """Wraps a ``BasePolicy``-compatible object, applying an observation
    transform before every ``infer`` call.

    This class itself satisfies the ``BasePolicy`` interface (``infer`` /
    ``reset``) so it composes cleanly with other wrappers such as
    ``ActionChunkBroker`` and ``PolicyClientAdapter``.
    """

    def __init__(self, policy: Any, transform: BaseObsTransform) -> None:
        self._policy = policy
        self._transform = transform
        self.last_transformed_obs: dict[str, Any] | None = None

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        transformed = self._transform(obs)
       
        self.last_transformed_obs = transformed

        # A transform builds its output dict from scratch, so a client-side
        # control key (the ``__...__`` request channel, injected by
        # PolicyClientAdapter rather than produced by the env) does not survive
        # the mapping unless it is forwarded here.
        
        # Forward dunder-prefixed control keys (e.g. __episode_seed__,
        # __save_first_batch_dir__, __ll_input_trace__) that the server pops before inference.
        if isinstance(obs, dict):
            for k, v in obs.items():
                if k.startswith("__") and k.endswith("__") and k not in transformed:
                    transformed[k] = v
        result = self._policy.infer(transformed)
        return self._transform.action_postprocess(result)

    def reset(self) -> None:
        if hasattr(self._policy, "reset"):
            self._policy.reset()
        if hasattr(self._transform, "reset"):
            self._transform.reset()

    def reset_envs(self, env_ids: Any) -> None:
        """Reset only the state belonging to ``env_ids``.

        Forwarded to the wrapped policy and transform when they support it. A
        transform with no per-env reset is left untouched rather than falling
        back to :meth:`reset`, whose per-episode state is global: collapsing it
        at one env's episode boundary would corrupt the envs still mid-episode
        (e.g. ``ManiflowObsTransform`` would drop its per-env ``pd_mode`` column
        back to a single scalar for every env).
        """
        for target in (self._policy, self._transform):
            reset_envs = getattr(target, "reset_envs", None)
            if reset_envs is not None:
                reset_envs(env_ids)

    @property
    def transform(self) -> BaseObsTransform:
        return self._transform


def find_obs_transform_policy(policy: Any) -> ObsTransformPolicy | None:
    """The :class:`ObsTransformPolicy` wrapper in a policy chain, if any."""
    for node in walk_policy_chain(policy):
        if isinstance(node, ObsTransformPolicy):
            return node
    return None


def find_obs_transform(policy: Any) -> BaseObsTransform | None:
    """The :class:`BaseObsTransform` inside a wrapped policy, if any.

    A diagnostic that wants to render the arrays inference actually sends has to
    reach through the client wrappers (see :func:`walk_policy_chain`) rather than
    build a second transform, which would drift from the negotiated one the
    moment a server-advertised setting changes.
    """
    node = find_obs_transform_policy(policy)
    return node.transform if node is not None else None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _lazy_registry() -> dict[str, type[BaseObsTransform]]:
    """Build the registry on first access so subclass modules are imported."""
    from toyblocks.serving.obs_transforms.maniflow import ManiflowObsTransform
    from toyblocks.serving.obs_transforms.maniflow_blocks_table import (
        ManiflowBlocksTableObsTransform,
    )

    return {
        "maniflow": ManiflowObsTransform,
        "maniflow_blocks_table": ManiflowBlocksTableObsTransform,
    }


_REGISTRY_CACHE: dict[str, type[BaseObsTransform]] | None = None


def get_obs_transform_registry() -> dict[str, type[BaseObsTransform]]:
    """Return the name -> class mapping of available observation transforms."""
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        _REGISTRY_CACHE = _lazy_registry()
    return _REGISTRY_CACHE
