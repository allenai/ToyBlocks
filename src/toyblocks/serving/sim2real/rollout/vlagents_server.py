"""Start a vlagents server for the ManiFlow policy from a checkpoint.

Thin, maniflow-specific wrapper around ``python -m vlagents start-server maniflow
--kwargs '...'`` (see vlagents/README.md): instead of a raw JSON kwargs blob, it
exposes the checkpoint (and the other constructor args of
:class:`~vlagents.policies.ManiFlowPolicy`) as plain CLI flags.

Once running, point a client's ``vlagents_host``/``vlagents_port`` (e.g.
``robot-control-stack/examples/inference/franka.py``'s config) at this server's
``--host``/``--port``.

Example::

python -m toyblocks.serving.sim2real.rollout.vlagents_server \
    --checkpoint $CK \
   --stats-path $CK/stats.json \
   --train-config $CK/config.json \
   --joint-only-state    
   # only required for 9d obs mode

``--checkpoint`` also takes the checkpoint file itself
(``$CK/steps_140000_model.safetensors``); given a directory it loads the
highest ``steps_<N>_model.safetensors`` found there (or in ``$CK/checkpoints/``)
and warns when the directory holds more than one.
   

missing keys in config.json:

 framework:
  encoder_output_dim: 768

    shape_meta:
    action:
      shape:
        - 9

"""

import argparse
import json
import logging
from pathlib import Path
import re
from typing import Any

import numpy as np

from vlagents import register_agent
from vlagents.__main__ import start_server
from vlagents.server import AgentService
from vlagents.policies.interface import Act, Agent, Obs, SingleAct

logger = logging.getLogger(__name__)

AGENT_NAME = "maniflow"

# Target-centered front crop the client sends when it runs with
# ``--use-target-front-crop`` (see sim2real/rollout/inference.py).
TARGET_CROP_KEY = "target_front_crop_rgb"

CHECKPOINT_SUBDIR = "checkpoints"


def _checkpoint_sort_key(path: Path) -> tuple[int, float]:
    """Order checkpoints by training step, falling back to mtime.

    Files are named ``steps_<N>_model.safetensors``; anything without a step
    in its name sorts below every numbered one and is then broken by mtime.
    """
    match = re.search(r"steps_(\d+)", path.name)
    step = int(match.group(1)) if match else -1
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (step, mtime)


def resolve_checkpoint(checkpoint: str) -> str:
    """Resolve ``--checkpoint`` to a single ``*.safetensors`` file.

    A file path is returned untouched. A directory is searched for
    ``*.safetensors`` -- first flat (the three-file serving layout:
    ``config.yaml`` + ``stats.json`` + ``steps_<N>_model.safetensors``), then in
    a ``checkpoints/`` subdirectory (the raw training run layout) -- and the
    latest by training step wins. When more than one candidate exists the
    choice is ambiguous, so warn with the full list; pass the file explicitly
    to silence it.
    """
    candidate = Path(checkpoint).expanduser()
    if not candidate.is_dir():
        return str(candidate)

    found = sorted(
        (p for p in candidate.glob("*.safetensors") if p.is_file()),
        key=_checkpoint_sort_key,
    )
    if not found:
        found = sorted(
            (p for p in (candidate / CHECKPOINT_SUBDIR).glob("*.safetensors") if p.is_file()),
            key=_checkpoint_sort_key,
        )
    if not found:
        raise FileNotFoundError(
            f"no *.safetensors under {candidate} or {candidate / CHECKPOINT_SUBDIR}; "
            f"pass --checkpoint pointing at the checkpoint file itself"
        )

    latest = found[-1]
    if len(found) > 1:
        logger.warning(
            "%d checkpoints in %s: %s -- picking the latest (%s). Pass "
            "--checkpoint <file> to choose a different one.",
            len(found),
            candidate,
            ", ".join(p.name for p in found),
            latest.name,
        )
    else:
        logger.info("resolved checkpoint directory %s to %s", candidate, latest.name)
    return str(latest)


class ManiFlowPolicy(Agent):
    """vlagents wrapper around the ManiFlow inference adapter.

    Rather than re-implementing ManiFlow's normalization, this delegates to
    ``toyblocks.serving.serve_maniflow_policy.ManiFlowInferAdapter`` (built via
    ``load_safetensors_policy``), which owns state q99-normalization, per-key
    action denormalization, relative->absolute cumsum, and gripper/play_mode
    binarization. This class only translates a vlagents :class:`Obs` into the
    adapter's obs dict and buffers the returned action chunk.
    """

    def __init__(
        self,
        default_checkpoint_path: str = "",
        device: str = "cuda:0",
        stats_path: str = "",
        train_config_path: str | None = None,
        action_mode: str | None = None,
        use_ema: bool = False,
        num_inference_steps: int | None = None,  # flow ode steps
        rename_map: dict[str, str] | None = None,
        state_key: str = "agent_pos",
        max_depth_m: float = 2.5,
        debug_visualize_heatmap: bool = False,
        debug_visualize_dir: str = "heatmap_debug",
        debug_visualize_crop: bool = False,
        joint_only_state: bool = False,
        include_play_mode: bool = True,
        **kwargs,
    ) -> None:
        """For now we assume wrapped maniflow will take 19d state:
        [7 joint pos, 7 joint vel, 3 gripper pos, 1 gripper width, 1 PD], but we will only input the joint angles for now.

        action 9d: [7 joint pos, 1 gripper in [-1, 1.], 1 play mode]

        Whether the policy consumes language, heatmap, or depth conditioning is
        auto-detected from the loaded checkpoint's encoder in
        :meth:`initialize`
        """
        super().__init__(default_checkpoint_path=default_checkpoint_path, **kwargs)
        self.device = device
        self.stats_path = stats_path
        self.config_path = train_config_path
        self.action_mode = action_mode
        self.use_ema = use_ema
        self.num_inference_steps = num_inference_steps
        self.rename_map = (
            rename_map or {}
        )  # optional if we ever need to rename obs->policy_dict
        self.state_key = state_key
        self.max_depth_m = float(max_depth_m)
        self.debug_visualize_heatmap = bool(debug_visualize_heatmap)
        self.debug_visualize_dir = str(debug_visualize_dir)
        self._debug_heatmap_counter = 0
        self.debug_visualize_crop = bool(debug_visualize_crop)
        self._debug_crop_counter = 0
        self.joint_only_state = joint_only_state
        self.path = self.checkpoint_path or self.default_checkpoint_path
        self.include_play_mode = include_play_mode
        if self.checkpoint_step is not None:
            self.path = self.path.format(checkpoint_step=self.checkpoint_step)
        # Accept a checkpoint directory and pick the latest step out of it, so
        # the path reported over RPC is always the concrete file that loaded.
        self.path = resolve_checkpoint(self.path)
        # Populated by initialize() by introspecting the loaded encoder.
        self.use_language: bool = False
        self.use_heatmap: bool = False
        self.use_depth: bool = False
        self.depth_img_size: int = 224

    def initialize(self):
        try:
            import hvla  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "hvla is not installed. It ships with the `maniflow` extra as an editable "
                "install of maniflow-tb (e.g. `uv sync --extra maniflow`)."
            ) from e

        from toyblocks.serving.serve_maniflow_policy import (
            Args,
            load_safetensors_policy,
            ManiFlowInferAdapter,
        )

        args = Args(
            checkpoint=self.path,
            train_config=self.config_path,
            use_ema=self.use_ema,
            stats_path=self.stats_path,
            action_mode=self.action_mode,
        )
        # None keeps the checkpoint's flow integration steps.
        self.adapter: ManiFlowInferAdapter = load_safetensors_policy(
            args, num_inference_steps=self.num_inference_steps
        )
        self.adapter._policy.to(self.device)

        policy = self.adapter._policy
        encoder = policy.obs_encoder
        self.use_language = bool(getattr(policy, "language_conditioned", False))
        self.use_heatmap = bool(getattr(encoder, "use_heatmap", False))
        self.use_depth = bool(getattr(encoder, "use_ptmap", False)) and bool(
            getattr(encoder, "depth_keys", []) or []
        )
        
        # Depth preprocessing HxW must match the encoder that later
        # back-projects depth into a pointmap; otherwise intrinsics scaling
        # is off and the pointmap silently corrupts. Pin to the encoder's
        # own img_size (guaranteed to match the checkpoint).
        self.depth_img_size = int(getattr(encoder, "img_size", 224))

        logger.info(
            "auto-detected conditioning from checkpoint: language=%s heatmap=%s "
            "depth=%s (depth_img_size=%d, max_depth_m=%.2f)",
            self.use_language,
            self.use_heatmap,
            self.use_depth,
            self.depth_img_size,
            self.max_depth_m,
        )

    def _build_obs_dict(self, obs: Obs) -> dict[str, Any]:
        """Build an obs dict for the adapter.

        Args: obs dict that contains keys that policy needs
        """
        assert (
            len(obs.obs) == 1
        ), "ManiFlowPolicy currently expects exactly one robot observation"
        _, single_obs = next(iter(obs.obs.items()))
        obs_dict: dict[str, Any] = {}
        for source_key, value in single_obs.cameras.items():
            key = self.rename_map.get(source_key, source_key)
            obs_dict[key] = np.asarray(value)
            if self.debug_visualize_crop and key == TARGET_CROP_KEY:
                self._save_target_crop(obs_dict[key])

        state_parts: list[np.ndarray] = []
        assert single_obs.joints is not None
        state_parts.append(np.asarray(single_obs.joints, dtype=np.float32))

        if not self.joint_only_state:
            state_parts.append(np.asarray(single_obs.info["velo"], dtype=np.float32))
            state_parts.append(np.asarray(single_obs.info["ee_xyz"], dtype=np.float32))

        assert single_obs.gripper is not None
        # convert gripper from rcs convention (0=closed, 1=open) to maniflow convention (1=open, 0=closed)
        state_parts.append(np.asarray([1.0 - single_obs.gripper], dtype=np.float32))
        # state_parts.append(np.asarray([single_obs.gripper], dtype=np.float32))

        if self.include_play_mode or not self.joint_only_state:
            # at the moment 19d (so joint_only_state=False) always includes play mode
            state_parts.append(
                np.asarray([single_obs.info["play_mode"]], dtype=np.float32)
            )

        obs_dict[self.state_key] = np.concatenate(state_parts).astype(np.float32)

        if obs.language_instruction is not None and self.use_language:
            obs_dict["task_name"] = [obs.language_instruction]

        target_heatmap = single_obs.info.get("target_heatmap")
        if target_heatmap is not None and self.use_heatmap:
            obs_dict["target_heatmap"] = np.asarray(target_heatmap, dtype=np.float32)
        elif target_heatmap is not None and not self.use_heatmap:
            logger.warning(
                "client sent target_heatmap %s but use_heatmap=false; dropping",
                np.asarray(target_heatmap).shape,
            )
        elif self.use_heatmap:
            logger.debug("use_heatmap=true but obs.info has no target_heatmap")

        if (
            self.debug_visualize_heatmap
            and target_heatmap is not None
            and self.use_heatmap
        ):
            self._save_heatmap_overlay(single_obs, target_heatmap)


        #   - depth: [B, 1, 1, img_size, img_size] float32 meters, NEAREST resize
        #   - camera_intrinsics: [B, 1, N_cams, 4] float32, K rescaled to img_size
        depths = single_obs.info.get("depths")
        if depths and self.use_depth:
            self._add_depth_to_obs_dict(single_obs, depths, obs_dict)
        elif depths and not self.use_depth:
            logger.warning(
                "client sent depth cameras %s but the loaded checkpoint does not "
                "consume depth (encoder.use_ptmap/depth_keys); dropping",
                list(depths.keys()),
            )
        elif self.use_depth:
            raise ValueError(
                "the loaded checkpoint consumes depth but obs.info has no "
                "depths; check the client's InferenceConfig.include_depth and "
                "that the camera set exposes depth streams"
            )

        # print(obs)
        # breakpoint()
        return obs_dict

    def _add_depth_to_obs_dict(
        self,
        single_obs,
        depths: dict[str, Any],
        obs_dict: dict[str, Any],
    ) -> None:
        """Preprocess depth (NEAREST resize, mm->m, add batch/obs/channel dims)
        and pack ``camera_intrinsics`` in encoder order.

        The encoder iterates ``depth_keys`` sorted alphabetically
        (:mod:`hvla.model.modules.maniflow.flow_policy` sorts them out of
        ``shape_meta``), so we stack intrinsics in the same sorted order so
        that ``obs['camera_intrinsics'][cam_i]`` lines up with
        ``obs['<cam_i>_depth']``.
        """
        from PIL import Image as _PILImage

        intrinsics_by_key = single_obs.info.get("camera_intrinsics") or {}
        target = self.depth_img_size or 224

        for depth_key in sorted(depths.keys()):
            raw = np.asarray(depths[depth_key])
            if raw.ndim != 2:
                logger.warning(
                    "skipping depth %r with unexpected shape %s (expected (H, W))",
                    depth_key,
                    raw.shape,
                )
                continue
            depth_m = raw.astype(np.float32) / 1000.0
            depth_m[raw == 0] = 0.0
            depth_m[depth_m > self.max_depth_m] = 0.0
            if depth_m.shape != (target, target):
                depth_m = np.array(
                    _PILImage.fromarray(depth_m).resize(
                        (target, target), _PILImage.NEAREST
                    ),
                    dtype=np.float32,
                )
            # (B=1, T_obs=1, C=1, H, W) to match franka_multimodal_collate_fn.
            depth_batched = depth_m[None, None, None, :, :]
            out_key = self.rename_map.get(depth_key, depth_key)
            obs_dict[out_key] = depth_batched

        # Build camera_intrinsics in the encoder's sorted-key order. Depth
        # intrinsics are load-bearing here: the encoder does
        # X = (u - cx) * depth / fx, Y = (v - cy) * depth / fy, so a missing
        # or all-zero K would silently produce a zero XYZ pointmap. Fail loud.
        intrinsics_stack: list[list[float]] = []
        for depth_key in sorted(depths.keys()):
            raw = np.asarray(depths[depth_key])
            if raw.ndim != 2:
                continue
            native_h, native_w = raw.shape
            K_raw = intrinsics_by_key.get(depth_key)
            if K_raw is None:
                raise ValueError(
                    f"the loaded checkpoint consumes depth and client sent "
                    f"depth {depth_key!r} but no camera_intrinsics for it. "
                    f"Depth->pointmap back-projection requires the depth "
                    f"stream's K matrix."
                )
            K = np.asarray(K_raw, dtype=np.float32).reshape(-1)
            if K.size < 9:
                raise ValueError(
                    f"camera_intrinsics for {depth_key!r} has {K.size} elements; "
                    f"expected >=9 (3x3 K) or >=12 (3x4 [K | 0])."
                )
            # Support 3x3 (parquet-style) or 3x4 (RCS obs stores color/depth
            # intrinsics as a 3x4 projection matrix; K lives in [:3, :3]).
            if K.size >= 12:
                K3 = K[:12].reshape(3, 4)[:, :3]
            else:
                K3 = K[:9].reshape(3, 3)
            fx = float(K3[0, 0])
            fy = float(K3[1, 1])
            cx = float(K3[0, 2])
            cy = float(K3[1, 2])
            if fx <= 0.0 or fy <= 0.0:
                raise ValueError(
                    f"camera_intrinsics for {depth_key!r} has non-positive focal "
                    f"length (fx={fx}, fy={fy}); back-projection would divide "
                    f"by zero or flip axes. Check camera calibration."
                )
            sx = target / float(native_w)
            sy = target / float(native_h)
            intrinsics_stack.append([fx * sx, fy * sy, cx * sx, cy * sy])

        if intrinsics_stack:
            # (B=1, T_obs=1, N_cams, 4) to match the training collate.
            obs_dict["camera_intrinsics"] = np.asarray(
                [[intrinsics_stack]], dtype=np.float32
            )

    def _save_target_crop(self, crop: np.ndarray) -> None:
        """Save ``target_front_crop_rgb`` exactly as it arrived, so the framing
        the encoder sees can be checked against the goal point that produced it.
        An all-black PNG means the client had no goal point."""
        from PIL import Image

        out_dir = Path(self.debug_visualize_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"crop_{self._debug_crop_counter:06d}.png"
        Image.fromarray(np.asarray(crop).astype(np.uint8)).save(path)
        self._debug_crop_counter += 1
        if self._debug_crop_counter % 25 == 1:
            logger.info("debug_visualize_crop: wrote %s", path)

    def _save_heatmap_overlay(
        self,
        single_obs,
        target_heatmap: np.ndarray,
    ) -> None:
        """Save a PNG of the target heatmap (already 224x224, built by
        ``goal_utils.build_target_heatmap``) blended over front_rgb resized to
        match, so we can visually confirm click -> heatmap -> obs alignment.
        """
        from PIL import Image
        import cv2

        hm = np.asarray(target_heatmap, dtype=np.float32).squeeze()
        if hm.ndim == 3:  # (3, H, W) - three identical channels
            hm = hm[0]
        H, W = hm.shape

        rgb = np.asarray(single_obs.cameras["front_rgb"])
        rgb = np.asarray(Image.fromarray(rgb).resize((W, H), Image.BILINEAR))

        hm_color = cv2.cvtColor(
            cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET),
            cv2.COLOR_BGR2RGB,
        )
        overlay = cv2.addWeighted(rgb, 0.6, hm_color, 0.4, 0)

        out_dir = Path(self.debug_visualize_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"heatmap_{self._debug_heatmap_counter:06d}.png"
        Image.fromarray(overlay).save(path)
        self._debug_heatmap_counter += 1
        if self._debug_heatmap_counter % 25 == 1:
            logger.info("debug_visualize_heatmap: wrote %s", path)

    def act(self, obs: Obs) -> Act:
        """Args:
        obs dict
        returns chunk (T, D) where D = 8 with 7 joint angles + gripper in [0., 1.]
        """
        super().act(obs)
        result = self.adapter.infer(
            self._build_obs_dict(obs)
        )  # maniflow returns (T, B, D)
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim == 3:  # pop off batch
            actions = actions[:, 0, :]

        actions_without_play_mode = actions[..., :-1]
        # postprocess_actions emits the gripper in the BinaryJointAction
        # convention (-1=close, +1=open) the sim needs; rcs wants 0=closed,
        # 1=open. ``1 - g`` was right for the old 0/1 output, but on +-1 it
        # sends 2 for close and 0 -- closed -- for open.
        actions_without_play_mode[..., -1] = (actions_without_play_mode[..., -1] + 1.0) / 2.0

        assert (
            len(obs.obs) == 1
        ), "ManiFlowPolicy currently expects exactly one robot observation"
        robot_name = next(iter(obs.obs))
        return Act(
            acts=[
                {
                    robot_name: SingleAct(
                        action=action_step[:-1].astype(np.float32),
                        gripper=float(action_step[-1]),
                        done=False,
                        info={"play_mode": float(actions[step_idx, -1])},
                    )
                }
                for step_idx, action_step in enumerate(actions_without_play_mode)
            ]
        )


register_agent(AGENT_NAME, ManiFlowPolicy)


def _exposed_checkpoint_path(service: AgentService) -> str:
    """RPC: the checkpoint this server loaded, as ``conn.root.checkpoint_path()``.

    vlagents' ``start_server`` always builds a plain :class:`AgentService`, so
    the class itself is the only place to hang an extra call. rpyc resolves
    ``root.checkpoint_path`` to ``exposed_checkpoint_path``; a client talking to
    a server without this raises AttributeError and falls back to its own config
    (see ``remote_agent_utils.remote_checkpoint_path``).
    """
    return str(getattr(service.agent, "path", "") or "")


AgentService.exposed_checkpoint_path = _exposed_checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Path to the ManiFlow checkpoint (safetensors), or to a checkpoint "
            "directory -- then the latest step in it (or in its checkpoints/ "
            "subdirectory) is used, with a warning when several are present."
        ),
    )
    parser.add_argument(
        "--stats-path",
        required=True,
        help="Path to the stats.json used for state normalization.",
    )
    parser.add_argument(
        "--train-config", default=None, help="Optional path to the training config."
    )
    parser.add_argument("--device", default="cuda:0", help="cpu or cuda:N")
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--joint-only-state",
        default=False,
        action="store_true",
        help="Whether to only use joint angles in the state.",
    )
    parser.add_argument(
        "--max_depth_m",
        type=float,
        default=2.5,
        help=(
            "Depth values (post mm->m) above this are zeroed, matching "
            "franka_dataset.MAX_DEPTH_M."
        ),
    )
    parser.add_argument(
        "--debug-visualize-heatmap",
        dest="debug_visualize_heatmap",
        action="store_true",
        help=(
            "Save a PNG per act() call showing the 224x224 target heatmap "
            "blended over the BILINEAR-resized front_rgb (i.e. exactly what "
            "the encoder sees). Only fires when the loaded checkpoint uses "
            "heatmap conditioning and the client actually sent a heatmap."
        ),
    )
    parser.add_argument(
        "--debug-visualize-dir",
        dest="debug_visualize_dir",
        default="heatmap_debug",
        help="Output directory for --debug-visualize-heatmap PNGs.",
    )
    parser.add_argument(
        "--debug-visualize-crop",
        dest="debug_visualize_crop",
        action="store_true",
        help=(
            "Save a PNG per act() call of the target_front_crop_rgb camera as it "
            "arrived from the client (into --debug-visualize-dir). Only fires "
            "when the client runs with --use-target-front-crop."
        ),
    )
    parser.add_argument(
        "--no-play-mode",
        default=True,
        action="store_false",
        help="Whether not to include play mode in the state.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.info(
        "serving %s on %s:%d (checkpoint=%s)",
        AGENT_NAME,
        args.host,
        args.port,
        args.checkpoint,
    )

    kwargs = dict(
        default_checkpoint_path=args.checkpoint,
        stats_path=args.stats_path,
        train_config_path=args.train_config,
        device=args.device,
        use_ema=args.use_ema,
        joint_only_state=args.joint_only_state,
        include_play_mode=args.no_play_mode,
        max_depth_m=args.max_depth_m,
        debug_visualize_heatmap=args.debug_visualize_heatmap,
        debug_visualize_dir=args.debug_visualize_dir,
        debug_visualize_crop=args.debug_visualize_crop,
    )
    start_server(AGENT_NAME, json.dumps(kwargs), port=args.port, host=args.host)


if __name__ == "__main__":
    main()
