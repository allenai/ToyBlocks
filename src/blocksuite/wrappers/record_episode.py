"""RecordEpisode wrapper for vectorized Isaac Lab environments.

Adapted from ManiSkill's RecordEpisode to work with Isaac Lab's GPU-vectorized
environments that perform automatic partial resets inside ``step()``.
"""

from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np
import torch

from blocksuite.tasks.base_env import BaseEnv
from blocksuite.utils.common_utils import to_numpy
from blocksuite.utils.dataset_recording_utils import (
    IsaacDataRecorderBase,
    _extract_frame,
)
from blocksuite.vecgen.recording.lerobot_recorder import IsaacDataRecorderLeRobot
from blocksuite.utils.visualization_utils import images_to_video, tile_images


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _index_dict_array(data, index):
    """Index into a (possibly nested-dict-of) array along the first axis."""
    if isinstance(data, dict):
        return {k: _index_dict_array(v, index) for k, v in data.items()}
    return data[index]


# Env-batched keys the eval loop injects into the obs that the HL latency video draws.
_HL_VIDEO_OBS_KEYS: tuple[str, ...] = (
    "target_heatmap",
    "hl_overlay_text",
    "hl_fresh_heatmap",
    "hl_fresh_text",
    "hl_conditioning_image",
)


def _owned_copy(x: Any) -> Any:
    """A CPU copy of ``x`` that later writes to the source cannot change."""
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu", copy=True).numpy()
    if isinstance(x, np.ndarray):
        return x.copy()
    if isinstance(x, list):
        return list(x)
    if isinstance(x, tuple):
        return tuple(x)
    return x


def _rows_copy(x: Any, rows: tuple[int, ...]) -> np.ndarray:
    """``x[rows]`` as a fresh CPU array: one transfer for the whole batch."""
    if isinstance(x, torch.Tensor):
        if rows != tuple(range(x.shape[0])):
            x = x[list(rows)]
        return x.detach().to("cpu", copy=True).numpy()
    return np.asarray(x)[list(rows)]


@dataclass
class _VideoObsSnapshot:
    """One step's sensor-obs video inputs, copied off the GPU in one go.

    ``rgb[cam][j]`` is env ``env_ids[j]``'s frame; ``hl`` holds the
    :data:`_HL_VIDEO_OBS_KEYS` present in the obs, still batched over all envs.
    """

    env_ids: tuple[int, ...]
    rgb: dict[str, np.ndarray]
    hl: dict[str, Any]


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class RecordEpisode(gym.Wrapper):
    """Record trajectories and/or videos for vectorized Isaac Lab environments.

    This wrapper handles the fact that Isaac Lab environments perform automatic
    partial resets inside ``step()`` rather than through explicit ``reset()``
    calls.  Done environments are detected each step via the ``_final_info``
    mask in the info dict, and their trajectory buffers are flushed to disk
    immediately.

    Trajectory data is stored as an HDF5 file paired with a JSON metadata
    sidecar.  Each trajectory group ``traj_{id}`` contains:

    - ``obs``:  observations with shape ``[T+1, ...]`` (includes initial obs)
    - ``actions``:  shape ``[T, action_dim]``
    - ``rewards``:  shape ``[T]``
    - ``terminated`` / ``truncated``:  boolean, shape ``[T]``
    - ``success`` / ``fail`` (optional):  boolean, shape ``[T]``

    Args:
        env: Gymnasium-wrapped Isaac Lab environment (may already be wrapped
            by e.g. :class:`FlattenObservationWrapper`).
        output_dir: Directory for output files.
        save_trajectory: Save trajectory data to HDF5.
        trajectory_name: Base filename (no extension). Defaults to a
            timestamp.
        save_rendering_video: Save rendered videos (from ``env.render()`` or
            the first camera sensor on the base env).
        info_on_video: *Reserved* – overlay info text on video frames (not
            yet implemented).
        save_on_reset: Flush remaining incomplete trajectories / videos when
            :meth:`reset` is called.
        save_video_trigger: ``f(elapsed_steps) -> bool`` controlling when
            frames are captured.  ``None`` means always capture.
        max_steps_per_video: Flush the video buffer every *N* steps.
            **Required** when ``num_envs > 1`` because partial resets make
            per-episode video cuts ambiguous.
        clean_on_close: Prune empty trajectories on :meth:`close`.
        record_reward: Include per-step rewards in trajectory data.
        video_fps: Frames per second for output MP4 files.
        data_recorder: Trajectory backend implementing :class:`IsaacDataRecorderBase`.
            Use ``None`` when ``save_trajectory`` is False. When ``save_trajectory``
            is True, pass a concrete recorder (e.g. :class:`~blocksuite.utils.dataset_recording_utils.IsaacDataRecorderHDF5`
            or :class:`~blocksuite.vecgen.recording.lerobot_recorder.IsaacDataRecorderLeRobot`).
            The active recorder may be read or replaced via the :attr:`data_recorder`
            property.
        source_type: Short label (e.g. ``"rl"``, ``"human"``).
        source_desc: Longer description of data provenance.
        video_env_ids: Subset of environment indices to composite into each
            video frame when the render batch is ``(num_envs, H, W, C)``.
            A single id returns that env's image; multiple ids are tiled
            (same layout as the full-env grid, with ``nrows`` from the subset
            size). ``None`` keeps the tiled grid of *all* envs.
        save_sensor_obs_video: When True, append frames from ``obs["images"]``
            each step (one video per finished episode per env index in
            ``video_env_ids``). Mutually exclusive with the render-based
            ``save_rendering_video`` path for the same rollout.
        video_camera: Camera name for :func:`~blocksuite.utils.dataset_recording_utils._extract_frame`
            when ``save_sensor_obs_video`` is True. ``None`` tiles all cameras.
        rgb_video_subdir: Subdirectory of ``output_dir`` for observation-based MP4s.
        rendering_video_subdir: Subdirectory of ``output_dir`` for render-based MP4s
            (``save_rendering_video=True``).
        on_video_saved: Optional ``f(env_idx, episode_idx, mp4_path)`` called
            after each sensor-obs video is written, on the encode thread.
        should_save_episode: Optional ``f(env_idx) -> bool`` checked at flush
            time. ``False`` skips the disk write and per-env counter bump
            (saved files remain sequential). Used by queue eval to discard
            "phantom" episodes (queue exhausted before this env got a setup).
        should_save_in_flight_episode: Same, but consulted for an episode still
            running when the rollout ends (the ``close`` / ``_partial`` flushes).
            Queue eval keys this off ``current_setups`` rather than
            ``previous_setups``, since the in-flight episode is the current one.
    """

    def __init__(
        self,
        env,
        output_dir: str,
        save_trajectory: bool = True,
        trajectory_name: Optional[str] = None,
        save_rendering_video: bool = True,
        info_on_video: bool = False,
        save_on_reset: bool = True,
        save_video_trigger: Optional[Callable[[int], bool]] = None,
        max_steps_per_video: Optional[int] = None,
        clean_on_close: bool = True,
        record_reward: bool = True,
        video_fps: int = 30,
        data_recorder: Optional[IsaacDataRecorderBase] = None,
        source_type: Optional[str] = None,
        source_desc: Optional[str] = None,
        video_env_ids: Optional[tuple[int, ...]] = None,
        save_sensor_obs_video: bool = False,
        video_camera: Optional[str] = None,
        rgb_video_subdir: str = "sensor_obs_videos",
        rendering_video_subdir: str = "env_rendering_videos",
        on_video_saved: Optional[Callable[[int, int, Path], None]] = None,
        should_save_episode: Optional[Callable[[int], bool]] = None,
        should_save_in_flight_episode: Optional[Callable[[int], bool]] = None,
        hl_overlay_camera: str = "front_camera",
        save_hl_latency_video: bool = False,
        hl_latency_video_subdir: str = "hl_latency_videos",
    ) -> None:
        super().__init__(env)

        self.output_dir = Path(output_dir)
        if save_trajectory or save_rendering_video or save_sensor_obs_video:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.video_fps = video_fps
        self._elapsed_steps = 0
        self._video_id = -1
        self._video_steps = 0
        self._closed = False

        self.save_video_trigger = save_video_trigger
        self.max_steps_per_video = max_steps_per_video
        self.save_on_reset = save_on_reset
        self._save_rendering_video = save_rendering_video
        self._save_sensor_obs_video = save_sensor_obs_video
        self._rgb_video_camera = video_camera
        self._rgb_video_subdir = rgb_video_subdir
        self._rendering_video_subdir = rendering_video_subdir
        self.save_trajectory = save_trajectory
        self.clean_on_close = clean_on_close
        self.record_reward = record_reward
        self.info_on_video = info_on_video

        self._last_obs: Optional[Any] = None
        self._rgb_obs_frame_buffers: dict[int, list[np.ndarray]] = {}
        self._rgb_videos_saved = 0
        self._rgb_episode_counters: dict[int, int] = {}
        self._on_video_saved = on_video_saved
        self.should_save_episode: Optional[Callable[[int], bool]] = should_save_episode
        self.should_save_in_flight_episode: Optional[Callable[[int], bool]] = (
            should_save_in_flight_episode
        )

        self._hl_overlay_camera = hl_overlay_camera

        # HL-latency side-by-side video (real-time fresh HL | aged LL-view):
        # shares the sensor-obs per-episode lifecycle, draws on hl_overlay_camera.
        self._save_hl_latency_video = save_hl_latency_video
        self._hl_latency_video_subdir = hl_latency_video_subdir
        self._hl_latency_frame_buffers: dict[int, list[np.ndarray]] = {}

        if self.num_envs > 1 and save_rendering_video:
            assert max_steps_per_video is not None, (
                "For vectorized environments with num_envs > 1, "
                "max_steps_per_video must be specified to periodically flush "
                "videos, since partial resets make per-episode video saving "
                "ambiguous."
            )

        self._data_recorder = data_recorder
        # One-shot: consumed by the next step().
        self._skip_recording_env_ids: set[int] = set()

        if save_trajectory:
            assert (self._data_recorder)
            
        self.render_images: list[np.ndarray] = []
        self.video_nrows = int(np.sqrt(self.num_envs))
        if video_env_ids is not None:
            seen: set[int] = set()
            deduped: list[int] = []
            for e in video_env_ids:
                if e < 0 or e >= self.num_envs:
                    raise ValueError(
                        f"video_env_ids contains {e}, which is out of range for "
                        f"num_envs={self.num_envs}."
                    )
                if e not in seen:
                    seen.add(e)
                    deduped.append(e)
            self._video_env_ids: Optional[tuple[int, ...]] = (
                tuple(deduped) if deduped else None
            )
        else:
            self._video_env_ids = None

        if self._save_sensor_obs_video:
            if self._video_env_ids is None:
                raise ValueError(
                    "save_sensor_obs_video requires video_env_ids (subset of env indices)."
                )
            self._rgb_obs_frame_buffers = {e: [] for e in self._video_env_ids}
            self._rgb_episode_counters = {e: 0 for e in self._video_env_ids}
            if self._save_hl_latency_video:
                self._hl_latency_frame_buffers = {e: [] for e in self._video_env_ids}

        # Sensor-obs video work off the sim loop: one worker builds frames (one step
        # in flight, so frames stay in order); finished episodes encode on a pool.
        self._video_build_pool: Optional[ThreadPoolExecutor] = None
        self._video_build_future: Optional[Future] = None
        self._video_encode_pool: Optional[ThreadPoolExecutor] = None
        self._video_encode_futures: list[Future] = []
        if self._save_sensor_obs_video:
            self._video_build_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="videobuild"
            )
            self._video_encode_pool = ThreadPoolExecutor(
                max_workers=4, thread_name_prefix="videoencode"
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def data_recorder(self) -> Optional[IsaacDataRecorderBase]:
        return self._data_recorder

    @data_recorder.setter
    def data_recorder(self, recorder: Optional[IsaacDataRecorderBase]):
        self._data_recorder = recorder

    @property
    def num_envs(self) -> int:
        return self.base_env.num_envs

    @property
    def base_env(self) -> BaseEnv:
        return self.env.unwrapped

    @property
    def save_rendering_video(self) -> bool:
        if not self._save_rendering_video:
            return False
        if self.save_video_trigger is not None:
            return self.save_video_trigger(self._elapsed_steps)
        return True

    def set_video_env_ids(self, env_ids: tuple[int, ...]) -> None:
        """Switch which sub-envs receive RGB observation video frames."""
        if not self._save_sensor_obs_video:
            return
        self._wait_video_build()
        deduped: list[int] = []
        seen: set[int] = set()
        for env_idx in env_ids:
            if env_idx < 0 or env_idx >= self.num_envs:
                raise ValueError(
                    f"video_env_ids contains {env_idx}, which is out of range for "
                    f"num_envs={self.num_envs}."
                )
            if env_idx not in seen:
                seen.add(env_idx)
                deduped.append(env_idx)
        self._video_env_ids = tuple(deduped)
        for env_idx in self._video_env_ids:
            self._rgb_obs_frame_buffers.setdefault(env_idx, [])
            self._rgb_episode_counters.setdefault(env_idx, 0)
            if self._save_hl_latency_video:
                self._hl_latency_frame_buffers.setdefault(env_idx, [])

    def sync_recording_observation(self) -> Any:
        """Refresh ``_last_obs`` after sim state was written outside ``step``/``reset``.

        Replay teleports to recorded frame 0 via ``set_initial_state()`` after
        ``reset()``. The first sensor-obs video frame is taken from ``_last_obs``
        at the start of the first ``step()``; without this refresh it still
        shows the pre-teleport spawn layout.
        """
        obs = self.unwrapped._get_observations()
        self._last_obs = obs
        return obs

    def skip_recording_next_step(self, env_ids: Iterable[int]) -> None:
        """Leave ``env_ids`` out of the trajectory recording of the next :meth:`step` only.

        The step still happens for them, it just produces no row; the next
        recorded step's action then lands on their last recorded observation.
        Envs that finish their episode in that step are recorded anyway, so the
        next episode's reset frame is still captured.
        """
        self._skip_recording_env_ids.update(int(i) for i in env_ids)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_env_info(self) -> Optional[dict]:
        base = self.env.unwrapped
        if base.spec is None:
            return None
        return dict(
            env_id=base.spec.id,
            env_kwargs=getattr(base.spec, "kwargs", {}),
            max_episode_steps=getattr(base, "max_episode_length", None),
        )

    def _clear_env_buffer(self, env_idx: int):
        if self._data_recorder is None:
            return
        if isinstance(self._data_recorder, IsaacDataRecorderLeRobot):
            self._data_recorder.discard_episode(env_idx)
        else:
            self._data_recorder.clear_env_buffer(env_idx)

    def _rgb_obs_video_dir(self) -> Path:
        d = self.output_dir / self._rgb_video_subdir
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _rendering_video_dir(self) -> Path:
        d = self.output_dir / self._rendering_video_subdir
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _append_rgb_obs_frames(self, obs: Any) -> None:
        if not isinstance(obs, dict) or "images" not in obs:
            return
        assert self._video_env_ids is not None
        assert self._video_build_pool is not None
        # One step in flight: the previous step's frames land in the buffers
        # before this step's are queued, so per-env frame order is preserved.
        self._wait_video_build()
        snap = self._snapshot_video_obs(obs)
        self._video_build_future = self._video_build_pool.submit(
            self._build_video_frames, snap
        )

    def _wait_video_build(self) -> None:
        """Block until queued frames are in the buffers; re-raises a build error."""
        if self._video_build_future is not None:
            future, self._video_build_future = self._video_build_future, None
            future.result()

    def _snapshot_video_obs(self, obs: dict) -> _VideoObsSnapshot:
        """Copy this step's video inputs to CPU: one transfer per camera, all envs.

        Runs on the step thread, before the sim overwrites its camera buffers.
        """
        assert self._video_env_ids is not None
        images = obs["images"]
        return _VideoObsSnapshot(
            env_ids=self._video_env_ids,
            rgb={
                cam_name: _rows_copy(cam_data["rgb"], self._video_env_ids)
                for cam_name, cam_data in images.items()
            },
            hl=(
                {k: _owned_copy(obs[k]) for k in _HL_VIDEO_OBS_KEYS if k in obs}
                if self._save_hl_latency_video
                else {}
            ),
        )

    def _build_video_frames(self, snap: _VideoObsSnapshot) -> None:
        """Build and buffer each video env's frames (runs on the build thread)."""
        obs = snap.hl
        for j, env_idx in enumerate(snap.env_ids):
            env_obs_slice = {
                "images": {cam_name: {"rgb": rgb[j]} for cam_name, rgb in snap.rgb.items()}
            }
            frame = _extract_frame(env_obs_slice, self._rgb_video_camera)
            if frame is not None:
                self._rgb_obs_frame_buffers[env_idx].append(frame)
            if self._save_hl_latency_video:
                self._append_hl_latency_frame(env_idx, env_obs_slice, obs)

    @staticmethod
    def _hl_heatmap_2d(x, env_idx: int) -> Optional[np.ndarray]:
        """One env's 2D heatmap from a batched (B,1,3,H,W)/(B,H,W)/... key.

        Every HL key the eval loop injects is batched over envs, so a leading
        axis shorter than ``env_idx`` means the producer collapsed the batch and
        this env's heatmap does not exist; that raises rather than silently
        drawing env 0's map on every env's video.
        """
        if x is None:
            return None
        a = to_numpy(x)
        if a.ndim == 0:
            return None
        a = a[env_idx]
        h, w = a.shape[-2], a.shape[-1]
        return np.asarray(a).reshape(-1, h, w)[0]

    @staticmethod
    def _hl_text(x, env_idx: int) -> Optional[str]:
        if x is None:
            return None
        if isinstance(x, (list, tuple)):
            return str(x[env_idx]) if env_idx < len(x) else None
        return str(x)

    def _append_hl_latency_frame(self, env_idx: int, env_obs_slice: dict, obs: dict) -> None:
        """Compose the real-time (fresh HL) | LL-view (aged HL) side-by-side panel.

        Left: current frame + ``hl_fresh_heatmap`` / ``hl_fresh_text``.
        Right: the aged cached ``hl_conditioning_image`` (else current frame) +
        ``target_heatmap`` / ``hl_overlay_text`` (both the latent LL-view).
        No-op until the fresh keys are present (i.e. HL latency is active).
        """
        from blocksuite.utils.visualization_utils import overlay_heatmap_and_text

        if obs.get("hl_fresh_heatmap") is None and obs.get("hl_fresh_text") is None:
            return
        if self._hl_overlay_camera not in env_obs_slice["images"]:
            # Returning here would leave the requested latency video silently
            # empty for the whole run.
            available = sorted(env_obs_slice["images"].keys())
            raise ValueError(
                f"hl_overlay_camera={self._hl_overlay_camera!r} not found in obs images "
                f"(available: {available}). Pass --hl-overlay-camera to match the camera "
                "the target heatmap is aligned to."
            )
        cur = _extract_frame(env_obs_slice, self._hl_overlay_camera)
        if cur is None:
            return

        left = overlay_heatmap_and_text(
            cur,
            self._hl_heatmap_2d(obs.get("hl_fresh_heatmap"), env_idx),
            self._hl_text(obs.get("hl_fresh_text"), env_idx),
        )

        cond = obs.get("hl_conditioning_image")
        right_frame = cur
        if cond is not None:
            cond_np = to_numpy(cond)
            if cond_np.ndim >= 4 and cond_np.shape[0] > env_idx:
                right_frame = np.asarray(cond_np[env_idx]).astype(np.uint8)
        right = overlay_heatmap_and_text(
            right_frame,
            self._hl_heatmap_2d(obs.get("target_heatmap"), env_idx),
            self._hl_text(obs.get("hl_overlay_text"), env_idx),
        )

        self._hl_latency_frame_buffers[env_idx].append(
            np.concatenate([left, right], axis=1)
        )

    @property
    def data_recorder(self) -> Optional[IsaacDataRecorderBase]:
        """The recorder episodes are flushed to, if any."""
        return self._data_recorder

    def _should_save_env_episode(self, env_idx: int) -> bool:
        """Whether the next flush for ``env_idx`` should hit disk (default ``True``)."""
        if self.should_save_episode is None:
            return True
        return bool(self.should_save_episode(env_idx))

    def _should_save_in_flight_episode(self, env_idx: int) -> bool:
        """Whether ``env_idx``'s still-running episode should hit disk."""
        if self.should_save_in_flight_episode is None:
            return True
        return bool(self.should_save_in_flight_episode(env_idx))

    def _hl_latency_video_dir(self) -> Path:
        d = self.output_dir / self._hl_latency_video_subdir
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _flush_rgb_env_video(
        self, env_idx: int, partial: bool = False, save: bool = True,
    ) -> None:
        # The episode's last frame was queued at the start of this step.
        self._wait_video_build()
        buf = self._rgb_obs_frame_buffers.get(env_idx)
        latency_buf = (
            self._hl_latency_frame_buffers.get(env_idx)
            if self._save_hl_latency_video
            else None
        )
        if not buf and not latency_buf:
            return
        if save:
            suffix = "_partial" if partial else ""
            ep_idx = self._rgb_episode_counters.get(env_idx, 0)
            name = f"env{env_idx}_episode_{ep_idx:04d}{suffix}"
            rgb_video: Optional[tuple[list[np.ndarray], Path]] = None
            if buf:
                rgb_video = (buf, self._rgb_obs_video_dir())
                self._rgb_videos_saved += 1
            hl_videos: list[tuple[list[np.ndarray], Path]] = []
            # Latency side-by-side shares the episode index so the episodes
            # table (EpisodeResultsTable) pairs it with the plain clip by path.
            if latency_buf:
                hl_videos.append((latency_buf, self._hl_latency_video_dir()))
            # The buffers are replaced (not cleared) below, so the encode thread
            # owns these lists.
            self._submit_video_encode(env_idx, ep_idx, name, rgb_video, hl_videos)
            self._rgb_episode_counters[env_idx] = ep_idx + 1
        self._rgb_obs_frame_buffers[env_idx] = []
        if latency_buf is not None:
            self._hl_latency_frame_buffers[env_idx] = []

    def _submit_video_encode(
        self,
        env_idx: int,
        ep_idx: int,
        name: str,
        rgb_video: Optional[tuple[list[np.ndarray], Path]],
        hl_videos: list[tuple[list[np.ndarray], Path]],
    ) -> None:
        assert self._video_encode_pool is not None
        self._reap_video_encodes(wait=False)
        self._video_encode_futures.append(
            self._video_encode_pool.submit(
                self._encode_env_videos, env_idx, ep_idx, name, rgb_video, hl_videos
            )
        )

    def _encode_env_videos(
        self,
        env_idx: int,
        ep_idx: int,
        name: str,
        rgb_video: Optional[tuple[list[np.ndarray], Path]],
        hl_videos: list[tuple[list[np.ndarray], Path]],
    ) -> None:
        """Write one episode's MP4s (runs on an encode thread)."""
        if rgb_video is not None:
            frames, out_dir = rgb_video
            images_to_video(frames, str(out_dir), name, fps=self.video_fps, verbose=False)
            if self._on_video_saved is not None:
                self._on_video_saved(env_idx, ep_idx, out_dir / f"{name}.mp4")
        for frames, out_dir in hl_videos:
            images_to_video(frames, str(out_dir), name, fps=self.video_fps, verbose=False)

    def _reap_video_encodes(self, wait: bool) -> None:
        """Drop finished encodes, re-raising any error; ``wait`` blocks on all of them."""
        pending: list[Future] = []
        for future in self._video_encode_futures:
            if wait or future.done():
                future.result()
            else:
                pending.append(future)
        self._video_encode_futures = pending

    def _flush_rgb_done_envs(self, info: dict) -> None:
        if "_final_info" not in info:
            return
        mask = to_numpy(info["_final_info"])
        done_env_ids = np.flatnonzero(mask).tolist()
        for env_idx in done_env_ids:
            self._flush_rgb_env_video(
                env_idx, partial=False, save=self._should_save_env_episode(env_idx),
            )

    def _flush_rgb_partial_all(self) -> None:
        self._wait_video_build()
        for env_idx in list(self._rgb_obs_frame_buffers.keys()):
            if self._rgb_obs_frame_buffers.get(env_idx):
                self._flush_rgb_env_video(
                    env_idx,
                    partial=True,
                    save=self._should_save_in_flight_episode(env_idx),
                )

    def capture_image(self) -> Optional[np.ndarray]:
        """Return an RGB uint8 image of the current env state, or *None*.

        Tries ``env.render()`` first, then falls back to reading the first
        camera sensor on the base environment (useful in headless mode with
        ``always_spawn_cameras=True``).

        Override this method for custom rendering logic.
        """
        img = self.env.render()

        if img is None and hasattr(self.base_env, "cameras"):
            for cam in self.base_env.cameras.values():
                rgb = cam.data.output.get("rgb")
                if rgb is not None:
                    img = rgb
                    break

        if img is None:
            return None

        if isinstance(img, torch.Tensor):
            img = to_numpy(img)
        else:
            img = np.asarray(img)

        if img.ndim == 3:
            return img
        if img.ndim == 4:
            if img.shape[0] == 1:
                return img[0]
            if self._video_env_ids is not None:
                idx = list(self._video_env_ids)
                sub = img[idx]
                if sub.shape[0] == 1:
                    return sub[0]
                nrows = int(np.ceil(np.sqrt(len(idx))))
                return tile_images(list(sub), nrows=nrows)
            return tile_images(list(img), nrows=self.video_nrows)
        return None

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, *args, **kwargs):
        if self.save_on_reset:
            if self._save_rendering_video and self.num_envs == 1:
                self.flush_video()
            if self._save_sensor_obs_video:
                self._flush_rgb_partial_all()
            if self.save_trajectory:
                self.flush_trajectory(ignore_empty=True)

        obs, info = super().reset(*args, **kwargs)
        self._last_obs = obs

        if self._data_recorder is not None:
            if isinstance(self._data_recorder, IsaacDataRecorderLeRobot):
                # LeRobot recorder pulls state from the scene directly; the
                # reset state is held as each env's first row.
                for i in range(self.num_envs):
                    self._data_recorder.discard_episode(i)
                self._data_recorder.capture_reset_frames(range(self.num_envs))
            else:
                obs_np = to_numpy(obs)
                for i in range(self.num_envs):
                    self._data_recorder.clear_env_buffer(i)
                    self._data_recorder.append_obs(
                        i, _index_dict_array(obs_np, i)
                    )

        return obs, info

    def step(self, action):
        if self._save_sensor_obs_video and self._last_obs is not None:
            self._append_rgb_obs_frames(self._last_obs)

        # Capture the first frame (s_0) on the first recording step so that
        # repeated reset() calls without steps don't produce empty videos.
        if self.save_rendering_video and self._video_steps == 0:
            frame = self.capture_image()
            if frame is not None:
                self.render_images.append(frame)

        obs, reward, terminated, truncated, info = super().step(action)
        skip_recording = self._skip_recording_env_ids
        self._skip_recording_env_ids = set()

        if self._save_sensor_obs_video:
            self._flush_rgb_done_envs(info)

        # ---- trajectory recording ----
        if self._data_recorder is not None:
            action_np = to_numpy(action)
            terminated_np = to_numpy(terminated)
            truncated_np = to_numpy(truncated)

            has_success = "success" in info
            success_np = to_numpy(info["success"]) if has_success else None

            done_mask = (
                to_numpy(info["_final_info"])
                if "_final_info" in info
                else (terminated_np | truncated_np)
            )

            if isinstance(self._data_recorder, IsaacDataRecorderLeRobot):
                # Fast batched path: one GPU prefetch + async frame build for
                # all envs, then route per-env messages to the shard queues.
                if has_success:
                    self._data_recorder.set_success_flags_batched(success_np)
                if not isinstance(action_np, dict):
                    self._data_recorder.set_pending_actions_batched(action_np)
                # Only envs that auto-reset inside this step expose the next
                # episode's s_0 on the same transition (_final_info in info).
                done_idxs = np.flatnonzero(done_mask).tolist()
                reset_env_ids = done_idxs if "_final_info" in info else []
                # An env with no episode to run (queue drained) is not recorded at all, rather
                # than encoded and discarded: its episode would never be saved.
                self._data_recorder.record_step_batched(
                    [
                        i for i in range(self.num_envs)
                        if (i not in skip_recording or done_mask[i])
                        and self._should_save_in_flight_episode(i)
                    ],
                    reset_env_ids=reset_env_ids,
                )
                for i in done_idxs:
                    self._flush_env_trajectory(
                        i, save=self._should_save_env_episode(i),
                    )
            else:
                # Per-env path (HDF5 recorder and any other subclass that
                # accepts external obs/action pairs).
                obs_np = to_numpy(obs)
                reward_np = to_numpy(reward)
                has_final_obs = "final_observation" in info
                has_fail = "fail" in info
                fail_np = to_numpy(info["fail"]) if has_fail else None

                for i in range(self.num_envs):
                    if i in skip_recording and not done_mask[i]:
                        continue
                    action_i = (
                        _index_dict_array(action_np, i)
                        if isinstance(action_np, dict)
                        else action_np[i]
                    )
                    self._data_recorder.append_step(
                        i,
                        action=action_i,
                        reward=reward_np[i],
                        terminated=terminated_np[i],
                        truncated=truncated_np[i],
                        success=success_np[i] if has_success else None,
                        fail=fail_np[i] if has_fail else None,
                    )

                    if done_mask[i]:
                        if has_final_obs:
                            final_obs_np = to_numpy(info["final_observation"])
                            self._data_recorder.append_obs(
                                i, _index_dict_array(final_obs_np, i)
                            )
                        else:
                            self._data_recorder.append_obs(
                                i, _index_dict_array(obs_np, i)
                            )
                        self._flush_env_trajectory(
                            i, save=self._should_save_env_episode(i),
                        )
                        if has_final_obs:
                            self._data_recorder.append_obs(
                                i, _index_dict_array(obs_np, i)
                            )
                    else:
                        self._data_recorder.append_obs(
                            i, _index_dict_array(obs_np, i)
                        )

        # ---- video recording ----
        if self.save_rendering_video:
            self._video_steps += 1
            frame = self.capture_image()
            if frame is not None:
                self.render_images.append(frame)
            if (
                self.max_steps_per_video is not None
                and self._video_steps >= self.max_steps_per_video
            ):
                self.flush_video()

        self._last_obs = obs
        self._elapsed_steps += 1
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Flushing
    # ------------------------------------------------------------------

    def _flush_env_trajectory(self, env_idx: int, save: bool = True):
        """Flush one environment's buffer to the data recorder."""
        if not self.save_trajectory or self._data_recorder is None:
            return
        if isinstance(self._data_recorder, IsaacDataRecorderLeRobot):
            # An env that was never recorded (no episode to run) has nothing to save.
            if save and self._data_recorder.active_episodes[env_idx]:
                self._data_recorder.save_episode(env_idx)
            else:
                self._data_recorder.discard_episode(env_idx)
            return
        self._data_recorder.flush_env(env_idx, save=save)

    def flush_trajectory(
        self,
        ignore_empty: bool = True,
        env_idxs: Optional[np.ndarray] = None,
        save: bool = True,
    ):
        """Flush trajectory buffers for the given (or all) environments."""
        if not self.save_trajectory or self._data_recorder is None:
            return
        if isinstance(self._data_recorder, IsaacDataRecorderLeRobot):
            rec = self._data_recorder
            for env_idx in range(self.num_envs) if env_idxs is None else env_idxs.tolist():
                if not rec.active_episodes[env_idx]:
                    continue
                if save:
                    rec.save_episode(env_idx)
                else:
                    rec.discard_episode(env_idx)
            return
        self._data_recorder.flush(ignore_empty=ignore_empty, env_idxs=env_idxs, save=save)

    def flush_video(
        self,
        name: Optional[str] = None,
        suffix: str = "",
        save: bool = True,
    ):
        """Flush buffered frames to an MP4 file."""
        if len(self.render_images) <= 1:
            self.render_images.clear()
            self._video_steps = 0
            return
        if save:
            self._video_id += 1
            video_name = name or str(self._video_id)
            if suffix:
                video_name = f"{video_name}_{suffix}"
            images_to_video(
                self.render_images,
                str(self._rendering_video_dir()),
                video_name=video_name,
                fps=self.video_fps,
                verbose=False,
            )
        self.render_images.clear()
        self._video_steps = 0

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        if self._closed:
            return
        self._closed = True

        if self.save_trajectory and self._data_recorder is not None:
            if self.save_on_reset:
                # Per env so the in-flight filter applies: a rollout that ends
                # because the episode queue drained leaves phantom episodes
                # buffered, and a single flush would save them as real ones.
                for env_idx in range(self.num_envs):
                    save = self._should_save_in_flight_episode(env_idx)
                    # A filtered-out episode is flushed even when it looks empty:
                    # the recorder saves anything still marked active when it
                    # closes, so it has to be explicitly discarded here.
                    self.flush_trajectory(
                        ignore_empty=save,
                        env_idxs=np.array([env_idx]),
                        save=save,
                    )
            if isinstance(self._data_recorder, IsaacDataRecorderLeRobot):
                self._data_recorder.finalize()
            else:
                self._data_recorder.close(clean=self.clean_on_close)

        if self._save_rendering_video and self.save_on_reset:
            self.flush_video()

        if self._save_sensor_obs_video:
            self._flush_rgb_partial_all()
            # Every queued episode video is on disk before close() returns.
            self._reap_video_encodes(wait=True)
            assert self._video_build_pool is not None and self._video_encode_pool is not None
            self._video_build_pool.shutdown(wait=True)
            self._video_encode_pool.shutdown(wait=True)
            if self._rgb_videos_saved > 0:
                print(
                    f"Saved {self._rgb_videos_saved} RGB observation video(s) to "
                    f"{self._rgb_obs_video_dir()}"
                )

        return super().close()
