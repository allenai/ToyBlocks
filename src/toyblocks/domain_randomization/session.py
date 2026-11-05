"""One object a client holds to drive the whole DR lifecycle.

`interpreter.py` is the working implementation -- a verbatim lift of render's replay_job body --
but its entry points are loose functions threading a 66-field namespace, and their call order
carries constraints that are silent when you get them wrong (build flags before the sim exists,
camera bases captured before any offset is applied, cameras seated in their own call because
apply_wave deliberately does not touch them). render.py encodes that order by being the code it
was extracted from. A second client -- a gym wrapper -- has nothing to copy it from.

DRSession is that order, and nothing else. Every method is a one-line delegation to the same
interpreter/build functions render calls, in the same sequence, so the two clients cannot drift:
there is no sampling, no Isaac work and no policy here. If a behaviour question has an answer, it
lives in interpreter.py; this file only says what happens when.

Lifecycle, in order:

    s = DRSession(flags, salt=..., seeding="hash")
    kit = DRSession.kit_args(flags)   # PURE, pre-launch: carb settings only take effect as
                                      # launch-time kit_args, not runtime carb.set
    s.configure_build(env_cfg)        # BEFORE the sim is built -- mutates env_cfg
    s.bind(sim, dt=dt, ...)           # once the sim exists and one frame has rendered
    s.sample_episodes([k])            # per reset, for a caller that streams episodes
    s.apply_wave(sim, wave, active)   # per batch: everything except cameras
    s.apply_cameras(sim, wave, active, dt=dt)   # per batch: the cameras
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from . import build as _build
from . import config as _config
from . import interpreter as _interp
from .assets import AssetPaths
from .camera import _capture_camera_bases


class DRSession:
    """Holds the flags, the seeding decision and the interpreter's `ir` namespace."""

    def __init__(self, flags: Optional[dict], *, salt: int, seeding: str = "hash",
                 assets: Optional[AssetPaths] = None,
                 material_baseline_revert: bool = False):
        self.flags: dict = flags or {}
        self.salt = int(salt)
        self.seeding = seeding
        self.assets = assets or AssetPaths.default()
        # off by default = render's behaviour today; on, a key the chosen material
        # omits reverts to the asset instead of keeping the previous episode's.
        self.material_baseline_revert = bool(material_baseline_revert)
        self.ir = None                      # set by bind()
        self.cam_base_pose: Optional[dict] = None
        self.cam_base_K: Optional[dict] = None

    # ---- pre-launch -------------------------------------------------------------------
    @staticmethod
    def kit_args(flags: Optional[dict]) -> str:
        """`--/a/b=c` string for flags['carb_settings']. PURE: no Isaac, safe before launch.

        The offscreen tiled-camera render ignores a runtime carb.set for RTX lighting settings,
        so these only land if they are passed at launch.
        """
        cs = (flags or {}).get("carb_settings")
        if not isinstance(cs, dict) or not cs:
            return ""
        def kv(k, v):
            v = ("true" if v else "false") if isinstance(v, bool) else v   # kit wants lowercase
            return f"--/{str(k).lstrip('/')}={v}"
        return " ".join(kv(k, v) for k, v in cs.items())

    @staticmethod
    def load_flags(path: str, *, strict: bool = False, verbose: bool = True) -> dict:
        return _config.load_flags(path, strict=strict, verbose=verbose)

    # ---- build time, BEFORE gym.make / the sim -----------------------------------------
    def configure_build(self, env_cfg, **kw) -> None:
        """Mutate env_cfg + the scene preset from the yaml. Roughly half a flag yaml is
        build-time and cannot be applied by anything installed after the env is made."""
        _build.apply_build_flags(env_cfg, self.flags, **kw)

    def apply_scene_flags(self, sim, *, show_gizmos: bool = False) -> None:
        """The three stage mutators render runs right after building the sim, same order."""
        _build.apply_lighting_flags(self.flags, sim)
        if not show_gizmos:
            _build.hide_debug_visuals()
        _build.hide_scene_objects(self.flags)

    # ---- once, after the sim exists -----------------------------------------------------
    def bind(self, sim, *, dt: float, episodes: Sequence[dict] = (), episode_keys=(),
             override_cams: Optional[dict] = None, capture_bases: bool = True):
        """Run interpreter.setup() and snapshot the pristine camera bases.

        The snapshot must happen BEFORE any per-episode offset is applied and be reused for every
        problem sharing this sim -- re-capturing from live cam.data after a prior job returns the
        already-offset pose.
        """
        self.ir = _interp.setup(sim, self.flags, episodes=episodes, override_cams=override_cams,
                                salt=self.salt, episode_keys=list(episode_keys),
                                assets=self.assets, indexed_seeding=self.seeding,
                                material_baseline_revert=self.material_baseline_revert)
        if capture_bases and self.ir.randomize_cams:
            self.cam_base_pose, self.cam_base_K = _capture_camera_bases(
                sim, self.ir.randomize_cams, self.ir.rand_static, dt)
        return self.ir

    # ---- per reset / per batch -----------------------------------------------------------
    def sample_episodes(self, episode_keys) -> None:
        """Sample DR for episodes this caller has only just learned about. Prefix-free seeding
        only -- interpreter.sample_episodes raises under `stream`."""
        _interp.sample_episodes(self.ir, episode_keys)

    def apply_wave(self, sim, wave, active=None):
        """Everything except cameras. `active` is the env slots to touch (default: all)."""
        return _interp.apply_wave(self.ir, sim, wave, active)

    def apply_cameras(self, sim, wave, active, *, dt: float, camera_base: str = "scene",
                      rtx_warmup_frames: int = 1):
        """The cameras, with their prender and post-warm-up ordering. Must be called too --
        apply_wave deliberately does not touch cameras."""
        return _interp.apply_wave_cameras(
            self.ir, sim, wave, active, dt=dt, camera_base=camera_base,
            cam_base_pose=self.cam_base_pose, cam_base_K=self.cam_base_K,
            rtx_warmup_frames=rtx_warmup_frames)
