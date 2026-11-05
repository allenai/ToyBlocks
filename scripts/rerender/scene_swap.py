"""Make the live Isaac scene match the CURRENT problem's block set, in-process.

rerender builds ONE Isaac scene, from the first job's env, and reuses it for every job in the
process (rerender.py, build_simulator). A problem whose blocks differ therefore renders into a
scene that lacks its objects -- observed on bigblocks_flip_a, where 30 problems declare
['block_a'] and 20 declare ['block_a', '<distractor>']: the mismatched ones produced meta/ and
render_params with completed=true and NO data/ and NO videos at all.

This module reconciles the difference instead of skipping the problem: it stops the timeline,
deletes the block prims that are no longer wanted, spawns the ones that are new, and plays again
so Isaac Lab re-binds physics handles. That is a supported transition, not a trick -- every asset
subscribes _initialize_callback to the timeline PLAY event and _invalidate_initialize_callback to
STOP (isaaclab/assets/asset_base.py), and RigidObject._initialize_impl re-resolves its prims from
cfg.prim_path on every play, so a prim spawned after build IS discovered. The same stop->play
already happens once per process today, in RenderSim.__init__'s sim.reset().

NOTHING happens when consecutive problems share a block set: signature() is a pure-Python parse of
the env source and reconcile() returns False without touching the timeline or the stage. A remote
run with one problem per worker therefore pays exactly zero.

isaaclab / pxr imports are function-local so this module can be imported before the Isaac app is
launched (same convention as toyblocks.domain_randomization.build).
"""
from __future__ import annotations

import ast
import contextlib
import copy
import os
from typing import Any, Dict, List, Optional, Sequence

# Blocks live at this per-env path in every generated env (`/World/envs/env_.*/{name}`).
PRIM_PATTERN = "/World/envs/env_.*/{name}"
_SETUP_FN = "_setup_scene_objects"


# ---------------------------------------------------------------------------------------------
# reading a problem's block set WITHOUT importing its env module
# ---------------------------------------------------------------------------------------------
def _setup_fn_node(src: str):
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == _SETUP_FN:
            return node
    return None


def block_specs(env_file: Optional[str]) -> Optional[Dict[str, dict]]:
    """{block_name: spec} for a generated env, parsed from source (no import, no Isaac).

    The generated `_setup_scene_objects` declares `block_names` and `block_specs` as plain
    literals, so ast.literal_eval reads them exactly. Returns None when the env cannot be read or
    does not declare them literally -- callers must treat None as "unknown", never as "empty".
    """
    if not env_file or not os.path.exists(env_file):
        return None
    try:
        src = open(env_file).read()
    except OSError:
        return None
    fn = _setup_fn_node(src)
    if fn is None:
        return None
    names, specs = None, None
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            continue
        try:
            if node.targets[0].id == "block_names":
                names = ast.literal_eval(node.value)
            elif node.targets[0].id == "block_specs":
                specs = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            return None
    if names is None or specs is None:
        return None
    return {n: specs[n] for n in names if n in specs}


def signature(env_file: Optional[str]) -> Optional[frozenset]:
    """The env's block-name set, or None if it cannot be determined."""
    bs = block_specs(env_file)
    return None if bs is None else frozenset(bs)


# ---------------------------------------------------------------------------------------------
# block sizes: which spawn slot renders an episode at its recorded size
# ---------------------------------------------------------------------------------------------
#: Recorded vs spawned edge length (metres) that still count as the same size. The recorder stores
#: ``observation.state.cube_scale`` as float32, so equal sizes differ by ~1e-9; distinct profiles
#: differ by millimetres.
SIZE_MATCH_TOL = 1e-5


def env_size_table(env_file: Optional[str]) -> Optional[List[float]]:
    """Edge length of every spawn slot of a generated env, or None if it has no size table.

    The generated ``_setup_scene_objects`` declares ``randomizations`` as a literal whose
    ``block_sizes`` maps each block to one ``[x, y, z]`` per slot. Every block of a slot shares one
    size, and the recorder writes the first block's edge as ``observation.state.cube_scale`` (see
    ``toyblocks.vecgen.recorder``), so the first non-empty list is the table episodes match.
    """
    if not env_file or not os.path.exists(env_file):
        return None
    try:
        fn = _setup_fn_node(open(env_file).read())
    except (OSError, SyntaxError):
        return None
    if fn is None:
        return None
    for node in ast.walk(fn):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "randomizations"):
            try:
                rand = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                return None
            for sizes in (rand.get("block_sizes") or {}).values():
                if sizes:
                    return [float(s[0]) for s in sizes]
            return None
    return None


def match_size_slots(table: Sequence[float], sizes: Sequence[float],
                     tol: float = SIZE_MATCH_TOL) -> List[Optional[int]]:
    """For each recorded size, the spawn slot whose edge is closest to it (None if none is within
    ``tol``). Slots that share an edge are interchangeable; the closest, then lowest, wins."""
    out: List[Optional[int]] = []
    for v in sizes:
        best = min(range(len(table)), key=lambda s: (abs(table[s] - float(v)), s), default=None)
        out.append(best if best is not None and abs(table[best] - float(v)) <= tol else None)
    return out


def round_robin_slots(num_envs: int, num_slots: int) -> List[int]:
    """The template's default env -> spawn-slot map (env ``i`` -> slot ``i % num_slots``).

    rerender always renders at this map: it is fixed per (num_envs, table), so a problem's waves
    depend only on its own episodes -- never on which fragment or which earlier problem the
    process rendered -- and a process that renders one problem never respawns its scene.
    """
    return [i % num_slots for i in range(num_envs)]


# ---------------------------------------------------------------------------------------------
# building the spawn configs the generated env WOULD have built
# ---------------------------------------------------------------------------------------------
class _ShimScene:
    def __init__(self):
        self.rigid_objects: Dict[str, Any] = {}


class _ShimEnv:
    """Just enough `self` for the generated `_setup_scene_objects` to run.

    It reads num_envs / device / cfg and writes randomizations, MAX_NON_SIM_STATE_RANDOMIZATIONS
    and scene.rigid_objects[...]; nothing else.
    """

    def __init__(self, num_envs: int, device: str, cfg: Any):
        self.num_envs = num_envs
        self.device = device
        self.cfg = cfg
        self.scene = _ShimScene()


@contextlib.contextmanager
def _capture_cfgs():
    """Temporarily make rigid_objects.create_* return a cfg instead of spawning.

    The generated function does `from toyblocks.utils import rigid_objects` INSIDE its body, so
    patching the module's attributes is the interception point. Passing return_cfg=True reuses the
    real builders -- per-env size tiling, friction tiling and the cuboid/usd dispatch all stay
    exactly as the template computes them, so this cannot drift from generated envs.
    """
    from toyblocks.utils import rigid_objects as ro

    originals = {n: getattr(ro, n) for n in ("create_cuboid", "create_toy_block")}

    def _wrap(fn):
        def _inner(*a, **kw):
            kw["return_cfg"] = True
            return fn(*a, **kw)
        return _inner

    for n, fn in originals.items():
        setattr(ro, n, _wrap(fn))
    try:
        yield
    finally:
        for n, fn in originals.items():
            setattr(ro, n, fn)


def build_cfgs(env_file: str, num_envs: int, device: str, cfg: Any,
               env_slots: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """{block_name: RigidObjectCfg} for `env_file`, without spawning anything.

    Runs the generated `_setup_scene_objects` itself rather than reimplementing it: the body is a
    codegen'd copy of a template, so any reimplementation here would silently drift the moment the
    template changes. ``env_slots`` (one spawn slot per env) is handed to it as ``cfg.env_slots``,
    on a copy so the live env's cfg is left as built.
    """
    if env_slots is not None:
        cfg = copy.copy(cfg)
        cfg.env_slots = tuple(int(s) for s in env_slots)
    src = open(env_file).read()
    fn = _setup_fn_node(src)
    if fn is None:
        raise ValueError(f"{env_file}: no {_SETUP_FN} to read block configs from")
    # Compile the function alone, at module level, then call it with the shim as `self`.
    mod = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(mod)
    # The generated function imports rigid_objects / OBJECT_ASSET_DIR itself, but uses names the
    # env module imports at MODULE level (torch). Provide those; anything still missing surfaces as
    # a NameError naming it, which is the right failure -- reconcile() turns it into a skip rather
    # than rendering the problem into a scene that lacks its objects.
    import torch                                                    # noqa: PLC0415
    ns: Dict[str, Any] = {"torch": torch}
    exec(compile(mod, filename=env_file, mode="exec"), ns)          # noqa: S102 - codegen'd source
    shim = _ShimEnv(num_envs, device, cfg)
    with _capture_cfgs():
        ns[_SETUP_FN](shim)
    return dict(shim.scene.rigid_objects)


# ---------------------------------------------------------------------------------------------
# the boundary transaction
# ---------------------------------------------------------------------------------------------
def scene_signature(sim) -> frozenset:
    """Block names currently registered in the live scene.

    Tracked on the simulator so it survives across problems, seeded from the scene the process
    built. Only block-like entries count -- the table and other scene-preset assets are never
    swapped.
    """
    sig = getattr(sim, "_rr_block_sig", None)
    if sig is not None:
        return sig
    return frozenset(n for n in sim.scene.rigid_objects if n.startswith("block"))


def reconcile(sim, env_file: Optional[str], log=print, warmup: int = 0,
              env_slots: Optional[Sequence[int]] = None) -> bool:
    """Make the live scene hold exactly this problem's blocks. True if the scene was changed.

    Returns False immediately -- no timeline, no stage work -- when the set already matches, which
    is the common case and the whole reason this is cheap for one-problem-per-worker runs.

    ``env_slots`` (one spawn slot per env) also pins block SIZES: each env's blocks must be spawned
    at that slot of THIS problem's size table. Every PDDL package draws its own table, so a scene
    built for another problem (or spawned round-robin) holds the wrong sizes even when the block
    names match. When the (table, env_slots) pair differs from what the scene was spawned with,
    every block is deleted and respawned.
    """
    want_specs = block_specs(env_file)
    if want_specs is None:
        log(f"[scene_swap] cannot read blocks from {env_file}; leaving the scene as built")
        return False
    want = frozenset(want_specs)
    have = scene_signature(sim)
    size_key = None
    if env_slots is not None:
        size_key = (tuple(env_size_table(env_file) or ()), tuple(int(s) for s in env_slots))
    resize = size_key is not None and size_key != getattr(sim, "_rr_size_key", None)
    if want == have and not resize:
        return False

    to_delete = sorted(have) if resize else sorted(have - want)
    to_add = sorted(want) if resize else sorted(want - have)
    log(f"[scene_swap] scene {sorted(have)} -> {sorted(want)} "
        f"(delete={to_delete or '-'}, add={to_add or '-'}"
        f"{', respawn at env_slots ' + str(list(size_key[1])) if resize else ''})")

    import isaacsim.core.utils.prims as prim_utils           # noqa: PLC0415 - needs a live app
    import isaaclab.sim as sim_utils                          # noqa: PLC0415
    from isaaclab.assets import RigidObject                   # noqa: PLC0415

    sc = sim.sim                                              # isaaclab SimulationContext
    stage = sim.scene.stage

    # Build the new cfgs BEFORE touching the sim: a parse or codegen error must not leave the stage
    # half-mutated with the timeline stopped.
    new_cfgs: Dict[str, Any] = {}
    if to_add:
        all_cfgs = build_cfgs(env_file, sim.num_envs, str(sim.device), sim.base_env.cfg,
                              env_slots=env_slots)
        missing = [n for n in to_add if n not in all_cfgs]
        if missing:
            raise ValueError(f"{env_file}: {_SETUP_FN} produced no config for {missing}")
        new_cfgs = {n: all_cfgs[n] for n in to_add}

    # The recorder from the PREVIOUS problem is finalized; drop it before the sim moves.
    sim.data_recorder = None

    # Stop with the app-control-on-stop handler disabled: SimulationContext subscribes a STOP
    # callback that spins `while not is_playing(): self.render()`, so a bare stop() never returns
    # when the app was launched by AppLauncher. SimulationContext.reset() sets the same guard.
    sc._disable_app_control_on_stop_handle = True
    try:
        sc.stop()

        for name in to_delete:
            obj = sim.scene.rigid_objects.pop(name, None)
            if obj is not None:
                # Drop the PLAY/STOP/PRIM_DELETION subscriptions first: they live on the timeline
                # event stream, not in scene.rigid_objects, so a popped-but-subscribed asset would
                # still try to re-initialize on the next play and fail to find its prim.
                with contextlib.suppress(Exception):
                    obj._clear_callbacks()
                del obj
            paths = sim_utils.find_matching_prim_paths(PRIM_PATTERN.format(name=name))
            for p in paths:
                prim_utils.delete_prim(p)
            log(f"[scene_swap]   deleted {len(paths)} prim(s) for {name}")

        if to_delete:
            # Drain the USD change notices (PRIM_DELETION) before respawning: a deferred notice
            # would otherwise match a NEW object with the SAME name and clear its callbacks, so it
            # would never initialize on play. bigblocks_flip_a reuses distractor names across
            # problems, so this path is real, not theoretical.
            import omni.kit.app                                # noqa: PLC0415 - needs a live app
            omni.kit.app.get_app().update()

        for name, cfg in new_cfgs.items():
            sim.scene.rigid_objects[name] = RigidObject(cfg=cfg)
            log(f"[scene_swap]   spawned {name}")
    finally:
        sc._disable_app_control_on_stop_handle = False

    # reset() plays and re-initializes physics, and re-raises the exception that
    # AssetBase._initialize_callback otherwise swallows (it sets _is_initialized=True even when
    # init failed), so a broken re-bind is loud instead of silently producing an empty scene.
    sc.reset()

    # Optional extra RTX passes after the play. A stop/play resets the renderer's temporal state,
    # and RT2 runs at ~1 sample/pixel relying entirely on the DLSS denoiser, so the frames right
    # after a swap can be noisier than the same episode rendered in a fresh process. Measured on
    # bigblocks_flip_a (P after Q vs P alone): the swap costs 1.7 dB of RGB PSNR on top of the
    # 4.5 dB that being the second problem in a process already costs today. Depth is bit-exact
    # either way, so this is purely a shading-convergence knob.
    for _ in range(max(0, int(warmup))):
        sc.render()

    bad = [n for n, o in sim.scene.rigid_objects.items() if not getattr(o, "is_initialized", True)]
    if bad:
        raise RuntimeError(f"[scene_swap] assets failed to initialize after play: {bad}")

    # Re-derive what the wrapper cached from the pre-swap physics views. env_base_poses is on the
    # hot path (set_rigid_object_pose(local=True) -> _transform_to_world_frame), so a stale copy
    # would place every object wrongly rather than crash.
    _rederive_sim_caches(sim, log)
    sim._rr_block_sig = want
    # Without env_slots, added blocks took the template's round-robin sizes next to survivors of
    # whatever sizes they had, so the scene matches no size key (None) any more.
    sim._rr_size_key = size_key
    return True


def _rederive_sim_caches(sim, log=print) -> None:
    """Rebuild RenderSim state captured from the previous physics views."""
    # The play re-parses the articulation from USD, dropping the arm gains RenderSim applied
    # (they matter with --physics_step); put them back.
    if hasattr(sim, "_apply_motion_profile"):
        try:
            sim._apply_motion_profile()
        except Exception as e:                                 # noqa: BLE001 - report, never mask
            log(f"[scene_swap] WARNING could not re-apply the arm gain profile "
                f"({type(e).__name__}: {e})")
    try:
        import torch                                          # noqa: PLC0415
        robot_pose = sim.scene["robot"].data.root_pose_w
        for env_idx in list(getattr(sim, "env_base_poses", {})):
            sim.env_base_poses[env_idx] = robot_pose[env_idx].clone()
    except Exception as e:                                     # noqa: BLE001 - report, never mask
        log(f"[scene_swap] WARNING could not re-derive env_base_poses ({type(e).__name__}: {e})")
    try:
        from toyblocks.domain_randomization.render_sim import fix_object_name_mappings
        sim.prim_scene_obj_mapping = fix_object_name_mappings(sim.scene, sim.scene.rigid_objects)
        sim.obj_prim_scene_mapping = {v: k for k, v in sim.prim_scene_obj_mapping.items()}
    except Exception as e:                                     # noqa: BLE001
        log(f"[scene_swap] WARNING could not re-derive prim mappings ({type(e).__name__}: {e})")
