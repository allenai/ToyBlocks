from __future__ import annotations

import json
import math
import torch
import numpy as np
from typing import Any, Protocol, overload


class _CameraOverridesCfg(Protocol):
    """Minimal surface needed by :func:`apply_camera_overrides_json`."""

    camera_overrides_json: str | None
    camera_cfgs: dict[str, Any]
@overload
def to_numpy(x: dict) -> dict: ...
@overload
def to_numpy(x: list) -> list: np.ndarray: ...
@overload
def to_numpy(x: np.ndarray) -> np.ndarray: ...
@overload
def to_numpy(x: torch.Tensor) -> np.ndarray: ...
def to_numpy(x: torch.Tensor | list | dict):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, list):
        return np.array(x)
    elif isinstance(x, dict):
        return {k: to_numpy(v) for k, v in x.items()}
    else:
        raise ValueError(f"Unsupported type: {type(x)}")

def per_env_hwc_uint8(frames: Any, env_idx: int) -> np.ndarray:
    """One env's ``(H, W, 3)`` uint8 frame from a batched or unbatched array.

    Takes camera RGB as the eval loop holds it, so the live obs arrives as a
    tensor on the sim device while a cached conditioning frame is already host
    numpy and already unbatched -- hence the conditional index. The env is
    indexed before the host copy: one frame crosses the bus per step instead of
    the whole batch. Alpha is dropped and float renders are scaled so callers
    always get the 3-channel uint8 grid they assume.
    """
    if torch.is_tensor(frames):
        if frames.ndim == 4:
            frames = frames[env_idx]
        arr = to_numpy(frames)
    else:
        arr = np.asarray(frames)
        if arr.ndim == 4:
            arr = arr[env_idx]
    if arr.ndim != 3:
        raise ValueError(f"expected (H, W, C) or (N, H, W, C) frames; got {arr.shape}")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if np.issubdtype(arr.dtype, np.floating):
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    return arr

def to_torch(x: dict, device: torch.device | str | None = None) -> dict:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    elif isinstance(x, list):
        return torch.tensor(x).to(device)
    elif isinstance(x, dict):
        return {k: to_torch(v, device=device) for k, v in x.items()}

def torch_clone_dict(data: dict) -> dict:
    """
    Recursively clones all torch tensors in a dictionary.
    If the input was a torch tensor, it will return a clone of the tensor.
    """
    if isinstance(data, torch.Tensor):
        return data.clone()

    output_dict = {}
    for key, value in data.items():
        if isinstance(value, dict):
            output_dict[key] = torch_clone_dict(value)
        elif isinstance(value, torch.Tensor):
            output_dict[key] = value.clone()
        else:
            output_dict[key] = value
    return output_dict

def flatten_state_dict(state_dict: dict) -> dict:
    """
    Flattens a nested dictionary into a single vector
    """
    states = []
    for value in state_dict.values():
        if isinstance(value, dict):
            sub = flatten_state_dict(value)
            if sub is None:
                continue  # empty sub-group contributes no elements
            states.append(sub)
        elif isinstance(value, torch.Tensor):
            states.append(value)
        elif isinstance(value, np.ndarray):
            states.append(value)
        else:
            states.append(value)
    if not states:
        return None  # empty group: no elements to concatenate
    if isinstance(states[0], torch.Tensor):
        return torch.cat(states, dim=-1)
    elif isinstance(states[0], np.ndarray):
        return np.concatenate(states, axis=-1)
    else:
        raise ValueError(f"Unsupported type: {type(states[0])}")


def _apply_pinhole_spawn_override_value(key: str, val: Any) -> Any:
    if key == "clipping_range":
        if not isinstance(val, (list, tuple)) or len(val) != 2:
            raise ValueError(f"spawn.{key} must be a length-2 array [near, far]")
        return (float(val[0]), float(val[1]))
    if key == "vertical_aperture" and val is None:
        return None
    if key == "lock_camera":
        return bool(val)
    if key == "projection_type":
        return str(val)
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        return val
    raise ValueError(f"spawn.{key}: unsupported JSON type {type(val).__name__}")


def apply_camera_overrides_json(cfg: _CameraOverridesCfg) -> None:
    """Merge ``cfg.camera_overrides_json`` into ``cfg.camera_cfgs`` (offset and spawn)."""
    raw_s = cfg.camera_overrides_json
    if not raw_s:
        return
    try:
        raw = json.loads(raw_s)
    except json.JSONDecodeError as e:
        raise ValueError(f"camera_overrides_json: invalid JSON ({e})") from e
    if not isinstance(raw, dict):
        raise ValueError("camera_overrides_json: root must be a JSON object")
    for name, body in raw.items():
        if not isinstance(body, dict):
            raise ValueError(f"camera_overrides_json[{name!r}]: value must be an object")
        cam = cfg.camera_cfgs.get(name)
        if cam is None:
            # Loud, but never fatal: a run that is otherwise fine must not die over one camera
            # name. This runs AFTER the scene preset and the robot's cameras are merged
            # (base_env._setup_scene), so every camera that will exist already does -- an
            # unresolved name here is a typo or a camera from another robot/scene.
            print(
                f"[camera_overrides] WARNING {name!r}: no such camera, override NOT applied. "
                f"Available: {sorted(cfg.camera_cfgs)}",
                flush=True,
            )
            continue
        if hasattr(cam, "offset"):
            pos = cam.offset.pos
            rot = cam.offset.rot
            if "pos" in body:
                p = body["pos"]
                if not isinstance(p, (list, tuple)) or len(p) != 3:
                    raise ValueError(f"camera_overrides_json[{name!r}].pos must be a length-3 array")
                pos = tuple(float(x) for x in p)
            if "rot" in body:
                r = body["rot"]
                if not isinstance(r, (list, tuple)) or len(r) != 4:
                    raise ValueError(f"camera_overrides_json[{name!r}].rot must be a length-4 array")
                rot = tuple(float(x) for x in r)
            if "pos" in body or "rot" in body:
                cam.offset = cam.offset.replace(pos=pos, rot=rot)
        # fov_deg -> focal_length, converted with THIS camera's horizontal_aperture. The
        # conversion cannot be done by the caller: the DR flags path runs before the robot's
        # cameras exist, and the apertures differ per camera (gripper 6.916 vs the 20.955
        # default), so a caller-side conversion silently sets the wrong focal length.
        if "fov_deg" in body:
            if not getattr(cam, "spawn", None):
                raise ValueError(f"camera_overrides_json[{name!r}].fov_deg: camera has spawn=None")
            fov = float(body["fov_deg"])
            if not 0.0 < fov < 180.0:
                raise ValueError(
                    f"camera_overrides_json[{name!r}].fov_deg: {fov} is not in (0, 180)"
                )
            ha = float(getattr(cam.spawn, "horizontal_aperture", 20.955))
            cam.spawn = cam.spawn.replace(
                focal_length=ha / (2.0 * math.tan(math.radians(fov) / 2.0))
            )
        spawn_body = body.get("spawn")
        if spawn_body is not None:
            if not getattr(cam, "spawn", None):
                raise ValueError(f"camera_overrides_json[{name!r}].spawn: camera has spawn=None")
            if not isinstance(spawn_body, dict):
                raise ValueError(f"camera_overrides_json[{name!r}].spawn must be an object")
            kwargs: dict[str, Any] = {}
            for sk, sv in spawn_body.items():
                if sk == "func":
                    raise ValueError(f"camera_overrides_json[{name!r}].spawn: overriding 'func' is not allowed")
                if not hasattr(cam.spawn, sk):
                    raise ValueError(
                        f"camera_overrides_json[{name!r}].spawn: unknown field {sk!r} "
                        f"for {type(cam.spawn).__name__}"
                    )
                kwargs[sk] = _apply_pinhole_spawn_override_value(sk, sv)
            cam.spawn = cam.spawn.replace(**kwargs)
