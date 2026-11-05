"""Named actuator profiles for overriding default robot physics at env creation.

Each profile is a dict mapping actuator-group names to ``ImplicitActuatorCfg``
instances, matching the ``ArticulationCfg.actuators`` schema.  Profiles are
selected via ``BaseEnvCfg.actuator_profile`` (CLI: ``--env-cfg.actuator-profile <name>``).

Example usage::

    python scripts/evaluate.py --task ToyBlocks-Lift-v0 --env-cfg.actuator-profile tamp
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

from isaaclab.actuators.actuator_pd_cfg import ImplicitActuatorCfg

# ---------------------------------------------------------------------------
# TODO: these settings will need to be updated to match the TAMP repo
# ---------------------------------------------------------------------------

TAMP_ACTUATORS = {
    "panda_shoulder": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[1-4]"],
        effort_limit_sim=87.0,
        velocity_limit_sim=2.175,
        stiffness=880.0,
        damping=80.0,
    ),
    "panda_forearm": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[5-7]"],
        effort_limit_sim=87.0,
        velocity_limit_sim=2.61,
        stiffness=880.0,
        damping=80.0,
    ),
    "gripper_drive": ImplicitActuatorCfg(
        joint_names_expr=["finger_joint"],
        effort_limit_sim=200.0,
        velocity_limit_sim=10.0,
        stiffness=17.0,
        damping=0.02,
    ),
    "gripper_finger": ImplicitActuatorCfg(
        joint_names_expr=[".*_inner_finger_joint"],
        effort_limit_sim=50.0,
        velocity_limit_sim=10.0,
        stiffness=0.2,
        damping=0.001,
    ),
    "gripper_passive": ImplicitActuatorCfg(
        joint_names_expr=[".*_inner_knuckle_joint", "right_outer_knuckle_joint"],
        effort_limit_sim=1.0,
        velocity_limit_sim=10.0,
        stiffness=0.0,
        damping=0.0,
    ),
}


# ---------------------------------------------------------------------------
# Registry — add new profiles here.
# ---------------------------------------------------------------------------

ACTUATOR_PROFILES: dict[str, dict] = {
    "tamp": TAMP_ACTUATORS,
}


# ---------------------------------------------------------------------------
# PD gain (de)serialization.
# ---------------------------------------------------------------------------

GAIN_FIELDS: tuple[str, ...] = (
    "stiffness",
    "damping",
    "effort_limit",
    "velocity_limit",
    "effort_limit_sim",
    "velocity_limit_sim",
    "armature",
    "friction",
    "dynamic_friction",
    "viscous_friction",
)
"""``ActuatorBaseCfg`` fields that are recorded / overridable."""

# ``effort_limit`` and ``velocity_limit`` have ``*_sim`` twins; robot configs set one
# or the other (ToyBlocks defaults use the plain field, the TAMP profile the ``_sim``
# variant), so reads fall back between them.
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "effort_limit": ("effort_limit", "effort_limit_sim"),
    "velocity_limit": ("velocity_limit", "velocity_limit_sim"),
}

ARM_JOINT_NAMES: tuple[str, ...] = tuple(f"panda_joint{i}" for i in range(1, 8))
"""The 7 Franka arm joints, in the canonical order used by TAMP stiffness profiles."""


def _json_gain(value):
    """Coerce a gain value to a JSON type, or ``None`` if it is unset/unusable."""
    if isinstance(value, dict):
        return {k: float(v) for k, v in value.items()}
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def actuators_to_gains(actuators: dict) -> dict[str, dict]:
    """Serialize an ``ArticulationCfg.actuators`` dict to plain JSON types.

    Each actuator group becomes ``{"joint_names_expr": [...], <gain field>: value}``
    where a value is either a scalar or a ``{joint regex: value}`` mapping (matching
    the ``ActuatorBaseCfg`` schema). Unset fields are omitted.
    """
    gains: dict[str, dict] = {}
    for name, cfg in actuators.items():
        entry: dict = {}
        joint_names_expr = getattr(cfg, "joint_names_expr", None)
        if isinstance(joint_names_expr, (list, tuple)):
            entry["joint_names_expr"] = [str(e) for e in joint_names_expr]
        for field_name in GAIN_FIELDS:
            value = _json_gain(getattr(cfg, field_name, None))
            if value is not None:
                entry[field_name] = value
        gains[name] = entry
    return gains


def apply_gains_to_actuators(actuators: dict, gains: dict[str, dict]) -> None:
    """Override PD gains / limits on ``actuators`` in place from a serialized ``gains``.

    Actuator groups present in ``gains`` but not in ``actuators`` are skipped with a
    warning -- this happens when a checkpoint was trained with a robot config whose
    actuator grouping differs from the one being loaded. ``joint_names_expr`` is never
    overridden; only gain fields are.
    """
    for name, entry in gains.items():
        cfg = actuators.get(name)
        if cfg is None:
            warnings.warn(
                f"Actuator group {name!r} in the recorded gains has no counterpart in the "
                f"robot config (groups: {sorted(actuators)}); its gains are ignored.",
                stacklevel=2,
            )
            continue
        for field_name, value in entry.items():
            if field_name in GAIN_FIELDS:
                setattr(cfg, field_name, value)


def apply_scalar_gain_overrides(
    actuators: dict, field_name: str, overrides: dict[str, float]
) -> None:
    """Set a single gain field per actuator group in place.

    Keys are actuator group names; the key ``"*"`` applies to every group. Unknown
    group names raise so a typo'd CLI override fails loudly instead of silently
    training with the default gains.
    """
    for group, value in overrides.items():
        targets = list(actuators) if group == "*" else [group]
        for target in targets:
            if target not in actuators:
                raise ValueError(
                    f"Unknown actuator group {target!r} for {field_name} override. "
                    f"Available groups: {sorted(actuators)} (use '*' for all)."
                )
            setattr(actuators[target], field_name, float(value))


def resolve_checkpoint_config_path(path: str | Path) -> Path:
    """Resolve a user-supplied checkpoint reference to its ``config.json``.

    Accepts the ``config.json`` itself, the run directory containing it, or any
    checkpoint file (``ckpt_*.pt`` / ``final_ckpt.pt``) inside that directory.
    """
    p = Path(path)
    if p.is_dir():
        return p / "config.json"
    if p.name == "config.json":
        return p
    return p.parent / "config.json"


def load_gains_from_checkpoint_config(path: str | Path) -> dict[str, dict] | None:
    """Read the ``actuator_gains`` block out of an RL run's ``config.json``.

    Returns ``None`` (with a warning) when the file is missing or predates gain
    recording, so callers degrade to the robot config's own gains.
    """
    cfg_path = resolve_checkpoint_config_path(path)
    if not cfg_path.is_file():
        warnings.warn(
            f"No config.json found at {cfg_path}; using the robot config's PD gains.",
            stacklevel=2,
        )
        return None
    try:
        with open(cfg_path) as f:
            gains = json.load(f).get("actuator_gains")
    except Exception as exc:  # noqa: BLE001 - degrade rather than abort a rollout
        warnings.warn(f"Could not read PD gains from {cfg_path}: {exc}", stacklevel=2)
        return None
    if not gains:
        warnings.warn(
            f"{cfg_path} has no 'actuator_gains' (trained before gain recording); "
            "using the robot config's PD gains.",
            stacklevel=2,
        )
        return None
    return gains


def _resolve_joint_gain(entry: dict, field_name: str, joint_name: str) -> float | None:
    """Value of ``field_name`` for ``joint_name`` within one actuator group entry."""
    for key in _FIELD_ALIASES.get(field_name, (field_name,)):
        value = entry.get(key)
        if value is None:
            continue
        if isinstance(value, dict):
            for expr, per_joint in value.items():
                if re.fullmatch(expr, joint_name):
                    return float(per_joint)
            continue
        return float(value)
    return None


def gains_to_joint_vectors(
    gains: dict[str, dict],
    joint_names: tuple[str, ...] = ARM_JOINT_NAMES,
    fields: tuple[str, ...] = ("stiffness", "damping", "effort_limit", "velocity_limit"),
) -> dict[str, tuple[float, ...]]:
    """Expand serialized ``gains`` into one per-joint tuple per requested field.

    Each joint is matched against the actuator groups' ``joint_names_expr`` regexes.
    Used to translate ToyBlocks actuator groups into TAMP's flat 7-DOF
    ``StiffnessProfile`` representation.

    Raises:
        ValueError: if any joint has no recorded value for a requested field.
    """
    vectors: dict[str, tuple[float, ...]] = {}
    for field_name in fields:
        values: list[float] = []
        for joint_name in joint_names:
            value = None
            for entry in gains.values():
                exprs = entry.get("joint_names_expr") or ()
                if not any(re.fullmatch(expr, joint_name) for expr in exprs):
                    continue
                value = _resolve_joint_gain(entry, field_name, joint_name)
                if value is not None:
                    break
            if value is None:
                raise ValueError(
                    f"Recorded gains have no {field_name} for joint {joint_name!r}."
                )
            values.append(value)
        vectors[field_name] = tuple(values)
    return vectors
