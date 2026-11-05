"""
Centralized stiffness / damping / effort-limit profiles for the Franka arm.

Add new profiles to ``STIFFNESS_PROFILES`` and they become available everywhere
via ``set_stiffness_profile(name)`` on IsaacRobot / IsaacRobotVectorized.

Copied from ``tmp3.Backends._shared.stiffness_profiles`` (TAMP main) so the vecgen path
runs without TAMP; keep the two in sync while the legacy pipeline still uses TAMP's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass(frozen=True)
class StiffnessProfile:
    """Immutable PD-gain profile for 7-DOF Franka arm joints."""

    stiffness: Tuple[float, ...]
    damping: Tuple[float, ...]
    effort_limit: Tuple[float, ...]
    velocity_limit: Tuple[float, ...]
    settle_steps: int = 10
    pd_mode: float = 1.0  # scalar written into recorded data (0.0 = softest, 1.0 = stiffest)


STIFFNESS_PROFILES: Dict[str, StiffnessProfile] = {
    "stiff": StiffnessProfile(
        stiffness=(600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0),
        damping=(50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0),
        effort_limit=(87.0, 87.0, 87.0, 87.0, 87.0, 87.0, 87.0),
        velocity_limit=(2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61),
        settle_steps=12,
        pd_mode=1.0,
    ),
    # Intermediate profile for sim2real-oriented datagen: half of "stiff" on
    # the main joints (low-kp data transfers poorly to the real robot, but
    # kp=600 contact skills are brittle), damping ~critically scaled from
    # "stiff" (kd ~ sqrt(kp)), wrist effort limits at the real FR3's 12 Nm.
    "medium": StiffnessProfile(
        stiffness=(300.0, 300.0, 300.0, 300.0, 150.0, 100.0, 50.0),
        damping=(35.0, 35.0, 35.0, 35.0, 20.0, 15.0, 10.0),
        effort_limit=(87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0),
        velocity_limit=(2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61),
        settle_steps=12,
        pd_mode=1.0,
    ),
    "soft": StiffnessProfile(
        # stiffness=(4.63, 5.41, 8.06, 7.28, 8.04, 7.18, 4.63),
        stiffness=(100., 100., 100., 100., 75., 150., 50.),
        damping=(20., 20., 20., 20., 7.5, 15.0, 5.0),
        effort_limit=(87.0, 87.0, 87.0, 87.0, 87.0, 87.0, 87.0),
        velocity_limit=(2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61),
        settle_steps=12,
        pd_mode=0.0,
    ),
    # Matches the ``fr3_robotiq_gripper_softer`` robot config's arm PD gains
    "softer": StiffnessProfile(
        stiffness=(40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0),
        damping=(13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0),
        effort_limit=(87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0),
        velocity_limit=(2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61),
        settle_steps=12,
        pd_mode=0.0,
    ),
    "flip": StiffnessProfile(
        stiffness=(200.0, 200.0, 200.0, 200.0, 50.0, 50.0, 30.0),
        damping=(40.0, 40.0, 40.0, 40.0, 10.0, 10.0, 8.0),
        effort_limit=(87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0),
        velocity_limit=(2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61),
        settle_steps=12,
        pd_mode=1.0,
    ),
}


def register_profile(name: str, profile: StiffnessProfile) -> None:
    """Register (or overwrite) a named stiffness profile at runtime."""
    STIFFNESS_PROFILES[name] = profile


def get_profile(name: str) -> StiffnessProfile:
    if name not in STIFFNESS_PROFILES:
        available = sorted(STIFFNESS_PROFILES.keys())
        raise ValueError(
            f"Unknown stiffness profile '{name}'. Available: {available}"
        )
    return STIFFNESS_PROFILES[name]


def get_profile_tensors(
    name: str,
    device: torch.device,
    batch_size: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (stiffness, damping, effort) tensors shaped ``[batch_size, 7]``."""
    p = get_profile(name)
    stiffness = torch.tensor([p.stiffness], device=device).expand(batch_size, -1)
    damping = torch.tensor([p.damping], device=device).expand(batch_size, -1)
    effort = torch.tensor([p.effort_limit], device=device).expand(batch_size, -1)
    velocity_limit = torch.tensor([p.velocity_limit], device=device).expand(batch_size, -1)
    return stiffness, damping, effort, velocity_limit


def classify_profile(stiffness_value: float) -> str:
    """Best-match profile name for a measured first-joint stiffness reading."""
    best_name = "unknown"
    best_dist = float("inf")
    for name, profile in STIFFNESS_PROFILES.items():
        dist = abs(stiffness_value - profile.stiffness[0])
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name


def pd_mode_for_stiffness(stiffness_value: float) -> float:
    """Return the ``pd_mode`` scalar for the closest matching profile."""
    name = classify_profile(stiffness_value)
    return STIFFNESS_PROFILES[name].pd_mode if name in STIFFNESS_PROFILES else 1.0


# Maps a blocksuite robot config ``id`` to the stiffness profile that reproduces
# its arm PD gains.
ROBOT_ID_TO_PROFILE: Dict[str, str] = {
    "franka_robotiq_gripper": "soft",
    "franka_robotiq_gripper_soft": "soft",
    "fr3_robotiq_gripper_soft": "soft",
    "fr3_robotiq_gripper_softer": "softer",
    "fr3_robotiq_gripper_sysid": "medium",
}


def profile_for_robot_id(robot_id: str | None, default: str = "soft") -> str:
    """Return the stiffness profile name matching a robot config ``id``.

    Falls back to ``default`` for unknown / missing ids so callers degrade
    gracefully rather than crashing.
    """
    if robot_id is None:
        return default
    return ROBOT_ID_TO_PROFILE.get(robot_id, default)
