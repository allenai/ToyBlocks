"""ManiFlow observation transform for ``color_only_blocks_table`` (front + gripper cams)."""

from __future__ import annotations

from typing import Any

from toyblocks.serving.obs_transforms.maniflow import (
    ManiflowObsTransform,
    PromptSource,
    state_dim_include_flags,
)


class ManiflowBlocksTableObsTransform(ManiflowObsTransform):
    """ManiFlow inputs when the scene uses ``front_camera`` + ``gripper_camera``.

    The default :class:`~toyblocks.serving.obs_transforms.maniflow.ManiflowObsTransform`
    maps ``external_camera`` → ``front_rgb`` and ``wrist_camera`` → ``gripper_rgb``.
    Scene preset ``color_only_blocks_table`` instead exposes ``front_camera`` /
    ``left_camera`` from :mod:`toyblocks.scenes` plus ``gripper_camera`` from the
    Franka robot bundle — use this class (registry name ``maniflow_blocks_table``)
    with ``--obs_transform maniflow_blocks_table``.

    The ``state_dim`` the policy server advertises (see
    :data:`~toyblocks.serving.obs_transforms.maniflow.STATE_DIM_LAYOUTS`)
    configures the state blocks: 8 → 7 arm + 1 gripper, 9 → that plus pd_mode,
    19 → extended state (joint_pos + joint_vel + ee_xyz + gripper + pd_mode).
    Absent metadata defaults to 19, matching checkpoints trained with the
    ``franka_multimodal`` dataloader. Any ``include_*`` flag passed explicitly
    overrides the inference; a flag that contradicts ``state_dim`` raises in the
    base class.
    """

    def __init__(
        self, prompt: PromptSource = None, state_dim: int | None = None, **kwargs: Any
    ) -> None:
        kwargs.setdefault("external_cam", "front_camera")
        kwargs.setdefault("wrist_cam", "gripper_camera")
        for flag, value in state_dim_include_flags(state_dim).items():
            kwargs.setdefault(flag, value)
        # The state layout the checkpoint's state_dim has to agree with, resolved
        # here from metadata; worth having in the log of every eval.
        print(
            f"[INFO]: ManiflowBlocksTableObsTransform state_dim={state_dim!r} "
            f"include_joint_vel={kwargs.get('include_joint_vel')} "
            f"include_ee_pos={kwargs.get('include_ee_pos')} "
            f"include_pd_mode={kwargs.get('include_pd_mode')}",
            flush=True,
        )
        super().__init__(prompt=prompt, state_dim=state_dim, **kwargs)
