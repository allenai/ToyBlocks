"""PD-mode gain switching controller with Schmitt-trigger hysteresis + debounce.

Handles closed-loop gain transitions for policies that emit a continuous
``pd_mode`` channel (0=compliant, 1=stiff).

Fully vectorized: tracks independent per-environment state so each env
can switch modes at different times.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def _joint_vectors_to_gains_dict(
    vectors: dict[str, tuple[float, ...]],
    n_arm: int,
) -> dict[str, list[float]]:
    return {
        "stiffness": list(vectors["stiffness"][:n_arm]),
        "damping": list(vectors["damping"][:n_arm]),
        "effort": list(vectors["effort_limit"][:n_arm]),
    }


def load_default_pd_gains(n_arm: int = 7) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Load stiff/compliant PD gains from blocksuite robot configs.

    Stiff gains come from ``actuator_profiles.TAMP_ACTUATORS`` (880/80). Compliant
    gains come from ``fr3_softer.FRANKA_DROID_ARTICULATION_CFG`` (40/13).
    """
    from blocksuite.robots.actuator_profiles import (
        TAMP_ACTUATORS,
        actuators_to_gains,
        gains_to_joint_vectors,
    )
    from blocksuite.robots.configs.fr3_softer import FRANKA_DROID_ARTICULATION_CFG

    fields = ("stiffness", "damping", "effort_limit")
    stiff_gains = _joint_vectors_to_gains_dict(
        gains_to_joint_vectors(actuators_to_gains(TAMP_ACTUATORS), fields=fields),
        n_arm,
    )
    compliant_gains = _joint_vectors_to_gains_dict(
        gains_to_joint_vectors(
            actuators_to_gains(FRANKA_DROID_ARTICULATION_CFG.actuators),
            fields=fields,
        ),
        n_arm,
    )
    return stiff_gains, compliant_gains


def load_pd_gains_from_dataset(
    dataset_path: str | Path,
    n_arm: int = 7,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Extract stiff and compliant PD gains from a LeRobot dataset.

    Reads the ``observation.state.current_stiffness``, ``current_damping``,
    ``current_effort_limit`` and ``action.pd_mode`` columns from the first
    parquet chunk and derives a single representative gain vector for each
    mode (pd_mode >= 0.5 → stiff, < 0.5 → compliant).

    Returns ``(stiff_gains, compliant_gains)`` dicts (keys: stiffness,
    damping, effort). Raises if the dataset is missing or predates gain
    recording. When the dataset lacks frames for one mode, falls back to
    blocksuite defaults (stiff=TAMP_ACTUATORS, compliant=fr3_softer).
    """
    import numpy as np

    dataset_path = Path(dataset_path)
    data_dir = dataset_path / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"No data/ directory in {dataset_path}; cannot load PD gains.")

    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {data_dir}; cannot load PD gains.")

    import pyarrow.parquet as pq

    required_cols = [
        "observation.state.current_stiffness",
        "observation.state.current_damping",
        "observation.state.current_effort_limit",
        "action.pd_mode",
    ]

    table = pq.read_table(parquet_files[0], columns=required_cols)
    for col in required_cols:
        if col not in table.column_names:
            raise ValueError(f"Dataset {dataset_path} missing column {col!r}; cannot load PD gains.")

    stiff_arr = np.array(table["observation.state.current_stiffness"].to_pylist())
    damp_arr = np.array(table["observation.state.current_damping"].to_pylist())
    effort_arr = np.array(table["observation.state.current_effort_limit"].to_pylist())
    pd_mode_arr = np.array(table["action.pd_mode"].to_pylist()).flatten()

    stiff_mask = pd_mode_arr >= 0.5
    compliant_mask = pd_mode_arr < 0.5

    default_stiff_gains, default_compliant_gains = load_default_pd_gains(n_arm)

    if not stiff_mask.any() or not compliant_mask.any():
        logger.warning(
            "Dataset %s does not contain both stiff and compliant frames "
            "(stiff frames: %d, compliant: %d); using blocksuite defaults for "
            "missing mode(s) (stiff=TAMP_ACTUATORS, compliant=fr3_softer).",
            dataset_path,
            int(stiff_mask.sum()),
            int(compliant_mask.sum()),
        )

    def _representative_row(arr: np.ndarray, mask: np.ndarray) -> list[float]:
        rows = np.unique(arr[mask][:, :n_arm], axis=0)
        if len(rows) > 1:
            logger.info(
                "Multiple unique gain vectors found; using the first of %d.",
                len(rows),
            )
        return rows[0].tolist()

    if stiff_mask.any():
        stiff_gains: dict[str, list[float]] = {
            "stiffness": _representative_row(stiff_arr, stiff_mask),
            "damping": _representative_row(damp_arr, stiff_mask),
            "effort": _representative_row(effort_arr, stiff_mask),
        }
    else:
        stiff_gains = default_stiff_gains

    if compliant_mask.any():
        compliant_gains: dict[str, list[float]] = {
            "stiffness": _representative_row(stiff_arr, compliant_mask),
            "damping": _representative_row(damp_arr, compliant_mask),
            "effort": _representative_row(effort_arr, compliant_mask),
        }
    else:
        compliant_gains = default_compliant_gains

    logger.info("Loaded PD gains from dataset %s", dataset_path)
    logger.info("  stiff:     %s", stiff_gains)
    logger.info("  compliant: %s", compliant_gains)

    return stiff_gains, compliant_gains


def read_pd_gains_from_robot(base_env, n_arm: int = 7) -> dict[str, list[float]]:
    """Read the current arm PD gains from the live robot articulation.

    Returns a dict with keys ``stiffness``, ``damping``, ``effort`` — each a
    list of ``n_arm`` floats — reflecting whatever the robot config set.
    """
    robot = base_env.robot.articulation
    gains: dict[str, list[float]] = {
        "stiffness": robot.data.joint_stiffness[0, :n_arm].detach().cpu().tolist(),
        "damping": robot.data.joint_damping[0, :n_arm].detach().cpu().tolist(),
        "effort": robot.data.joint_effort_limits[0, :n_arm].detach().cpu().tolist(),
    }
    logger.info("Read PD gains from robot config: %s", gains)
    return gains


# pd_mode >= this -> switch to stiff gains (model outputs in [0, 1])
PD_STIFF_THRESH = 0.7

# pd_mode <= this -> switch to compliant gains; values in (COMPLIANT, STIFF) are a
# dead zone where gains stay unchanged, preventing thrashing on noisy predictions
PD_COMPLIANT_THRESH = 0.3

# Number of consecutive same-direction predictions required before committing a switch
PD_MODE_DEBOUNCE = 0

# ``(compliant_ticks, stiff_ticks)``: TAMP control ticks the arm is held still for
# after a gain switch, x ``decimation`` raw substeps. These are the legacy
# counts the standalone dataset replay used; the datasets themselves do not
# record settle steps (the TAMP orchestrator suppresses recording while a new
# profile settles), so this is a fixed convention applied whenever a
# ground-truth dataset is given, not a value read from the parquet files.
DATASET_SWITCH_SETTLE_TICKS: tuple[int, int] = (5, 10)


class PDModeController:
    """Manages PD-mode gain switching with hysteresis and debounce.

    Two layers of protection against thrashing:

    1. Schmitt-trigger hysteresis: a prediction must clear ``stiff_thresh``
       to be considered "stiff" or fall below ``compliant_thresh`` to be
       considered "compliant". Values in the dead zone leave gains untouched.
    2. Debounce: ``debounce`` consecutive predictions of the opposite mode
       are required before actually switching.

    Supports vectorized environments: each env has independent state.
    """

    def __init__(
        self,
        base_env,
        device: torch.device,
        *,
        num_envs: int = 1,
        stiff_gains: dict[str, list[float]],
        compliant_gains: dict[str, list[float]],
        stiff_thresh: float = PD_STIFF_THRESH,
        compliant_thresh: float = PD_COMPLIANT_THRESH,
        debounce: int = PD_MODE_DEBOUNCE,
        initial_mode: int = -1,  # -1 means PD mode starts uninitialized. If called with 0 or 1, it will start in the given mode.
        switch_settle_ticks: tuple[int, int] | None = None,
        defer_settle: bool = False,
    ):
        """
        Args:
            switch_settle_ticks: ``(compliant_ticks, stiff_ticks)`` TAMP control
                ticks to hold the arm still after a mode switch committed by
                :meth:`apply`, run as ``ticks[new_mode] * decimation`` raw sim
                substeps with the arm joints frozen and the finger target left
                active (the TAMP datagen settle). ``None`` writes gains with no
                settle period, letting the normal ``env.step()`` substeps absorb
                the transition. Initial writes (:meth:`initialize_after_reset`
                and ``initial_mode``) never settle.
            defer_settle: Write switched gains immediately and leave the settle
                to the caller (:class:`DeferredSettle`, sized by
                :meth:`settle_ticks`). The raw substeps advance every env in the
                scene, so in a vectorized scene the in-place settle would hand
                every non-switching env unobserved physics time; the eval defers
                at every env count so single- and multi-env runs settle alike.
        """
        self._base_env = base_env
        self._device = device
        self._num_envs = num_envs
        self._stiff_gains = stiff_gains
        self._compliant_gains = compliant_gains
        self._stiff_thresh = stiff_thresh
        self._compliant_thresh = compliant_thresh
        self._debounce = debounce
        self._switch_settle_ticks: tuple[int, int] | None = (
            (int(switch_settle_ticks[0]), int(switch_settle_ticks[1]))
            if switch_settle_ticks is not None
            else None
        )
        self._defer_settle = defer_settle

        # Per-env state: -1 means uninitialized
        self._prev_pd_mode = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self._pending = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self._pending_count = torch.zeros(num_envs, dtype=torch.long, device=device)

        # Mode whose gains are physically in the sim. Unlike ``_prev_pd_mode``,
        # which ``reset_envs`` clears to re-arm the first-time path, this only
        # moves on an actual switch, so it survives an episode boundary just
        # like the gains do.
        self._applied_mode = torch.full(
            (num_envs,), float("nan"), dtype=torch.float32, device=device
        )

        if initial_mode in (0, 1):
            target_pd_mode = torch.full((num_envs,), initial_mode, dtype=torch.long, device=device)
            mask_env_ids = torch.ones(num_envs, dtype=torch.bool, device=device)
            self._commit(target_pd_mode, mask_env_ids, settle=False)
            logger.info("PDModeController: initialized all envs to mode %d", initial_mode)

    @property
    def applied_modes(self) -> torch.Tensor:
        """Per-env mode currently in effect (0 compliant / 1 stiff).

        NaN for an env whose gains no switch has touched yet. This is the
        binary mode the gains reflect, not the continuous prediction that asked
        for it — the dead zone and debounce mean the two can disagree.
        """
        return self._applied_mode

    def reset_envs(self, env_mask: torch.Tensor) -> None:
        """Reset state for environments that just finished an episode.

        The next call to ``apply`` for these envs will bypass debounce
        (first-time initialization logic), ensuring the robot starts
        at the correct stiffness for the new episode.
        """
        self._prev_pd_mode[env_mask] = -1
        self._pending[env_mask] = -1
        self._pending_count[env_mask] = 0

    def initialize_after_reset(
        self,
        env_mask: torch.Tensor | None = None,
        target_mode: int | torch.Tensor = 1,
    ) -> None:
        """Write actuator gains for ``target_mode`` immediately and mark the
        given envs as already in that mode.

        Use this right after ``env.reset()`` (initial or auto-reset) so the
        very first policy step already sees the training-time impedance
        instead of whatever gains the robot's default cfg happens to apply
        on reset. Without this, the transient between reset (softer default)
        and the first policy step (STIFF written by ``apply``) leaks one
        step of wrong dynamics into the beginning of every episode.

        After this call, :meth:`current_pd_mode` reports ``target_mode`` for
        the given envs truthfully, letting the observation transform put a
        non-lying ``pd_mode`` scalar into ``agent_pos`` at step 0.

        Args:
            env_mask: Boolean tensor of shape ``(num_envs,)`` selecting which
                envs to (re)initialize. ``None`` means all envs.
            target_mode: 1 = stiff (default), 0 = compliant. Either a scalar
                (broadcast to every env) or a ``(num_envs,)`` tensor giving a
                per-env mode, so different envs can start under different
                impedances (e.g. from per-episode setups whose recorded
                first-frame ``action.pd_mode`` disagrees).
        """
        if env_mask is None:
            mask = torch.ones(self._num_envs, dtype=torch.bool, device=self._device)
        elif env_mask.dtype != torch.bool:
            mask = env_mask.bool()
        else:
            mask = env_mask
        if not mask.any():
            return
        if isinstance(target_mode, torch.Tensor):
            if target_mode.numel() != self._num_envs:
                raise ValueError(
                    f"target_mode tensor has {target_mode.numel()} entries; "
                    f"expected {self._num_envs}"
                )
            target = target_mode.to(dtype=torch.long, device=self._device).reshape(-1)
        else:
            target = torch.full(
                (self._num_envs,), int(target_mode), dtype=torch.long, device=self._device
            )
        # Force ``_commit`` to run unconditionally on the given envs by clearing
        # any pending debounce and marking them uninitialized; ``_commit``
        # updates ``_prev_pd_mode`` for the switched envs at the end.
        self._prev_pd_mode[mask] = -1
        self._pending[mask] = -1
        self._pending_count[mask] = 0
        # No settle: the recorded frame-0 state was just restored atomically by
        # the reset, and advancing physics here would leave episode-order
        # dependent PhysX contact history in the scene.
        self._commit(target, mask, settle=False)

    def current_pd_mode(self) -> torch.Tensor:
        """Per-env last-committed pd_mode as a float tensor in ``{0.0, 1.0}``.

        Envs that have never been committed (still ``-1``) are reported as
        ``1.0`` to match the transform's conservative default; call
        :meth:`initialize_after_reset` first if you want this to be truthful
        for every env.
        """
        return torch.where(
            self._prev_pd_mode >= 0,
            self._prev_pd_mode.float(),
            torch.ones(self._num_envs, dtype=torch.float32, device=self._device),
        )

    def apply(
        self, pd_mode_vals: torch.Tensor, env_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Switch actuator gains based on (noisy) continuous pd_mode values.

        Args:
            pd_mode_vals: Tensor of shape ``(num_envs,)`` with continuous
                pd_mode predictions per environment.
            env_mask: Optional ``(num_envs,)`` bool tensor. Envs outside it are
                left untouched by this call, debounce state included.

        Returns:
            ``(num_envs,)`` bool tensor of the envs whose gains were switched.
        """
        if pd_mode_vals.ndim == 0:
            pd_mode_vals = pd_mode_vals.unsqueeze(0)

        stiff_mask = pd_mode_vals >= self._stiff_thresh
        compliant_mask = pd_mode_vals <= self._compliant_thresh
        dead_zone_mask = ~stiff_mask & ~compliant_mask

        pd_mode_binary = torch.where(
            stiff_mask, torch.ones_like(self._prev_pd_mode), torch.zeros_like(self._prev_pd_mode)
        )

        # Environments in the dead zone get no action
        actionable = ~dead_zone_mask
        if env_mask is not None:
            actionable = actionable & env_mask.to(device=actionable.device, dtype=torch.bool)

        # First-time initialization: bypass debounce
        first_time = (self._prev_pd_mode == -1) & actionable

        # Same-as-current: reset pending
        same_as_current = (pd_mode_binary == self._prev_pd_mode) & actionable & ~first_time
        self._pending[same_as_current] = -1
        self._pending_count[same_as_current] = 0

        # Different from current: accumulate debounce
        diff_from_current = actionable & ~first_time & ~same_as_current
        match_pending = diff_from_current & (pd_mode_binary == self._pending)
        new_pending = diff_from_current & ~match_pending

        self._pending_count[match_pending] += 1
        self._pending[new_pending] = pd_mode_binary[new_pending]
        self._pending_count[new_pending] = 1

        # Determine which envs actually switch
        debounce_ready = diff_from_current & (self._pending_count >= self._debounce)
        switch_mask = first_time | debounce_ready

        if switch_mask.any():
            self._commit(pd_mode_binary, switch_mask, settle=not self._defer_settle)
        return switch_mask

    def settle_ticks(self) -> torch.Tensor:
        """Per-env settle length, in control ticks, for the mode each env is now in.

        The deferred counterpart of :meth:`_settle_substeps`: one tick is one
        ``env.step()``, i.e. ``decimation`` substeps. Zero when no
        ``switch_settle_ticks`` were configured.
        """
        if self._switch_settle_ticks is None:
            return torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        compliant_ticks, stiff_ticks = self._switch_settle_ticks
        return torch.where(
            self._prev_pd_mode == 1,
            torch.full((self._num_envs,), stiff_ticks, dtype=torch.long, device=self._device),
            torch.full((self._num_envs,), compliant_ticks, dtype=torch.long, device=self._device),
        )

    def _settle_substeps(self, mode: int) -> int:
        """Raw sim substeps to hold the arm after switching into *mode*.

        TAMP advances one control tick per render interval, and
        ``set_actuator_gains`` settles via raw ``sim.step()`` calls, so the
        recorded tick count is scaled by the env's decimation to reproduce the
        same elapsed physics time.
        """
        if self._switch_settle_ticks is None:
            return 0
        ticks = self._switch_settle_ticks[1] if mode == 1 else self._switch_settle_ticks[0]
        decimation = int(getattr(self._base_env.cfg, "decimation", 1))
        return ticks * decimation

    def _commit(
        self,
        pd_mode_binary: torch.Tensor,
        switch_mask: torch.Tensor,
        *,
        settle: bool,
    ) -> None:
        """Apply gains for the switching environments and reset their debounce.

        With ``settle=False`` (or no ``switch_settle_ticks``) the gains are
        written immediately and the normal ``env.step()`` substeps absorb the
        transition. With ``settle=True`` and ``switch_settle_ticks`` set, the
        write is followed by the TAMP-style settle: ``ticks[new_mode] *
        decimation`` raw substeps with the switching envs' arm targets held at
        their current position and their joint velocities left untouched, as
        TAMP's stiffness-profile switch does (finger target left active, so an
        in-progress close is not interrupted). The tick count depends on the mode being
        switched *into*, so when ``switch_mask`` holds envs going both ways the
        write is issued once per mode present (at most two calls).
        """
        device = self._device
        robot = self._base_env.robot.articulation
        n_joints = robot.data.joint_pos.shape[1]
        n_arm = min(7, n_joints)

        stiff = robot.data.joint_stiffness.clone()
        damp = robot.data.joint_damping.clone()
        eff = robot.data.joint_effort_limits.clone()

        stiff_gains_t = torch.tensor(
            self._stiff_gains["stiffness"][:n_arm], device=device, dtype=torch.float32
        )
        compliant_gains_stiff_t = torch.tensor(
            self._compliant_gains["stiffness"][:n_arm], device=device, dtype=torch.float32
        )
        stiff_damp_t = torch.tensor(
            self._stiff_gains["damping"][:n_arm], device=device, dtype=torch.float32
        )
        compliant_damp_t = torch.tensor(
            self._compliant_gains["damping"][:n_arm], device=device, dtype=torch.float32
        )
        stiff_eff_t = torch.tensor(
            self._stiff_gains["effort"][:n_arm], device=device, dtype=torch.float32
        )
        compliant_eff_t = torch.tensor(
            self._compliant_gains["effort"][:n_arm], device=device, dtype=torch.float32
        )

        for i in range(self._num_envs):
            if not switch_mask[i]:
                continue
            mode = int(pd_mode_binary[i].item())
            if mode == 1:
                stiff[i, :n_arm] = stiff_gains_t
                damp[i, :n_arm] = stiff_damp_t
                eff[i, :n_arm] = stiff_eff_t
            else:
                stiff[i, :n_arm] = compliant_gains_stiff_t
                damp[i, :n_arm] = compliant_damp_t
                eff[i, :n_arm] = compliant_eff_t

        if not settle or self._switch_settle_ticks is None:
            self._base_env.set_actuator_gains(
                stiffness=stiff,
                damping=damp,
                effort_limits=eff,
                settle_steps=0,
                settle_env_mask=switch_mask,
            )
        else:
            # One write per target mode present in the mask: the settle length
            # is a property of the mode being entered. Freezing only the arm
            # matches TAMP, which keeps the finger target active during the
            # settle so a close in flight is not interrupted right before a
            # contact-rich phase.
            for mode in (0, 1):
                mode_mask = switch_mask & (pd_mode_binary == mode)
                if not bool(mode_mask.any()):
                    continue
                self._base_env.set_actuator_gains(
                    stiffness=stiff,
                    damping=damp,
                    effort_limits=eff,
                    settle_steps=self._settle_substeps(mode),
                    settle_env_mask=mode_mask,
                    settle_joint_ids=list(range(n_arm)),
                    settle_zero_velocity=False,
                )

        # Print diagnostics for switching envs
        switch_indices = switch_mask.nonzero(as_tuple=False).flatten()
        for i in switch_indices:
            idx = int(i.item())
            mode = int(pd_mode_binary[idx].item())
            mode_name = "stiff" if mode == 1 else "compliant"
            logger.debug("[PD mode] env %d: Switched to %s (pd_mode=%d)", idx, mode_name, mode)

        if self._num_envs == 1 and switch_mask[0]:
            self._print_diagnostic(stiff, damp, eff, n_arm, int(pd_mode_binary[0].item()), robot)

        # Update state for switching envs
        self._prev_pd_mode[switch_mask] = pd_mode_binary[switch_mask]
        self._applied_mode[switch_mask] = pd_mode_binary[switch_mask].to(torch.float32)
        self._pending[switch_mask] = -1
        self._pending_count[switch_mask] = 0

    def _print_diagnostic(self, stiff, damp, eff, n_arm, pd_mode_binary, robot) -> None:
        """Print side-by-side comparison of PD gains after a switch."""
        applied_stiff = stiff[0, :n_arm].cpu().tolist()
        applied_damp = damp[0, :n_arm].cpu().tolist()
        applied_eff = eff[0, :n_arm].cpu().tolist()

        header = (
            f"  {'Joint':>8}  {'Stiff(mode0)':>12} {'Stiff(mode1)':>12} {'Stiff(applied)':>14}  |  "
            f"{'Damp(mode0)':>11} {'Damp(mode1)':>11} {'Damp(applied)':>13}  |  "
            f"{'Effort(mode0)':>13} {'Effort(mode1)':>13} {'Effort(applied)':>15}"
        )
        sep = "  " + "-" * len(header.lstrip())
        print(header)
        print(sep)
        for j in range(n_arm):
            s0 = self._compliant_gains["stiffness"][j]
            s1 = self._stiff_gains["stiffness"][j]
            sa = applied_stiff[j]
            d0 = self._compliant_gains["damping"][j]
            d1 = self._stiff_gains["damping"][j]
            da = applied_damp[j]
            e0 = self._compliant_gains["effort"][j]
            e1 = self._stiff_gains["effort"][j]
            ea = applied_eff[j]
            s_flag = (
                " "
                if (sa == s0 and pd_mode_binary == 0) or (sa == s1 and pd_mode_binary == 1)
                else "*"
            )
            d_flag = (
                " "
                if (da == d0 and pd_mode_binary == 0) or (da == d1 and pd_mode_binary == 1)
                else "*"
            )
            e_flag = (
                " "
                if (ea == e0 and pd_mode_binary == 0) or (ea == e1 and pd_mode_binary == 1)
                else "*"
            )
            print(
                f"  {j:>8}  {s0:>12.3f} {s1:>12.3f} {sa:>13.3f}{s_flag}  |  "
                f"{d0:>11.3f} {d1:>11.3f} {da:>12.3f}{d_flag}  |  "
                f"{e0:>13.3f} {e1:>13.3f} {ea:>14.3f}{e_flag}"
            )
        print(sep)
        print("  (* = applied value differs from expected for the selected mode)")

        n_total = stiff.shape[1]
        if n_total > n_arm:
            gripper_stiff = stiff[0, n_arm:].cpu().tolist()
            gripper_damp = damp[0, n_arm:].cpu().tolist()
            gripper_eff = eff[0, n_arm:].cpu().tolist()
            try:
                joint_names = robot.joint_names
            except Exception:
                joint_names = None
            gh = f"  {'Joint':>8}  {'Name':>28}  {'Stiffness':>10} {'Damping':>10} {'Effort':>10}"
            gsep = "  " + "-" * len(gh.lstrip())
            print()
            print("  [Gripper joints — unchanged by PD mode, showing current sim values]")
            print(gh)
            print(gsep)
            for k in range(n_total - n_arm):
                idx = n_arm + k
                jn = joint_names[idx] if joint_names and idx < len(joint_names) else "?"
                print(
                    f"  {idx:>8}  {jn:>28}  "
                    f"{gripper_stiff[k]:>10.3f} {gripper_damp[k]:>10.4f} {gripper_eff[k]:>10.3f}"
                )
            print(gsep)


def settle_hold_action(base_env) -> torch.Tensor:
    """``(num_envs, arm_dim + 1)`` action that holds every env where the legacy settle did.

    The arm part is each env's current arm joint positions, which is a hold only
    for an absolute joint-position arm term (``arm_joint_gripper_binary``). The
    gripper part is the binary command (``-1`` close, ``+1`` open) that
    reproduces the finger target already on the articulation: the legacy settle
    froze the arm and left that target untouched. ``base_env.actions`` would be
    the wrong source -- auto-reset does not clear it, so a switch on an
    episode's first step would hold with the previous episode's last command.
    """
    controllers = base_env.robot.controller
    arm, gripper = controllers.get("arm"), controllers.get("gripper")
    required = ("_open_command", "_close_command", "_joint_ids")
    if arm is None or gripper is None or not all(hasattr(gripper, a) for a in required):
        raise ValueError(
            "the deferred PD-switch settle needs an 'arm' joint action plus a binary "
            f"'gripper' action term to build its hold action; got controllers "
            f"{sorted(controllers)}."
        )
    data = base_env.robot.articulation.data
    arm_pos = data.joint_pos[:, arm._joint_ids].clone()
    target = data.joint_pos_target[:, gripper._joint_ids]
    open_cmd = gripper._open_command.to(target.device)
    close_cmd = gripper._close_command.to(target.device)
    # Signed distance past the open/close midpoint, along the closing direction
    # of each finger joint, so the rule holds whichever way a joint closes.
    past_midpoint = (target - (open_cmd + close_cmd) / 2) * (close_cmd - open_cmd)
    closed = (past_midpoint > 0).all(dim=-1)
    gripper_cmd = torch.where(
        closed,
        torch.full_like(closed, -1.0, dtype=arm_pos.dtype),
        torch.full_like(closed, 1.0, dtype=arm_pos.dtype),
    )
    return torch.cat([arm_pos, gripper_cmd.unsqueeze(-1)], dim=-1)


class DeferredSettle:
    """Per-env PD-switch settle run as ordinary control steps.

    The in-place settle (``PDModeController._commit`` with ``settle=True``) runs
    raw ``sim.step()`` calls, which PhysX applies to every env in the scene, so
    in a vectorized scene every switch hands all the other envs unobserved
    physics time. Here an env that switches instead spends its settle ticks as
    ordinary ``env.step()`` calls under a hold action (see
    :func:`settle_hold_action`), then executes the action that triggered the
    switch, while the other envs keep stepping under their own policy actions.
    The eval uses it for every env count, including one, so single- and
    multi-env runs settle the same way.

    For a switch at step ``t`` with ``k`` ticks: steps ``t .. t+k-1`` hold, step
    ``t+k`` executes the deferred action, and the env is :attr:`busy` (its policy
    must not advance) from ``t+1`` through ``t+k``. The caller is expected to keep
    hold steps out of the recording and the episode clock.

    Differences from the in-place settle, accepted because they do not change
    actions or recorded rows (they apply at every env count):

    * Any episode-end request -- success with ``--episode_end goal``, oracle
      done, HL-model done, the expert timeout -- can land on a hold step and end
      the episode mid-hold, so the deferred action never runs; the in-place
      settle always ran it. Success requires a released gripper and the gripper
      command does not change during a hold, so a success end there should be
      rare.
    * Only ``episode_length_buf`` is compensated, so ``first_success_step`` and
      the expert-timeout budget skip hold steps. The oracle's ``compute()`` and
      its backstop's ``_success_streak``, the HL latency clock
      (``hl_sim_time_ms``), ``RecordMetrics`` (one ``step_metrics`` row per hold
      step) and interval events all advance on hold steps.
    * Hold frames appear in the sensor-obs video, which freezes for ``k`` frames
      per switch; tools matching its frames to recorded rows by index will
      misalign (the LeRobot dataset skips hold rows).
    * Each switch costs ``k`` extra rendered ``env.step()`` calls, where the raw
      substeps were never rendered. An env that thrashes between modes spends
      extra steps holding; its timeout budget is unchanged but the run takes
      longer.
    * A hold is a full ``env.step()``: it re-applies the action terms every
      substep and updates ``prev_actions``, which the bare ``sim.step()`` calls
      did not, so the physics is close to the in-place settle but not
      bit-identical.
    """

    def __init__(self, num_envs: int, device: torch.device) -> None:
        self._num_envs = num_envs
        self._device = device
        self._remaining = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._deferred_valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._hold_action: torch.Tensor | None = None
        self._deferred_action: torch.Tensor | None = None

    @property
    def busy(self) -> torch.Tensor:
        """``(num_envs,)`` bool: envs still holding or about to run their deferred action."""
        return (self._remaining > 0) | self._deferred_valid

    def start(
        self,
        env_mask: torch.Tensor,
        ticks: torch.Tensor,
        hold_action: torch.Tensor,
        deferred_action: torch.Tensor,
    ) -> None:
        """Begin a settle for the ``env_mask`` envs.

        Args:
            env_mask: ``(num_envs,)`` bool tensor of the envs that just switched.
            ticks: ``(num_envs,)`` long tensor of hold steps per env.
            hold_action: ``(num_envs, action_dim)`` action to hold with.
            deferred_action: ``(num_envs, action_dim)`` action to run afterwards.
        """
        env_mask = env_mask.to(device=self._device, dtype=torch.bool)
        if not bool(env_mask.any()):
            return
        ticks = ticks.to(self._device)
        hold_action = hold_action.to(self._device)
        deferred_action = deferred_action.to(self._device)
        if self._hold_action is None or self._deferred_action is None:
            self._hold_action = torch.zeros_like(hold_action)
            self._deferred_action = torch.zeros_like(deferred_action)
        self._remaining[env_mask] = ticks[env_mask].to(self._remaining.dtype)
        self._hold_action[env_mask] = hold_action[env_mask].to(self._hold_action.dtype)
        self._deferred_action[env_mask] = deferred_action[env_mask].to(
            self._deferred_action.dtype
        )
        self._deferred_valid[env_mask] = True

    def compose(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Overlay this step's hold / deferred actions onto ``action``.

        Consumes one hold tick per holding env, so call it exactly once per
        ``env.step()``.

        Returns:
            ``(action_to_step, holding)``: the action to pass to ``env.step()``
            and the ``(num_envs,)`` bool mask of envs spending this step holding.
        """
        holding = self._remaining > 0
        if self._hold_action is None or self._deferred_action is None:
            return action, holding
        run_deferred = self._deferred_valid & ~holding
        if not bool(holding.any() or run_deferred.any()):
            return action, holding
        out = action.clone()
        out[holding.to(out.device)] = self._hold_action[holding].to(out)
        out[run_deferred.to(out.device)] = self._deferred_action[run_deferred].to(out)
        self._deferred_valid &= ~run_deferred
        self._remaining[holding] -= 1
        return out, holding

    def clear(self, env_mask: torch.Tensor) -> None:
        """Drop any settle in progress for the ``env_mask`` envs (episode boundary)."""
        env_mask = env_mask.to(device=self._device, dtype=torch.bool)
        self._remaining[env_mask] = 0
        self._deferred_valid[env_mask] = False
