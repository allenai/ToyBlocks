# System identification — FR3

Make the simulated FR3 move like *our* FR3 under the *same* joint-space
controller. We drive the real arm through a free-space excitation, record it,
then replay that recording in sim and fit the joint dynamics parameters until
the trajectories match.

Two engines, one recording:

| | fit | parameters |
|---|---|---|
| MuJoCo | `fit_mujoco.py` | `dof_frictionloss`, `dof_damping`, `dof_armature` |
| Isaac / PhysX | `fit_isaac.py` | joint `armature`, joint `friction` |

## Files

| | |
|---|---|
| `common.py` | the multi-sine trajectory, recording + metadata loading, the plots |
| `collect.py` | step 1 — run the excitation on the real robot, record it |
| `fit_mujoco.py` | step 2a — closed-loop replay fit in MuJoCo, with plots |
| `fit_isaac.py` | step 2b — batched single-step fit in Isaac, with plots |
| `sysid_data/` | a reference recording (2026-09-02, 25 Hz, ~1000 rows) |

## 1. Collect

```bash
cd /home/yejin/code/robot-control-stack/blocksuite

# plan the trajectory, print the safety table, write metadata — no motion
python -m blocksuite.serving.sim2real.sysid.collect \
    --record-path outputs/sysid_data_kp300 --dry-run

# the real run (prompts before moving; keep the e-stop ready)
python -m blocksuite.serving.sim2real.sysid.collect \
    --record-path outputs/sysid_data_kp300
```

Writes `outputs/sysid_data_kp300/real/*.parquet` and `outputs/sysid_data_kp300/sysid_run_meta.json`.
Defaults: 40 s at the config's rate (25 Hz), 6 harmonics from 0.2 Hz, per-joint
frequency offsets so no two joints share a frequency. Vary with `--duration`,
`--amp-scale`, `--center`, `--seed`.

**The env comes from `env_setup.get_env(load_inference_config())`** — the same
call `vlagent_rcs_client.py` and `episode_replay.py` make. Gains, torque limits,
joint-limit overrides, TAM switch, control mode, rate and per-step clamp are
whatever the policy runs use. Nothing is redeclared in the sysid code, so a
recording can never describe a plant we do not deploy.

`sysid_run_meta.json` records the gains actually in force (read back from
`env.stiffness_profile`), the rate, the clamp, the excitation settings and both
the commanded and measured centre. Both fits read it.

For workspace coverage — needed before mass/COM/inertia are identifiable at all
— record extra runs at other configurations with `--center`.

## 2a. Fit — MuJoCo

```bash
python -m blocksuite.serving.sim2real.sysid.fit_mujoco outputs/sysid_data_kp300
```

Replays the whole recording open-loop in a headless free-space sim
(`duobench.utils.single_arm.SingleArm`, no task scene, no rendering) at the
recorded gains, and runs per-joint coordinate descent over
`[frictionloss, damping, armature]`. One sweep, 40 rollouts per joint.

Writes `calibrated_params.json` and `sysid_fit/`:
`rms_per_joint.png` (baseline vs fitted), `overlay_jN.png` (real vs sim before
and after, plus the error trace, for the three joints that changed most),
`convergence.png`.

Apply by hand into `assets/robots/fr3/fr3.xml` — as commit `d2449a4e` did.

**Keep it modest.** More passes keep lowering the training RMS but push damping
and armature to their bounds: the three parameters are collinear at these
frequencies, so a lower RMS can mean less physical parameters. `--passes 1`
(the default) stays physical.

## 2b. Fit — Isaac

```bash
python -m blocksuite.serving.sim2real.sysid.fit_isaac outputs/sysid_data_kp300 \
    --headless --num-envs 512 --max-iter 40
```

Different method, because Isaac's advantage is parallelism and PhysX is not
differentiable:

- **Batched single-transition prediction error.** Seed env *i* with the real
  state at time *i*, command the recorded target, step one control step, compare
  the predicted `(q, dq)` against the real next state. 512 transitions evaluate
  in one batched step. Re-seeding every step kills drift, so the objective stays
  well conditioned — unlike the free rollout the MuJoCo fit uses.
- **CMA-ES** over 14 bounded parameters (7 armature + 7 friction), since there
  are no gradients to be had.
- **Velocity-weighted loss** (`--w_q 0.1`, `--w_dq 1.0`, residuals scaled by
  `1/dt`): a position-only loss hides over-damping.

Writes `isaac_calibrated_params.json` and `sysid_fit_isaac/` (same three figure
kinds, plus the CMA-ES trace). Values are pushed at runtime with
`write_joint_armature_to_sim` / `write_joint_friction_coefficient_to_sim`; bake
them into the robot config or USD by hand.

Needs `uv sync --extra isaac` and `cmaes`. `--task` only supplies an FR3
articulation to borrow; the scene's objects are never touched.

## Gains are fixed, never fit

Both fits read `kp`/`kd` from the recording's metadata and hold them there.
This is the single most important invariant in the pipeline: if sim replays at
different gains than the recording, the optimiser explains the *gain* mismatch
as *dynamics*, and the fitted friction and armature describe a plant that does
not exist.

There is one source of truth, `env_setup.PD_KP` / `PD_KD` / `PD_TORQUE_LIMIT`,
read by both branches of `get_env` (hardware and sim) and by `fit_mujoco.py`'s
replay env. Current profile: `kp300`.

`fit_isaac.py` deliberately does **not** reuse frankatwin's control law.
frankatwin's Isaac sysid was the reference for the optimizer and the parameter
handling, but it fits under a Cartesian impedance controller; we drive the arm
in joint space, so the sim must too.

## What this does not do

- **No held-out split.** Both fits report training error. The plots show how
  well the fit reproduces the run it was fitted on, which is not the same as
  generalisation. To validate, record a second run at a different `--center` (or
  a different amplitude) and check the fitted values reproduce. If joint 5's
  armature sits at its bound in one fit and not the other, the fit found noise.
- **No mass / COM / inertia.** A single-centre excitation identifies them only
  weakly; they stay at their CAD/USD values. Workspace-coverage runs would
  change that.
- **25 Hz.** Friction and motor delay live largely above it. The original Isaac plan
  flagged this and it is still true: the velocity-weighted loss mitigates the
  finite-difference noise but cannot recover what was never sampled. A faster
  logging path would be the single biggest improvement to fit quality.

## Known sim/real asymmetries

Not gain mismatches, but they show up in a sim-vs-real residual all the same:

- `env_setup.override_joint_limits` is applied on the **hardware** branch only,
  so sim joint 4 stops at −2.7478 where the real arm reaches −2.9921.
- The sim branch of `get_env` builds a `FlippingEnvConfig` task scene; hardware
  is a bare DROID env. (`fit_mujoco.py`'s own replay env avoids this — it uses
  the bare `SingleArm`.)

## Provenance

`collect.py`, `common.py` and `fit_mujoco.py` started as Tobias Juelg's
`sysid.py` on `origin/tobi/sim2real` (Sept 2026), split into collect/fit and
rewired to go through `env_setup`. The MJCF values currently in
`assets/robots/fr3/fr3.xml` came from that original at kp100 gains, so they do
not correspond to the current `kp300` profile.
