"""System identification for the FR3. See SYSID.md.

  collect.py     step 1  -- run the excitation on the real robot and record it
  fit_mujoco.py  step 2a -- closed-loop replay fit in MuJoCo, with plots
  fit_isaac.py   step 2b -- batched single-step fit in Isaac/PhysX, with plots
  common.py      the excitation, recording/metadata loading, and the plots
"""
