from dataclasses import dataclass, replace
from typing import Literal

import tyro

from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher()
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from blocksuite.robots.configs.fr3_soft import FRANKA_DROID_ARTICULATION_CFG


@dataclass
class Args:
    device: Literal["cpu", "cuda"] = "cpu"


def main() -> int:
    args = tyro.cli(Args)

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device)
    sim_ctx = SimulationContext(sim_cfg)

    plane_cfg = sim_utils.GroundPlaneCfg()
    plane_cfg.func("/World/floor", plane_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/light", light_cfg)

    franka_cfg = replace(FRANKA_DROID_ARTICULATION_CFG, prim_path="/World/robot")
    franka = Articulation(cfg=franka_cfg)

    sim_dt = sim_ctx.get_physics_dt()

    sim_ctx.reset()

    while simulation_app.is_running():
        sim_ctx.step()
        franka.update(sim_dt)

    simulation_app.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
