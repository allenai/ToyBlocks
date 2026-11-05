"""
Base arguments for all CLI commands that use the isaac app launcher, serves as a typed replacement for argparse
"""
from dataclasses import dataclass, field, make_dataclass
from toyblocks.tasks.base_env_cfg import BaseEnvCfg
import sys
from typing import TypeVar
from tyro.conf import Suppress
@dataclass
class BaseArgs:
    """Base arguments for all CLI commands that open isaac, serves as a typed replacement for argparse"""
    disable_fabric: bool = False
    """Disable fabric and use USD I/O operations."""
    headless: bool = False
    """Force display off at all times."""
    enable_cameras: bool = False
    """Enable camera sensors and relevant extension dependencies."""
    device: str = "cuda:0"
    """The device to run the simulation on. Can be "cpu", "cuda", "cuda:N", where N is the device ID"""
    verbose: bool = False
    """Enable verbose-level log output from the SimulationApp."""
    info: bool = False
    """Enable info-level log output from the SimulationApp."""
    experience: str = ""
    """The experience file to load when launching the SimulationApp. If an empty string is provided, the experience file is determined based on the headless flag."""
    kit_args: str = ""
    """Command line arguments for Omniverse Kit as a string separated by a space delimiter. Example usage: --kit_args "--ext-folder=/path/to/ext1 --ext-folder=/path/to/ext2"""
    rendering_mode: str = "balanced"
    """Sets the rendering mode. Preset settings files can be found in apps/rendering_modes. Can be "performance", "balanced", or "quality". Individual settings can be overwritten by using the RenderCfg class."""
    env_cfg: Suppress[BaseEnvCfg] = field(default_factory=BaseEnvCfg)
    """The environment configuration to use. This is automatically populated by the add_env_cfg_args decorator."""
    
T = TypeVar('T', bound=BaseArgs)
def add_env_cfg_args(args: T) -> T:
    """
    if you want a script/cli tool that has a --task argument and a corresponding typed env_cfg argument, decorate the Args class with this

    e.g.

    @add_env_cfg_args
    @dataclass
    class MyArgs(BaseArgs):
        # ...
    
    tyro.cli(MyArgs)
    """
    from isaaclab_tasks.utils import parse_env_cfg
    # Save the task id from sys.argv if specified, otherwise use Args default and add it to sys.argv
    specified_task = None
    for i, arg in enumerate(sys.argv):
        if arg.startswith("--task"):
            # Could be --task=TaskName or --task TaskName
            if "=" in arg:
                specified_task = arg.split("=")[1]
            elif (i + 1) < len(sys.argv):
                specified_task = sys.argv[i + 1]
            break
    if specified_task is None:
        specified_task = args.task
    # now we need to find the correct env config
    env_cfg: BaseEnvCfg = parse_env_cfg(
        specified_task, #device=args.device, num_envs=args.num_envs, use_fabric=not args.disable_fabric
    )
    args = make_dataclass(BaseArgs.__name__, [("env_cfg", env_cfg.__class__, field(default_factory=env_cfg.__class__))], bases=(args,))
    return args

def parse_cli(args_cls):
    """``tyro.cli`` that names an unrecognized option instead of failing to build a parser.

    ``add_env_cfg_args`` exposes ``--env-cfg.*`` by rebuilding the dataclass without
    ``Suppress``, which brings in config fields tyro cannot describe (dicts of config objects
    with no default). Those parse fine, but tyro introspects every field to render the usage
    block, so one mistyped flag reported a type error naming no flag. Collecting the unknown
    arguments instead never enters that path.
    """
    import tyro

    args, unknown = tyro.cli(args_cls, return_unknown_args=True)
    if unknown:
        print(f"Unrecognized options: {' '.join(unknown)}", file=sys.stderr)
        raise SystemExit(2)
    return args
