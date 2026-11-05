"""
Code to launch the Isaac App. Must be run at the top of every scrip that uses the simulator or imports anything isaac related
"""
import argparse
import sys
from isaaclab.app import AppLauncher

def launch_app():
    # Import the venv's Warp before the Omniverse app starts: Isaac Sim's omni.warp.core
    # extension otherwise puts its bundled (older) Warp first on sys.path, and cuRobo
    # needs the Warp version pinned in pyproject.
    import warp  # noqa: F401

    parser = argparse.ArgumentParser(description="Launch the Isaac App.")
    AppLauncher.add_app_launcher_args(parser)
    parser._option_string_actions.pop("-h")
    parser._option_string_actions.pop("--help")
    args, _ = parser.parse_known_args()
    # Remove any instance of '-h' and '--help' from sys.argv to avoid triggering the default isaac help message
    original_argv = sys.argv
    sys.argv = [arg for arg in sys.argv if arg not in ("-h", "--help")]
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    sys.argv = original_argv
    return simulation_app