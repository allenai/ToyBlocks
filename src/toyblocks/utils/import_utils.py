from pathlib import Path
import sys
from typing import overload

@overload
def resolve_module_from_path(path: str) -> str | None: ...
@overload
def resolve_module_from_path(path: Path) -> str | None: ...
def resolve_module_from_path(path: str | Path) -> str | None:
    """
    Resolve a module from a filepath (e.g. src/toyblocks/tamp/blocks/ -> toyblocks.tamp.blocks).

    Finds the root module from the filepath and returns the module path string that can be imported via importlib.import_module.

    Returns None if no module is found.
    """
    if isinstance(path, str):
        path = Path(path)
    module_path_parts = path.parts
    for i, module in enumerate(module_path_parts):
        if module in sys.modules:
            return ".".join(module_path_parts[i:])
    return None