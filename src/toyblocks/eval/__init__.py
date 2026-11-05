"""Unified evaluation logic for ToyBlocks Isaac Lab environments."""


def evaluate(*args, **kwargs):
    """Lazy import so Isaac-free submodules (e.g. ``pd_mode_controller``) load without Isaac Sim."""
    from toyblocks.eval.run_eval import evaluate as _evaluate

    return _evaluate(*args, **kwargs)
