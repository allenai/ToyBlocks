"""Reset-state sampling for autogen env packages without TAMP.

A port of the stage-1 predicate samplers (``blocksuite.tamp.blocks.predicates.*``
``sample_vectorized`` / ``evaluate_vectorized``) onto :class:`SceneSim`, a thin view of
the Isaac Lab scene that offers the handful of simulator calls those samplers made
through ``tmp3``'s ``IsaacLabSimulatorVectorized``.
"""

from blocksuite.vecgen.sampling.predicates import SAMPLERS, PredicateSampler, make_sampler
from blocksuite.vecgen.sampling.scene import SceneSim

__all__ = ["SAMPLERS", "PredicateSampler", "SceneSim", "make_sampler"]
