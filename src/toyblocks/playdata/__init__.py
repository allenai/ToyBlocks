"""Play-data collection: goal policies for the vecgen play engine (``scripts/vecgen/run.py --play``).

Each environment plays a chain of goals from *whatever state it is currently in*: a goal policy
(random goals, the balanced pnp / flip / rotate mix, or a DFA strategy from :mod:`.dfa`) picks
the next goal from the symbolic state, FF plans it and vecgen executes it as one recorded
episode, with no reset between goals (:mod:`toyblocks.vecgen.play`). This package holds the
TAMP-free pieces: goal policies, the abandon follow-up, domain filtering, the symbolic state.
"""
