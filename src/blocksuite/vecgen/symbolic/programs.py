"""Turn an FF plan over ``domain.pddl`` into a skill program.

Domain actions and their positional parameters (see ``src/blocksuite/tamp/blocks/domain.pddl``):

    pnp_table_table  g b1 t o o2      -> pick(b1) place_table(b1 o2)
    pnp_table_block  g b1 b2 t o o2   -> pick(b1) place_on(b1 b2 o2)
    pnp_block_table  g b1 b2 t o o2   -> pick(b1) place_table(b1 o2)
    pnp_block_block  g b1 b2 b3 o o2  -> pick(b1) place_on(b1 b3 o2)
    reorient         g b d o o2 t     -> flip(b d o o2)
    rotate           g b d o o2 t     -> rotate(b d o o2)
    move_to_side     g b t            -> pick(b) place_side(b)
    move_to_center   g b t            -> pick(b) place_table(b <current ori>)
    init / move_to / done / stop      -> ignored
"""

from __future__ import annotations

from blocksuite.vecgen.program import Program, SkillCall
from blocksuite.vecgen.symbolic.pddl import Fact

IGNORED = {"init", "move_to", "done", "reach-goal", "stop"}
UNSUPPORTED = {"push_to_right", "push_next_to", "isolate"}


class UnsupportedAction(ValueError):
    pass


def skills_for_action(action: Fact) -> list[SkillCall]:
    name, a = action.pred, action.args
    if name in IGNORED:
        return []
    if name in UNSUPPORTED:
        raise UnsupportedAction(name)
    if name == "pnp_table_table":
        _, b1, _t, _o, o2 = a
        return [SkillCall("pick", (b1,)), SkillCall("place_table", (b1, o2))]
    if name == "pnp_table_block":
        _, b1, b2, _t, _o, o2 = a
        return [SkillCall("pick", (b1,)), SkillCall("place_on", (b1, b2, o2))]
    if name == "pnp_block_table":
        _, b1, _b2, _t, _o, o2 = a
        return [SkillCall("pick", (b1,)), SkillCall("place_table", (b1, o2))]
    if name == "pnp_block_block":
        _, b1, _b2, b3, _o, o2 = a
        return [SkillCall("pick", (b1,)), SkillCall("place_on", (b1, b3, o2))]
    if name == "reorient":
        _, b, d, o, o2, _t = a
        return [SkillCall("flip", (b, d, o, o2))]
    if name == "rotate":
        _, b, d, o, o2, _t = a
        return [SkillCall("rotate", (b, d, o, o2))]
    if name == "move_to_side":
        _, b, _t = a
        return [SkillCall("pick", (b,)), SkillCall("place_side", (b, ""))]
    if name == "move_to_center":
        _, b, _t = a
        return [SkillCall("pick", (b,)), SkillCall("place_table", (b, ""))]
    raise UnsupportedAction(name)


def program_from_plan(
    actions: list[Fact],
    problem_id: str,
    reset_state_id: int,
    states: list[tuple[str, ...]] | None = None,
) -> Program:
    """``states[i]`` (FF's state before ``actions[i]``, see ``FFResult.states``) rides on
    the skill calls of action ``i`` as ``hl_state``."""
    skills: list[SkillCall] = []
    for i, act in enumerate(actions):
        hl_state = states[i] if states and i < len(states) else ()
        for sc in skills_for_action(act):
            skills.append(
                SkillCall(sc.name, sc.args, hl_name=act.pred, hl_args=act.args, hl_state=hl_state)
            )
    return Program(skills=skills, problem_id=problem_id, reset_state_id=reset_state_id)
