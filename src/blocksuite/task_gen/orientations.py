"""Orientation data and transition tables for the blocks domain.

All 24 orientations (o1-o24) represent die configurations.
Transition tables map (from_ori -> to_ori) for each movement type.
"""

ORIENTATIONS = [f'o{i}' for i in range(1, 25)]

# (ori_id, up_face, front_face, top_align)
ORIENTATION_DEFS = [
    ('o1', 'one', 'three', 'front'),
    ('o2', 'one', 'five', 'left'),
    ('o3', 'one', 'four', 'back'),
    ('o4', 'one', 'five', 'right'),
    ('o5', 'three', 'two', 'front'),
    ('o6', 'three', 'five', 'left'),
    ('o7', 'three', 'one', 'back'),
    ('o8', 'three', 'six', 'right'),
    ('o9', 'two', 'four', 'back'),
    ('o10', 'two', 'five', 'right'),
    ('o11', 'two', 'three', 'front'),
    ('o12', 'two', 'six', 'left'),
    ('o13', 'four', 'one', 'back'),
    ('o14', 'four', 'five', 'right'),
    ('o15', 'four', 'two', 'front'),
    ('o16', 'four', 'six', 'left'),
    ('o17', 'five', 'two', 'front'),
    ('o18', 'five', 'four', 'left'),
    ('o19', 'five', 'one', 'back'),
    ('o20', 'five', 'three', 'right'),
    ('o21', 'six', 'one', 'back'),
    ('o22', 'six', 'four', 'right'),
    ('o23', 'six', 'two', 'front'),
    ('o24', 'six', 'three', 'left'),
]

FLIP_FORWARD = {
    'o1': 'o5', 'o2': 'o17', 'o3': 'o15', 'o4': 'o23',
    'o5': 'o9', 'o6': 'o18', 'o7': 'o3', 'o8': 'o22',
    'o9': 'o13', 'o10': 'o19', 'o11': 'o7', 'o12': 'o21',
    'o13': 'o1', 'o14': 'o20', 'o15': 'o11', 'o16': 'o24',
    'o17': 'o12', 'o18': 'o16', 'o19': 'o4', 'o20': 'o8',
    'o21': 'o2', 'o22': 'o14', 'o23': 'o10', 'o24': 'o6',
}
FLIP_BACKWARD = {
    'o1': 'o13', 'o2': 'o21', 'o3': 'o7', 'o4': 'o19',
    'o5': 'o1', 'o6': 'o24', 'o7': 'o11', 'o8': 'o20',
    'o9': 'o5', 'o10': 'o23', 'o11': 'o15', 'o12': 'o17',
    'o13': 'o9', 'o14': 'o22', 'o15': 'o3', 'o16': 'o18',
    'o17': 'o2', 'o18': 'o6', 'o19': 'o10', 'o20': 'o14',
    'o21': 'o12', 'o22': 'o8', 'o23': 'o4', 'o24': 'o16',
}
FLIP_LEFT = {
    'o1': 'o20', 'o2': 'o14', 'o3': 'o22', 'o4': 'o8',
    'o5': 'o17', 'o6': 'o2', 'o7': 'o21', 'o8': 'o12',
    'o9': 'o18', 'o10': 'o6', 'o11': 'o24', 'o12': 'o16',
    'o13': 'o19', 'o14': 'o10', 'o15': 'o23', 'o16': 'o4',
    'o17': 'o15', 'o18': 'o3', 'o19': 'o7', 'o20': 'o11',
    'o21': 'o13', 'o22': 'o9', 'o23': 'o5', 'o24': 'o1',
}
FLIP_RIGHT = {
    'o1': 'o24', 'o2': 'o6', 'o3': 'o18', 'o4': 'o16',
    'o5': 'o23', 'o6': 'o10', 'o7': 'o19', 'o8': 'o4',
    'o9': 'o22', 'o10': 'o14', 'o11': 'o20', 'o12': 'o8',
    'o13': 'o21', 'o14': 'o2', 'o15': 'o17', 'o16': 'o12',
    'o17': 'o5', 'o18': 'o9', 'o19': 'o13', 'o20': 'o1',
    'o21': 'o7', 'o22': 'o3', 'o23': 'o15', 'o24': 'o11',
}
FLIP_TABLES = {
    'forward': FLIP_FORWARD,
    'backward': FLIP_BACKWARD,
    'left_flip': FLIP_LEFT,
    'right_flip': FLIP_RIGHT,
}

YAW_CW = {
    'o1': 'o2', 'o2': 'o3', 'o3': 'o4', 'o4': 'o1',
    'o5': 'o6', 'o6': 'o7', 'o7': 'o8', 'o8': 'o5',
    'o9': 'o10', 'o10': 'o11', 'o11': 'o12', 'o12': 'o9',
    'o13': 'o14', 'o14': 'o15', 'o15': 'o16', 'o16': 'o13',
    'o17': 'o18', 'o18': 'o19', 'o19': 'o20', 'o20': 'o17',
    'o21': 'o22', 'o22': 'o23', 'o23': 'o24', 'o24': 'o21',
}
YAW_CCW = {v: k for k, v in YAW_CW.items()}


def flip(ori: str, direction: str) -> str:
    """Apply a flip transition. direction in {forward, backward, left_flip, right_flip}."""
    return FLIP_TABLES[direction][ori]


def yaw(ori: str, direction: str) -> str:
    """Apply a yaw rotation. direction in {cw, ccw}."""
    if direction == "cw":
        return YAW_CW[ori]
    elif direction == "ccw":
        return YAW_CCW[ori]
    raise ValueError(f"Unknown yaw direction: {direction}")


def static_physics_predicates():
    """Return static physics init facts as symbolic predicates."""
    from blocksuite.task_gen.predicates import Pred

    preds = []
    for ori, up, front, align in ORIENTATION_DEFS:
        preds.append(Pred("up_face", ori, up))
        preds.append(Pred("front_face", ori, front))
        preds.append(Pred("top_align", ori, align))

    for from_ori, to_ori in sorted(YAW_CW.items(), key=lambda x: int(x[0][1:])):
        preds.append(Pred("yaw_next", "cw", from_ori, to_ori))
    for from_ori, to_ori in sorted(YAW_CCW.items(), key=lambda x: int(x[0][1:])):
        preds.append(Pred("yaw_next", "ccw", from_ori, to_ori))

    for direction, table in FLIP_TABLES.items():
        for from_ori, to_ori in sorted(table.items(), key=lambda x: int(x[0][1:])):
            preds.append(Pred("flip_next", direction, from_ori, to_ori))

    return preds


def generate_static_physics() -> str:
    """Render static physics predicates as a PDDL init block (legacy helper)."""
    from blocksuite.task_gen.predicates import And

    return And(*static_physics_predicates()).render(indent=8)
