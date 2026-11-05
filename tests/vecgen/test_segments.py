import torch

from toyblocks.vecgen.program import EnvProgramState, Phase, Program, SkillCall
from toyblocks.vecgen.segments import Segment, SegmentTable, interpolate_rows


def test_segment_table_gather_and_hold():
    dev = torch.device("cpu")
    table = SegmentTable(num_envs=3, dof=2, capacity=8, device=dev)
    seg_a = Segment(targets=torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]), hold=2, label="a")
    seg_b = Segment(targets=torch.tensor([[9.0, 9.0]]), hold=0, label="b")
    table.install([0, 2], [seg_a, seg_b])
    assert table.active.tolist() == [True, False, True]
    assert table.labels == ["a", "", "b"]

    targets, _v, _a, _c, finished = table.gather()  # tick 1
    assert targets[0].tolist() == [1.0, 1.0] and targets[2].tolist() == [9.0, 9.0]
    assert finished.tolist() == [False, False, True]  # single-row, no hold: done at once
    assert table.active.tolist() == [True, False, False]

    targets, _v, _a, _c, finished = table.gather()  # tick 2
    assert targets[0].tolist() == [2.0, 2.0] and finished[0].item() is False
    targets, _v, _a, _c, finished = table.gather()  # tick 3: last row
    assert targets[0].tolist() == [3.0, 3.0] and finished[0].item() is False
    targets, _v, _a, _c, finished = table.gather()  # tick 4: hold 1 (last row repeated)
    assert targets[0].tolist() == [3.0, 3.0] and finished[0].item() is False
    targets, _v, _a, _c, finished = table.gather()  # tick 5: hold 2 -> finished
    assert targets[0].tolist() == [3.0, 3.0] and finished[0].item() is True
    assert not table.active.any()


def test_segment_table_cancel_and_capacity():
    table = SegmentTable(num_envs=1, dof=1, capacity=2, device=torch.device("cpu"))
    table.install([0], [Segment(targets=torch.zeros(2, 1))])
    table.cancel([0])
    assert not table.active[0]
    try:
        table.install([0], [Segment(targets=torch.zeros(3, 1))])
    except ValueError:
        pass
    else:
        raise AssertionError("capacity overflow not rejected")


def test_interpolate_rows():
    start = torch.zeros(2, 3)
    end = torch.ones(2, 3)
    rows = interpolate_rows(start, end, 5)
    assert rows.shape == (2, 5, 3)
    assert torch.allclose(rows[:, 0], start) and torch.allclose(rows[:, -1], end)
    assert torch.allclose(rows[:, 2], torch.full((2, 3), 0.5))


def test_program_state_lifecycle():
    prog = Program([SkillCall("pick", ("b1",)), SkillCall("place", ("b1", "b2"))], "p", 0)
    st = EnvProgramState(env_id=0)
    st.assign(prog, episode_id=7)
    assert st.phase is Phase.READY and str(st.current) == "pick(b1)"
    assert st.retry(max_attempts=2, reason="miss") is True and st.attempts == 1
    assert st.retry(max_attempts=2, reason="miss") is False and st.phase is Phase.FAILURE
    st.assign(prog, episode_id=8)
    st.advance()
    assert str(st.current) == "place(b1, b2)"
    st.advance()
    assert st.phase is Phase.SUCCESS and st.current is None


def test_settle_tol_ends_hold_once_converged():
    """A ``settle_tol`` segment holds its last row until the measured joints reach it."""
    dev = torch.device("cpu")
    table = SegmentTable(num_envs=2, dof=2, capacity=8, device=dev)
    target = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    table.install(
        [0, 1],
        [
            Segment(targets=target.clone(), hold=10, settle_tol=0.05),
            Segment(targets=target.clone(), hold=10),  # no tolerance: fixed hold
        ],
    )
    assert table.needs_joint_pos()
    lagging = torch.tensor([[0.0, 0.5], [0.0, 0.5]])
    # rows are not cut short even if already converged; only the hold is gated
    _p, _v, _a, _c, fin = table.gather(torch.tensor([[0.0, 1.0], [0.0, 1.0]]))
    assert fin.tolist() == [False, False]
    _p, _v, _a, _c, fin = table.gather(lagging)  # last row, still lagging
    assert fin.tolist() == [False, False]
    _p, _v, _a, _c, fin = table.gather(lagging)  # hold 1, lagging -> keep holding
    assert fin.tolist() == [False, False]
    _p, _v, _a, _c, fin = table.gather(torch.tensor([[0.0, 0.97], [0.0, 0.97]]))  # within 0.05
    assert fin.tolist() == [True, False]
    assert table.active.tolist() == [False, True]
    assert not table.needs_joint_pos()


def test_settle_tol_hold_is_a_cap():
    dev = torch.device("cpu")
    table = SegmentTable(num_envs=1, dof=1, capacity=4, device=dev)
    table.install([0], [Segment(targets=torch.tensor([[1.0]]), hold=2, settle_tol=0.01)])
    far = torch.tensor([[0.0]])
    fins = [table.gather(far)[4].item() for _ in range(3)]
    assert fins == [False, False, True]  # never converged: ends after the 2 hold ticks


def test_settle_tol_hold_ends_when_error_stops_shrinking():
    """A joint parked in its deadband short of the tolerance must not burn the whole hold cap."""
    from toyblocks.vecgen.segments import SETTLE_STALL_TICKS

    dev = torch.device("cpu")
    table = SegmentTable(num_envs=1, dof=1, capacity=4, device=dev)
    table.install([0], [Segment(targets=torch.tensor([[1.0]]), hold=100, settle_tol=0.01)])
    stuck = torch.tensor([[0.87]])  # 0.13 rad short, never improving
    ticks = 0
    while True:
        ticks += 1
        if table.gather(stuck)[4].item():
            break
    assert ticks == 1 + 1 + SETTLE_STALL_TICKS  # last row, first hold tick (best), then stall
