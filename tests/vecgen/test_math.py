import math

import torch

from toyblocks.vecgen.math import (
    pose_compose,
    pose_inverse,
    quat_angle,
    quat_apply,
    quat_from_axis_angle,
    quat_from_euler_xyz,
    quat_inverse,
    quat_mul,
    quat_yaw,
    quat_z_axis,
    yaw_of,
)


def test_quat_basic_identities():
    ax = torch.tensor([[0.0, 0.0, 1.0]])
    q = quat_from_axis_angle(ax, torch.tensor([math.pi / 2]))
    v = torch.tensor([[1.0, 0.0, 0.0]])
    r = quat_apply(q, v)
    assert torch.allclose(r, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6)
    assert torch.allclose(quat_mul(q, quat_inverse(q)), torch.tensor([[1.0, 0, 0, 0]]), atol=1e-6)
    assert torch.allclose(quat_angle(q), torch.tensor([math.pi / 2]), atol=1e-6)
    assert torch.allclose(quat_yaw(torch.tensor([math.pi / 2])), q, atol=1e-6)
    assert torch.allclose(yaw_of(q), torch.tensor([math.pi / 2]), atol=1e-6)


def test_euler_top_down_grasp_points_down():
    q = quat_from_euler_xyz(torch.zeros(1), torch.full((1,), math.pi), torch.zeros(1))
    z = quat_z_axis(q)
    assert torch.allclose(z, torch.tensor([[0.0, 0.0, -1.0]]), atol=1e-6)


def test_pose_compose_inverse():
    a = torch.tensor([[0.1, 0.2, 0.3, *quat_yaw(torch.tensor([0.7])).tolist()[0]]])
    b = torch.tensor([[0.5, -0.1, 0.0, *quat_yaw(torch.tensor([-0.2])).tolist()[0]]])
    ab = pose_compose(a, b)
    back = pose_compose(pose_inverse(a), ab)
    assert torch.allclose(back, b, atol=1e-6)
