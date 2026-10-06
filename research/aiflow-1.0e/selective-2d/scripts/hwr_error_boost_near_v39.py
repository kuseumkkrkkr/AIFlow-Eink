"""Preserve every near pair; double only teacher-right/student-wrong rows."""
import numpy as np
import torch

from cloud_hwr_snapshot import configure

ERROR_BOOST = 2.0


def error_boost_gap(z, teacher_z, truth):
    near, correct = configure().near_mask(z, teacher_z, truth)
    focused = correct & (z.detach().argmax(1) != truth)
    weights = 1 + (ERROR_BOOST - 1) * focused.to(z.dtype)
    deficit = torch.relu((teacher_z.gather(1, truth[:, None]) - teacher_z) -
                         (z.gather(1, truth[:, None]) - z)) * near
    loss = (deficit.square() * weights[:, None]).sum() / near.sum().clamp_min(1)
    return loss, correct, int((deficit.detach() > 0).sum())


def selftest():
    run = configure()
    teacher = torch.full((2, 372), -10.)
    teacher[0, :5] = torch.tensor([5., 4., 3., 2., 1.])
    teacher[1, 1] = 5.
    truth = torch.tensor([0, 0])
    student = teacher.clone()
    student[0, 0] = 3.
    student.requires_grad_(True)
    baseline = run.near_gap_loss(student, teacher, truth)[0]
    value, correct, _ = error_boost_gap(student, teacher, truth)
    actual_gradient = torch.autograd.grad(value, student, retain_graph=True)[0]
    baseline_gradient = torch.autograd.grad(baseline, student)[0]
    assert correct.tolist() == [True, False] and torch.equal(value, ERROR_BOOST * baseline)
    assert torch.equal(actual_gradient, ERROR_BOOST * baseline_gradient)
    assert (actual_gradient[1] == 0).all() and actual_gradient[0, 0] < 0
    assert torch.allclose(value, error_boost_gap(student + 13, teacher + 7, truth)[0], atol=1e-6)
    almost = teacher.clone(); almost[0, 0] = 4.5
    assert torch.equal(error_boost_gap(almost, teacher, truth)[0], run.near_gap_loss(almost, teacher, truth)[0])
    empty, mask, _ = error_boost_gap(student, teacher, truth + 9)
    assert float(empty) == 0. and not mask.any()
    # Independent full NumPy pair mask, loss and derivative calculation.
    rng = np.random.default_rng(39)
    t, s = [rng.normal(size=(8, 372)).astype(np.float32) for _ in range(2)]
    y = t.argmax(1); y[::3] = (y[::3] + 1) % 372
    s[1, y[1]] = 10.
    eligible = t.argmax(1) == y
    near = np.zeros_like(s, dtype=bool)
    for row in range(8):
        near[row, np.argsort(-t[row], kind='stable')[:5]] = True
        near[row, np.argsort(-s[row], kind='stable')[:5]] = True
        near[row, y[row]] = False
    near &= eligible[:, None]
    focus = eligible & (s.argmax(1) != y)
    weights = 1 + (ERROR_BOOST - 1) * focus
    delta = np.maximum((t[np.arange(8), y, None] - t) - (s[np.arange(8), y, None] - s), 0) * near
    denominator = max(int(near.sum()), 1)
    expected = (delta**2 * weights[:, None]).sum() / denominator
    gradient = 2 * delta * weights[:, None] / denominator
    gradient[np.arange(8), y] -= gradient.sum(1)
    sz = torch.tensor(s, requires_grad=True)
    loss, mask, _ = error_boost_gap(sz, torch.tensor(t), torch.tensor(y))
    assert np.isclose(float(loss.detach()), expected, atol=1e-6)
    assert np.array_equal(mask.numpy(), eligible)
    assert np.allclose(torch.autograd.grad(loss, sz)[0].numpy(), gradient, atol=1e-6)
    print('PASS: unchanged pair coverage/denominator, detached error boost, teacher mask, correct-row equality, shift, empty mask and independent NumPy gradients', flush=True)


if __name__ == '__main__':
    selftest()
