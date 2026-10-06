"""Retain the teacher gap to the student's closest non-truth rival only."""
import torch
from torch.nn import functional as F


def closest_gap_loss(z, teacher_z, truth):
    assert z.shape == teacher_z.shape and z.shape[1] == 372 and not teacher_z.requires_grad
    correct = teacher_z.argmax(1) == truth
    rival = z.detach().masked_fill(F.one_hot(truth, 372).bool(), -torch.inf).argmax(1)
    target = truth[:, None]
    rival = rival[:, None]
    teacher_gap = teacher_z.gather(1, target) - teacher_z.gather(1, rival)
    student_gap = z.gather(1, target) - z.gather(1, rival)
    deficit = torch.relu(teacher_gap - student_gap).squeeze(1) * correct
    loss = deficit.square().sum() / correct.sum().clamp_min(1)
    return loss, correct, int((deficit.detach() > 0).sum())


def selftest():
    teacher = torch.full((2, 372), -10.)
    teacher[0, :3] = torch.tensor([5., 4., 3.])
    teacher[1, 1] = 5.
    truth = torch.tensor([0, 0])
    student = teacher.clone()
    student[0, 0] = 3.
    student.requires_grad_(True)
    loss, correct, active = closest_gap_loss(student, teacher, truth)
    g = torch.autograd.grad(loss, student)[0]
    assert correct.tolist() == [True, False] and active == 1 and float(loss) == 4.
    assert g[0, 0] < 0 and g[0, 1] > 0 and (g[0, 2:] == 0).all() and (g[1] == 0).all()
    shifted, _, _ = closest_gap_loss(student + 13., teacher + 7., truth)
    assert torch.allclose(loss, shifted, atol=1e-6)
    empty, mask, _ = closest_gap_loss(student, teacher, truth + 9)
    assert float(empty) == 0. and not mask.any()
    tied = torch.zeros((1, 372))
    tied_teacher = tied.clone()
    tied_teacher[0, 0] = 1.
    tied.requires_grad_(True)
    loss, _, _ = closest_gap_loss(tied, tied_teacher, torch.tensor([0]))
    g = torch.autograd.grad(loss, tied)[0]
    assert g[0, 1] > 0 and (g[0, 2:] == 0).all()
    teacher = torch.tensor([[5., 4., 0.] + [-10.] * 369])
    student = torch.tensor([[4., 3., 3.5] + [-10.] * 369], requires_grad=True)
    loss, _, _ = closest_gap_loss(student, teacher, torch.tensor([0]))
    g = torch.autograd.grad(loss, student)[0]
    assert g[0, 2] > 0 and g[0, 1] == 0, 'Student rival was not selected'
    print('PASS: closest-rival gradient direction, teacher mask, shift, empty mask, ties and student selection')


if __name__ == '__main__':
    selftest()
