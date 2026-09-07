"""Teacher-student operator recovery: y = P* x + noise, with P* known.

The controlled setting the theory is stated in. Because the target operator is known and
the student's operator is directly computable, "did the method move P where it was asked
to?" is answered by a number rather than inferred from a loss curve -- which is what lets
depth, conditioning, and update mismatch be separated at all.

The target's spectrum is the experimental knob: an orthogonal P* asks the student to
realize a perfectly conditioned operator, so any spectral anisotropy that appears during
training was introduced by the optimizer, not requested by the problem.
"""
from __future__ import annotations

import torch

from olo.tasks.base import Task, mse, relative_operator_error


class TeacherStudent(Task):
    def __init__(
        self,
        d: int,
        n: int = 64,
        d_out: int | None = None,
        target: str = "orthogonal",
        rank: int | None = None,
        noise: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.d_in = int(d)
        self.d_out = int(d if d_out is None else d_out)
        self.n = int(n)
        self.target_kind = target
        self.noise = float(noise)

        gen = torch.Generator().manual_seed(int(seed))
        self.P_star = _make_target(self.d_out, self.d_in, target, rank, gen)
        self.X = torch.randn(self.n, self.d_in, generator=gen)
        self.Y = self.X @ self.P_star.T
        if noise > 0:
            self.Y = self.Y + noise * torch.randn(self.Y.shape, generator=gen)

    def to(self, device, dtype):
        self.X = self.X.to(device=device, dtype=dtype)
        self.Y = self.Y.to(device=device, dtype=dtype)
        self.P_star = self.P_star.to(device=device, dtype=dtype)
        return self

    def train_batch(self, step, batch_size=None):
        if batch_size is None or batch_size >= self.n:
            return self.X, self.Y
        # Deterministic cycling rather than random sampling: two methods compared at the
        # same step must see the same data, or the comparison measures the draw.
        start = (step * batch_size) % self.n
        idx = torch.arange(start, start + batch_size) % self.n
        return self.X[idx], self.Y[idx]

    def loss(self, yhat, y):
        return mse(yhat, y)

    def evaluate(self, net) -> dict[str, float]:
        """Both numbers always; `primary` is whichever the model makes meaningful.

        Recovering P* is the right question only when the student *has* a single operator.
        A rectifier network's Jacobian is constrained by the data solely along each
        sample's own direction, so J(x) can stay far from P* while the network fits the
        teacher exactly -- reading such a run on operator error would report failure for a
        model that has solved the task.
        """
        with torch.no_grad():
            err = relative_operator_error(net, self.X, self.P_star)
            loss = float(self.loss(net(self.X), self.Y))
        primary = err if net.input_independent else loss
        return {"primary": primary, "rel_operator_error": err, "loss": loss}

    def describe(self):
        return {
            **super().describe(),
            "n": self.n,
            "target": self.target_kind,
            "noise": self.noise,
        }


def _make_target(d_out, d_in, kind, rank, gen) -> torch.Tensor:
    if kind == "gaussian":
        return torch.randn(d_out, d_in, generator=gen) / (d_in ** 0.5)
    if kind in ("orthogonal", "isotropic", "low_rank"):
        a = torch.randn(d_out, d_in, generator=gen)
        u, _, vh = torch.linalg.svd(a, full_matrices=False)
        if kind == "low_rank":
            r = int(rank if rank is not None else max(1, min(d_out, d_in) // 4))
            s = torch.zeros(min(d_out, d_in))
            s[:r] = 1.0
            return u @ torch.diag(s) @ vh
        return u @ vh
    raise ValueError(
        f"unknown teacher-student target {kind!r}; known: orthogonal, gaussian, low_rank"
    )
