"""Underdetermined matrix sensing: y_s = <M_s, P*>, with m < d^2 measurements.

The implicit-bias experiment. In dense teacher-student regression with enough full-rank
samples the observations pin the operator down uniquely, so there is nothing for an
implicit bias to select and the question "is correcting the factorization bias desirable?"
cannot even be posed. Here many operators interpolate the data exactly, so which one a
method reaches is entirely a property of its trajectory.

That makes this the setting where operator projection earns its keep as a *control*: run
the same three-factor model under (a) a method applied to the factors, (b) the same method
applied directly to P, and (c) projection, which uses (a)'s parameterization while
realizing (b)'s trajectory. The difference between (a) and (c) is then the factorization
bias itself, isolated -- with representation and initialization held fixed.

Configure (b) and (c) through the target: `optim.als.target: {type: shadow, inner: adam}`.
"""
from __future__ import annotations

import torch

from olo.tasks.base import Task, relative_operator_error


class MatrixSensing(Task):
    def __init__(
        self,
        d: int = 20,
        m: int = 160,
        rank: int = 2,
        n_test: int = 400,
        seed: int = 0,
    ) -> None:
        self.d_in = int(d)
        self.d_out = int(d)
        self.m = int(m)
        self.rank = int(rank)
        if m >= d * d:
            raise ValueError(
                f"m={m} >= d^2={d*d}: the measurements determine the operator uniquely, "
                "so there is no implicit bias left to measure. Use fewer measurements."
            )

        gen = torch.Generator().manual_seed(int(seed))
        u = torch.linalg.qr(torch.randn(d, rank, generator=gen))[0]
        v = torch.linalg.qr(torch.randn(d, rank, generator=gen))[0]
        self.P_star = u @ v.T

        self.M = torch.randn(self.m, d, d, generator=gen) / d**0.5
        self.y = torch.einsum("sij,ij->s", self.M, self.P_star)
        self.M_val = torch.randn(n_test, d, d, generator=gen) / d**0.5
        self.y_val = torch.einsum("sij,ij->s", self.M_val, self.P_star)
        self.M_test = torch.randn(n_test, d, d, generator=gen) / d**0.5
        self.y_test = torch.einsum("sij,ij->s", self.M_test, self.P_star)

    def to(self, device, dtype):
        for name in ("P_star", "M", "y", "M_val", "y_val", "M_test", "y_test"):
            setattr(self, name, getattr(self, name).to(device=device, dtype=dtype))
        return self

    # The "batch" is the measurement basis: each sensing matrix acts on the operator, so
    # the network is probed by feeding it the rows of M_s and reading the trace.
    def train_batch(self, step, batch_size=None):
        if batch_size is None or batch_size >= self.m:
            return self.M, self.y
        start = (step * batch_size) % self.m
        idx = torch.arange(start, start + batch_size) % self.m
        return self.M[idx], self.y[idx]

    def forward(self, net, x):
        """x is a batch of sensing matrices, so the network is probed as a whole operator."""
        return self.apply_operator(net.operator(self._probe(net)), x)

    def apply_operator(self, P, x):
        if P.dim() == 3:
            P = P.mean(dim=0)
        return torch.einsum("sij,ij->s", x, P)

    def _probe(self, net) -> torch.Tensor:
        eye = torch.eye(net.d_in, device=net.weights[0].device, dtype=net.weights[0].dtype)
        return eye

    def operator_gradient(self, net, x, y):
        P = net.operator(self._probe(net)).detach().mean(dim=0).requires_grad_(True)
        loss = self.loss(self.apply_operator(P, x), y)
        (G,) = torch.autograd.grad(loss, P)
        return G.detach().unsqueeze(0)

    def loss(self, yhat, y):
        return 0.5 * ((yhat - y) ** 2).mean()

    def evaluate(self, net) -> dict[str, float]:
        return self._metrics(net, self.M_val, self.y_val, "val")

    def test(self, net) -> dict[str, float]:
        return self._metrics(net, self.M_test, self.y_test, "test")

    def _metrics(self, net, M, y, split: str) -> dict[str, float]:
        with torch.no_grad():
            P = _operator_of(net)
            train = float(0.5 * ((torch.einsum("sij,ij->s", self.M, P) - self.y) ** 2).mean())
            held = float(0.5 * ((torch.einsum("sij,ij->s", M, P) - y) ** 2).mean())
            sv = torch.linalg.svdvals(P.to(torch.float64))
            p = sv / sv.sum().clamp(min=1e-300)
            eff_rank = float((-(p * p.clamp(min=1e-300).log()).sum()).exp())
            rel = float((P - self.P_star).norm() / self.P_star.norm())
        return {
            "primary": held,
            "train_mse": train,
            "held_out_mse": held,
            f"{split}_mse": held,
            "effective_rank": eff_rank,
            "rel_recovery_error": rel,
        }

    def describe(self):
        return {**super().describe(), "m": self.m, "rank": self.rank}


def _operator_of(net) -> torch.Tensor:
    """The composed operator, probed with the identity so it works for any model here."""
    eye = torch.eye(net.d_in, device=net.weights[0].device, dtype=net.weights[0].dtype)
    return net.operator(eye).mean(dim=0)
