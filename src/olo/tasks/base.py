"""The task interface: data, loss, and what "doing well" means for this problem.

`batch_mean` matters more than it looks. Every task here reports a loss averaged over the
batch, and ALS has to undo that averaging to recover each sample's own operator target
(see `olo.optim.base.output_gradient`). A task whose loss is a sum, not a mean, must say
so, or its operator steps would be off by a factor of the batch size.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Task(ABC):
    #: loss() averages over the batch (so per-sample gradients need rescaling by B)
    batch_mean: bool = True

    d_in: int
    d_out: int

    @abstractmethod
    def train_batch(self, step: int, batch_size: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch for the given step; full batch when `batch_size` is None."""

    @abstractmethod
    def loss(self, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Scalar training loss."""

    @abstractmethod
    def evaluate(self, net) -> dict[str, float]:
        """Validation metrics. Must include 'primary': what training is steered by.

        Called during training, so it decides early stopping and model selection. It must
        therefore never touch the test split -- selecting on test and then reporting test
        is how a comparison quietly becomes meaningless.
        """

    def test(self, net) -> dict[str, float]:
        """Test metrics, for after training only.

        Kept separate from `evaluate` on purpose: nothing in the training loop calls this,
        so no run can tune against it. `olo.evaluate` runs it on saved checkpoints.
        """
        return {}

    #: target operator when the task defines one (teacher-student, matrix sensing)
    P_star: torch.Tensor | None = None

    # -- how the loss sees the operator -------------------------------------

    def forward(self, net, x: torch.Tensor) -> torch.Tensor:
        """Predictions for this batch. Override when the input is not a feature vector."""
        return net(x)

    def apply_operator(self, P: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Predictions as a function of the operator alone, holding the gates fixed.

        Every operator-space method needs dL/dP, and the honest way to get it is to write
        the predictions in terms of P and differentiate. Regression reads J(x) x; matrix
        sensing reads <M_s, P>. Defining this one function per task is what lets the
        solver, the target, and the mismatch diagnostic share a single definition of the
        operator gradient instead of each re-deriving it.
        """
        return torch.einsum("bij,bj->bi", P.expand(x.shape[0], *P.shape[1:]), x)

    def operator_gradient(self, net, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """dL/dP at the current weights, per sample where the operator is per-sample.

        The batch averaging in `loss` is undone so that each sample's operator target is
        its own gradient, not a B-th of it -- otherwise operator steps would silently
        shrink as the batch grows.
        """
        with torch.no_grad():
            P0 = net.operator(x)
        P = P0.detach().requires_grad_(True)
        loss = self.loss(self.apply_operator(P, x), y)
        (G,) = torch.autograd.grad(loss, P)
        scale = x.shape[0] if (self.batch_mean and not net.input_independent) else 1.0
        return G.detach() * scale

    def to(self, device, dtype):
        return self

    def describe(self) -> dict[str, object]:
        return {"type": type(self).__name__, "d_in": self.d_in, "d_out": self.d_out}


def mse(yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Half squared error, summed over outputs and averaged over the batch.

    The half makes the per-sample output gradient exactly the residual (yhat - y), so the
    operator gradient is the clean rank-one (yhat_b - y_b) x_b^T with no stray factor of
    two propagating into every target and learning rate in the repo.
    """
    return 0.5 * ((yhat - y) ** 2).sum(dim=-1).mean()


def relative_operator_error(net, x: torch.Tensor, P_star: torch.Tensor) -> float:
    """||P - P*||_F / ||P*||_F, averaged over samples when the operator is per-sample."""
    with torch.no_grad():
        P = net.operator(x)
        star = P_star.to(device=P.device, dtype=P.dtype).unsqueeze(0)
        return float((P - star).flatten(1).norm(dim=1).mean() / star.norm())
