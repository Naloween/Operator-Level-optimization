"""MNIST-1D: a 40-dimensional procedurally generated classification benchmark.

Greydanus & Kobak's MNIST-1D (https://github.com/greydanus/mnist1d). Ten classes built from
hand-drawn 1D templates, then randomly translated, sheared, correlated-noised and downsampled
to 40 points. It is the right benchmark for this project for three reasons:

* **40 inputs, not 784.** The Jacobian `J(x)` is `10 x 40`, so its whole spectrum is
  meaningful and cheap, and the per-sample operator machinery is affordable at every step.
* **No degenerate coordinates.** MNIST's border pixels are constant, which makes a
  rectangular-identity first layer see nothing at all and a ReLU network die outright — a
  fact about the pixel ordering that is easy to mistake for a fact about the architecture.
  MNIST-1D has no such coordinates.
* **Procedurally generated**, so it needs no download and is reproducible from a seed.

Generation costs a few seconds, so a process-level cache keyed by `(num_samples, seed)` keeps
a sweep from paying it once per run.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from olo.tasks.base import Task

_CACHE: dict[tuple[int, int], tuple] = {}


def _generate(num_samples: int, seed: int):
    key = (num_samples, seed)
    if key not in _CACHE:
        from mnist1d.data import get_dataset_args, make_dataset

        args = get_dataset_args()
        args.num_samples = int(num_samples)
        args.seed = int(seed)
        d = make_dataset(args)
        _CACHE[key] = (torch.as_tensor(d["x"]).float(), torch.as_tensor(d["y"]).long(),
                       torch.as_tensor(d["x_test"]).float(), torch.as_tensor(d["y_test"]).long())
    return _CACHE[key]


class MNIST1D(Task):
    n_classes = 10
    stall_metric = ("val_accuracy", "max")

    def __init__(self, n_train: int | None = None, n_val: int = 500,
                 n_test: int | None = None, batch_size: int = 128,
                 num_samples: int = 5000, seed: int = 0) -> None:
        X, y, Xte, yte = _generate(num_samples, seed)

        # Validation is carved out of the *train* split; the official test split is touched
        # only by test(), so nothing in the training loop can select on it.
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(X.shape[0], generator=g)
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        if n_train is not None:
            tr_idx = tr_idx[:n_train]
        self.Xtr, self.ytr = X[tr_idx], y[tr_idx]
        self.Xva, self.yva = X[val_idx], y[val_idx]
        self.Xte, self.yte = (Xte, yte) if n_test is None else (Xte[:n_test], yte[:n_test])

        self.d_in = int(self.Xtr.shape[1])
        self.d_out = self.n_classes
        self.batch_size = int(batch_size)

        mu, sd = self.Xtr.mean(), self.Xtr.std().clamp(min=1e-6)   # scalar, as for the images
        self.Xtr, self.Xva, self.Xte = ((t - mu) / sd for t in (self.Xtr, self.Xva, self.Xte))
        self._perm = torch.randperm(self.Xtr.shape[0], generator=g)

    def to(self, device, dtype):
        for n in ("Xtr", "Xva", "Xte"):
            setattr(self, n, getattr(self, n).to(device=device, dtype=dtype))
        for n in ("ytr", "yva", "yte"):
            setattr(self, n, getattr(self, n).to(device=device))
        return self

    def train_batch(self, step, batch_size=None):
        bs = batch_size or self.batch_size
        n = self.Xtr.shape[0]
        idx = self._perm[torch.arange((step * bs) % n, (step * bs) % n + bs) % n]
        return self.Xtr[idx], self.ytr[idx]

    def loss(self, yhat, y):
        return F.cross_entropy(yhat, y)

    def evaluate(self, net) -> dict[str, float]:
        loss, acc = self._score(net, self.Xva, self.yva)
        return {"primary": loss, "val_loss": loss, "val_accuracy": acc}

    def test(self, net) -> dict[str, float]:
        loss, acc = self._score(net, self.Xte, self.yte)
        tr_loss, tr_acc = self._score(net, self.Xtr, self.ytr)
        return {"primary": loss, "test_loss": loss, "test_accuracy": acc,
                "train_loss": tr_loss, "train_accuracy": tr_acc}

    @torch.no_grad()
    def _score(self, net, X, y) -> tuple[float, float]:
        logits = net(X)
        return (float(F.cross_entropy(logits, y)),
                float((logits.argmax(-1) == y).double().mean()))

    def describe(self):
        return {**super().describe(), "dataset": "MNIST1D",
                "n_train": int(self.Xtr.shape[0]), "n_val": int(self.Xva.shape[0]),
                "n_test": int(self.Xte.shape[0]), "batch_size": self.batch_size}
