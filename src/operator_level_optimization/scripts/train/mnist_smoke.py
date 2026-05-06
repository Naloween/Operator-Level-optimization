from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from operator_level_optimization.core.optim.kfac import KFAC
from operator_level_optimization.core.optim.muon import Muon
from operator_level_optimization.core.optim.operator import OperatorLevelMLP
from operator_level_optimization.core.optim.shampoo import Shampoo
from operator_level_optimization.core.optim.soap import SOAP


class MLPForMNIST(nn.Module):
    def __init__(self, hidden_dim: int = 64, depth: int = 2, bias: bool = False):
        super().__init__()
        dims = [784] + [hidden_dim] * depth + [10]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=bias))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))


def make_loader(data_root: Path, batch_size: int, subset: int) -> torch.utils.data.DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    ds = torchvision.datasets.MNIST(root=str(data_root), train=True, download=True, transform=transform)
    if subset > 0:
        ds = torch.utils.data.Subset(ds, list(range(min(subset, len(ds)))))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)


def make_optimizer(name: str, model: nn.Module, lr: float, lam: float, sweeps: int):
    params = list(model.parameters())
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "heavyball":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "muon":
        return Muon(params, lr=lr)
    if name == "kfac":
        return KFAC(params, lr=lr, factor_decay=0.95, damping=0.1, momentum=0.0, weight_decay=0.0, inv_floor=1e-2)
    if name == "shampoo":
        return Shampoo(params, lr=lr, beta=0.9, momentum=0.0, weight_decay=0.0, eps=1e-4)
    if name == "soap":
        return SOAP(params, lr=lr, weight_decay=0.0, correct_bias=True)
    if name == "als_exact":
        return OperatorLevelMLP(
            params,
            lr=lr,
            lam=lam,
            approximation="als",
            n_sweeps=sweeps,
            als_layer_solve="exact",
            als_reverse_sweep=True,
            als_gateperm_warmstart=True,
            als_gateperm_warmstart_once=True,
            als_lam_anchor_post_warmstart=True,
        )
    raise ValueError(f"unknown optimizer {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Small MNIST smoke test across optimizers.")
    ap.add_argument("--out_dir", type=str, default="outputs/smoke/mnist")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--subset", type=int, default=512)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_als", type=float, default=1.0)
    ap.add_argument("--als_lam", type=float, default=1e-4)
    ap.add_argument("--als_sweeps", type=int, default=2)
    ap.add_argument(
        "--optimizers",
        nargs="+",
        default=["heavyball", "adam", "muon", "kfac", "shampoo", "soap", "als_exact"],
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    loader = make_loader(Path("data") / "mnist_smoke", args.batch_size, args.subset)
    criterion = nn.CrossEntropyLoss()
    results: dict[str, dict[str, float | int | str]] = {}

    for opt_name in args.optimizers:
        torch.manual_seed(0)
        model = MLPForMNIST(hidden_dim=args.hidden_dim, depth=args.depth, bias=False).to(device=device, dtype=torch.float64)
        lr = args.lr_als if opt_name == "als_exact" else args.lr
        opt = make_optimizer(opt_name, model, lr=lr, lam=args.als_lam, sweeps=args.als_sweeps)
        if hasattr(opt, "attach_hooks"):
            opt.attach_hooks(model)
        losses: list[float] = []
        it = iter(loader)
        for _ in range(args.steps):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)
            x = x.to(device=device, dtype=torch.float64)
            y = y.to(device=device)
            opt.zero_grad(set_to_none=True) if hasattr(opt, "zero_grad") else None
            logits = model(x)
            loss = criterion(logits, y)
            if not getattr(opt, "needs_backward", False):
                loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        results[opt_name] = {
            "steps": args.steps,
            "first_loss": losses[0],
            "last_loss": losses[-1],
            "device": str(device),
        }

    (out_dir / "mnist_smoke_results.json").write_text(json.dumps(results, indent=2))
    print(f"[done] wrote {out_dir / 'mnist_smoke_results.json'}")


if __name__ == "__main__":
    main()
