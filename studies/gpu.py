"""GPU implementation of the operator step: contexts, the step map, and LSQR, in torch.

Why this exists. The operator step costs ~3.7 GFLOP per training step, but spread over roughly
1600 small batched matmuls (L layers x 2 applications x k Krylov iterations). On CPU that is
compute-bound at about a second per step, which caps every experiment. On GPU the same work is
launch-bound and should be one to two orders faster.

Precision. Training arms run in float32: fp64 on this class of GPU runs at 1/64 rate, and the
training step does not need it. The VALIDATION diagnostics -- which must reproduce the closed form
to 1e-14 -- stay on CPU in float64. That split is deliberate; do not move the validation here.

Interfaces mirror `nonlinear.py` / `crelu.py` so results can be cross-checked against them.
"""
from __future__ import annotations
import torch


def init_net(d_in, d, d_out, L, arch, seed, device, dtype=torch.float32):
    g = torch.Generator(device="cpu").manual_seed(seed)
    dims_in = [d_in] + ([2 * d] * (L - 1) if arch == "crelu" else [d] * (L - 1))
    dims_out = [d] * (L - 1) + [d_out]
    Ws = []
    for l in range(L):
        m, k = dims_out[l], dims_in[l]
        W = torch.randn(m, k, generator=g, dtype=torch.float64) * (2.0 / k) ** 0.5
        Ws.append(W.to(device=device, dtype=dtype))
    return Ws


def forward(Ws, X, arch):
    """Returns (logits, per-layer gate tensors)."""
    L = len(Ws)
    if arch == "crelu":
        z = X @ Ws[0].T
        zs = []
        for l in range(1, L):
            zs.append(z)
            c = torch.cat([z.clamp(min=0), (-z).clamp(min=0)], dim=1)
            z = c @ Ws[l].T
        return z, zs
    h = X
    gates = []
    for l in range(L - 1):
        z = h @ Ws[l].T
        d = (z > 0).to(X.dtype)
        gates.append(d)
        h = z * d
    return h @ Ws[-1].T, gates


def contexts(Ws, gs, X, arch):
    """A_l, B_l with P(x) = A_l(x) W_l B_l(x), batched over inputs."""
    L, n = len(Ws), X.shape[0]
    dev, dt = X.device, X.dtype
    d_in, d_out = Ws[0].shape[1], Ws[-1].shape[0]
    B = [torch.eye(d_in, device=dev, dtype=dt).expand(n, d_in, d_in)]
    for l in range(L - 1):
        WB = Ws[l].unsqueeze(0) @ B[-1]
        if arch == "crelu":
            z = gs[l]
            B.append(torch.cat([(z > 0).to(dt).unsqueeze(2) * WB,
                                -((z < 0).to(dt)).unsqueeze(2) * WB], dim=1))
        else:
            B.append(gs[l].unsqueeze(2) * WB)
    A = [torch.eye(d_out, device=dev, dtype=dt).expand(n, d_out, d_out)]
    for l in range(L - 1, 0, -1):
        AW = A[-1] @ Ws[l].unsqueeze(0)
        if arch == "crelu":
            z = gs[l - 1]
            w = AW.shape[2] // 2
            A.append(AW[:, :, :w] * (z > 0).to(dt).unsqueeze(1)
                     - AW[:, :, w:] * (z < 0).to(dt).unsqueeze(1))
        else:
            A.append(AW * gs[l - 1].unsqueeze(1))
    return A[::-1], B


class StepMap:
    """M: dW -> (sum_l A_l(x) dW_l B_l(x))_x, and its adjoint. Flat vectors in and out."""

    def __init__(self, A, B, shapes):
        self.A, self.B, self.shapes = A, B, shapes
        self.At = [a.transpose(1, 2).contiguous() for a in A]
        self.Bt = [b.transpose(1, 2).contiguous() for b in B]
        self.sizes = [s[0] * s[1] for s in shapes]
        self.offs = torch.cumsum(torch.tensor([0] + self.sizes), 0).tolist()
        self.n, self.d_out, self.d_in = A[0].shape[0], A[0].shape[1], B[0].shape[2]

    def mv(self, v):
        out = None
        for l, sh in enumerate(self.shapes):
            dW = v[self.offs[l]:self.offs[l + 1]].view(sh)
            t = (self.A[l] @ dW) @ self.B[l]
            out = t if out is None else out + t
        return out.reshape(-1)

    def rmv(self, u):
        H = u.view(self.n, self.d_out, self.d_in)
        return torch.cat([((self.At[l] @ H) @ self.Bt[l]).sum(0).reshape(-1)
                          for l in range(len(self.shapes))])


def lsqr(M, b, iters, atol=0.0):
    """LSQR (Paige-Saunders), matrix-free, in torch. Returns the solution after `iters` steps."""
    u = b.clone()
    beta = torch.linalg.vector_norm(u)
    if beta == 0:
        return torch.zeros(M.offs[-1], device=b.device, dtype=b.dtype)
    u = u / beta
    v = M.rmv(u)
    alpha = torch.linalg.vector_norm(v)
    if alpha == 0:
        return torch.zeros(M.offs[-1], device=b.device, dtype=b.dtype)
    v = v / alpha
    w = v.clone()
    x = torch.zeros_like(v)
    phibar, rhobar = beta, alpha
    b0 = beta
    for _ in range(iters):
        u = M.mv(v) - alpha * u
        beta = torch.linalg.vector_norm(u)
        if beta > 0:
            u = u / beta
        v = M.rmv(u) - beta * v
        alpha = torch.linalg.vector_norm(v)
        if alpha > 0:
            v = v / alpha
        rho = torch.hypot(rhobar, beta)
        if rho == 0:
            break
        cs, sn = rhobar / rho, beta / rho
        theta = sn * alpha
        rhobar = -cs * alpha
        phi, phibar = cs * phibar, sn * phibar
        x = x + (phi / rho) * w
        w = v - (theta / rho) * w
        if atol > 0 and phibar <= atol * b0:
            break
    return x


def op_step(Ws, X, R, eta, k, arch, atol=0.0):
    """The lambda-free operator step: dW realising as much of -eta*G as k Krylov steps reach."""
    _, gs = forward(Ws, X, arch)
    A, B = contexts(Ws, gs, X, arch)
    G = R.unsqueeze(2) * X.unsqueeze(1)                     # per-sample rank-one operator gradient
    shapes = [tuple(W.shape) for W in Ws]
    M = StepMap(A, B, shapes)
    x = lsqr(M, (-eta * G).reshape(-1), k, atol)
    return [x[M.offs[l]:M.offs[l + 1]].view(shapes[l]) for l in range(len(Ws))]


def coord_grad(Ws, X, R, arch):
    _, gs = forward(Ws, X, arch)
    A, B = contexts(Ws, gs, X, arch)
    G = R.unsqueeze(2) * X.unsqueeze(1)
    return [(A[l].transpose(1, 2) @ G @ B[l].transpose(1, 2)).sum(0) for l in range(len(Ws))]
