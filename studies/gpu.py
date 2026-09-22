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


def _semi_orthogonal(m, k, g):
    """A matrix with orthonormal rows (k>=m) or columns (k<m), from the QR of a Gaussian."""
    a = torch.randn(max(m, k), min(m, k), generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)      # fix the sign convention
    return q if m >= k else q.T


def init_net(d_in, d, d_out, L, arch, seed, device, dtype=torch.float32, init="he"):
    """`init`: he (default, unchanged) | xavier | orthogonal | looks_linear.

    `looks_linear` is the reason the CReLU model is in the paper: with W = [O | -O] the gate
    D(z) = [diag(1[z>0]); -diag(1[z<0])] satisfies W D(z) = O for EVERY z, so P(x) = O_L...O_1 is
    orthogonal at any depth and for every input. That gives a genuinely nonlinear network which is
    perfectly conditioned by construction, so conditioning and gating can be varied independently
    -- which no amount of tuning a ReLU network can achieve.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    doubled = arch == "crelu"
    dims_in = [d_in] + ([2 * d] * (L - 1) if doubled else [d] * (L - 1))
    dims_out = [d] * (L - 1) + [d_out]
    Ws = []
    for l in range(L):
        m, k = dims_out[l], dims_in[l]
        if init == "he":
            W = torch.randn(m, k, generator=g, dtype=torch.float64) * (2.0 / k) ** 0.5
        elif init == "xavier":
            W = torch.randn(m, k, generator=g, dtype=torch.float64) * (1.0 / k) ** 0.5
        elif init == "orthogonal":
            W = _semi_orthogonal(m, k, g)
        elif init == "looks_linear":
            if doubled and k == 2 * (d if l else d_in) and l > 0:
                O = _semi_orthogonal(m, k // 2, g)
                W = torch.cat([O, -O], dim=1)
            else:
                W = _semi_orthogonal(m, k, g)
        else:
            raise ValueError(f"unknown init {init!r}")
        Ws.append(W.to(device=device, dtype=dtype))
    return Ws


_FGLN_CACHE = {}


def _fgln_gate(width, seed, layer, device, dtype):
    """Fixed 0/1 diagonal, independent of the input -- the FGLN model of N1."""
    key = (width, seed, layer, str(device), str(dtype))
    if key not in _FGLN_CACHE:
        g = torch.Generator(device="cpu").manual_seed(9176 + 31 * layer + seed)
        m = (torch.rand(width, generator=g, dtype=torch.float64) > 0.5).to(torch.float64)
        _FGLN_CACHE[key] = m.to(device=device, dtype=dtype)
    return _FGLN_CACHE[key]


FGLN_SEED = 0


LEAKY_ALPHA = 0.1


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
    # "leaky" keeps every gate strictly positive, so no unit can ever die. Leaky ReLU is still
    # positively homogeneous, so f(x) = P(x) x holds exactly and the contexts below are unchanged:
    # the gate tensor is used as a plain diagonal multiplier either way. That makes `leaky` a
    # control for gate death that holds depth, width and the operator factorisation fixed.
    alpha = LEAKY_ALPHA if arch == "leaky" else 0.0
    for l in range(L - 1):
        z = h @ Ws[l].T
        if arch == "linear":                     # gates all one: P is shared across inputs
            d = torch.ones_like(z)
        elif arch == "fgln":                     # fixed gates: also input-independent, but masked
            d = _fgln_gate(z.shape[1], FGLN_SEED, l, z.device, z.dtype).expand_as(z)
        else:
            d = torch.where(z > 0, torch.ones_like(z), torch.full_like(z, alpha))
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


def lsqr(M, b, iters, atol=0.0, stall=0.0):
    """LSQR (Paige-Saunders), matrix-free, in torch. Returns the solution after `iters` steps.

    `stall > 0` turns on best-iterate tracking. Without reorthogonalisation the Lanczos basis loses
    orthogonality on ill-conditioned step maps, and LSQR then *diverges* after it has converged --
    measured here on a depth-4 ReLU net, the residual plateaus by k~200 and the iterate blows up to
    1e16 by k=2000. A fixed large `iters` therefore returns garbage, silently. With `stall` set, the
    iterate with the smallest NORMAL-EQUATION residual ||M^T(Mx-b)|| is kept, and the loop exits on
    convergence or on clear divergence. See the in-loop comment for why the two cheaper tests
    (`phibar`, and ||Mx-b|| itself) both fail. Default 0.0 keeps the original behaviour exactly.
    """
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
    best_ng = torch.as_tensor(float("inf"), device=b.device, dtype=b.dtype)
    best_x = x
    ng0 = torch.linalg.vector_norm(M.rmv(b)).clamp_min(1e-300) if stall > 0 else None
    for it in range(1, iters + 1):
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
        if stall > 0 and (it % 10 == 0):
            # Stop on the NORMAL-EQUATION residual ||M^T(Mx-b)||, the actual optimality condition
            # for min ||Mx-b||. Two cheaper-looking tests both fail here. `phibar` is monotonically
            # non-increasing by construction (phibar <- sn*phibar, sn<=1) so it never signals
            # anything. And ||Mx-b|| alone is blind to the iterate drifting into ker(M), which is
            # how this diverges in practice: on a leaky net ||x|| reached 2.9e9 with ||Mx-b||
            # unchanged to 7 digits. M^T(Mx-b) sees the first and is unmoved by the second, so it
            # stops at the right iterate in both cases.
            ng = torch.linalg.vector_norm(M.rmv(M.mv(x) - b))
            if ng < best_ng:
                best_ng, best_x = ng, x.clone()
            if ng <= stall * ng0 or ng > best_ng * 1e6:
                # The 1e6 margin is deliberately loose: this residual is NOT monotone, it
                # oscillates during convergence, and a tight margin exits on an upswing (a x10
                # margin stopped ReLU L=8 at 1e-3 when it reaches 1e-11 by k=4000). Correctness
                # does not rest on this test -- best_x already holds the best iterate seen, and
                # neither a blow-up nor a ker(M) drift can improve ng -- so it is only an early
                # exit to save iterations once the solve has clearly failed.
                return best_x
        if atol > 0 and phibar <= atol * b0:
            break
    return x


_OP_STEP_DRIFT_RATIO = 1e6


def op_step(Ws, X, R, eta, k, arch, atol=0.0):
    """The lambda-free operator step: dW realising as much of -eta*G as k Krylov steps reach.

    `k` is the bias dial, so this solve is deliberately truncated and deliberately UNDAMPED --
    unlike `solve_min_norm`, which adds Tikhonov damping and stall detection because it is meant
    to converge. The consequence is that quality here is not monotone in `k`. Measured at L=2,
    width 128, MNIST-1D: cos against a converged reference is 1.0000 at k = 200 and k = 1000, and
    then 0.61 (CReLU) / 0.96 (ReLU) at k = 2000, where ||dW|| has gone from 0.85 to 7.9e13 while
    the operator residual moved only 1.66 -> 2.12. That is the ker(M) drift recorded in the
    notebook's section 1.3: past convergence the iterate accumulates null-space components, which
    cost nothing in the residual LSQR minimises and everything in the step actually taken.

    Every result in the paper uses k <= 200 and is below this threshold, but nothing in the code
    said so, so a later sweep at larger k would have returned a silently ruined step. The guard
    raises instead, following the same principle as the exact-ALS dimension guard: a method's
    limits belong in the code, not in the reader's memory.
    """
    _, gs = forward(Ws, X, arch)
    A, B = contexts(Ws, gs, X, arch)
    G = R.unsqueeze(2) * X.unsqueeze(1)                     # per-sample rank-one operator gradient
    shapes = [tuple(W.shape) for W in Ws]
    M = StepMap(A, B, shapes)
    b = (-eta * G).reshape(-1)
    x = lsqr(M, b, k, atol)
    nx = torch.linalg.vector_norm(x)
    nb = torch.linalg.vector_norm(b)
    if nb > 0 and nx > _OP_STEP_DRIFT_RATIO * nb:
        raise RuntimeError(
            f"op_step: the undamped solve has drifted into ker(M) at k={k} "
            f"(||dW||={float(nx):.3e} against ||target||={float(nb):.3e}). The step map's null "
            f"space is large and LSQR's residual cannot see motion inside it, so more iterations "
            f"make the step worse, not better. Use a smaller k (<= 200 is verified safe here), "
            f"or solve with gpu.solve_min_norm, which damps and stall-detects.")
    return [x[M.offs[l]:M.offs[l + 1]].view(shapes[l]) for l in range(len(Ws))]


def coord_grad(Ws, X, R, arch):
    _, gs = forward(Ws, X, arch)
    A, B = contexts(Ws, gs, X, arch)
    G = R.unsqueeze(2) * X.unsqueeze(1)
    return [(A[l].transpose(1, 2) @ G @ B[l].transpose(1, 2)).sum(0) for l in range(len(Ws))]

class _Damped:
    """The augmented operator [M; lam I], whose least-squares solution is the Tikhonov one.

    Solving min ||Mx-b||^2 + lam^2||x||^2 is equivalent to an undamped least-squares problem on
    [M; lam I] against [b; 0], so it needs no change to the LSQR recursion. Damping is what makes
    the solve usable on a step map with a large null space: LSQR's failure mode here is drift into
    ker(M), and ker([M; lam I]) is trivial for any lam > 0, so the iterate cannot drift. Measured
    nullity of the step map at L=6, width 8: 19% for CReLU, 53% for ReLU, but 83% for deep linear
    and 93% for FGLN -- the input-independent architectures are the degenerate ones, because every
    layer moves the same shared operator and the rank is capped at d_out*d_in however many
    parameters there are.
    """

    def __init__(self, M, lam):
        self.M, self.lam, self.offs = M, lam, M.offs
        self.rows = None

    def mv(self, v):
        out = self.M.mv(v)
        self.rows = out.numel()
        return torch.cat([out, self.lam * v])

    def rmv(self, u):
        k = u.numel() - self.offs[-1]
        return self.M.rmv(u[:k]) + self.lam * u[k:]


def solve_min_norm(M, b, iters=400, lam_rel=1e-6, stall=0.0):
    """Minimum-norm least-squares solve of M x ~ b, stabilised by relative Tikhonov damping.

    `lam_rel` is relative to an estimate of the largest singular value of M, so the damping is
    scale-free. It biases the solution by O(lam_rel), which is far below the effects being
    measured, and in exchange makes the solve converge on step maps where the undamped iteration
    silently returns a diverged iterate.
    """
    v = M.rmv(b)
    nv = torch.linalg.vector_norm(v)
    if nv == 0:
        return torch.zeros(M.offs[-1], device=b.device, dtype=b.dtype)
    sigma = torch.linalg.vector_norm(M.mv(v / nv))        # one power step: a lower bound on s_max
    lam = lam_rel * sigma.clamp_min(1e-30)
    D = _Damped(M, lam)
    bb = torch.cat([b, torch.zeros(M.offs[-1], device=b.device, dtype=b.dtype)])
    return lsqr(D, bb, iters, 0.0, stall)

