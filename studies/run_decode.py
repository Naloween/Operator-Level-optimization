"""Compute the post-hoc metrics, including the operator-step decoding, for every completed run."""
from __future__ import annotations
import torch, exp, posthoc, gpu

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    runs = exp.load("ref_mnist1d")
    cache = {}
    for i, (c, r, d) in enumerate(runs, 1):
        key = (c["n_train"], c["n_val"], c["n_test"], c["seed"])
        if key not in cache:
            cache[key] = exp.mnist1d(*key, dev)
        D = cache[key]
        Xp, Yp = D["Xtr"][:100], D["Ytr"][:100]

        def R_fn(Ws, Xp=Xp, Yp=Yp, arch=c["arch"]):
            o, _ = gpu.forward(Ws, Xp, arch)
            _, R, _ = exp.ce(o, Yp)
            return R

        posthoc.compute(d, c, Xp, dev, force=True, X_div=D["Xtr"][:512], R_fn=R_fn)
        print(f"{i}/{len(runs)} {c['arch']} {c['arm']}", flush=True)
    print("DECODE_DONE")

if __name__ == "__main__":
    main()
