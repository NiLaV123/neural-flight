"""
Export policy checkpoints to the flat float layout the wasm sim reads:
  [n_layers, d0..dn, obs_mean(d0), obs_var(d0), log_std(dn), (W[out,in], b[out]) per layer]

    python web/export_policy.py runs/fresh/iter_0.pt runs/fresh/iter_300.pt ... --out web/build/policies.json
Writes one JSON manifest with each policy as base64 float16 (or float32 with --f32).
"""
import argparse
import base64
import json
import os
import re

import numpy as np
import torch


def flatten(ck):
    sd = ck["model"]
    lin = sorted({int(m.group(1)) for k in sd if (m := re.match(r"actor\.(\d+)\.weight", k))})
    Ws = [sd[f"actor.{i}.weight"].float().numpy() for i in lin]
    bs = [sd[f"actor.{i}.bias"].float().numpy() for i in lin]
    dims = [Ws[0].shape[1]] + [w.shape[0] for w in Ws]
    parts = [np.array([len(Ws)] + dims, np.float32),
             sd["norm.mean"].float().numpy(), sd["norm.var"].float().numpy(), sd["log_std"].float().numpy()]
    for w, b in zip(Ws, bs):
        parts += [w.reshape(-1), b]
    return np.concatenate([p.astype(np.float32).reshape(-1) for p in parts])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--out", default="web/build/policies.json")
    p.add_argument("--f32", action="store_true")
    a = p.parse_args()
    out = []
    for path in a.ckpts:
        ck = torch.load(path, map_location="cpu")
        flat = flatten(ck)
        arr = flat.astype(np.float32 if a.f32 else np.float16)
        out.append(dict(name=os.path.basename(path).replace(".pt", ""), iter=int(ck.get("iter", 0)),
                        ring_radius=float(ck.get("ring_radius", 2.0)), dtype="f32" if a.f32 else "f16",
                        n=int(flat.size), data=base64.b64encode(arr.tobytes()).decode()))
        print(f"{path}: iter {out[-1]['iter']}, {flat.size} floats")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
