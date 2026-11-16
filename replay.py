"""
Fly a trained policy and write an interactive 3D replay (replay.html).

    python replay.py --ckpt checkpoints/best.pt --birds 4 --seconds 60
"""
import argparse
import json

import torch

from bird_env import SPECIES_INFO, BirdEnv
from train import ActorCritic, env_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/latest.pt")
    p.add_argument("--birds", type=int, default=4)
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="replay.html")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = env_config(ck)
    env = BirdEnv(args.birds, "cpu", ring_radius=float(ck.get("ring_radius", 2.5)), **cfg)
    radius = env.ring_radius
    ac = ActorCritic(env.N_OBS, BirdEnv.N_ACT)
    ac.load_state_dict(ck["model"]); ac.eval()
    env.MAX_STEPS = 10 ** 9
    env.RING_TIMEOUT = 10 ** 9

    n_steps = int(args.seconds / env.DT)
    birds = [dict(frames=[], rings=[dict(p=env.ring_pos[b].tolist(), n=env.ring_n[b].tolist(), t0=0, t1=None)],
                  end=None) for b in range(args.birds)]
    alive = torch.ones(args.birds, dtype=torch.bool)
    r3 = lambda xs: [round(x, 3) for x in xs]
    with torch.no_grad():
        for t in range(n_steps):
            a = ac.act_deterministic(env.obs())
            _, term, _, info = env.step(a)
            for b in range(args.birds):
                if not alive[b]:
                    continue
                B = birds[b]
                B["frames"].append(r3(env.pos[b].tolist()) + r3(env.quat[b].tolist()) + r3(env.joints[b].tolist()))
                if info["passed"][b]:
                    B["rings"][-1]["t1"] = t
                    B["rings"].append(dict(p=env.ring_pos[b].tolist(), n=env.ring_n[b].tolist(), t0=t, t1=None))
                if term[b]:
                    alive[b] = False
                    B["end"] = "crashed" if env.pos[b, 2] < 0 else "out of bounds"
            if not alive.any():
                break
    for b, B in enumerate(birds):
        print(f"bird {b}: {sum(r['t1'] is not None for r in B['rings'])} rings in {len(B['frames']) * env.DT:.1f}s"
              + (f" ({B['end']})" if B["end"] else ""))

    meta = dict(dt=env.DT, ring_radius=radius, half_span=env.HALF_SPAN, chord=env.CHORD,
                shoulder=list(env.SHOULDER), tail_pos=list(env.TAIL_POS), incidence=env.WING_INCIDENCE,
                iter=ck.get("iter", 0), j_lo=env.j_lo.tolist(), j_hi=env.j_hi.tolist(),
                species=env.species, species_info=SPECIES_INFO[env.species], scale=env.SCALE,
                colors=list(env.COLORS), tail_area=list(env.TAIL_AREA), wing_momentum=env.wing_dynamics)
    html = open("viewer_template.html", encoding="utf-8").read().replace("__DATA__", json.dumps(dict(meta=meta, birds=birds)))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
