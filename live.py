"""
Fly the newest checkpoint of a run in real time and stream it as JSON lines on stdout.
The hub runs this for its live view while training; it switches to each new checkpoint
as training saves it (every 10 iterations), so you watch the bird improve.

    python live.py --run runs/hawk            # follows runs/hawk/latest.pt
    python live.py --ckpt runs/hawk/best.pt   # one fixed checkpoint

Messages (one JSON object per line, field "t" is the type):
  wait   no checkpoint yet          meta   bird/physics settings + training iteration
  reset  a new flight starts        f      one frame: pos(3) quat(4) joints(9) energy power
  ring   ring passed, next ring     th     thermals changed          end   flight ended
"""
import argparse
import json
import os
import sys
import time

import torch

from bird_env import SPECIES_INFO, BirdEnv
from train import ActorCritic, env_config

torch.set_num_threads(1)          # stay out of the way of training


def emit(**msg):
    sys.stdout.write(json.dumps(msg, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def r3(xs):
    return [round(x, 3) for x in xs]


def meta_of(env, ck):
    return dict(dt=env.DT, ring_radius=env.ring_radius, half_span=env.HALF_SPAN, chord=env.CHORD,
                shoulder=list(env.SHOULDER), tail_pos=list(env.TAIL_POS), incidence=env.WING_INCIDENCE,
                iter=ck.get("iter", 0), j_lo=env.j_lo.tolist(), j_hi=env.j_hi.tolist(),
                species=env.species, species_info=SPECIES_INFO[env.species], scale=env.SCALE,
                colors=list(env.COLORS), tail_area=list(env.TAIL_AREA), wing_momentum=env.wing_dynamics,
                physics=env.physics, energy_budget=env.energy_budget, soaring=env.soaring,
                z0=env.Z0, thermal_top=env.THERMAL_TOP)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", help="run folder to follow (uses its latest.pt)")
    p.add_argument("--ckpt", help="a fixed checkpoint instead")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed (1 = real time)")
    args = p.parse_args()
    path = args.ckpt or os.path.join(args.run, "latest.pt")

    env = ac = cfg = None
    mtime, k, next_check = None, 0, 0.0
    t_next = time.monotonic()
    while True:
        now = time.monotonic()
        if now >= next_check:                                   # look for a new checkpoint once a second
            next_check = now + 1.0
            try:
                m = os.path.getmtime(path)
            except OSError:
                m = None
            if m is None and ac is None:
                emit(t="wait", text="Waiting for the first checkpoint (training saves one every 10 iterations)…")
                time.sleep(1.0)
                continue
            if m is not None and m != mtime:
                try:
                    ck = torch.load(path, map_location="cpu")
                except Exception:                               # file still being written; try again shortly
                    time.sleep(0.3)
                    continue
                mtime = m
                new_cfg = env_config(ck)
                if env is None or new_cfg != cfg:
                    cfg = new_cfg
                    env = BirdEnv(1, "cpu", ring_radius=float(ck.get("ring_radius", 2.5)), **cfg)
                    env.MAX_STEPS = 10 ** 9
                    ac = ActorCritic(env.N_OBS, BirdEnv.N_ACT)
                    fresh = True
                else:
                    env.ring_radius = float(ck.get("ring_radius", env.ring_radius))
                    fresh = False
                ac.load_state_dict(ck["model"]); ac.eval()
                emit(t="meta", meta=meta_of(env, ck))
                if fresh:
                    env.reset(torch.ones(1, dtype=torch.bool))
                    emit(t="reset", k=k, ring=dict(p=env.ring_pos[0].tolist(), n=env.ring_n[0].tolist()),
                         wind=[env.wind_ref[0].item(), *env.wind_dir[0].tolist()],
                         th=[r3(x) for x in env.therm[0].tolist()] if env.soaring else None)
                    last_th = env.therm[0].clone()

        with torch.no_grad():
            _, term, trunc, info = env.step(ac.act_deterministic(env.obs()))
        emit(t="f", f=r3(env.pos[0].tolist()) + r3(env.quat[0].tolist()) + r3(env.joints[0].tolist())
             + [round(info["energy"][0].item(), 3), round(info["power"][0].item(), 1)])
        if info["passed"][0]:
            emit(t="ring", k=k + 1, p=env.ring_pos[0].tolist(), n=env.ring_n[0].tolist())
        if env.soaring and not torch.equal(env.therm[0], last_th):
            last_th = env.therm[0].clone()
            emit(t="th", k=k, th=[r3(x) for x in last_th.tolist()])
        k += 1
        if term[0] or trunc[0]:
            why = "crashed" if env.pos[0, 2] < 0 else ("out of bounds" if env.pos[0, 2] > 150 else "lost the ring")
            emit(t="end", why=why)
            time.sleep(1.2 / args.speed)
            env.reset(torch.ones(1, dtype=torch.bool))
            emit(t="reset", k=k, ring=dict(p=env.ring_pos[0].tolist(), n=env.ring_n[0].tolist()),
                 wind=[env.wind_ref[0].item(), *env.wind_dir[0].tolist()],
                 th=[r3(x) for x in env.therm[0].tolist()] if env.soaring else None)
            last_th = env.therm[0].clone()
            t_next = time.monotonic()
        t_next += env.DT / args.speed
        delay = t_next - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        elif delay < -0.5:                                      # fell behind (slow machine): don't try to catch up
            t_next = time.monotonic()


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, BrokenPipeError):
        pass
