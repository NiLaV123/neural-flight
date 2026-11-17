"""
PPO training of a bird-flight policy. Simulation, rollouts and learning all run on
the GPU (Apple MPS, CUDA, or CPU fallback).

    python train.py --envs 4096 --iters 3000
    python train.py --species eagle --out runs/eagle
"""
import argparse
import json
import math
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn

from bird_env import SPECIES, BirdEnv


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class RunningNorm(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n))
        self.register_buffer("var", torch.ones(n))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        d = bm - self.mean
        tot = self.count + bc
        self.mean += d * bc / tot
        self.var = (self.var * self.count + bv * bc + d ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def forward(self, x):
        return ((x - self.mean) / (self.var.sqrt() + 1e-6)).clamp(-8, 8)


def mlp(i, o, h=(256, 256, 128), out_gain=1.0):
    layers, last = [], i
    for w in h:
        l = nn.Linear(last, w)
        nn.init.orthogonal_(l.weight, math.sqrt(2)); nn.init.zeros_(l.bias)
        layers += [l, nn.ELU()]
        last = w
    l = nn.Linear(last, o)
    nn.init.orthogonal_(l.weight, out_gain); nn.init.zeros_(l.bias)
    return nn.Sequential(*layers, l)


class ActorCritic(nn.Module):
    def __init__(self, n_obs, n_act):
        super().__init__()
        self.norm = RunningNorm(n_obs)
        self.actor = mlp(n_obs, n_act, out_gain=0.01)
        self.critic = mlp(n_obs, 1, out_gain=1.0)
        self.log_std = nn.Parameter(torch.full((n_act,), -0.5))

    def dist(self, obs):
        x = self.norm(obs)
        mu = self.actor(x)
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu)), self.critic(x).squeeze(-1)

    def value(self, obs):
        return self.critic(self.norm(obs)).squeeze(-1)

    def act_deterministic(self, obs):
        return self.actor(self.norm(obs))


def env_config(ck):
    """Bird settings stored in a checkpoint (or a run's config.json). Older checkpoints predate some
    settings: no "physics" means wings-v1 if they had wing momentum, else the original massless hawk."""
    physics = ck.get("physics") or ("wings-v1" if ck.get("wing_dynamics", False) else "massless")
    return dict(species=ck.get("species", "hawk"), physics=physics, energy_cost=ck.get("energy_cost", 0.0),
                energy_budget=ck.get("energy_budget", False), soaring=ck.get("soaring", False))


def save_checkpoint(ck, path):
    """Publish a complete file so live view and replay can read it during training."""
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as f:
        temp_path = Path(f.name)
    try:
        torch.save(ck, temp_path)
        # On Windows an open checkpoint can briefly block replacement.
        for attempt in range(10):
            try:
                temp_path.replace(path)
                return
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.05)
    finally:
        temp_path.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--envs", type=int, default=8192)
    p.add_argument("--steps", type=int, default=32, help="rollout length per iteration")
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--device", default="auto")
    p.add_argument("--lr", type=float, help="starting learning rate (default: 0.0003 or the checkpoint's rate)")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--mb-size", type=int, default=16384)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent", type=float, default=0.0)
    p.add_argument("--resume", default="")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--snapshots", default="", help="comma-separated iterations to save as iter_<N>.pt, e.g. 0,10,30,60")
    p.add_argument("--species", default="hawk", choices=list(SPECIES))
    p.add_argument("--physics", default="wings-v2", choices=list(BirdEnv.PHYSICS),
                   help="wings-v2: full wing dynamics (default); wings-v1: simpler wing momentum; "
                        "massless: the original ideal-servo wings")
    p.add_argument("--massless-wings", action="store_true", help="same as --physics massless")
    p.add_argument("--energy-budget", action="store_true",
                   help="muscles drain an energy reserve; rings are food; exhausted birds must glide (wings-v2)")
    p.add_argument("--soaring", action="store_true", help="wind, thermals and sink (wings-v2)")
    p.add_argument("--energy-cost", type=float, default=0.0,
                   help="reward penalty per joule of muscle work per kg of bird (wing momentum only), e.g. 0.02")
    args = p.parse_args()
    for name in ("envs", "steps", "iters", "epochs", "mb_size"):
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.lr is not None and (not math.isfinite(args.lr) or args.lr <= 0):
        p.error("--lr must be positive and finite")
    try:
        snaps = {int(x) for x in args.snapshots.split(",") if x.strip()}
    except ValueError:
        p.error("--snapshots must be comma-separated iteration numbers")
    if any(it < 0 for it in snaps):
        p.error("--snapshots requires nonnegative iteration numbers")

    dev = pick_device(args.device)
    print("device:", dev)
    torch.manual_seed(0)
    cfg = dict(species=args.species, physics="massless" if args.massless_wings else args.physics,
               energy_cost=args.energy_cost, energy_budget=args.energy_budget, soaring=args.soaring)
    ck = None
    if args.resume:
        ck = torch.load(args.resume, map_location=dev)
        cfg = env_config(ck)  # a checkpoint only makes sense with the bird it was trained on
    env = BirdEnv(args.envs, dev, compile=not args.no_compile, **cfg)
    env.ring_radius = 2 * env.RING_MIN_R
    print(f"species: {env.species} | physics: {env.physics} | energy budget: {'on' if env.energy_budget else 'off'} | "
          f"soaring: {'on' if env.soaring else 'off'} | mass {env.MASS} kg | span {2 * env.HALF_SPAN:.2f} m", flush=True)
    ac = ActorCritic(env.N_OBS, BirdEnv.N_ACT).to(dev)
    opt = torch.optim.Adam(ac.parameters(), lr=args.lr if args.lr is not None else 3e-4)
    start_it = 0
    if ck is not None:
        ac.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        if args.lr is not None:
            for group in opt.param_groups:
                group["lr"] = args.lr
        env.ring_radius = ck.get("ring_radius", env.ring_radius)
        start_it = ck.get("iter", -1) + 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(dict(cfg, wing_dynamics=cfg["physics"] != "massless", envs=args.envs), fh, indent=2)

    def checkpoint(it):
        return dict(model=ac.state_dict(), opt=opt.state_dict(), iter=it, ring_radius=env.ring_radius, **cfg)

    T, N = args.steps, args.envs
    buf = {k: torch.zeros(T, N, *s, device=dev) for k, s in
           dict(obs=(env.N_OBS,), act=(BirdEnv.N_ACT,), logp=(), val=(), rew=(), done=()).items()}

    obs = env.obs()
    ep_ret = torch.zeros(N, device=dev)
    ep_len = torch.zeros(N, device=dev)
    stats = dict(ret=[], len=[], rings=[])
    power_sum, power_n, energy_sum = 0.0, 0, 0.0
    lr = opt.param_groups[0]["lr"]
    best_rings = -1.0

    last_it = start_it + args.iters - 1
    for it in range(start_it, last_it + 1):
        t0 = time.time()
        # ------------------------------------------------ rollout (all on GPU)
        with torch.no_grad():
            for t in range(T):
                d, v = ac.dist(obs)
                a = d.sample()
                rew, _, trunc, info = env.step(a)
                done = info["done"]
                ep_ret += rew; ep_len += 1
                if trunc.any():  # bootstrap value of time-limit truncation
                    rew = rew + args.gamma * trunc.float() * ac.value(env.obs())
                buf["obs"][t] = obs; buf["act"][t] = a; buf["logp"][t] = d.log_prob(a).sum(-1)
                buf["val"][t] = v; buf["rew"][t] = rew; buf["done"][t] = done.float()
                power_sum += info["power"].sum(); power_n += N; energy_sum += info["energy"].sum()
                if done.any():
                    idx = done.nonzero().squeeze(-1)
                    stats["ret"] += ep_ret[idx].tolist()
                    stats["len"] += ep_len[idx].tolist()
                    stats["rings"] += info["rings"][idx].float().tolist()
                    ep_ret[idx] = 0; ep_len[idx] = 0
                    env.reset(done)
                obs = env.obs()
            last_val = ac.value(obs)

            # GAE
            adv = torch.zeros_like(buf["rew"])
            gae = torch.zeros(N, device=dev)
            for t in reversed(range(T)):
                nv = last_val if t == T - 1 else buf["val"][t + 1]
                nd = 1.0 - buf["done"][t]
                delta = buf["rew"][t] + args.gamma * nv * nd - buf["val"][t]
                gae = delta + args.gamma * args.lam * nd * gae
                adv[t] = gae
            ret = adv + buf["val"]

        # ------------------------------------------------ PPO update
        b_obs = buf["obs"].reshape(T * N, -1); b_act = buf["act"].reshape(T * N, -1)
        b_logp = buf["logp"].reshape(-1); b_adv = adv.reshape(-1); b_ret = ret.reshape(-1)
        b_val = buf["val"].reshape(-1)
        if b_adv.numel() > 1:
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std(unbiased=False) + 1e-8)
        for _ in range(args.epochs):
            perm = torch.randperm(T * N, device=dev)
            for i in perm.split(args.mb_size):
                d, v = ac.dist(b_obs[i])
                logp = d.log_prob(b_act[i]).sum(-1)
                ratio = (logp - b_logp[i]).exp()
                A = b_adv[i]
                pl = -torch.min(ratio * A, ratio.clamp(1 - args.clip, 1 + args.clip) * A).mean()
                v_clip = b_val[i] + (v - b_val[i]).clamp(-args.clip, args.clip)
                vl = torch.max((v - b_ret[i]) ** 2, (v_clip - b_ret[i]) ** 2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pl + 0.5 * vl - args.ent * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
                opt.step()
                with torch.no_grad():  # adaptive LR on KL
                    kl = ((ratio - 1) - (logp - b_logp[i])).mean().item()
                if kl > 0.02: lr = max(lr / 1.5, 1e-5)
                elif kl < 0.005: lr = min(lr * 1.5, 1e-3)
                for g in opt.param_groups: g["lr"] = lr
            with torch.no_grad():
                ac.log_std.clamp_(-3.0, 0.0)
        # Keep the same normalizer for rollout log probabilities and all PPO epochs.
        ac.norm.update(b_obs)

        # ------------------------------------------------ logging / curriculum
        if dev.type == "mps": torch.mps.synchronize()
        dt = time.time() - t0
        save_latest = it % 10 == 0 or it == last_it
        is_best = False
        if save_latest and stats["ret"]:
            n = len(stats["ret"])
            mr = sum(stats["rings"]) / n
            print(f"it {it:5d} | sps {T*N/dt:8.0f} | ep_ret {sum(stats['ret'])/n:7.2f} | "
                  f"ep_len {sum(stats['len'])/n:6.0f} | rings/ep {mr:5.2f} | max {max(stats['rings']):3.0f} | "
                  f"ring_r {env.ring_radius:.2f} | std {ac.log_std.exp().mean():.3f} | lr {lr:.1e} | "
                  f"power {float(power_sum) / max(power_n, 1):7.2f} W | energy {float(energy_sum) / max(power_n, 1):.2f}",
                  flush=True)
            if mr > 4 and env.ring_radius > env.RING_MIN_R:
                env.ring_radius = max(env.RING_MIN_R, env.ring_radius * 0.95)
            is_best = mr > best_rings and n > 50
            if is_best:
                best_rings = mr
            stats = dict(ret=[], len=[], rings=[])
            power_sum, power_n, energy_sum = 0.0, 0, 0.0
        if save_latest or it in snaps:
            ck = checkpoint(it)
            if save_latest:
                save_checkpoint(ck, out / "latest.pt")
            if is_best:
                save_checkpoint(ck, out / "best.pt")
        if it in snaps:
            save_checkpoint(ck, out / f"iter_{it}.pt")
            print(f"saved snapshot iter_{it}.pt", flush=True)


if __name__ == "__main__":
    main()
