"""
PPO training of a bird-flight policy. Simulation, rollouts and learning all run on
the GPU (Apple MPS, CUDA, or CPU fallback).

    python train.py --envs 4096 --iters 3000
"""
import argparse
import math
import os
import time

import torch
import torch.nn as nn

from bird_env import BirdEnv


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--envs", type=int, default=8192)
    p.add_argument("--steps", type=int, default=32, help="rollout length per iteration")
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--device", default="auto")
    p.add_argument("--lr", type=float, default=3e-4)
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
    args = p.parse_args()

    dev = pick_device(args.device)
    print("device:", dev)
    torch.manual_seed(0)
    env = BirdEnv(args.envs, dev, ring_radius=4.0, compile=not args.no_compile)
    ac = ActorCritic(BirdEnv.N_OBS, BirdEnv.N_ACT).to(dev)
    opt = torch.optim.Adam(ac.parameters(), lr=args.lr)
    start_it = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=dev)
        ac.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        env.ring_radius = ck.get("ring_radius", env.ring_radius)
        start_it = ck.get("iter", 0)
    os.makedirs(args.out, exist_ok=True)

    snaps = {int(x) for x in args.snapshots.split(",") if x.strip()}
    T, N = args.steps, args.envs
    buf = {k: torch.zeros(T, N, *s, device=dev) for k, s in
           dict(obs=(BirdEnv.N_OBS,), act=(BirdEnv.N_ACT,), logp=(), val=(), rew=(), done=()).items()}

    obs = env.obs()
    ep_ret = torch.zeros(N, device=dev)
    ep_len = torch.zeros(N, device=dev)
    stats = dict(ret=[], len=[], rings=[])
    lr = args.lr
    best_rings = -1.0

    for it in range(start_it, start_it + args.iters):
        t0 = time.time()
        # ------------------------------------------------ rollout (all on GPU)
        with torch.no_grad():
            for t in range(T):
                ac.norm.update(obs)
                d, v = ac.dist(obs)
                a = d.sample()
                rew, term, trunc, info = env.step(a)
                done = info["done"]
                if trunc.any():  # bootstrap value of time-limit truncation
                    rew = rew + args.gamma * trunc.float() * ac.value(env.obs())
                buf["obs"][t] = obs; buf["act"][t] = a; buf["logp"][t] = d.log_prob(a).sum(-1)
                buf["val"][t] = v; buf["rew"][t] = rew; buf["done"][t] = done.float()
                ep_ret += rew; ep_len += 1
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
        mb = args.mb_size; n_mb = T * N // mb
        for _ in range(args.epochs):
            perm = torch.randperm(T * N, device=dev)
            for k in range(n_mb):
                i = perm[k * mb:(k + 1) * mb]
                d, v = ac.dist(b_obs[i])
                logp = d.log_prob(b_act[i]).sum(-1)
                ratio = (logp - b_logp[i]).exp()
                A = b_adv[i]; A = (A - A.mean()) / (A.std() + 1e-8)
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

        # ------------------------------------------------ logging / curriculum
        if dev.type == "mps": torch.mps.synchronize()
        dt = time.time() - t0
        if it % 10 == 0 and stats["ret"]:
            n = len(stats["ret"])
            mr = sum(stats["rings"]) / n
            print(f"it {it:5d} | sps {T*N/dt:8.0f} | ep_ret {sum(stats['ret'])/n:7.2f} | "
                  f"ep_len {sum(stats['len'])/n:6.0f} | rings/ep {mr:5.2f} | max {max(stats['rings']):3.0f} | "
                  f"ring_r {env.ring_radius:.2f} | std {ac.log_std.exp().mean():.3f} | lr {lr:.1e}", flush=True)
            if mr > 4 and env.ring_radius > 2.0:
                env.ring_radius = max(2.0, env.ring_radius * 0.95)
            ck = dict(model=ac.state_dict(), opt=opt.state_dict(), iter=it, ring_radius=env.ring_radius)
            torch.save(ck, os.path.join(args.out, "latest.pt"))
            if mr > best_rings and n > 50:
                best_rings = mr
                torch.save(ck, os.path.join(args.out, "best.pt"))
            stats = dict(ret=[], len=[], rings=[])
        if it in snaps:
            torch.save(dict(model=ac.state_dict(), opt=opt.state_dict(), iter=it, ring_radius=env.ring_radius),
                       os.path.join(args.out, f"iter_{it}.pt"))
            print(f"saved snapshot iter_{it}.pt", flush=True)


if __name__ == "__main__":
    main()
