# Reference trajectory from the PyTorch sim for verifying the wasm port.
import json, sys, torch
sys.path.insert(0, '.')
from bird_env import BirdEnv
from train import ActorCritic
torch.manual_seed(3)
ck = torch.load('runs/fresh/iter_300.pt', map_location='cpu')
ac = ActorCritic(BirdEnv.N_OBS, BirdEnv.N_ACT); ac.load_state_dict(ck['model']); ac.eval()
env = BirdEnv(4, 'cpu', ring_radius=2.0)
init = torch.cat([env.pos, env.vel, env.quat, env.omega, env.joints, env.ring_pos, env.ring_n], -1)
ref = dict(init=init.tolist(), obs0=env.obs().tolist(), act0=ac.act_deterministic(env.obs()).tolist(), pos=[], passed=[])
with torch.no_grad():
    for t in range(150):
        _, _, _, info = env.step(ac.act_deterministic(env.obs()))
        ref['pos'].append(env.pos.tolist()); ref['passed'].append(info['passed'].tolist())
json.dump(ref, open('web/build/ref.json', 'w'))
print('ok')
