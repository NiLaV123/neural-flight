# Neural Flight

A neural network that learns to fly a bird-like creature through rings. The bird is controlled one surface at a time (each wing's flap, sweep and twist, plus the tail's pitch, tilt and spread), and its flight is physically simulated: rigid-body momentum and rotation, with lift and drag on every wing segment. There is no scripted thrust; the bird has to discover flapping to stay up.

- `bird_env.py`: batched PyTorch simulator, runs entirely on the GPU (Apple MPS or CUDA)
- `train.py`: PPO training
- `replay.py`: flies a trained policy and writes an interactive 3D replay page

## How the simulation works

- **Rigid body:** 0.8 kg bird with gravity, momentum, an inertia tensor and Euler's rotation equations, integrated at 200 Hz.
- **Blade-element aerodynamics:** each wing is split into 3 segments, plus one surface for the tail. The airflow each segment feels includes the bird's motion, its rotation, and the segment's own motion from the joints, so a downstroke really pushes air. Lift and drag come from the angle of attack, with a stall model.
- **Servo joints:** the network outputs 9 target angles at 50 Hz, and each joint moves toward its target at a limited speed.
- **Task:** fly through a ring (crossing its plane inside the radius, from the correct side). A new ring then appears 18–35 m ahead at a random angle and height.
- **Reward:** progress toward the ring, +10 for each pass, −5 for a crash, and a small penalty on control effort.

## Training

```bash
uv venv && uv pip install -r requirements.txt
.venv/bin/python train.py                                  # ~115k steps/s on an M2 GPU
.venv/bin/python train.py --resume checkpoints/latest.pt   # continue a run
```

The training setup:

- **8,192 birds** fly in parallel, with the physics fused into GPU kernels by `torch.compile`.
- **PPO** with GAE, adaptive learning rate and running observation normalisation.
- **Ring curriculum:** rings start at 4 m radius and shrink to 2 m as the birds improve.
- **Learning speed:** it learns to fly through rings in about 150–200 iterations (around 10 minutes).

On macOS, use `caffeinate -i` so the machine doesn't sleep mid-run.

## Replays

```bash
.venv/bin/python replay.py --ckpt runs/fresh/iter_300.pt --birds 4 --seconds 60 --out replay.html
```

This flies the policy in the PyTorch simulator and writes a recorded three.js replay using `viewer_template.html`. `runs/fresh/make_replays.sh` makes one for every snapshot.

## Known simplifications

- The wings have no mass, and flapping costs no energy, so the learned gait is a fast flutter at about 11 Hz.
- Wing motion doesn't push back on the body.
