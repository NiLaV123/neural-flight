# Neural Flight

A neural network that learns to fly a bird-like creature through rings. The bird is controlled one surface at a time (each wing's flap, sweep and twist, plus the tail's pitch, tilt and spread), and its flight is physically simulated: rigid-body momentum and rotation, with lift and drag on every wing segment. There is no scripted thrust; the bird has to discover flapping to stay up.

- `bird_env.py`: batched PyTorch simulator, runs entirely on the GPU (Apple MPS or CUDA)
- `train.py`: PPO training
- `replay.py`: flies a trained policy and writes an interactive 3D replay page
- `live.py`: flies the newest checkpoint of a run in real time (the hub's live view)
- `hub.py`: a local dashboard that runs all of the above for you

## Quick start: the Hub

```bash
./start.sh        # macOS / Linux
start.bat         # Windows (or double-click it)
```

The first time, this creates `.venv` and installs the requirements. It then opens the **Neural Flight Hub** at http://127.0.0.1:8765, where you can:

- **Train**: pick a bird species, the wing physics, and whether to add the energy budget and soaring. Then pick a preset (quick test, standard or long), or set the iterations, parallel birds, device and snapshots yourself. You can start from scratch or continue from any checkpoint.
- **Live view**: while a run trains, its newest checkpoint flies in real time right in the hub, and the bird switches to each new checkpoint as it is saved. The live view starts with training, or you can press ▶ Start live view for any run.
- **Watch it fly**: press ▶ Watch it fly, during or after training, to record the run's best checkpoint and play it in the page.
- **Watch progress**: see live charts of rings per episode, return, ring radius and episode length, plus the current steps per second, for any run.
- **Follow activity**: open the output of each running job, or stop a training run.
- **Browse checkpoints**: every run in `checkpoints/` and `runs/<name>/`, each with Replay and Continue training buttons.
- **Make replays**: fly a checkpoint and watch the recorded 3D replay right in the page. Replays are saved in `replays/`.

The hub uses only the Python standard library and runs the same `train.py` and `replay.py` commands shown below, so the command line still works the same way. You can use `--port` to change the port or `--no-browser` to stop it opening a browser.

### Windows and NVIDIA GPUs

Training runs on NVIDIA GPUs through CUDA, on Windows and Linux.

- **Windows:** `start.bat` needs Python 3 from python.org with "Add python.exe to PATH" ticked. On first run, `setup_env.py` reads your NVIDIA driver's CUDA version and installs the matching CUDA build of PyTorch. A plain `pip install torch` would install the CPU-only build on Windows. The hub's top bar should then show **CUDA**. If it shows CPU, update your NVIDIA driver, delete `.venv` and run `start.bat` again.
- **`torch.compile`:** on Windows it often isn't available because it needs Triton. When that happens, training says so once and carries on without it. The results are the same, but training is slower.
- **Linux:** the default `pip install torch` already includes CUDA, so `./start.sh` works as is.

## Bird species

| Species | Mass | Wingspan | Character |
|---|---|---|---|
| `sparrow` | 30 g | 0.24 m | tiny and light, can beat its wings very fast |
| `pigeon` | 350 g | 0.66 m | agile all-rounder |
| `hawk` | 0.8 kg | 1.2 m | the original bird |
| `eagle` | 4.5 kg | 2.1 m | broad wings, slow powerful beats |
| `albatross` | 8.5 kg | 3.0 m | long thin wings, needs speed |

Each species gets its own size, inertia, wing and tail area, muscle strength, launch speed, ring spacing and smallest ring size. All of these scale from a few measurements in `SPECIES` in `bird_env.py`, so adding a new bird takes one line. A policy only flies the species it was trained on, so each species needs its own training run:

```bash
.venv/bin/python train.py --species eagle --out runs/eagle
```

## How the simulation works

- **Rigid body:** each bird (0.8 kg for the hawk) with gravity, momentum, an inertia tensor and Euler's rotation equations, integrated at 200 Hz.
- **Blade-element aerodynamics:** each wing is split into 3 segments, plus one surface for the tail. The airflow each segment feels includes the bird's motion, its rotation, and the segment's own motion from the joints, so a downstroke really pushes air. Lift and drag come from the angle of attack, with a stall model.
- **Wing physics** (`--physics`):
  - `wings-v2` (default): full wing dynamics. Each wing has mass (7% of the bird) spread along its span and chord. Flap, sweep and twist are all driven by torque-limited muscles, against the wing's inertia, its aerodynamic load and its weight. Lift acts just behind the twist axis, so the air pushes on the twist too. The body and wings are solved together each physics step as one 12-degree-of-freedom system: body translation and rotation plus six wing joints. That covers recoil, the time-varying inertia tensor, and centrifugal, Coriolis and gyroscopic coupling. When a wing hits its joint stop, the body takes up the wing's momentum. Muscle work is reported as flapping power.
  - `wings-v1`: the earlier, simpler model. Flap and sweep have mass and momentum, and twist is a servo.
  - `massless`: the original model, with ideal servo joints and massless wings. The checkpoints in `checkpoints/` and `runs/fresh/` use it.
  - Every checkpoint remembers its physics and keeps working with it.
- **Energy budget** (`--energy-budget`): the bird starts with a reserve of 2,500 J per kg. Muscle work and a resting metabolism drain it, and each ring passed restores 20%. When the reserve runs out, the muscles drop to 10% strength, so the bird has to glide.
- **Soaring** (`--soaring`): adds wind and rising air. Each episode gets a random wind of up to 7 m/s at 10 m height, with a realistic boundary-layer profile that is stronger higher up. Rising thermals form and dissipate around the bird, and the air sinks gently between them. The bird feels the wind, and its observations include the nearest thermal. Combine soaring with the energy budget to reward birds that gain height in thermals instead of flapping.
- **Task:** fly through a ring (crossing its plane inside the radius, from the correct side). A new ring then appears 18–35 m ahead at a random angle and height.
- **Reward:** progress toward the ring, +10 for each pass, −5 for a crash, and a small penalty on control effort. `--energy-cost` can also charge for the muscles' work, which pushes the bird toward an efficient wingbeat.

## Training

```bash
uv venv && uv pip install -r requirements.txt
.venv/bin/python train.py                                  # hawk with full wing dynamics
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

- Each wing segment's mass is modelled as two point masses, and the tail has no mass.
- The wing integrator is first order. With very violent flapping it loses a little momentum, which shrinks with smaller time steps.
- Wind is sampled at the bird's position rather than per wing segment, and thermals are smooth columns without turbulence.
- With massless wings, the learned gait is a fast flutter at about 11 Hz.
