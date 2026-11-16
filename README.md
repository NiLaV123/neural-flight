# Neural Flight

A neural network that learns to fly a bird-like creature through rings. The bird is controlled one surface at a time (each wing's flap, sweep and twist, plus the tail's pitch, tilt and spread), and its flight is physically simulated: rigid-body momentum and rotation, with lift and drag on every wing segment. There is no scripted thrust; the bird has to discover flapping to stay up.

The project has two parts:

| | What it is | Use it for |
|---|---|---|
| **Training** (`bird_env.py`, `train.py`) | Batched PyTorch simulator + PPO, runs entirely on the GPU (Apple MPS or CUDA) | Teaching the network to fly |
| **Live web demo** (`web/`, `docs/index.html`) | The same simulator and network ported to Rust → WebAssembly, rendered with WebGPU | Watching and playing with the trained birds |

## Live demo

Open `docs/index.html` in a browser with WebGPU (current Chrome or Edge, or Safari 26+). It is a single self-contained file. The trained network flies live in your browser; this is not a recording. From the page you can:

- scrub through **training stages** (iteration 0 → 300) to watch random flailing turn into flight
- change **wind and turbulence** (the bird never saw wind during training), **action noise**, **ring size**, **flock size** (1–64) and **time scale**
- switch between **chase, orbit and bird's-eye** cameras

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

## Results

The models below were evaluated in the wasm simulator, with 32 birds × 60 s × 3 seeds each:

| Model | Rings per bird per minute | Crashes (96 bird-minutes) |
|---|---|---|
| `runs/fresh/iter_300.pt` (**best**) | **63.3** | **0** |
| `runs/fresh/best.pt` (iter 280) | 61.1 | 0 |
| `checkpoints/latest.pt` (first run, iter 250) | 57.0 | 4 |
| `runs/fresh/iter_200.pt` | 49.6 | 13 |

## Rebuilding the web demo

You need Rust with the `wasm32-unknown-unknown` target:

```bash
.venv/bin/python web/build.py --run runs/fresh     # compiles wasm, exports snapshots, writes docs/index.html
```

`web/test/verify.mjs` checks the wasm port against the PyTorch simulator and measures flight performance. Observations match to within 2e-7, and positions match to 4 decimal places for the first second of flight.

Recorded replays from the PyTorch simulator can still be made with `replay.py`, which uses the three.js viewer in `viewer_template.html`.

## Known simplifications

- The wings have no mass, and flapping costs no energy, so the learned gait is a fast flutter at about 11 Hz.
- Wing motion doesn't push back on the body.
