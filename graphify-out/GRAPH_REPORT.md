# Graph Report - neural-flight  (2026-11-17)

## Corpus Check
- Corpus is ~18,691 words - fits in a single context window. You may not need a graph.

## Summary
- 208 nodes · 425 edges · 11 communities (6 shown, 5 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 41 edges (avg confidence: 0.89)
- Token cost unavailable for semantic extraction. The agent tool did not provide token counts. Structural extraction used no model tokens.

## Graph Freshness
- Built from commit: `8abcacf5`
- Check for changes to the files you are investigating since the source commit above.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- PPO Training and Playback
- Flight Models and Visualization
- Dashboard Job Management
- Aerodynamic Simulation
- Flight Environment Lifecycle
- Live Streaming and HTTP
- CUDA and Dependency Setup
- Observation Normalization
- Training Smoke Tests
- Batch Replay Generation
- Shell Launcher

## God Nodes (most connected - your core abstractions)
1. `BirdEnv` - 36 edges
2. `main()` - 17 edges
3. `LiveView` - 12 edges
4. `ActorCritic` - 10 edges
5. `Job` - 9 edges
6. `start_training()` - 9 edges
7. `list_runs()` - 8 edges
8. `main()` - 8 edges
9. `Handler` - 7 edges
10. `main()` - 7 edges

## Surprising Connections (you probably didn't know these)
- `Training Progress Charts` --shares_data_with--> `parse_log()`  [INFERRED]
  hub.html → hub.py
- `Checkpoint Browser` --shares_data_with--> `list_runs()`  [INFERRED]
  hub.html → hub.py
- `Job Activity` --shares_data_with--> `Job`  [INFERRED]
  hub.html → hub.py
- `Training Configuration Form` --references--> `start_training()`  [INFERRED]
  hub.html → hub.py
- `Live View Iframe` --shares_data_with--> `LiveView`  [INFERRED]
  hub.html → hub.py

## Import Cycles
- None detected.

## Communities (11 total, 5 thin omitted)

### Community 0 - "PPO Training and Playback"
Cohesion: 0.11
Nodes (15): start_training(), emit(), main(), meta_of(), r3(), Neural Flight, main(), NumPy (+7 more)

### Community 1 - "Flight Models and Visualization"
Cohesion: 0.09
Nodes (23): _species(), Checkpoint Browser, Neural Flight Hub Dashboard, Job Activity, Live View Iframe, Replay Viewer Modal, Training Progress Charts, Dashboard State Polling (+15 more)

### Community 2 - "Dashboard Job Management"
Cohesion: 0.10
Nodes (8): check_env(), Job, load_species(), main(), run_config(), safe_ckpt(), start_replay(), Watch It Fly

### Community 3 - "Aerodynamic Simulation"
Cohesion: 0.12
Nodes (9): jac(), mv(), quat_mul(), quat_to_mat(), rot_x(), rot_y(), rot_z(), skew() (+1 more)

### Community 5 - "Live Streaming and HTTP"
Cohesion: 0.15
Nodes (7): Handler, list_replays(), list_runs(), LiveView, log_path(), parse_log(), run_dirs()

### Community 6 - "CUDA and Dependency Setup"
Cohesion: 0.18
Nodes (8): ckpt_sort_key(), PyTorch, ensure_triton(), gpu_works(), install(), main(), nvidia(), pick_builds()

## Knowledge Gaps
- **2 isolated node(s):** `Three.js`, `NumPy`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 55 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BirdEnv` connect `Flight Environment Lifecycle` to `PPO Training and Playback`, `Aerodynamic Simulation`?**
  _High betweenness centrality (0.284) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `main()` (e.g. with `PPO Training` and `.obs()`) actually correct?**
  _`main()` has 5 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Three.js`, `NumPy` to the rest of the system?**
  _2 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `PPO Training and Playback` be split into smaller, more focused modules?**
  _Cohesion score 0.10810810810810811 - nodes in this community are weakly interconnected._
- **Why does `Neural Flight` connect `PPO Training and Playback` to `Flight Models and Visualization`, `Dashboard Job Management`, `Aerodynamic Simulation`?**
  _High betweenness centrality (0.097) - this node is a cross-community bridge._
- **Should `Flight Models and Visualization` be split into smaller, more focused modules?**
  _Cohesion score 0.08712121212121213 - nodes in this community are weakly interconnected._
- **Why does `main()` connect `PPO Training and Playback` to `Flight Models and Visualization`, `Dashboard Job Management`, `Aerodynamic Simulation`, `Flight Environment Lifecycle`?**
  _High betweenness centrality (0.085) - this node is a cross-community bridge._

## Extraction Audit

- The raw extraction reported 81 references without an extracted endpoint. The exported graph uses 23 external-symbol placeholders to retain links for unresolved imports and calls.
- The undirected export combines 9 relationships with identical endpoints. The diagnostic found 0 edges with empty endpoints and 0 self-loops.
- Graphify detected Python, shell scripts, Markdown, HTML and requirements text. The Windows batch launcher, training logs and binary checkpoints were outside its supported-file scan.
- The repository tracks the HTML, JSON, report, and update manifest. Graphify caches and machine-specific settings are ignored.
