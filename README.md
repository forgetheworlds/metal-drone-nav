# Metal Drone Navigation

**A from-scratch reinforcement-learning training engine that runs entirely in raw Metal on Apple Silicon — no PyTorch, no TensorFlow, no Python in the training loop — that trains a drone navigation policy to fly through cluttered environments above a real learned flight controller.**

Built on an Apple M3 (10 GPU cores, 16 GB unified memory). Everything reported here is simulation evidence. Numerical parity against the upstream references is checked; reliable transfer to hardware is **not** established.

| Start here | |
|---|---|
| Every graph, training record and video, with its source data | [docs/RESULTS_GALLERY.md](docs/RESULTS_GALLERY.md) |
| What "done" means and why each choice was made | [GOAL.md](GOAL.md) |
| Every measured number, its machine, commit and exact command | [docs/BENCHMARKS.md](docs/BENCHMARKS.md) |
| Current phase and what counts as evidence | [docs/NEXT_PHASE.md](docs/NEXT_PHASE.md) |
| Coding standard this repository is written to | [docs/CODE_DIRECTION.md](docs/CODE_DIRECTION.md) |
| Hypothesis → experiment → result, **including rejected hypotheses** | [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md) |

---

## What this is, in plain terms

A drone needs two control layers:

1. **A low-level controller** that keeps the aircraft stable and tracks motion commands. This project integrates the frozen [RAPTOR](https://github.com/rl-tools/raptor) policy (Science Robotics 2026) unchanged and validates its software behaviour against upstream fixtures and simulator trajectories. It does not establish real flight control.
2. **A navigation policy** that looks through an onboard depth sensor and decides *how to move* — climb over the counter, drop under the beam, slow down in the doorway, swerve from the moving sphere.

This repository builds the second layer and, more importantly, **the entire training system for it**, specialised for one machine.

### The intended goal

[GOAL.md](GOAL.md) specifies the outcome: a quadrotor that moves quickly through previously unseen, cluttered 3-D environments using onboard local perception.

```text
depth sensor + ego state + relative goal
        ↓
navigation policy  →  [vx, vy, vz, yaw_rate]
        ↓
trajectory adapter →  frozen RAPTOR motor controller → 4 motor commands
```

It must vary its speed with environmental complexity, react to moving collision threats quickly enough to evade them, keep working under disturbance, and generalise to layouts it was not trained on. The contract behind every optimisation in this repository is *maximise validated policy-improvement iterations per unit wall time*.

**The full goal is active and incomplete.** The current milestone — a policy that reaches 25/30 connected-room development levels and two watchable native Webots flights — is real but narrow. Read the limitations section before citing any number on this page.

---

## How it works

```text
   depth sensor (16×20 rays, 20 Hz) + ego state + relative goal
                          ↓
        ┌─────────────────────────────────────────────┐
        │  Deployable actor (184 floats, ~12k params) │
        └─────────────────────────────────────────────┘
                          ↓  body velocity + yaw rate
        ┌─────────────────────────────────────────────┐
        │  Trajectory adapter (persistent reference)  │
        └─────────────────────────────────────────────┘
                          ↓  100 Hz
        ┌─────────────────────────────────────────────┐
        │  RAPTOR learned motor controller (frozen)   │
        └─────────────────────────────────────────────┘
                          ↓  4 motor commands
        ┌─────────────────────────────────────────────┐
        │  L2F RK4 quadrotor dynamics (100 Hz)        │
        └─────────────────────────────────────────────┘
                          ↓ rewards / termination
          rollout storage → GAE → PPO → Adam  (all Metal)
```

- **No ML framework in the hot path.** Simulation, synthetic depth, policy forward, rollouts, GAE, PPO backward and Adam are fixed-shape Metal kernels with buffers allocated once. Python appears only in the cold path (export, evaluation orchestration, plotting).
- **The deployed actor receives depth, ego state and the goal.** It receives no obstacle list or stored witness route. Ego sensing is ideal in the current Webots checks. The training critic may receive additional simulator truth and is discarded at export. Privileged waypoint tests are separate diagnostics.
- **Three build targets**, sharing one 320-ray sensor: `metal_nav` (661 raw-ray inputs), `metal_nav_pooled` (181), `metal_nav_guided` (184 — **selected**: geometry prior + learned residual). Actor weights do not load across input dimensions.

Why hand-written Metal rather than a framework: every dependency in the hot loop costs memory traffic, kernel-launch overhead and abstraction that do not contribute to that contract. Specialisation is what makes the measured training-speed optimisations possible. Full reasoning is in [GOAL.md](GOAL.md); the coding standard is [docs/CODE_DIRECTION.md](docs/CODE_DIRECTION.md).

---

## What works today

Each line links to the evidence that supports it. Nothing here is a hardware claim.

| Milestone | Result | Evidence |
|---|---|---|
| Numerical parity gates | RAPTOR oracle 5.96e-7, L2F 9.54e-7, PX4 transform 5.96e-8, integrated control loop 9.09e-6; 40,960 ray/clearance tests exact | [BENCHMARKS · Correctness gates](docs/BENCHMARKS.md#correctness-gates), [ray tests](docs/BENCHMARKS.md#primitive-ray-range-first-measured-ladder) |
| Training throughput | Full rollout + PPO update 1.66 s → 0.295 s → ~0.026 s; GPU beats the matched CPU reference by 1.85× at 8,192 envs | [BENCHMARKS · Optimizations](docs/BENCHMARKS.md#matched-complete-ppo-optimization), [gallery](docs/RESULTS_GALLERY.md#training-throughput) |
| Open-room stable arrival | Fresh seed 820001: original **95/128** → candidate **128/128**; mean arrival 8.19 s → 4.39 s under varied dynamics | [arrival evidence manifest](evidence/inputs/arrival-training/manifest.json) |
| Independent Webots stable arrival (18 static scenes) | Original **0/18 holds** despite entering the goal region in 17/18; arrival candidate **17/18 holds** | [webots-stable-arrival manifest](evidence/inputs/webots-stable-arrival/manifest.json), [figure](docs/RESULTS_GALLERY.md#webots-transfer) |
| Focused connected-room PPO | **25/30** development rooms at rollout 300 (1.2288 M transitions), no privileged waypoints at deployment | [room-training manifest](evidence/inputs/room-training/manifest.json) |
| Paired connected-room transfer | Same 30 development rooms: **25/30 Metal → 18/30 Webots**, 12 contacts in hard scenes | [paired proof](evidence/inputs/room-webots-transfer/proof.json), [figure](docs/RESULTS_GALLERY.md#webots-transfer) |
| Two native Webots recordings | Doorway 6.91 s stable hold, no contact; connected room first entry at 5.23 s, no contact | [native-flight-videos manifest](evidence/inputs/native-flight-videos/manifest.json) |

The final challenge-bank split (90 levels) has **never been evaluated or used for selection**. Development levels were used for selection and are labelled as such everywhere.

### Obstacles encountered, and how we improved

- **Throughput first.** The hot loop was rebuilt around measured bottlenecks: direct batch-reduced gradients removed 43.8 MiB of scratch, and M3 SIMD reductions plus a fused 8×8 `simdgroup_matrix` actor forward took the full rollout+update from 1.66 s to ~0.026 s. [BENCHMARKS](docs/BENCHMARKS.md#matched-complete-ppo-optimization) · [figure](docs/RESULTS_GALLERY.md#training-throughput)
- **Entering the goal was not the same as arriving.** The original policy entered the goal region in 17/18 Webots scenes but held in **0/18**. Adding an explicit arrival contract (≤0.35 m, speed ≤0.5 m/s for 0.2 s) and training for it produced **17/18** stable holds. [protocol](docs/BENCHMARKS.md#goal-entry-versus-stable-arrival-in-webots) · [figure](docs/RESULTS_GALLERY.md#webots-transfer)
- **Short obstacle tests did not predict long routes.** A frozen 270-scene bank (bent hallways, connected rooms, vertical over/under) showed the broad policies at **0/30** and **0/30** on hallways and rooms. A focused family-15 PPO run reached **25/30** development rooms. [bank results](docs/BENCHMARKS.md#frozen-challenge-bank-development-failures) · [figure](docs/RESULTS_GALLERY.md#long-route-challenge-bank-capability)
- **Continued training made it worse.** From the 25/30 peak the run collapsed to **10/30**, and a lower-learning-rate refinement did not help. The selected checkpoint is frozen and preserved; later training is not adopted. [room-training manifest](evidence/inputs/room-training/manifest.json)
- **Knob search did not beat 25/30.** A delegated 1500-rollout search over risk, learning rate and potential scale never exceeded it (best alternatives 18/30 and 19/30), and an independent exact-recipe seed-54 replication peaked at **22/30**. Those arms are preserved as failed evidence, not as improvements. [training records and failed arms](docs/RESULTS_GALLERY.md#training-records)
- **Route guidance helps the arrival baseline.** The arrival actor completes **30/30** connected-room routes when given privileged intermediate goals, but **0/30** with the final goal alone. This is an oracle diagnostic, not an autonomous score. [BENCHMARKS · Routing](docs/BENCHMARKS.md#routing-and-generator-diagnostics)
- **The new candidate is not a replacement.** On the legacy first-entry protocol it regressed tabletop 124→101/128 and mixed 115→108/128. The original, static and moving-threat policies and their checkpoints stay preserved. [retention figure](docs/RESULTS_GALLERY.md#arrival-candidate-vs-preserved-baselines)
- **Video evidence had to be rebuilt.** Early recordings were black or empty because of the observer-camera convention and hidden rendering. The current movies are unmodified Webots Supervisor output with an ffmpeg decoded-frame gate; one real rejected half-black capture is preserved as a failure. The earlier Blender cutaway reconstructions are **not** native footage and are not promoted. [recording protocol](webots/README.md#native-scene-recordings) · [rejected capture](docs/RESULTS_GALLERY.md#rejected-and-local-only-recordings)

---

## Actual Webots flight recordings

Native Webots R2025a main-view movies from live simulation. The observer camera follows the real GPS body position; the scene stays opaque and the drone keeps its declared size. The visible airframe uses simple primitives — it is **not** verified hardware CAD.

| Scene and policy | Result | Goal rule |
|---|---|---|
| Offset doorway · arrival policy | 6.91 s, no contact, 0.2 s hold | Within 0.35 m, speed ≤0.5 m/s for 0.2 s |
| Two offset doors with a table · room policy | 5.23 s, no contact | First entry within 0.35 m; final speed 1.56 m/s — **not** a stable stop |

[![Native doorway view](artifacts/videos/native-doorway-preview.jpg)](artifacts/videos/native-doorway.mp4)

*Offset doorway · arrival policy · 6.91 s stable hold with no contact — [watch the doorway flight](artifacts/videos/native-doorway.mp4).*

[![Native connected-room view](artifacts/videos/native-connected-rooms-preview.jpg)](artifacts/videos/native-connected-rooms.mp4)

*Two offset doors with a table · room policy · first entry at 5.23 s, no contact, final speed 1.56 m/s — [watch the connected-room flight](artifacts/videos/native-connected-rooms.mp4).*

These are **two selected successful examples**, not a success rate. Exact worlds, receipts, 100 Hz traces and video hashes: [native-flight-videos manifest](evidence/inputs/native-flight-videos/manifest.json). Reproduction commands: [webots/README.md](webots/README.md#native-scene-recordings). The full video catalogue, including raw takes and rejected captures, is in [the gallery](docs/RESULTS_GALLERY.md#videos).

---

## Evidence

- **[docs/RESULTS_GALLERY.md](docs/RESULTS_GALLERY.md)** — every figure, training record and video in the repository, each with a caption, the CSV/JSON it was built from, and a link to the authoritative document section.
- **[`evidence/inputs/`](evidence/inputs)** — the compact CSV, TSV, JSONL and trace inputs behind every figure, with hashes in their manifests. Checkpoints used for provenance are in [`assets/checkpoints/`](assets/checkpoints).
- Rebuild every figure and its input-hash manifest without running Metal:

```sh
python3 evidence.py --out artifacts
```

- [`artifacts/manifest.json`](artifacts/manifest.json) records the command, inputs, per-figure filters, metric definitions and the stated limitations of each figure.

`results/` and `build/` are gitignored working directories. Records that live only there (failed training arms, raw Webots runs, verification reports) are catalogued in the gallery and labelled as local-only.

---

## Limitations — read this before citing the results

- **Simulation only.** No real-flight performance has been tested. Real stereo and a physical flight-controller connection remain unvalidated.
- **The geometry-only baseline is strong and sometimes wins.** Mode 13 (no learned residual) reaches 96.1% on held two-doorways vs 89.1% for the learned mode 17, and 91.4% vs 89.8% on mixed. Learning clearly adds value on table/counter and moving spheres, but it is *not* uniformly better than the prior it builds on. Both are always reported side by side.
- **Desired speed is bounded; actual vehicle speed can overshoot it** (a requested 1.5 m/s cap has been measured at 7.34 m/s in one table case).
- **Doorway performance is sensitive to command delay** (50 ms alone: 85.2% → 76.6% on held two-door).
- **Wind is a force-equivalent simulator disturbance**, not a measured wind velocity. The dynamics ranges used for training are declared stress settings, not identified hardware uncertainty.
- **Collision is a conservative 0.18 m sphere**, not a mechanical airframe model; a pose-aware contact model exists as an audit and does not change scoring. Depth is clean and ego sensors are ideal in all reported runs.
- **Webots transfer evidence is limited to small static matrices.** A matched 30-room development matrix is in progress and has not been published yet. Sensor origin, sampling and motor startup still differ from Metal. None of this is broad generalization and none of it is hardware validation.
- **Moving-threat evidence and long-route evidence are separate.** Threat results come from a matched 24-case moving-sphere matrix; long-route results come from the frozen challenge bank. Neither substitutes for the other.
- **No GR2PO experiment has been run.** Every learning result here is PPO. PPO remains the verified control; alternatives require paired controlled evidence.

---

## Build and run

Requires Apple Silicon, macOS 15 or newer, CMake and Command Line Tools. Full Xcode is **not** required — Metal Shading Language sources compile at runtime via `MTLDevice`.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
./build/metal_nav_guided test
```

Train a fresh guided policy, then its table/counter curriculum, then evaluate:

```sh
./build/metal_nav_guided train 3000 7 results/guided.bin '' 1.5 4
./build/metal_nav_guided train 1200 5 results/guided-table-memory.bin results/guided.bin.best 1.5 4 .04 .0003 .0001 1 1
./build/metal_nav_guided eval assets/checkpoints/guided-table-memory.bin.best 17 8 800001 1.5 4
python3 evaluation.py --checkpoint assets/checkpoints/guided-table-memory.bin.best
```

`CHECKPOINT` saves exact resume state atomically; `CHECKPOINT.best` keeps the best validation score. Repeat the same command to resume. Validation seed **700001** differs from the legacy evaluation seeds **800001 / 900001**. Selected checkpoints are preserved under `assets/checkpoints/`.

### Command reference

```text
train ROLLOUTS FAMILY CHECKPOINT [WARMSTART] [SPEED] [DISTANCE]
      [RISK] [ENTROPY] [LEARNING_RATE] [VELOCITY_CONTRACT] [GEOMETRY_MEMORY]
eval CHECKPOINT MODE FAMILY SEED SPEED DISTANCE [SENSOR_DELAY] [WIND_ACCEL]
     [DEPTH_NOISE] [DROPOUT] [COMMAND_DELAY]
bank-eval CHECKPOINT BANK_JSONL SPLIT OUTPUT_CSV [MODE=17] [SPEED=1.5] [MAX_STEPS=400]
bank-witness BANK_JSONL train|dev OUTPUT_CSV [SPEED=1] [MAX_STEPS=1200]
task-eval CHECKPOINT OUTPUT_CSV STAGE FAMILY DOMAIN_AMPLITUDE [SEED] [MODE]
train-tasks ROLLOUTS CHECKPOINT WARMSTART STAGE FAMILY DOMAIN_AMPLITUDE [SEED]
```

Families: 0 open room, 1 boxes, 2 poles, 3 moving spheres, 4 offset doorway, 5 table/counter, 6 mixed, 7 training mixture 0–6, 8 **held-out** two-door composition. Eval modes: 1 random, 2 goal script, 4 full learned mean, 13 geometry prior alone, **17 overhead-gated residual (selected)**, 18/19 inference ablations on fixed weights. Speed is a desired-velocity cap in m/s, distance and noise are metres, wind is acceleration in m/s², dropout is a fraction; sensor delay counts 50 ms frames (≤6), command delay counts 50 ms navigation ticks (≤7). `eval` restores the checkpoint's contract and geometry-memory setting.

### Training under measured sensing and disturbance

Arguments after GEOMETRY_MEMORY are SENSOR_DELAY, WIND_ACCEL, DEPTH_NOISE, DROPOUT, COMMAND_DELAY and SELECTION_MODE. Defaults preserve the clean path:

```sh
./build/metal_nav_guided train 1000 7 results/guided-stress.bin assets/checkpoints/guided-table-memory.bin.best 1.5 4 .04 .0003 .0001 1 1 2 .5 .05 .1 1 17
```

That run trains with 100 ms sensing lag, 50 ms command lag, 0.5 m/s² disturbance, 0.05 m range noise and 10% pixel dropout. It is a candidate curriculum; the packaged policy changes only after independent validation. Measured outcomes of the resulting candidates are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md#disturbance-curricula-and-controlled-threat-acceptance-gaps).

### Deployed policy interface

`assets/navigation.bin` is the selected actor export (48,496 bytes, 12,104 FP32 parameters, source-checkpoint hash inside). [`deployment.hpp`](deployment.hpp) loads it and returns a body FLU velocity vector plus yaw rate — **no Metal dependency, critic, optimizer or motor output**. The caller supplies 184 floats in four groups:

| Indices | Value |
|---|---|
| 0–159 | Current and previous 2×2 min-pooled 8×10 ray ranges /12 |
| 160–172 | Unit goal direction, goal distance /10, body linear/angular velocity /4, world-up in body frame |
| 173–180 | Previous applied navigation fractions, sensor age, reference-position error ×2 |
| 181–183 | Geometry-prior XYZ latents from `guidance.hpp` |

The actor also carries a short depth/pose ring so the vehicle does not re-enter geometry that has left the sensor view. Native RAPTOR and physics run at 100 Hz; navigation and depth at 20 Hz. Requested intent uses a 1.5 m/s vector-norm cap and 0.5 rad/s yaw cap; actual speed can exceed it. **An actual sensor front-end and flight-controller connection still need integration and validation.**

```sh
./build/metal_nav_guided eval-policy assets/navigation.bin 8 800001
./build/metal_nav_guided policy-bench assets/navigation.bin
./build/metal_nav_guided export assets/checkpoints/guided-table-memory.bin.best assets/navigation.bin
```

### Validation and measurement

```sh
./build/metal_nav test            # parity gates: geometry, RAPTOR, L2F, PX4 transform, PPO/GAE/Adam, integrated loop
./build/metal_nav bench-depth     # ray-range scaling ladder
./build/metal_nav bench-raptor    # batched RAPTOR throughput
./build/metal_nav gpu-bench 2048 3 1
./build/metal_nav cpu-bench 3 2048 1
./build/metal_nav profile         # hardware encoder timestamps on M3
```

Machine, commit, configuration and caveats for every benchmark are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

---

## Repository map

| Path | What it holds |
|---|---|
| [GOAL.md](GOAL.md) | Outcome specification and execution contract — what "done" means and why |
| [docs/README.md](docs/README.md) | Index of the research and engineering documents |
| [docs/RESULTS_GALLERY.md](docs/RESULTS_GALLERY.md) | Every figure, training record and video, with source data and captions |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | Every measured number: machine, commit, command, and discarded measurements |
| [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md) | Hypothesis → experiment → result, including failed and rejected hypotheses |
| [docs/NEXT_PHASE.md](docs/NEXT_PHASE.md) | Current phase brief: general local navigation and what "evidence matters" means |
| [docs/RESEARCH_FIDELITY.md](docs/RESEARCH_FIDELITY.md) | Fidelity assessment and the recommended next ablation |
| `main.mm` | Objective-C++ host: device/queue/pipeline setup, harnesses, benchmarks, training orchestration |
| `*.metal` | GPU kernels — physics, depth, RAPTOR, PPO forward/backward, geometry memory |
| `*.hpp` | Portable C++ references and shared CPU/GPU source |
| [`deployment.hpp`](deployment.hpp) | CPU-only deployable actor interface |
| [`challenge_bank.py`](challenge_bank.py) | Deterministic saved AABB levels, mirrored-level option, geometric witness routes |
| [`evidence.py`](evidence.py) + [`evidence/inputs/`](evidence/inputs) | Reproducible figures and their compact inputs with hashes |
| [`evaluation.py`](evaluation.py) | Fixed-seed, stress, threat, ablation and requested-cap evaluation matrices → CSV |
| [`assets/checkpoints/`](assets/checkpoints) | Selected checkpoint snapshots used for evaluation provenance |
| [THIRD_PARTY_LICENSES.txt](THIRD_PARTY_LICENSES.txt) | MIT notices for RAPTOR / RLtools source and weights |

> `STATUS.md` is a local working-state note kept out of version control; it is not part of the public deliverable.

---

## Reference provenance

Pinned upstream commits:

- RAPTOR `2c789dfcf16cc96fe697704492b3bf79dd2cc5a0`
- RLtools `e43ae4bcda4556321a63f4eb5dcc826cd637aa39`

To regenerate the cold reference assets:

```sh
git clone https://github.com/rl-tools/raptor /tmp/raptor-reference
git -C /tmp/raptor-reference checkout 2c789dfcf16cc96fe697704492b3bf79dd2cc5a0
git -C /tmp/raptor-reference submodule update --init rl-tools data
python3 export_raptor.py /tmp/raptor-reference/data/raptor-policy-checkpoint.tar.gz
clang++ -std=c++17 -O2 -I/tmp/raptor-reference/rl-tools/include reference.cpp -o build/reference
./build/reference assets/physics.bin
```

MIT notices for RAPTOR/RLtools source and weights are in [THIRD_PARTY_LICENSES.txt](THIRD_PARTY_LICENSES.txt). **The navigation weights are trained locally by this engine.**

Selected checkpoint SHA-256: `fdd62374c9a1724fc12690d1ccb756985b6c9f8d5f945b290156c17a204cf7d9`
