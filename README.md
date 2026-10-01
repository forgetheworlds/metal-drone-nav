# Metal Drone Navigation

**A from-scratch reinforcement-learning training engine that runs entirely in raw Metal on Apple Silicon — no PyTorch, no TensorFlow, no Python in the training loop — that trains a drone navigation policy to fly through cluttered environments above a real learned flight controller.**

Built and validated on a MacBook-class Apple M3 (10 GPU cores, 16 GB unified memory) in a single day of automated research sessions.

---

## At a glance

| | |
|---|---|
| **What** | Simulation + training engine for a local depth-based quadrotor navigation policy |
| **How** | Simulation, depth sensing, policy inference, rollouts, PPO gradients and Adam all run as raw Metal compute kernels |
| **Low-level control** | The real [RAPTOR](https://github.com/rl-tools/raptor) foundation policy (Science Robotics 2026) flies the vehicle; the learned policy only decides *where to go* |
| **Physics** | L2F / [RLtools](https://github.com/rl-tools/rl-tools)-compatible Crazyflie dynamics, validated against official fixtures |
| **ML stack in the hot path** | **None.** Fixed-shape FP32 kernels, hand-derived PPO backward pass, buffers allocated once at setup |
| **Cold path only** | Python for asset export and evaluation orchestration |
| **Third-party code** | RAPTOR + RLtools, pinned by commit hash, MIT — see [THIRD_PARTY_LICENSES.txt](THIRD_PARTY_LICENSES.txt) |

---

## Headline results

### Performance

The same workload (raw 661-input actor, 128 environments × 32-step rollout, two PPO epochs, minibatch 256, FP32, with the actual controller/physics/sensor in the loop):

| Optimization | Full rollout + PPO update |
|---|---|
| Baseline Metal implementation | 1.66 s |
| Direct batch-reduced gradients (removed 43.8 MiB of scratch) | 0.295 s |
| M3 SIMD reductions + fused 8×8 matrix actor forward | **~0.026 s** |

Geometry-memory evaluation (128 envs × 200 navigation ticks) with **identical episode results**: 3.08 s → **0.143 s**.

Against a *matched, optimized* CPU reference (Accelerate SGEMM + GCD, identical worlds, rays, physics and PPO work):

| Environments | GPU | CPU | CPU/GPU |
|---:|---:|---:|---:|
| 128 | 0.082 s | 0.081 s | ~0.99× |
| 512 | 0.188 s | 0.310 s | 1.65× |
| 2,048 | 0.675 s | 1.230 s | 1.82× |
| 8,192 | 2.707 s | 5.016 s | **1.85×** |

> These are three separate measurements on different workloads and must not be combined into a single ratio. The large ~64× figure is versus the project's own original Metal implementation, not versus CPU. Full commands, configurations and caveats are in [BENCHMARKS.md](BENCHMARKS.md).

### Correctness

Every performance claim is gated on parity against official upstream references:

| Gate | Workload | Max absolute error |
|---|---|---:|
| RAPTOR official oracle | 128 envs × 16 recurrent steps | 5.96e-7 |
| L2F official physics fixtures | 512 fixtures | 9.54e-7 |
| Ray/collision geometry | 40,960 CPU/GPU rays + 128 clearances | exact match |
| PX4 observation transform | independent expanded formula | 5.96e-8 |
| Complete integrated control loop | 32 envs × 160 native ticks | 9.09e-6 |
| PPO / GAE / gradients / Adam | operator-level checks | within 3e-5 bounds |

### Navigation quality

Pure PPO from scratch — no imitation learning, no privileged actor. Final evaluation: 28 configurations × 128 episodes, held-out seeds. Machine-readable results in [assets/evaluation.csv](assets/evaluation.csv).

| Environment family | Learned (mode 17) | Geometry-only baseline (mode 13) | Goal script (mode 2) |
|---|---:|---:|---:|
| Table / counter | **96.9%** | 74.2% | 24.2% |
| Moving spheres | **98.4%** | 94.5% | 72.7% |
| Mixed (training families 0–6) | 89.8% | 91.4% | 57.8% |
| Held-out two-door composition | 89.1% | 96.1% | 45.3% |

On a fresh seed (900001): mixed 91.4%, held two-door 92.2%, table 98.4%, moving spheres 92.2%.

Earlier standalone evidence that the policy uses actual perception: box world **81.25%** success vs 47.66% goal-script, **0%** random, 47.66% blind-depth ablation.

**Robustness** — combined stress (100 ms sensor delay + 50 ms command delay + 0.05 m depth noise + 10% pixel dropout + 0.5 m/s² wind): held two-door success drops to **70.3%**.

---

## What this actually is, in plain terms

A drone needs two control layers:

1. **A low-level controller** that keeps the aircraft stable and tracks motion commands — this is *solved* here by using RAPTOR, a pretrained neural flight controller, unchanged and frozen.
2. **A navigation policy** that looks through an onboard depth sensor and decides *how to move* — climb over the counter, drop under the beam, slow down in the doorway, swerve from the moving sphere.

This project builds the second layer and, more importantly, **the entire training system for it**, specialized for one machine.

### Why hand-written Metal instead of PyTorch?

The GOAL.md contract is: *maximize validated policy-improvement iterations per unit wall time.* Every Python/tensor-framework dependency in the hot loop costs memory traffic, kernel-launch overhead and abstraction that does not contribute to that outcome. So the simulator, synthetic depth generation, policy forward pass, rollout storage, GAE, PPO clipping, backprop and Adam updates are all fixed-shape Metal kernels with buffers allocated once.

That specialization is what makes the optimizations above possible — e.g. the fused 8×8 `simdgroup_matrix` actor forward uses an Apple GPU primitive that a generic framework would not select for this exact shape.

### What's genuinely hard here

- **Runtime MSL compilation** — the kernels compile at run time from source, so full Xcode isn't required (verified against Command Line Tools only).
- **Hand-derived backward pass** — no autograd; PPO gradients are derived and implemented directly for this exact network.
- **Privileged-critic asymmetry** — the critic may see exact simulator state during training; the deployed actor sees only deployable depth + estimated ego pose, never privileged obstacle positions.
- **Parity-first discipline** — every subsystem is validated against official RAPTOR and L2F references before it is allowed into training.
- **Honest negative results** — failed hypotheses are preserved rather than deleted (see [RESEARCH_LOG.md](RESEARCH_LOG.md)).

---

## Architecture

```text
     depth sensor (16x20 ray range, 20 Hz)
                    │
                    ▼
   ┌──────────────────────────────────────────────┐
   │  Deployable actor  (184 floats)              │
   │  pooled range ×2 + goal/ego/control +        │
   │  3 geometry-prior latents + pose/depth ring  │
   └──────────────────────────────────────────────┘
                    │  body FLU velocity + yaw rate
                    ▼
   ┌──────────────────────────────────────────────┐
   │  Trajectory adapter                          │
   │  integrates desired velocity into a          │
   │  persistent position reference (PX4-style)   │
   └──────────────────────────────────────────────┘
                    │  100 Hz
                    ▼
   ┌──────────────────────────────────────────────┐
   │  RAPTOR learned motor controller (frozen)    │
   │  22→16 dense, GRU-16, 16→4                   │
   └──────────────────────────────────────────────┘
                    │  4 motor commands
                    ▼
   ┌──────────────────────────────────────────────┐
   │  L2F RK4 quadrotor dynamics (100 Hz)         │
   └──────────────────────────────────────────────┘
                    │
                    ▼  rewards / termination
        rollout storage → GAE → PPO → Adam
```

The actor sees **no exact simulator obstacle positions**. Only the separate critic, which is discarded at export time, gets privileged state.

**Three build targets**, all sharing one 320-ray sensor:

| Target | Actor inputs | Notes |
|---|---:|---|
| `metal_nav` | 661 | raw per-ray ranges |
| `metal_nav_pooled` | 181 | 2×2 min-pooled ranges |
| `metal_nav_guided` | 184 | **selected** — geometry prior + learned residual |

Actor weights cannot be loaded across different input dimensions.

---

## Limitations — read this before citing the results

This is stated plainly because it matters more than the scores:

- **Simulation only.** Real-flight performance has **not** been tested. There is no sensor front-end and no flight-controller connection in this repository.
- **The geometry-only baseline is strong and sometimes wins.** Mode 13 (no learned residual) reaches 96.1% on held two-doorways vs 89.1% for the learned mode 17, and 91.4% vs 89.8% on mixed. Learning clearly adds value on table/counter (96.9% vs 74.2%) and moving spheres, but it is *not* uniformly better than the prior it builds on. Both are reported side by side rather than only the flattering number.
- **Doorway performance is sensitive to command delay** (50 ms command delay alone: 85.2% → 76.6% on held two-door).
- **Desired speed is bounded; actual vehicle speed can overshoot it.**
- **Wind is a force-equivalent simulator disturbance**, not a measured wind velocity.
- **Transfer to real sensors is an open problem.** Depth-domain gap between simulated and stereo depth is a known issue in the literature (Depth Transfer, RA-L 2025) and is out of scope here.

---

## Build and run

Requires Apple Silicon, macOS 15 or newer, CMake and Command Line Tools. Full Xcode is **not** required — Metal Shading Language sources compile at runtime via `MTLDevice`. The verified implementation uses FP32 safe arithmetic and precise floating-point functions.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
./build/metal_nav_guided test
```

Train a fresh guided policy, then its table/counter curriculum:

```sh
./build/metal_nav_guided train 3000 7 results/guided.bin '' 1.5 4
./build/metal_nav_guided train 1200 5 results/guided-table-memory.bin results/guided.bin.best 1.5 4 .04 .0003 .0001 1 1
./build/metal_nav_guided eval results/guided-table-memory.bin.best 17 8 800001 1.5 4
python3 evaluation.py
```

`CHECKPOINT` saves exact resume state atomically; `CHECKPOINT.best` retains the best validation score, breaking ties by successful goal time. Repeat the same train command to resume. A warm-start path loads actor/critic and resets optimizer/exploration. One `results/training.tsv` records selection history. Validation seed **700001** differs from final evaluation seeds **800001 / 900001**. `results/` and `build/` are gitignored; trained checkpoints live on the local machine only.

### Command reference

```text
train ROLLOUTS FAMILY CHECKPOINT [WARMSTART] [SPEED] [DISTANCE]
      [RISK] [ENTROPY] [LEARNING_RATE] [VELOCITY_CONTRACT] [GEOMETRY_MEMORY]
eval CHECKPOINT MODE FAMILY SEED SPEED DISTANCE
     [SENSOR_DELAY] [WIND_ACCEL] [DEPTH_NOISE] [DROPOUT] [COMMAND_DELAY]
```

Speed is desired velocity cap in m/s; distance/noise are metres; wind is acceleration in m/s²; dropout is a fraction. Sensor delay counts 50 ms frames (≤6); command delay counts 50 ms navigation ticks (≤7). Contract 1 uses equal XYZ scaling with a vector norm cap; legacy contract 0 is retained for older checkpoints. Eval restores the checkpoint's contract and geometry-memory setting.

| Family | Geometry |
|---|---|
| 0 | Open room |
| 1 | Boxes |
| 2 | Poles |
| 3 | Moving spheres |
| 4 | Offset doorway |
| 5 | Table or counter |
| 6 | Mixed geometry |
| 7 | Training mixture 0–6 |
| 8 | **Held-out** two-door composition |

Eval modes: 1 random, 2 goal direction, 4 full learned mean; guided 13 geometry memory alone, 14 quarter learned residual, **17 overhead-gated residual (selected)**. Mode 17 keeps more learned residual for vertical obstacle clearance and reduces it near door frames. Mode 13 is reported alongside as the geometry-only baseline.

---

## Train under measured sensing and disturbance conditions

Additional train arguments after GEOMETRY_MEMORY are SENSOR_DELAY, WIND_ACCEL, DEPTH_NOISE, DROPOUT, COMMAND_DELAY and SELECTION_MODE. Defaults preserve the clean training path. The delays and corruption fields already exist in the simulator/checkpoint; training now exposes them directly. Selection mode17 evaluates the guided inference variant under the same conditions on held-out seed700001. New runs also retain their starting policy as a validation baseline. Resume uses additional rollout count.

```sh
./build/metal_nav_guided train 1000 7 results/guided-stress.bin results/guided-table-memory.bin.best 1.5 4 .04 .0003 .0001 1 1 2 .5 .05 .1 1 17
```

This trains with100ms sensing lag,50ms command lag,0.5m/s² disturbance,0.05m range noise and10% pixel dropout. It is a candidate curriculum; the packaged policy changes only after independent validation.

## Controlled threats and reaction time

```sh
python3 evaluation.py --threats
./build/metal_nav_guided threat-eval results/guided-table-memory.bin.best 17 0 2 1 800001
./build/metal_nav_guided reaction-latency results/guided-table-memory.bin.best 2 1
```

Threat arguments are checkpoint,mode,kind(0 approaching /1 crossing),threat speed,nominal TTC,seed,optional sensor and command delays. The vehicle starts flying at1.5m/s; the threat is a0.35m sphere. TTC is the nominal straight-line encounter parameter, not the measured policy trajectory. All motion, sensing and collision checks remain in Metal after scene setup. Goal-script and geometry-only baselines establish whether the scene requires avoidance.

`reaction-latency` clones a warmed episode into two identical runs. It verifies matching motor outputs before inserting an approaching sphere into one run, then measures the first applied-command and motor difference above1e-4. This diagnostic reports simulated latency upper bounds at50ms observation resolution. The measured bounds are50ms without delay and200ms with100ms sensing plus50ms command delay. They exclude real sensor/transport timing and do not prove evasion success.

Training family9 rehearses50% single doors,25% tables/counters and25% broad families0–6. Two-door family8 remains held out. This addresses the measured loss of table skills in a door-only curriculum.

## Compact policy interface

`assets/navigation.bin` is the selected actor export. [`deployment.hpp`](deployment.hpp) loads it and produces a body FLU velocity vector plus yaw rate. It has **no Metal dependency, critic, optimizer or motor output** — it is the deployable slice. The caller supplies 184 floats:

| Indices | Value |
|---|---|
| 0–79 | Current 2×2 minimum-pooled ray range /12 |
| 80–159 | Previous pooled range /12 |
| 160–162 | Unit goal direction in body FLU |
| 163 | Goal distance /10, capped 1.5 |
| 164–166 | Body linear velocity /4 |
| 167–169 | Body angular velocity /4 |
| 170–172 | World-up direction in body frame |
| 173–176 | Previous applied navigation fractions |
| 177 | Sensor age in seconds |
| 178–180 | Body reference-position error ×2, clipped ±1 |
| 181–183 | Geometry prior XYZ latents from `guidance.hpp` |

`nav_guidance_memory` prepares the prior from range data, capture poses, current estimated pose, goal and velocity. Use the same 16×20 ray directions from `wcamera`, 12 m range limit, 2×2 pooling and pose/depth ring semantics as the simulator. Depth values are distance along a ray, not camera-axis Z depth. Missing depth uses 12 m. Pose records hold xyz then row-major body-to-world rotation. This preparation is part of deployment and is excluded from actor-only latency.

The binary stores a versioned little-endian header, 12104 FP32 actor parameters and a source-checkpoint hash. Output velocity is capped at the recorded 1.5 m/s intent and yaw rate at 0.5 rad/s. RAPTOR owns motor control. Its adapter must integrate the velocity reference at 100 Hz and maintain recurrence. **An actual sensor front-end and flight-controller connection still need integration and validation.**

Use the packaged policy without a training checkpoint:

```sh
./build/metal_nav_guided eval-policy assets/navigation.bin 8 800001
./build/metal_nav_guided policy-bench assets/navigation.bin
./build/metal_nav_guided export results/guided-table-memory.bin.best assets/navigation.bin
```

`policy-bench` reports batch-1 CPU actor time only. Export parity is checked against the training actor. On the measured M3 build, the file is 48,496 bytes, actor parity error is zero over eight probes, and actor-only inference is about 5.28 microseconds. Sensor processing, geometry preparation and RAPTOR are excluded.

The actor sees two pooled range frames, goal/ego/control context and 3 geometry prior latents. A short depth/estimated-pose ring supplies local memory so the vehicle does not return into geometry that has left the forward sensor view. Native RAPTOR/physics runs at 100 Hz; navigation/depth at 20 Hz.

---

## Validation and measurement

```sh
./build/metal_nav bench-depth     # ray-range scaling ladder
./build/metal_nav bench-raptor    # batched RAPTOR throughput
./build/metal_nav bench-loop      # closed-loop throughput
./build/metal_nav gpu-bench 2048 3 1
./build/metal_nav cpu-bench 3 2048 1
./build/metal_nav profile         # hardware encoder timestamps on M3
```

`test` checks independent geometry, official RAPTOR outputs, official L2F fixtures, the PX4 observation transform, PPO/GAE/gradients/Adam and an integrated 160-native-tick CPU/GPU trajectory. `profile` uses hardware encoder timestamps (M3 supports encoder-stage sampling; dispatch-boundary sampling is unsupported). `METAL_NAV_SCALAR_ACTOR=1` selects the scalar GPU actor reference for A/B comparison. CPU benchmarks use Accelerate SGEMM and GCD running the *same* simulator and PPO work.

---

## Repository map

| File | What it holds |
|---|---|
| [GOAL.md](GOAL.md) | The outcome specification and execution contract — what "done" means and why each design choice was made |
| [BENCHMARKS.md](BENCHMARKS.md) | Every measured number: machine, commit, exact command, configuration, result, and which earlier measurements were discarded as invalid |
| [RESEARCH_LOG.md](RESEARCH_LOG.md) | Hypothesis → experiment → result → conclusion, **including rejected and failed hypotheses** |
| [M3_RESEARCH.md](M3_RESEARCH.md) | Apple M3 / Apple GPU family 9 feature research with cited Apple sources and a ranked experiment list |
| `main.mm` | Objective-C++ host: Metal device/queue/pipeline setup, test harnesses, benchmarks, training orchestration |
| `*.metal` | GPU kernels — physics, depth, RAPTOR, PPO forward/backward, geometry memory |
| `*.hpp` | Portable C++ reference implementations and shared types (also used as shared CPU/GPU source) |
| `deployment.hpp` | CPU-only deployable actor interface |
| `evaluation.py` | The 28-configuration evaluation matrix → CSV |
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
