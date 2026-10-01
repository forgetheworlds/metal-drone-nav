## Current working objective — updated 2026-10-01

Build on the verified Metal-native drone navigation engine in /Users/muadhsambul/RL, following GOAL.md, nextphase1.md and CODE_DIRECTION.md. Establish and improve a fast, robust local depth-navigation policy that maps goal/waypoint, perception and ego state to [vx, vy, vz, yaw_rate], then uses the actual RAPTOR controller to drive motors. Identify generalization limits through richer seeded 3D tasks, failure analysis and challenge-focused PPO training; test alternative methods such as GR2PO only through justified controlled comparisons. Validate the full stack in independent Webots physics, sensors, actuators and collisions, preserve genuinely held-out task splits, and measure speed, reliability, dynamics and sensing robustness. Preserve strong baselines and produce reproducible benchmarks, plots, rendered worlds and recorded videos that show both improvements and failures. Keep the specialized Metal hot path and verified RAPTOR/L2F/PPO behavior; follow the coding direction, use only Luna subagents and no skills.

The original specification below remains the technical baseline. The next-phase brief and coding direction extend it; this update does not claim completion.

---

# GOAL.md — Metal-Native High-Throughput Drone Navigation Research Engine

**Status:** Architectural specification and autonomous-agent execution contract  
**Date:** 2026-10-01  
**Primary target:** Apple Silicon + raw Metal/Metal Shading Language  
**Purpose:** Give an autonomous coding agent enough outcome, evidence, interface, research, and execution context to work productively for a long uninterrupted session without re-deriving the project from chat history.

---

## 0. Read this first

This project is **not** “port an RL framework to Metal.”

It is:

> **Build the smallest, fastest, evidence-driven Metal-native training system needed to train a local drone navigation/evasion policy at very high throughput, using a proven learned quadrotor control policy (RAPTOR) underneath it.**

The system should be aggressively specialized around this one research outcome. Generality is a cost and must justify itself.

The core principle is:

> **Preserve validated algorithmic behavior; remove abstraction, data movement, rendering, framework overhead, and functionality that does not contribute to the final outcome.**

Do not optimize for impressive demos or synthetic steps/second alone. Optimize for **time to a policy that succeeds on held-out navigation tasks**, while preserving correctness and useful fidelity.

---

# 1. Outcome engineering contract

## 1.1 Final real-world outcome

A quadrotor should be able to move quickly through previously unseen, cluttered environments while using onboard local perception to:

- travel toward a requested local/global goal;
- avoid static geometry such as walls, counters, tables, poles, doorways, ceilings, and clutter;
- choose routes above, below, or around obstacles when free space allows;
- vary its speed according to environmental complexity rather than flying at one fixed speed;
- react to moving collision threats quickly enough to evade them;
- continue useful navigation under wind and other disturbances;
- remain stable because low-level flight is handled by RAPTOR rather than relearned by the navigation policy;
- generalize to environments and obstacle layouts not seen during training;
- operate with low closed-loop latency from environmental change to changed motor response.

The intended deployed stack is conceptually:

```text
stereo cameras / depth sensor front-end
            ↓
compact local geometry representation
            +
ego state / IMU-derived state
            +
relative goal
            +
temporal context
            ↓
learned navigation / evasion policy
            ↓
high-level local motion reference
            ↓
RAPTOR trajectory-tracking adapter
            ↓
RAPTOR learned motor-control policy
            ↓
4 motor commands
```

The **navigation policy decides where/how to move locally**. RAPTOR decides how to make the aircraft execute that intent at motor level.

---

## 1.2 Research outcome

The research system should make meaningful navigation-policy experiments cheap enough that an autonomous research agent can test multiple hypotheses per working session rather than waiting hours for every idea.

The true research objective is:

> **Maximize validated policy-improvement iterations per unit wall time.**

Secondary performance metrics such as environment steps/s and GPU occupancy are useful only insofar as they improve that objective.

---

## 1.3 Engineering outcome

Build a purpose-built Apple-Silicon execution engine with:

- no PyTorch in the hot path;
- no TensorFlow in the hot path;
- no MLX in the hot path;
- no generic tensor runtime unless a measured result proves that a specific Apple primitive is faster and its cost is justified;
- no Python in the simulation/training hot loop;
- raw Metal compute for the scalable workload;
- static/precomputed shapes and memory layouts;
- no dynamic allocation in the steady-state training loop;
- simulation, depth generation, policy inference, rollout generation, and ideally PPO updates GPU-resident;
- CPU/host involvement primarily for setup, experiment orchestration, checkpointing, logging, and occasional evaluation.

Python is allowed for cold-path scripts, plotting, offline result analysis, and research automation.

---

## 1.4 Radical outcome-engineering mandate

The **outcomes and non-goals in this document are strict. The implementation is not.**

Do not confuse a specific outcome with a predetermined architecture. RAPTOR, RLtools/L2F, PPO, the proposed observation format, kernel boundaries, buffer layouts, and every other implementation detail are starting points or evidence sources. They are not sacred. Preserve them only while they remain the best measured path to the stated outcome.

This project should be technically aggressive. If a large speedup, latency reduction, memory reduction, or learning-efficiency gain appears physically achievable, investigate it even if there is no existing Apple-Silicon implementation, no paper has packaged it this way, or the normal framework ecosystem does something else. **Lack of precedent is not evidence of impossibility.** Distinguish:

- impossible because of mathematics, hardware limits, API constraints, or measured behavior;
- difficult because tooling is immature;
- uncommon because general frameworks optimize for portability/generality;
- simply not attempted yet.

Only the first category is a reason to stop. Verify purported limits against current Apple/Metal documentation, hardware behavior, source code, papers, and experiments before treating them as constraints.

### Be willing to rip things out

Aggressively delete or bypass:

- generic abstractions;
- compatibility layers;
- object hierarchies;
- redundant memory representations;
- host-side orchestration in the hot loop;
- framework conventions inherited from CUDA/PyTorch ecosystems;
- kernels whose work can be fused safely;
- intermediate buffers that can be eliminated;
- features that exist only because upstream software serves many unrelated users.

A clean rewrite of a narrow subsystem is preferable to preserving an upstream abstraction when the rewrite creates a material measurable advantage.

### Be willing to build custom solutions

Custom Metal/MSL kernels, fixed-shape neural operators, specialized memory allocators, purpose-built depth generation, hand-derived backward passes, custom fused PPO kernels, unusual scheduling strategies, or compile-time generated code are all acceptable when they materially improve the objective.

Do not write custom code for aesthetic purity. Use a standard Apple primitive when it is faster. But if a custom implementation has credible potential for a significant end-to-end gain, test it. The burden is measurement, not precedent.

A useful heuristic:

> If an optimization could plausibly move end-to-end time-to-threshold, experiment throughput, or deployed reaction latency by a meaningful amount, it deserves a measured probe.

### Failure is allowed; unmeasured wandering is not

Do not be afraid to fail. Radical probes are expected. Isolate them with commits/branches or small experimental implementations, define the success metric before the probe, measure, then keep or discard.

A failed custom kernel that disproves a hypothesis is useful. A week of architecture churn without a benchmark is not.

Do not protect sunk cost. If evidence shows a subsystem is unnecessary or a different decomposition is much faster, replace it.

### Think in second-order effects

Never evaluate a local optimization only by its microbenchmark. Before keeping it, ask what it does to the whole research/deployment system. Examples:

- more parallel environments may increase steps/s but worsen sample efficiency or time-to-solution;
- lower precision may speed kernels but destabilize PPO or alter RAPTOR behavior;
- kernel fusion may reduce dispatch overhead but increase register pressure or make the next kernel bandwidth-bound;
- a compressed depth representation may speed training but reduce dynamic-obstacle performance;
- moving work onto GPU may remove copies but increase memory pressure and reduce the number of concurrent experiments;
- a larger policy may improve reward but hurt deployed reaction latency;
- a more realistic simulator may improve transfer but reduce experiment throughput enough to slow overall progress;
- a faster local navigator may expose perception latency as the next dominant bottleneck.

Optimize the **system-level outcome**, not an isolated number. Record important second-order effects in `docs/DECISIONS.md` and `docs/BENCHMARKS.md`.

### Use the web and primary sources aggressively

The autonomous agent is expected to research while building. Use current web search, official documentation, research papers, source repositories, issues, discussions, profiler documentation, and hardware/API references whenever they can resolve an uncertainty or reveal a better implementation.

Prefer sources in roughly this order when facts conflict:

1. current official Metal/Apple developer documentation and headers;
2. current upstream source code at a pinned commit;
3. primary research papers and supplementary material;
4. authors' technical notes/issues/discussions;
5. high-quality secondary explanations.

Do not assume this specification or model memory is current on Metal capabilities. Verify current API/hardware claims. When borrowing an optimization from CUDA, another GPU architecture, a paper, or a different robotics stack, reason from the underlying hardware/dataflow rather than mechanically translating syntax.

Research should be question-driven. Examples:

- What exactly is RAPTOR's trajectory/reference transformation?
- Which Metal primitive is actually fastest for this fixed matrix shape?
- Can argument buffers, indirect command buffers, function constants, simdgroup operations, or fused kernels remove measurable overhead here?
- What is the real occupancy/register/bandwidth bottleneck on this Apple GPU?
- Which observation/action representation has the strongest evidence for high-speed local navigation?

Capture useful findings in `docs/RESEARCH_LOG.md` with links and, when applicable, pinned commit hashes.

### Physical possibility outranks convention

Do not reject an approach because "Metal RL does not exist," "people normally use PyTorch," "this is usually done in Isaac/Unity," or "there is no library for that." Those are ecosystem facts, not physical constraints.

When the outcome requires a capability and no suitable implementation exists, derive the minimum required implementation from first principles, validate it against references, and build it.

The project should be conservative about **claims** and radical about **means**.

---

# 2. Non-goals

Do **not** spend the initial project building:

- a general-purpose RL framework;
- a general-purpose tensor library;
- a generic autograd engine;
- a generic physics engine;
- a game engine;
- photorealistic RGB rendering;
- Unity/Unreal integration;
- a generic scene graph;
- support for arbitrary neural architectures;
- multiple unrelated RL algorithms;
- CUDA compatibility;
- multi-GPU support;
- distributed training;
- a UI;
- an LLM in the runtime flight loop;
- another low-level quadrotor controller unless RAPTOR is experimentally shown to be inadequate.

Do not reproduce all of RLtools. Use it as a validated reference and source of algorithms/behavior.

---

# 3. Research-grounded decisions

The following decisions should be considered the current design baseline unless implementation evidence contradicts them.

## 3.1 Use RAPTOR as the low-level learned flight policy

RAPTOR is a small recurrent foundation policy for quadrotor control. The official implementation consumes quadrotor state including position, rotation matrix, linear velocity, angular velocity, and previous motor action, and outputs four normalized motor commands. Its example simulator advances at 10 ms per step. It is specifically designed for broad vehicle transfer and embedded deployment.

**Decision:** Do not ask the new navigation policy to learn motor stabilization from scratch.

The training simulator must include a **batched RAPTOR forward pass** so thousands of parallel environments experience the actual learned low-level policy inside their closed loop.

RAPTOR weights are shared; recurrent state is per environment.

**Important:** Before implementing the navigation-to-RAPTOR adapter, inspect the official RAPTOR and PX4 integration source and reproduce its trajectory-tracking transformation exactly. Do not guess from prose.

Reference:
- https://github.com/rl-tools/raptor
- Science Robotics 2026, DOI: 10.1126/scirobotics.aec1481

---

## 3.2 Start from L2F/RLtools dynamics and performance ideas

RLtools/Learning-to-Fly demonstrated unusually fast quadrotor RL by tightly integrating simulation and learning in C++ and specializing heavily around fixed-size continuous-control problems. It is an excellent correctness and systems reference.

**Decision:** Use the L2F dynamics model and implementation as the first physics reference. Port/re-express only what the closed-loop navigation simulator requires.

Do not assume the CPU/Accelerate or CUDA execution architecture is optimal for Apple Silicon. Preserve semantics, not implementation baggage.

References:
- https://github.com/rl-tools/rl-tools
- https://www.jmlr.org/papers/v25/24-0248.html
- Learning to Fly in Seconds, arXiv:2311.13081

---

## 3.3 The navigation RL hot loop should not train from raw stereo RGB initially

Recent navigation work supports training on compact geometry/depth representations and dealing with the perception domain gap separately.

Depth Transfer (2025) trains navigation from simulated ground-truth depth in a learned latent and adapts real stereo depth into that representation. It reports a large degradation when naïvely swapping perfect simulated depth for stereo depth, and much better transfer after domain adaptation.

**Decision:** The simulation hot loop produces synthetic local depth/geometry directly. It does **not** generate left/right RGB frames and run stereo matching millions of times.

Deployment can use a stereo front-end that outputs depth/disparity or a compatible latent representation.

References:
- Depth Transfer: Learning to See Like a Simulator for Real-World Drone Navigation, IEEE RA-L 2025, DOI: 10.1109/LRA.2025.3617729
- https://research.tudelft.nl/en/publications/depth-transfer-learning-to-see-like-a-simulator-for-real-world-dr/

---

## 3.4 Temporal information is required for robust dynamic-obstacle behavior

A single depth frame can indicate that something is close, but it cannot reliably tell whether an obstacle is approaching rapidly, moving laterally, or static. MAVRL and newer dynamic-obstacle work show value from memory/temporal observations.

**Decision:** The navigation observation must contain temporal information.

V0 may use a simple two-frame or short frame stack because it is much easier to train with a hard-coded Metal PPO implementation. A recurrent policy (small GRU) is a later high-priority experiment and may become the final architecture if it materially improves dynamic obstacle handling or partial observability.

References:
- MAVRL, IEEE RA-L 2025, DOI: 10.1109/LRA.2024.3522778
- https://research.tudelft.nl/en/publications/mavrl-learn-to-fly-in-cluttered-environments-with-varying-speed/

---

## 3.5 Speed should be a learned/adaptive navigation variable

MAVRL specifically shows that fixed navigation speed is inferior to adapting speed to clutter/complexity.

**Decision:** Do not hard-code one forward velocity. The action space/reward must let the policy choose aggressive motion in open space and slower motion in difficult geometry.

---

## 3.6 Navigation output should remain above motor level

Relevant systems use several higher-level abstractions:

- Agile Autonomy / Learning High-Speed Flight in the Wild predicts short-horizon collision-free trajectories from onboard sensing.
- Deep Drone Racing predicts waypoint direction and desired speed and lets a trajectory/controller layer execute it.
- MAVRL uses a navigation layer above low-level control.
- RAPTOR/PX4 exposes trajectory-setpoint integration.

**Decision:** The new policy should not output motor commands. RAPTOR already owns that problem.

**V0 action candidate:** body-frame desired velocity vector plus yaw-rate or heading intent, converted by a deterministic adapter into a short-horizon position/velocity trajectory reference for RAPTOR.

If inspection of RAPTOR’s official trajectory interface makes a local waypoint + speed representation cleaner, prefer that. The invariant is that the action remains a **local navigation/motion command**, not motor actuation.

References:
- Learning High-Speed Flight in the Wild, Science Robotics 2021, DOI: 10.1126/scirobotics.abg5810
- RAPTOR official repository and PX4 integration

---

## 3.7 Use an asymmetric/privileged critic during training

Low-level L2F and high-performance drone RL systems have successfully used privileged simulation state during training while restricting the deployed actor to observations available on the real vehicle.

**Decision:** The actor sees only deployable local perception/state. The critic may see privileged simulator information such as exact state and compact obstacle descriptors if doing so improves training efficiency.

The deployed actor must never depend on privileged information.

---

## 3.8 PPO is the first navigation-training algorithm, not a permanent dogma

PPO is well-supported in highly parallel aerial RL and maps naturally onto large batched rollouts. RLtools already provides validated PPO semantics that can be studied.

**Decision:** Implement one fixed PPO training path first:

- rollout collection;
- GAE;
- clipped policy objective;
- value loss;
- entropy term;
- advantage normalization;
- fixed optimizer (Adam initially);
- fixed network architecture.

Do not build an algorithm abstraction layer.

If PPO proves too sample-inefficient, a privileged-expert / imitation stage inspired by Agile Autonomy is a later experiment. It is not V0 scope.

---

## 3.9 Evaluation is independent from training reward

AvoidBench demonstrates the value of multiple independent obstacle-avoidance metrics rather than a single training objective.

**Decision:** Never report “reward increased” as sufficient evidence of better navigation.

Reference:
- AvoidBench, ICRA 2023, DOI: 10.1109/ICRA48891.2023.10161097
- https://github.com/tudelft/AvoidBench

---

# 4. Core system architecture

```text
                HOST / RESEARCH ORCHESTRATION
        C++ / Objective-C++ + scripts outside hot path
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                       METAL GPU                              │
│                                                              │
│  N parallel environments                                     │
│                                                              │
│  world dynamics ──► collision ──► synthetic depth            │
│       ▲                                  │                   │
│       │                                  ▼                   │
│       │                           sensor timing/noise          │
│       │                                  │                   │
│       │                                  ▼                   │
│       │                         navigation policy actor        │
│       │                                  │                   │
│       │                           local motion command         │
│       │                                  │                   │
│       │                                  ▼                   │
│       │                       trajectory/RAPTOR adapter        │
│       │                                  │                   │
│       │                                  ▼                   │
│       └──────────────────────── batched RAPTOR ──► motors     │
│                                                              │
│                 rewards / termination / reset                 │
│                              │                               │
│                              ▼                               │
│                       rollout storage                         │
│                              │                               │
│                              ▼                               │
│                  GAE → PPO → backward → Adam                 │
│                              │                               │
│                              └──► updated nav policy          │
└──────────────────────────────────────────────────────────────┘
                           │
                           ▼
                METRICS / CHECKPOINT / EVAL
```

No RGB renderer is in the initial hot loop.

---

# 5. Policy contract

## 5.1 Actor observation: V0

V0 should use the smallest deployable observation that lets us test the full architecture.

Required information classes:

1. **Local depth/geometry**
   - low-resolution egocentric depth/range grid;
   - compile-time dimensions;
   - begin with a small resolution such as `16×20` or `32×24`;
   - represent distances in a normalized/clipped format;
   - support invalid/dropout values later.

2. **Temporal geometry**
   - at minimum current + previous depth observation, or current depth + depth delta;
   - keep the representation fixed-size.

3. **Goal**
   - relative goal direction in body frame;
   - relative goal distance;
   - optionally relative vertical component is naturally included in 3-D direction.

4. **Ego state**
   - body-frame linear velocity;
   - body-frame angular velocity;
   - gravity/body-up orientation representation or equivalent compact attitude signal;
   - previous navigation action.

Do not include exact obstacle positions in the actor observation.

## 5.2 Critic observation

The critic may additionally receive:

- exact position/velocity/orientation;
- exact relative goal;
- compact nearest-obstacle distances/velocities;
- wind/disturbance state;
- simulator timing state.

Keep privileged critic input compact. The purpose is training efficiency, not building another giant network.

## 5.3 Navigation action

Start with one narrow action contract and keep the adapter separate.

Preferred first candidate:

```text
NavigationCommand {
    desired_body_velocity_xyz
    desired_yaw_rate_or_heading_delta
}
```

The trajectory adapter converts this to the reference representation expected by RAPTOR/PX4 semantics.

Alternative first candidate if official RAPTOR source suggests it is cleaner:

```text
NavigationCommand {
    local_target_delta_xyz
    desired_speed
    desired_yaw_or_yaw_rate
}
```

Do not implement both before one end-to-end baseline works.

## 5.4 Policy frequency

Do not hard-code a literature-derived frequency as truth.

Requirements:

- make navigation policy rate independently configurable;
- make depth update rate independently configurable;
- RAPTOR native recurrent update timing must match its validated semantics;
- physics may use substeps if needed;
- model delays explicitly.

Initial navigation/depth rates should be selected conservatively (e.g. tens of Hz) and then benchmarked upward.

The final design should aim for low enough reaction latency that speed is limited by vehicle physics/environment, not avoidable software delay.

---

# 6. Simulator contract

## 6.1 World representation

V0 worlds must be generated from very cheap geometric primitives.

Required primitive classes:

- axis-aligned or oriented boxes;
- planes/walls/floor/ceiling;
- cylinders/capsules or equivalent pole/tree primitives;
- spheres for cheap moving-object tests.

Every primitive needs only data required for:

- ray/depth intersection;
- collision testing;
- optional motion over time.

Do not build a general mesh renderer first.

## 6.2 Environment families

Training and evaluation generators should create at least these scenario families:

### Indoor
- walls;
- corridors;
- doorways;
- counters/tables as box geometry;
- vertical clearance decisions;
- tight turns;
- mixed clutter.

### Sparse/open
- widely spaced obstacles;
- long high-speed traversals.

### Pole/forest-like
- vertical cylinders/capsules;
- variable density;
- gaps and slalom-like structures.

### Dense clutter
- multiple primitive types;
- narrow feasible paths;
- local minima / dead ends where possible.

### Dynamic threats
- crossing objects;
- head-on approaching objects;
- lateral sweeps;
- vertical sweeps;
- converging multiple obstacles;
- sudden obstacle motion after the episode begins.

### Disturbance
- constant wind;
- gusts;
- directional changes;
- parameter randomization later.

## 6.3 Dynamic-object model

Do not model a human.

Model collision-relevant geometry and motion:

```text
position
velocity
optional acceleration / scripted trajectory
primitive dimensions
```

A moving box/capsule/sphere is sufficient to train the geometric behavior “threat is on collision course → evade”.

## 6.4 Physics

Use L2F/RAPTOR-compatible quadrotor dynamics as the correctness baseline.

V0 priorities:

- correct rigid-body state evolution;
- thrust/torque behavior consistent with the reference;
- enough inertia/momentum fidelity that a navigation policy learns that high speed limits turning/stopping;
- RAPTOR in the loop;
- wind/disturbance input.

Do not add aerodynamic detail unless an ablation demonstrates that its absence changes navigation transfer/evaluation materially.

## 6.5 Collision

Start with a conservative collision body approximation such as a sphere or ellipsoid around the vehicle.

Collision checks must be independent of depth generation so evaluation cannot “cheat” through the observation model.

---

# 7. Synthetic depth contract

## 7.1 V0 depth generation

Generate depth directly from geometry using GPU ray intersection.

Do not render color.

For each depth sample:

```text
camera origin + camera model + ray
       ↓
nearest geometry intersection
       ↓
distance Z
       ↓
clamp/normalize
       ↓
policy observation
```

Depth should run at sensor/navigation rate, not necessarily every physics substep.

## 7.2 Performance strategy

Implement in stages:

1. correctness-first naive primitive traversal;
2. profile;
3. if ray traversal dominates, implement a simple spatial acceleration structure appropriate to procedurally generated worlds (uniform grid/BVH/etc.);
4. re-profile;
5. retain only optimizations that improve end-to-end training.

Do not begin with an elaborate BVH before measuring the naive kernel.

## 7.3 Sensor corruption

V0 can train with nearly perfect depth to establish learning.

Then add independently controllable corruption dimensions:

- distance-dependent depth noise;
- quantization;
- max range;
- missing pixels/dropout;
- bad edge measurements;
- frame drops;
- frame latency/jitter;
- camera FOV variation;
- stereo-like confidence failures.

Do not invent a complicated “realistic” noise model before hardware data exists.

The long-term preferred approach is empirical: measure the real stereo front-end and fit corruption/residual statistics or adapt the real representation toward the simulator latent, as motivated by Depth Transfer and Swift-style residual modeling.

---

# 8. Timing and reaction-latency contract

Closed-loop speed is an outcome, not a later optimization.

Model separate clocks for:

- physics integration;
- synthetic depth updates;
- navigation policy execution;
- RAPTOR policy updates;
- command/application delay;
- motor/vehicle response.

The simulator must be able to insert latency and jitter between these layers.

Measure at least:

- sensor observation age at policy use;
- navigation inference time;
- navigation-update → RAPTOR-command delay;
- total simulated event → changed motor-command delay;
- total event → meaningful vehicle-acceleration delay in simulation.

Dynamic-obstacle evaluations should vary time-to-collision and obstacle velocity, not just obstacle position.

---

# 9. Training task and reward

## 9.1 Task

An episode begins from randomized valid state and goal. The actor should reach the goal while avoiding collisions and unnecessary delay.

Task difficulty is controlled by:

- obstacle density;
- gap width;
- start/goal geometry;
- permitted speed;
- dynamic-obstacle frequency/speed;
- wind;
- sensor corruption;
- latency.

## 9.2 Reward principles

Keep reward small and interpretable.

Candidate terms:

- positive goal progress;
- terminal goal success bonus;
- large collision penalty;
- small time penalty or equivalent incentive for efficiency;
- optional action-change/jerk penalty if needed for pathological commands;
- optional proximity/risk shaping only if sparse collision feedback is insufficient.

Do not bake evaluation metrics wholesale into one giant reward.

The policy should learn speed selection from the tradeoff between progress/time and collision risk rather than from a fixed commanded speed.

## 9.3 Curriculum

Curriculum is permitted but must earn its complexity.

A reasonable sequence if direct mixed-task PPO fails or is slow:

1. goal reaching with no obstacles;
2. sparse static obstacles;
3. dense/narrow static obstacles;
4. higher speed;
5. dynamic obstacles;
6. sensor delay/noise;
7. wind/parameter variation;
8. mixed distribution.

If one mixed distribution trains as fast or better, delete the curriculum.

---

# 10. PPO implementation contract

The final hot path should be purpose-built, not generic.

Required components only:

1. fixed actor forward;
2. fixed critic forward;
3. action sampling/squashing if used;
4. rollout buffers;
5. GAE;
6. PPO ratio/clipping loss;
7. value loss;
8. entropy term;
9. backward pass for the exact chosen network;
10. gradient accumulation/reduction;
11. Adam update;
12. checkpoint/parameter export.

No generic graph/autograd API.

## 10.1 Numerical strategy

Start in FP32 for correctness.

Only introduce FP16/BF16/mixed precision after:

- CPU/reference parity is established;
- learning curves are stable;
- benchmark evidence shows the conversion is useful.

## 10.2 Actor architecture: V0

Favor implementation simplicity first.

V0 may flatten a very low-resolution depth/history grid into a fixed MLP so the complete Metal PPO path can work before custom convolution/backprop is added.

Example family, not a mandatory exact shape:

```text
[depth_t, depth_t-1, goal, ego state, prev action]
       ↓
fixed dense layer
       ↓
activation
       ↓
fixed dense layer
       ↓
actor head + value head
```

After the baseline works, test:

- tiny convolutional encoder;
- pooled depth sectors;
- small GRU/recurrent latent;
- depth latent trained offline.

Do not build all of these before baseline learning is demonstrated.

---

# 11. Metal-native implementation constraints

## 11.1 Language/tooling

Preferred project structure:

- C++ for portable scalar/reference math and core data types;
- Objective-C++ (`.mm`) or minimal Swift only where required to own Metal device/queue/pipelines;
- MSL (`.metal`) for GPU kernels;
- CMake/Xcode tooling as appropriate;
- small Python scripts only for plotting/offline analysis if useful.

Avoid introducing a large dependency solely for convenience.

## 11.2 Memory

At startup:

- calculate all maximum buffer sizes;
- allocate once;
- keep stable offsets/layouts;
- use shared/unified-memory capabilities intelligently but remember synchronization is still a cost;
- avoid host reads/writes inside every sim step.

Prefer struct-of-arrays or another layout only after considering actual access patterns. Measure rather than assuming.

Shared policy weights; per-environment:

- dynamics state;
- RAPTOR recurrent state;
- navigation temporal state;
- obstacle/world state or references;
- RNG state;
- reward/termination state;
- rollout storage as needed.

## 11.3 Dispatch/synchronization

Avoid the pattern:

```text
launch kernel
wait CPU
launch kernel
wait CPU
...
```

The GPU should execute long chunks of useful work with minimal host synchronization.

Explore kernel fusion only after correctness:

Potential fusion candidates:

- physics + simple reward/termination;
- depth normalization + actor input packing;
- PPO elementwise loss operations;
- optimizer elementwise operations.

Do not fuse components whose separation is useful for correctness/debugging until tests exist.

## 11.4 Static specialization

Exploit the fact that this system has fixed families of:

- sensor resolution;
- actor shape;
- critic shape;
- action dimension;
- RAPTOR shape;
- rollout length;
- environment layout.

Compile-time specialization is encouraged.

Do not build runtime dynamic-shape machinery.

## 11.5 RNG

Use deterministic per-environment RNG with an equivalent CPU implementation.

Given a global seed + environment ID + step/generation, runs should be reproducible enough to compare changes.

---

# 12. Correctness and parity requirements

Performance work cannot outrun validation.

Create scalar/reference implementations for critical calculations where practical.

## 12.1 RAPTOR parity

For fixed sampled observations and recurrent states:

```text
official RAPTOR reference
vs
Metal batched RAPTOR
```

Compare outputs and recurrent state updates within documented floating-point tolerance.

Do this before using RAPTOR inside training.

## 12.2 Physics parity

For fixed parameters/state/action and deterministic timesteps:

```text
L2F/reference step
vs
Metal step
```

Compare state trajectories over both one step and multiple steps.

## 12.3 Ray/depth parity

CPU geometric ray tests should verify edge cases:

- ray misses;
- ray starts inside/near primitive;
- grazing intersections;
- closest of multiple objects;
- moving primitive update;
- max-range clamp.

## 12.4 PPO parity

Build a tiny deterministic CPU reference for one PPO update on a tiny synthetic batch.

Compare:

- GAE/returns;
- log-prob ratio;
- clipped loss;
- value loss;
- gradients for selected weights;
- Adam update;
- resulting weights.

Exact bitwise equality is not required across CPU/GPU, but divergence must be bounded and explained.

## 12.5 Learning parity

A kernel can be mathematically close while breaking learning.

Maintain at least one tiny deterministic/easy navigation task that should learn reliably. Use it as a smoke test after major optimizer/network changes.

---

# 13. Evaluation suite

Training reward is not the score.

Maintain held-out seeds/worlds that training never sees.

Report at least:

## Navigation quality
- success rate;
- collision rate;
- mission progress on failure;
- time to goal;
- mean speed;
- peak speed;
- path length;
- path efficiency relative to a geometric baseline where calculable;
- minimum clearance;
- stuck/timeout rate.

## Dynamic avoidance
- success against crossing obstacles;
- success against approaching obstacles;
- success as a function of obstacle speed;
- success as a function of initial time-to-collision;
- near-miss/minimum separation.

## Robustness
- wind strength sweep;
- sensor-latency sweep;
- frame-drop sweep;
- depth-noise sweep;
- dynamics-parameter sweep;
- unseen obstacle-layout families.

## Computational
- end-to-end environment transitions/s;
- useful rollouts/s;
- training updates/s;
- wall-clock time to fixed evaluation thresholds;
- Metal kernel timing breakdown;
- memory footprint;
- host↔GPU synchronization frequency;
- actor inference latency at deployment-like batch size 1;
- batched actor/RAPTOR inference throughput.

The primary training-performance number is **time to reach a fixed held-out evaluation threshold**, not peak synthetic steps/s.

---

# 14. Research questions still intentionally open

These should be answered by controlled experiments, not preference.

## Q1. Minimum depth resolution

Compare at least a small set such as:

- very coarse (`16×12` or similar);
- `16×20`;
- `32×24`;
- `64×48` if compute permits.

Decision rule: choose the smallest representation whose held-out success/speed is within an acceptable margin of the best larger representation.

## Q2. Depth vs inverse-depth/disparity-like representation

Stereo disparity is related to inverse depth, so a normalized inverse-depth representation may be easier for local collision geometry.

Test rather than assume.

## Q3. Temporal mechanism

Compare:

1. single frame;
2. two-frame/depth-delta;
3. short frame stack;
4. small GRU.

Dynamic-obstacle tests are the deciding metric.

## Q4. Navigation action abstraction

After the first end-to-end action interface works, compare only if needed:

- desired body velocity + yaw;
- local waypoint + speed;
- desired acceleration/short trajectory.

Judge on learning speed, collision performance, aggressive maneuverability, and RAPTOR tracking feasibility.

## Q5. Sensor corruption fidelity

Start simple. Add each corruption independently and measure whether it improves robustness without unacceptable training cost.

## Q6. Physics fidelity

Use L2F baseline. Add complexity only where held-out/higher-fidelity comparisons prove it matters.

## Q7. Actor architecture

Start with fixed MLP/cheap depth representation. Compare tiny CNN/latent/recurrent designs only after the trainer is trustworthy.

## Q8. Curriculum

Compare direct mixed training against staged difficulty if convergence is poor.

## Q9. Multi-policy experiment batching

Once one shared policy + many envs works, evaluate partitioning the GPU into multiple policy experiments simultaneously.

This is useful only if it increases **research conclusions per hour**, not merely concurrency.

---

# 15. Performance targets

Do not fabricate a required million-env target before measurement.

Establish baselines on the actual machine first.

Required benchmark ladder:

```text
N = 1
N = 32
N = 128
N = 512
N = 2,048
N = 8,192
N = 32,768
...continue while memory and scaling remain useful
```

For each N, measure isolated and combined cost of:

- physics;
- RAPTOR inference;
- collision;
- depth generation;
- nav actor inference;
- rollout bookkeeping;
- PPO update;
- complete train step.

Performance acceptance is relative to valid references:

1. correctness established;
2. GPU path scales with environment count;
3. large-batch navigation workload beats a straightforward CPU/reference implementation materially;
4. optimizations improve time-to-target-performance, not only raw throughput.

A desirable project outcome is a multi-fold end-to-end speedup versus the best reasonable Apple CPU/Accelerate reference for the same navigation training task. Do **not** fake this by lowering sensor resolution, fidelity, PPO work, or evaluation difficulty without labeling the comparison.

---

# 16. Autonomous implementation protocol for Codex

You are authorized to make reversible engineering decisions without asking the user, as long as they serve this specification.

## 16.1 Required working style

Loop:

```text
inspect reference
→ form explicit hypothesis
→ implement smallest testable version
→ validate correctness
→ benchmark
→ profile
→ optimize measured bottleneck
→ revalidate
→ document evidence
```

Do not spend the session writing speculative architecture prose after the spec is understood.

## 16.2 Source archaeology before coding

Read enough of these sources to understand the actual interfaces/algorithms before reproducing them:

1. `rl-tools/raptor`
2. relevant RAPTOR integration code, especially trajectory tracking and recurrent executor semantics
3. `rl-tools/rl-tools` L2F environment/dynamics
4. RLtools PPO implementation only as an algorithm/reference source
5. relevant research papers listed at the end of this document

Do not clone/copy large unrelated framework sections into the final codebase.

Respect licenses. RLtools/RAPTOR are MIT. Treat other repositories primarily as research references unless their license is compatible and copying is intentional.

## 16.3 Repository documentation to maintain

Create/update:

- `README.md` — build/run/benchmark basics;
- `STATUS.md` — what currently works, what fails, next highest-leverage step;
- `DECISIONS.md` — important architectural choices and evidence;
- `BENCHMARKS.md` — machine, commit, configuration, exact commands, results;
- `RESEARCH_LOG.md` — hypothesis → experiment → result → conclusion;
- `docs/` for any deeper notes that are truly needed.

Do not hide failed experiments. Failed hypotheses are useful evidence.

## 16.4 Git discipline

Use small coherent commits after verified milestones.

Examples:

- reference RAPTOR harness;
- Metal buffer/runtime skeleton;
- RAPTOR parity kernel;
- L2F physics parity;
- ray-depth correctness;
- closed-loop batch sim;
- PPO reference update;
- Metal PPO update;
- first learning smoke test;
- first end-to-end benchmark;
- each meaningful performance optimization.

## 16.5 No-performance-claim rule

Never claim an optimization is faster without recording:

- before measurement;
- after measurement;
- exact workload;
- exact environment count;
- precision;
- machine/build mode;
- correctness status.

## 16.6 Stop/escalation rules

If an optimization breaks parity or learning and the cause is unclear:

- revert or isolate it;
- document the failure;
- continue from the last correct baseline.

If a research choice is ambiguous:

- prefer the simplest reversible choice;
- expose the choice as a compile-time/config experiment axis;
- continue.

If full PPO is not reachable in the available execution window:

- prioritize a correct, benchmarked closed-loop Metal simulator with RAPTOR + depth over a half-working trainer;
- leave exact next steps in `STATUS.md`.

A correct fast simulator is a valuable milestone by itself.

---

# 17. Recommended build order

This is priority order, not a promise that every item must be finished in one session.

## Phase 0 — Ground truth

- Build/run official RAPTOR reference locally.
- Build/run relevant L2F/RLtools references.
- Capture exact observation/action shapes, axis conventions, timestep semantics, recurrence behavior, and trajectory-target transformation.
- Record baseline timings.

**Gate:** Reference outputs can be generated deterministically for tests.

## Phase 1 — Minimal Metal runtime

- create Metal device/command queue/pipeline setup;
- compile one trivial compute kernel;
- implement static buffer allocation;
- implement deterministic timing/benchmark harness;
- establish Release build path.

**Gate:** reproducible kernel benchmark and test harness.

## Phase 2 — Batched RAPTOR on Metal

- reproduce exact RAPTOR forward network and recurrent state update;
- one shared parameter set;
- per-environment hidden/recurrent state;
- CPU/reference parity tests;
- batch scaling benchmarks.

**Gate:** Metal RAPTOR tracks official outputs within tolerance.

## Phase 3 — Batched L2F-compatible dynamics

- port minimum required state/dynamics;
- apply RAPTOR motor outputs;
- wind input;
- parity tests over trajectories;
- batch benchmarks.

**Gate:** closed-loop RAPTOR can stabilize/track simple reference in the Metal simulator similarly to reference.

## Phase 4 — Primitive worlds + collision + depth

- procedural worlds;
- cheap primitives;
- collision body;
- low-res ray-depth generation;
- moving obstacles;
- deterministic seeded generation;
- CPU ray/collision tests;
- throughput benchmark.

**Gate:** thousands of environments can generate valid depth/collisions without host stepping.

## Phase 5 — Navigation policy inference + adapter

- implement simplest fixed actor;
- actor observation packing;
- high-level command;
- exact RAPTOR target adapter;
- configurable navigation/depth rates and delay;
- scripted or random policy for pipeline tests.

**Gate:** complete observation → nav command → RAPTOR → dynamics loop runs entirely in batch.

## Phase 6 — Reward/evaluation

- goal progress;
- collision/success termination;
- held-out scene seeds;
- navigation metrics;
- dynamic threat metrics;
- latency metrics.

**Gate:** a hand-scripted/simple controller can be objectively compared to another one.

## Phase 7 — PPO CPU/reference

- tiny fixed actor/critic;
- deterministic toy batch;
- GAE/PPO/Adam scalar reference tests;
- simple toy learning task.

**Gate:** reference PPO learns the smoke-test task.

## Phase 8 — Metal PPO

- fixed-shape forward/backward;
- GAE;
- PPO loss;
- gradient reduction;
- Adam;
- parity tests;
- minimal CPU sync.

**Gate:** Metal PPO matches reference update and learns smoke test.

## Phase 9 — First actual navigation training

Start with easy static worlds and low-res depth/history.

**Gate:** held-out goal-reaching/avoidance success rises materially above random/scripted baselines.

## Phase 10 — Profile and optimize

Only now aggressively:

- fuse kernels;
- change layouts;
- reduce dispatches;
- improve ray acceleration;
- experiment with mixed precision;
- batch multiple policy experiments;
- add recurrence/tiny CNN if justified.

**Gate:** every retained optimization improves measured end-to-end outcome.

---

# 18. First experiments after the system works

Run these in roughly this order because they answer the highest-leverage product questions.

## E1 — Observation resolution

Same policy family, same training budget, same held-out environments.

Compare small depth resolutions.

Output:

- time to threshold success;
- held-out success;
- collision rate;
- speed;
- inference cost.

## E2 — Temporal information

Compare:

- current depth only;
- current + previous;
- current + depth delta;
- GRU later.

Use moving-obstacle evaluation.

## E3 — Action abstraction

Only after one baseline is working.

Compare velocity-like command vs local target+speed if evidence suggests a need.

## E4 — Adaptive speed

Verify the learned policy actually slows in difficult clutter and accelerates in open space.

Compare against fixed-speed ablations.

## E5 — Latency robustness

Train/evaluate across sensor and command delay distributions.

Plot success vs latency and speed.

## E6 — Dynamic avoidance

Sweep obstacle velocity and time-to-collision.

Test whether temporal input materially moves the failure boundary.

## E7 — Wind/disturbance

Measure whether RAPTOR handles most low-level disturbance without extra nav-policy complexity, and where navigation must adapt.

## E8 — Sensor corruption

Introduce one corruption at a time.

Do not combine all corruption until individual effects are understood.

---

# 19. Goodhart / failure modes to guard against

The autonomous agent must actively check for these.

### Fake speedup
Throughput rises because fidelity/work was silently removed.

**Guard:** matched workload benchmark.

### Reward hacking
Reward improves but held-out navigation does not.

**Guard:** independent evaluation suite.

### Simulator exploitation
Policy discovers behavior that relies on perfect depth, zero latency, unrealistic collision, or numerical quirks.

**Guard:** corruption/latency sweeps and higher-fidelity later validation.

### Over-parallelization
More environments increase steps/s but worsen wall-clock learning because updates become stale/inefficient.

**Guard:** time-to-evaluation-threshold metric.

### Unnecessary framework recreation
Agent spends time building abstractions for hypothetical future use.

**Guard:** every abstraction must serve an immediate required path.

### Premature neural complexity
Agent builds CNN/GRU/autograd before proving the simulator and PPO pipeline.

**Guard:** simplest fixed network first.

### Overfitting to training generators
Policy succeeds on generator seeds but not different geometry families.

**Guard:** held-out generator families and parameter ranges.

### RAPTOR mismatch
Navigation training uses an approximation instead of the actual lower-level learned controller.

**Guard:** keep batched RAPTOR in the loop after Phase 2.

### Unrealistic instantaneous sensing
Policy depends on observations with zero age.

**Guard:** explicit sensor/update clocks and latency.

---

# 20. Definition of meaningful success

A first serious milestone is reached when all of the following are true:

1. RAPTOR inference is reproduced in Metal with validated parity.
2. L2F-compatible quadrotor dynamics run in large batches on Metal.
3. Primitive environments generate collision-independent low-res depth directly on GPU.
4. The full navigation→RAPTOR→physics loop runs without per-step CPU synchronization.
5. A purpose-built PPO implementation can train a small navigation policy.
6. The policy reaches held-out goals and avoids static obstacles at a rate clearly above random/scripted trivial baselines.
7. The benchmark suite reports both training quality and throughput.
8. At least one profile-guided optimization produces a measured end-to-end improvement without reducing correctness.

A stronger milestone adds:

- adaptive speed;
- temporal/dynamic obstacle avoidance;
- latency robustness;
- wind robustness;
- sensor corruption;
- multi-policy experiment batching;
- a compact deployable navigation policy.

---

# 21. Research source map

These are starting points, not a requirement to reproduce every method.

## Low-level control / fast RL

### RAPTOR: A Foundation Policy for Quadrotor Control
- Science Robotics, 2026
- DOI: https://doi.org/10.1126/scirobotics.aec1481
- Code: https://github.com/rl-tools/raptor
- Use for: pretrained learned motor-level controller, recurrence, embedded deployment, vehicle generalization.

### Learning to Fly in Seconds
- arXiv: https://arxiv.org/abs/2311.13081
- Use for: ultra-fast quadrotor simulation/training, asymmetric actor-critic, curriculum, physics reference.

### RLtools: A Fast, Portable Deep Reinforcement Learning Library for Continuous Control
- JMLR 2024: https://www.jmlr.org/papers/v25/24-0248.html
- Code: https://github.com/rl-tools/rl-tools
- Use for: minimal C++ RL semantics, compile-time specialization, reference PPO/SAC/NN implementations, performance philosophy.

## Navigation / obstacle avoidance

### Learning High-Speed Flight in the Wild
- Science Robotics 2021
- DOI: https://doi.org/10.1126/scirobotics.abg5810
- Use for: high-speed local navigation, short-horizon trajectory output, privileged-learning philosophy, latency-aware autonomy.

### MAVRL: Learn to Fly in Cluttered Environments With Varying Speed
- IEEE RA-L 2025
- DOI: https://doi.org/10.1109/LRA.2024.3522778
- Code: https://github.com/tudelft/mavrl
- Use for: adaptive speed and temporal/memory-augmented depth representations.

### Reinforcement Learning for Collision-free Flight Exploiting Deep Collision Encoding
- ICRA 2024
- DOI: https://doi.org/10.1109/ICRA57147.2024.10610287
- arXiv: https://arxiv.org/abs/2402.03947
- Use for: compact collision-oriented depth latent and low-latency modular navigation.

### Depth Transfer: Learning to See Like a Simulator for Real-World Drone Navigation
- IEEE RA-L 2025
- DOI: https://doi.org/10.1109/LRA.2025.3617729
- Use for: training navigation on perfect simulated depth/latent while adapting real stereo depth toward the simulation representation.

### Champion-level drone racing using deep reinforcement learning
- Nature 2023
- DOI: https://doi.org/10.1038/s41586-023-06419-4
- Use for: separation of perception and learned control, real-data residual modeling, high-performance evaluation.

## Simulation / evaluation

### Aerial Gym Simulator: A Framework for Highly Parallelized Simulation of Aerial Robots
- IEEE RA-L 2025 / arXiv:2503.01471
- Use for: precedent for massively parallel aerial simulation and GPU-generated depth.

### AvoidBench
- ICRA 2023
- DOI: https://doi.org/10.1109/ICRA48891.2023.10161097
- Code: https://github.com/tudelft/AvoidBench
- Use for: independent navigation metrics and held-out benchmarking philosophy.

---

# 22. Final instruction to the autonomous coding agent

Do not try to impress the user by completing the most components.

Produce the **deepest verified vertical slice** possible.

The preferred order of value is:

```text
correct reference understanding
>
correct RAPTOR + physics Metal parity
>
correct massively parallel closed-loop simulator
>
correct depth/navigation pipeline
>
correct PPO trainer
>
first learned obstacle avoidance
>
performance optimization
>
extra features
```

If forced to choose between a flashy but unverifiable end-to-end demo and a smaller subsystem with strong parity tests and benchmark evidence, choose the verified subsystem.

At the end of the work session, leave the repository in a state where another capable agent can immediately continue. `STATUS.md`, `DECISIONS.md`, `BENCHMARKS.md`, tests, and reproducible commands are part of the product.

The north-star question for every line of code is:

> **Does this help us produce a fast, robust local-navigation policy for a real drone more quickly, or help us prove whether we have?**

If not, delete it or defer it.
