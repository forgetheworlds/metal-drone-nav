# Metal Drone Navigation

I want to build a drone that can take an instruction, look around an unfamiliar place, and carry out the task without someone piloting it.

![Native Webots doorway recording](artifacts/videos/native-doorway-preview.jpg)

*Offset doorway, arrival policy, 6.91 s stable hold with no contact. [Watch the native flight and see its recording limits](docs/RESULTS_GALLERY.md#native-webots-recordings).*

This repository works on the part between choosing a destination and reaching it. A slower cloud reasoning system can decide that the drone should go to the stairs, a doorway, or a point down a platform. Metal-nav takes the supplied 3-D destination, depth observations, and the drone's motion, then chooses how to move through the geometry in between.

Metal-nav does not control the motors. It sends desired body velocity and yaw-rate commands to RAPTOR, a separate learned flight controller. RAPTOR stabilizes the aircraft and produces four motor commands. The results here come from simulation.

I train the navigation policy with a raw Metal training loop on Apple Silicon.

## The idea

```text
Person's instruction
        ↓
Cloud reasoning: what matters, and where to go next?
        ↓
Chosen 3-D destination
        ↓
Metal-nav: move there through local geometry
        ↓  desired body velocity + yaw rate
RAPTOR: stabilize and control the aircraft
        ↓
4 motors
```

If another system has chosen the destination, moving through the geometry is navigation. That destination might be a doorway two metres away or the end of a visible hallway fifteen metres away. A blind corner that requires choosing which room or staircase matters next calls for another high-level decision.

Some experiments use short goals to isolate this navigation layer. That choice does not set a distance limit for the design. The published native static-transfer benchmark uses exposed development tasks with 1–3 m goals. It is one test of the navigation layer, not a full semantic mission. [Benchmark scope](docs/NATIVE_BC_TRANSFER.md#scope-and-limits).

## What the policy does

The policy gets a compact depth view, information about its motion, and the supplied goal. It runs at 20 Hz and produces three body-velocity commands and one yaw-rate command. A trajectory adapter carries those commands to RAPTOR, which runs at 100 Hz. The policy does not receive a map of obstacles or a generator's route witness.

The selected guided actor has 184 inputs, one 64-unit hidden layer, and four outputs: about 12,000 learned parameters. Its observation includes current and previous pooled depth, motion and goal context, and three geometry-prior values. The exact packing and export contract are in [deployment.hpp](deployment.hpp) and [goal.md](goal.md). The critic used during training can receive simulator information that the deployed actor does not receive.

```text
Depth + motion + supplied goal
                 ↓
       184 → 64 → 4 policy
                 ↓
      body vx, vy, vz + yaw rate
                 ↓
      trajectory adapter → RAPTOR → motors
```

## How training works

The training engine batches thousands of simulated drones. Most reported policy runs use 128 environments at once; the measured throughput ladder reaches 8,192. Each environment gets a start state, a destination, and a generated scene. The system advances the vehicle, renders depth, runs RAPTOR and the navigation actor, and records experience. PPO updates the actor; development tasks help select checkpoints. The generator can use a route witness to check a scene, while the policy flies from its observations and supplied goal.

The simulator, depth sensing, RAPTOR, policy inference, rollout collection, GAE, PPO backward pass, and Adam update run in raw Metal on Apple Silicon. There is no PyTorch, TensorFlow, or Python in the hot training loop. Python handles cold-path work such as experiment orchestration, review, and plots.

At 8,192 environments, a full rollout took 2.707 seconds on the GPU and 5.016 seconds on a matched CPU reference (1.85×). In another workload, GPU time for a full rollout and PPO update fell from 1.66 seconds to about 0.026 seconds after optimization. The [benchmark report](docs/BENCHMARKS.md#matched-complete-ppo-optimization) gives the setup and command for each test.

```text
Generate tasks → simulate flights → collect experience
       → PPO update → evaluate → keep or reject the run
```

I use the same loop to test hypotheses, not to assume every change helps. The [training strategy study](docs/TRAINING_STRATEGY_RESULTS.md) found that a larger collision cost reduced contacts but added timeouts, while a new task mixture improved blocked-route results and lost some old-task performance.

## What I train it to do

The intended skill is to reach a supplied destination through unfamiliar local geometry. Training scenes vary the actual decisions: move directly when clear, brake before a narrow gap, go around a blocked route, choose between openings, move above or below an obstacle, and regain the goal direction after a detour. Distance alone does not define difficulty. Clearance, braking room, heading change, visibility, occlusion, obstacle motion, and route choice also matter.

This image shows generated challenge geometry and geometric witness routes. Those green routes are diagnostics for the generator; they are not learned policy trajectories.

![Challenge-bank worlds and geometric witness routes](artifacts/challenge-bank-witness-worlds.png)

*Generated hallway, connected-room, and vertical-choice scenes. The routes show geometric witnesses, not learned flight.*

The repository also has studies of corner detours, arrival and braking, moving threats, depth history, and sensor visibility. [The results gallery](docs/RESULTS_GALLERY.md) links each figure to its source records.

## Results so far

In 128 native Webots flights, the frozen dynamic-trained policy (FULL) outperformed the fast policy on two separately predeclared 32-task panels. No policy training ran in Webots.

| Panel | Fast: success / contact / timeout | FULL: success / contact / timeout | Mean arrival on shared tasks, fast / FULL |
|---|---:|---:|---:|
| Diagnostic, 32 tasks | 29 / 3 / 0 | 32 / 0 / 0 | — |
| Challenge, 32 tasks | 13 / 19 / 0 | 30 / 2 / 0 | 4.08 s vs 3.70 s |

FULL completed 17 challenge tasks that fast missed; fast beat FULL on none, and both failed two. On the 13 challenge tasks both policies completed, fast averaged 4.08 s and FULL 3.70 s. These are exposed source-development tasks. See the [full protocol and evidence](docs/NATIVE_MOVING_TRANSFER.md).

![Moving-obstacle transfer results](artifacts/plots/native-moving-transfer.png)

*Frozen policies in Webots. The challenge panel adds faster approach and crossing cases. [Protocol and 916 hashed records](docs/NATIVE_MOVING_TRANSFER.md).*

A separate static transfer test compared frozen imitation and fast policies across 128 Webots tasks. Imitation completed 113/128, with 14 contacts; fast completed 105/128, with 23. Successful arrival averaged 6.17 seconds for imitation and 3.61 for fast. Both completed all 85 direct-route tasks; imitation gained on blocked routes.

![Native Webots reliability and arrival-time comparison](artifacts/plots/native-bc-transfer.png)

*Frozen policies on the same static task bank. Imitation completes more tasks and takes longer. [Protocol, paired outcomes, and evidence bundle](docs/NATIVE_BC_TRANSFER.md).*

The static test used exposed development scenes with short goals. The report retains all 256 flights, selection details, hashes, and the stable-arrival rule.

## What failed

More fixed rehearsal did not solve the loss of earlier skills. Four PPO runs completed 10,000 rollouts each across two matched seeds. Both predeclared gates failed: fresh blocked-task success fell, and the treatment missed old-task retention floors. I did not adopt that schedule. [Full outcomes and records](docs/REHEARSAL_RESULTS.md).

The learned policy also does not beat its geometry prior on every task. On one held-out two-door composition, the prior reached 96.1% while the learned policy reached 89.1%; on the mixed set the prior reached 91.4% and the learned policy 89.8%. I keep that baseline beside the learned results.

![Learned policy and geometry-only comparison](artifacts/static-scene-policy-comparison.png)

*Fixed static scene matrix, 128 episodes per condition. [Full table and protocol](docs/RESULTS_GALLERY.md#policy-quality-robustness-and-speed).*

Other studies show related tradeoffs. A larger collision penalty reduced contacts but produced more timeouts. Continued training after a strong connected-room checkpoint reduced its score. Corner imitation loss fell while completion stayed at 0/30. See [training comparisons](docs/TRAINING_STRATEGY_RESULTS.md), [local training review](docs/LOCAL_TRAINING_REVIEW.md), and [the research log](docs/RESEARCH_LOG.md).

The experimental [perception-support profile](docs/PERCEPTION_SUPPORT.md) has passed interface checks; its full training comparison is still running.

## Why Metal?

I built the training loop for the machine used in this work: an Apple M3. A framework could provide familiar training components, but it would also put more layers between the measured bottleneck and the code that controls it. Writing the fixed-shape simulation and PPO workload in Metal makes memory use, kernel boundaries, and the actor's deployment contract explicit.

That engineering matters because the useful unit is a tested policy update, not a fast kernel by itself. I keep numerical parity checks against CPU and upstream references, and I report whole-workload timing beside component measurements. The implementation and measured optimization ladder are in [BENCHMARKS.md](docs/BENCHMARKS.md); coding decisions follow [CODE_DIRECTION.md](docs/CODE_DIRECTION.md).

## Physical deployment

The intended onboard stack is a depth sensor and state estimate feeding Metal-nav, followed by a velocity-command interface to RAPTOR and the flight controller. The actor interface is designed around information that could be available onboard. Training and evaluation infrastructure may use privileged geometry to construct tasks, reject impossible starts, or score outcomes; those data are not actor inputs.

The next deployment work is the onboard sensor front end and flight-controller connection, followed by physical flight testing.

## Build and reproduce

The project requires Apple Silicon, macOS, CMake, and the Xcode Command Line Tools. Metal shader sources compile at runtime; full Xcode is not required.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
./build/metal_nav_guided test
```

Use [docs/EXECUTION.md](docs/EXECUTION.md) and each result report to reproduce training and evaluation with the listed checkpoint, task manifest, seed, and grader.

The key records are:

- [Goal and current task boundary](goal.md)
- [Code direction](docs/CODE_DIRECTION.md)
- [Native Webots transfer: 256 flights](docs/NATIVE_BC_TRANSFER.md)
- [Moving-obstacle transfer: 128 native flights](docs/NATIVE_MOVING_TRANSFER.md)
- [Fixed rehearsal study](docs/REHEARSAL_RESULTS.md)
- [Training strategy comparisons](docs/TRAINING_STRATEGY_RESULTS.md)
- [Local training review](docs/LOCAL_TRAINING_REVIEW.md)
- [Metal benchmarks and correctness gates](docs/BENCHMARKS.md)
- [Research log, including rejected hypotheses](docs/RESEARCH_LOG.md)
- [Next research phase](docs/RESEARCH_NEXT_PHASE.md)
- [Figures, videos, source data, and limitations](docs/RESULTS_GALLERY.md)
- [Writing standard for future repository documents](docs/WRITING_STANDARD.md)
- [Evidence inputs and hashes](evidence/inputs/)
- [Selected checkpoint files](assets/checkpoints/)
- [Challenge-bank design and evaluation](docs/COURSE_BANK_REVIEW.md)

Rebuild published figures from the checked-in evidence inputs without running Metal:

```sh
python3 evidence.py --out artifacts
```

The evidence inputs and manifests record hashes, commands, metric definitions, and filters. Some raw simulator runs remain in the local `results/` directory; linked reports identify the records they use.
