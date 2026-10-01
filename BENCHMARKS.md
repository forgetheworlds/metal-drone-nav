# Benchmarks

Machine: Apple M3, 10 GPU cores, 16 GB unified memory; Apple clang 17. Release `-O3`. Date: 2026-10-01.

Initial gate: runtime-compiled raw Metal kernel writes 4096 known float values. Every value agrees exactly. First cold GPU execution: 6.375 microseconds (not a sustained workload benchmark). Command: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j 4 && ./build/metal_nav`.

The sections below retain measurements in development order. Earlier failures describe earlier implementations; the final deliverable section states the selected policy and current boundaries.

## Primitive ray range, first measured ladder

Commit base `7fcf5d2` plus world implementation. FP32 safe/precise, eight AABBs per world plus six room bounds, 16x20 rays, same scalar geometry code on CPU and GPU. Command: `./build/metal_nav bench-depth`. Times are minimum of four GPU runs; CPU is a single scalar run. This is a sensor component benchmark, not training throughput or a comparison to an optimized CPU trainer.

| Environments | One thread/ray ms | One thread/world ms | Scalar CPU ms |
|---:|---:|---:|---:|
| 1 | 0.0123 | 2.001 | 0.0134 |
| 32 | 0.0216 | 2.014 | 0.396 |
| 128 | 0.0694 | 1.947 | 1.603 |
| 512 | 0.261 | 1.965 | 6.107 |
| 2,048 | 0.906 | 2.030 | 24.640 |
| 8,192 | 2.841 | 9.325 | 94.950 |
| 32,768 | 11.318 | 43.425 | 406.212 |

Analytic tests cover misses, inside exits, tangents, parallel slabs, cylinder sides/caps, moving geometry, range clamp and collision independent of depth. All 40,960 CPU/GPU rays and 128 clearances match exactly for the test scene set. Full workload impact remains to be measured.

## Correctness gates

Command: `./build/metal_nav test`, Release, FP32 safe/precise.

| Gate | Workload | Max absolute error |
|---|---|---:|
| RAPTOR official oracle | 128 envs ×16 recurrent steps | 5.96e-7 |
| RAPTOR recurrent state | CPU vs Metal | 1.19e-7 |
| L2F official physics | 64 states ×8 reference steps | 9.54e-7 |
| PX4 observation transform | independent expanded formula | 5.96e-8 |
| Complete control loop | 32 envs ×160 native ticks | state 9.09e-6; hidden 2.50e-6 |
| PPO operators | actor/critic, boundary GAE, gradients, Adam | within 3e-5 test bounds |

GPU fixed-target optimizer smoke: action-mean error 0.587→0.062; this does not establish navigation learning.

Scripted goal direction at requested horizontal speed≤2 m/s, 4m goal distance, 128 held-out first episodes: open success38.3%, boxes28.1%, poles4.7%, dynamic spheres34.4%. These low baseline results are recorded rather than hidden. They are not a trained policy result. GPU execution for each 200-step episode batch is roughly 0.05s; wall about0.05–0.08s on these short early-terminating scenes. Count actual live steps before quoting transitions/s.

## Matched complete PPO optimization

Before: per-sample parameter gradients followed by reduction. After: per-sample64-unit hidden deltas followed by direct batch sums. Same4096 transitions,128 envs,32-step horizon,two PPO epochs,256-sample minibatches,661 actor inputs,64 hidden units, actual RAPTOR/physics/depth, FP32 safe/precise, M3 Release.

Three worker probes: before1.6624/1.6556/1.6558s; after0.3055/0.2942/0.2947s. Median GPU execution improves5.62× for full rollout+update. Loss/reward traces agree within~1e-6 and direct-gradient CPU parity passes. Root confirms existing baseline warm updates1.65–1.67s and optimized updates0.294–0.297s through300 further rollouts. Gradient scratch falls from45,884,416 bytes to131,072 bytes; no simulation/sensor/PPO work is removed. Baseline commit355b8ee, optimization follows it. Commands: `./build/metal_nav train 10 0 results/probe.bin` for a fresh matched configuration; preserve separate before/after checkpoint paths.

## Real learning, open curriculum

Pure PPO, no expert/teacher. At rollout150 (614,400 transitions), validation success first reaches100% over128 unseen open episodes. It repeatedly reaches100% through rollout250; later faster policies can miss goals, so the newest checkpoint is not automatically the best. At rollout380 latest success82.8%, collisions0%, remaining cases time out. Time-to-goal~2.53s for successful latest episodes. A fresh final test seed and clutter training are next. Do not count open-world learning as obstacle-avoidance success.

## M3 measured forward and reduction work

Actual hardware encoder timestamps: original actor collection114–126ms, PPO actor forward~95ms, serial norm~84ms per complete128×32rollout. SIMD norm reduced norm to~5ms; fused8x8 forward in trainer alone0.217→0.124s; using the same fused kernel for collection reaches~0.026s. Scalar-actor fallback in the same current graph:0.210s at rollout3 vsSIMD0.0257s, initial losses agree to printed precision. No sensor/physics/PPO work is removed. Optional `profile` records611+ actual timestamp intervals. M3 has encoder-stage sampling; dispatch-boundary sampling is unsupported.

## Optimized CPU reference comparison

ReleaseFP32, identical box worlds,320rays/history, actual RAPTOR/native5ticks per nav,128/512/2048/8192envs,32horizon,two PPO epochs,batch256. Three consecutive full rollouts; setup/evaluation excluded from both. CPU uses Accelerate SGEMM and GCD for parallel geometry/physics; GPU timings include host encoding/wait in wall result.

| N | GPU wall3rollouts s | CPU wall3rollouts s | CPU/GPU |
|---:|---:|---:|---:|
| 128 | 0.0822 | 0.0811 | ~0.99 |
| 512 | 0.1881 | 0.3097 | 1.65 |
| 2048 | 0.6754 | 1.2297 | 1.82 |
| 8192 | 2.7073 | 5.0162 | 1.85 |

Commands: `./build/metal_nav gpu-bench N 3 1` and `./build/metal_nav cpu-bench 3 N 1`. Small-batchCPU is competitive. A CPU scratch-size error affected early N>256 attempts; corrected to max(N,256), those earlier times are discarded. First-rollout math/forward agrees within test bounds; accumulation order can change later learning traces.

## Fresh-seed learning evidence

Box-trained best checkpoint,128 freshseed800001 episodes at speed1/goal3m: learned81.25%success/18.75%collision; goal script47.66%/52.34%; random0%success. Mixed five-family curriculum at speed1.5/goal4m:1500rollouts in52.2s; first best validation81.25% at380rollouts/13.55s. Fresh mixed75.78% vsgoal script65.63%. Blind-depth box test47.66%. Generalization failure: unseen doorway6.25%, table/counter44.53%. The policy is useful but not robust; the goal remains active.

## Final deliverable evaluation

Source base `f758f57`, selected checkpoint `results/guided-table-memory.bin.best`, guided184 actor with recorded geometry memory and velocity contract1. Evaluation command `python3 evaluation.py`;28 sequential configurations,128 episodes each,1.5m/s intent,4m goal. Machine-readable evidence is preserved in `assets/evaluation.csv`; the local run is `results/evaluation.csv`. Each row includes actual printed metrics and GPU/host times. Baseline seed800001, second fresh seed900001, training selection seed700001.

| Family | Learned/guided mode17 | Geometry-only mode13 | Goal script mode2 | Guided fresh seed900001 |
|---|---:|---:|---:|---:|
| Mixed0–6 | 89.84% | 91.41% | 57.81% | 91.41% |
| Held two-door composition | 89.06% | 96.09% | 45.31% | 92.19% |
| Table/counter | 96.88% | 74.22% | 24.22% | 98.44% |
| Moving spheres | 98.44% | 94.53% | 72.66% | 92.19% |

Mode17 is a learned residual over a geometry prior with an overhead-range inference gate. The separate strong geometry-only baseline matters: learning adds vertical-clearance and moving-sphere performance, but the prior alone has higher door/mixed success. Successful goal times (guided vsgeometry, seed800001):mixed2.777 vs3.368s;held2.837 vs3.426s;table3.044 vs3.823s;dynamic2.748 vs3.264s. These averages have different successful subsets; they are not matched-trajectory speedups.

Held two-door mode17, one factor at a time:100ms sensor delay85.16%;constant0.5m/s² x acceleration84.38%;0.05m Gaussian range noise88.28%;10% dropped pixels87.50%;50ms command delay76.56%. Combining all five gives70.31% held-door success and89.06% table success on seed800001. No timeouts occurred in this matrix. Wind is a force-equivalent simulator disturbance, not a measured wind velocity. Doorway performance is sensitive to command delay; this policy is not certified for real flight.

All28 eval processes used~5.03s total host time,~2.87s reported GPU time. Evaluation timing is separate from training timing. The full guided memory update in the table curriculum is~28.4ms; the original~64× matched optimization ratio uses the raw661 workload, not this changed actor/memory graph.

Selected checkpoint SHA-256: `fdd62374c9a1724fc12690d1ccb756985b6c9f8d5f945b290156c17a204cf7d9`.
