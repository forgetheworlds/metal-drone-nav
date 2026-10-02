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

Source base `f758f57`, selected checkpoint `assets/checkpoints/guided-table-memory.bin.best`, guided184 actor with recorded geometry memory and velocity contract1. Evaluation command: `python3 evaluation.py --checkpoint assets/checkpoints/guided-table-memory.bin.best`; 28 sequential configurations, 128 episodes each, 1.5m/s intent and 4m goal. The captured CSV is packaged at `evidence/inputs/evaluation.csv`; its local run was written to ignored `results/`. Each row includes actual printed metrics and GPU/host times. Baseline seed800001, second fresh seed900001, training selection seed700001.

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

## Packaged policy verification

Commit eec01c4 includes the actor-only asset and loader. `./build/metal_nav_guided export assets/checkpoints/guided-table-memory.bin.best assets/navigation.bin`:48,496 bytes,184 observations,12,104 FP32 parameters,mode17,1.5m/s,source FNV64 `1d991b0048dec80a`. Eight observation probes match the fixed PPO CPU actor exactly (maximum error0). `./build/metal_nav_guided eval-policy assets/navigation.bin 8 800001` reproduces checkpoint episode results:89.0625%success,10.9375%collision,0%timeout,2.83684s successful goal time.

`./build/metal_nav_guided policy-bench assets/navigation.bin`:10,000 sequential CPU evaluations at batch1,5.28245us average on M3. Inputs change and previous-command features update each iteration. This excludes depth acquisition, geometry-memory preparation, RAPTOR and physical command response. It does not establish end-to-end reaction latency.

## Disturbance curricula and controlled-threat acceptance gaps

Same guided184/spherical/memory actor,FP32 onM3,128episodes/configuration,freshseed800001. Each candidate receives1000additional PPOrollouts withrisk.04,entropy.0003,learningrate.0001,sensor2frames,command1tick,acceleration.5m/s²,noise.05m,pixel-drop.1,validationselectionmode17. Broadstress warmsfromtheoriginaltable-memorybest;doorand rehearsalwarmfrombroadstressbest. These are curriculum candidates, not a controlled architecture-speed comparison. Full rows and checkpoint SHA-256 values are in `evidence/inputs/continuation-evaluation.csv`.

| Candidate | Clean held doors | Combined held doors | Clean tables | Combined tables |
|---|---:|---:|---:|---:|
| Packaged original | 89.06% | 70.31% | 96.88% | 89.06% |
| Broad stress | 85.94% | 77.34% | 97.66% | 92.19% |
| Door stress | 95.31% | 83.59% | 58.59% | 39.84% |
| Rehearsal stress | 89.06% | 93.75% | 78.13% | 76.56% |

Rehearsal is50%doors,25%tables,25%broad0–6;heldfamily8isneversampled. It raisescommand-delay-onlyhelddoor success76.56%→95.31%,butdoesnotretaintablequality. The original export is preserved. The next requirement is broadskillretentionacrosscleananddisturbedconditions.

Controlled threats: `python3 evaluation.py --threats`.24configurations×128episodes,flyingstartvx1.5m/s,approach/crossing,.5/2m/s,sphere.35m,nominalTTC.5/1s,modes17/13/2,seed800001. Goal-script collides100% inallcases. Selectedoldpolicy succeeds100% onslowapproach/1s and82.03%on2m/sapproach/1s,but0% on2m/sapproach/.5s andfastcrossing. This contradicts a broad fast-evasion claim. TTC is a nominal scene parameter,not the actual policy path's measuredcollisiontime.

Reaction diagnostic: `./build/metal_nav_guided reaction-latency assets/checkpoints/guided-table-memory.bin.best 0 0` and`... 2 1`. Pairclonedwarmedtrajectories,verifyidenticalmotorsbeforeaddingoneapproachingthreat,anddetectcommand/motordelta>1e-4. Simulatedresponseupperbounds50ms clean and200ms delayed,resolution50ms. These include modeledsample/queue delays; theyexclude actualsensor/transportdelay and do not prove evasion. Normaltrainingstillhasnoper-stepCPUwait;the diagnostic intentionallyreadsaftereachnavigationtick.

## Frozen challenge-bank development failures

The versioned challenge bank contains 270 explicit levels: 90 train, 90 dev and 90 final across family 14 (bent hallway), family 15 (connected rooms and doors), and family 16 (vertical over/under choice). The fixed-scene results below use only the 90 dev levels (30 per family). The final split has no evaluation rows and remains untouched. Every policy episode used a 1.5 m/s requested cap and a 400-step limit (20 seconds).

Success counts are out of 30 levels per family:

| Policy or inference mode | Family 14 corner | Family 15 rooms | Family 16 vertical |
|---|---:|---:|---:|
| Original guided PPO, mode 17 | 0 | 0 | 22 |
| Clean/stress PPO, mode 17 | 0 | 0 | 5 |
| Threat-joint PPO, mode 17 | 0 | 0 | 13 |
| Geometry prior, mode 13 | 0 | 18 | 28 |
| Goal script, mode 2 | 0 | 0 | 0 |

![Frozen challenge-bank development outcomes](../artifacts/challenge-bank-held-dev-outcomes.png)

The chart also shows collision/timeout counts and 95% Wilson intervals. This is a development-set comparison. The checkpoints were trained or selected under different procedures; it is not a single matched training experiment. The geometry prior is a non-learned baseline. The green paths below are geometric clearance witnesses; they do not show learned flight.

![Representative saved challenge geometry and witness paths](../artifacts/challenge-bank-witness-worlds.png)

One targeted family-14 run warm-started the original policy and trained for 1,000 PPO rollouts (4.096 million transitions). It logged 82,729 training episodes with a 99.994% collision rate. Its 128-episode family-14 validation at seed 700001 stayed at 0/128 successes before and after training. This shows that the tested schedule did not teach the needed detour. It does not show that the route is unlearnable. The 100 validation records and run log are packaged in `evidence/inputs/`.

## Privileged witness-route execution

The opt-in mirrored bank is generated with `--mirror-y`, which reflects alternate levels across `Y=0`. This reduces a fixed left/right detour bias; it does not add learned-policy data. The saved route is supplied directly as a sequence of waypoints to a scripted controller. Frozen RAPTOR and simulator physics execute the waypoints. These measurements test route execution with privileged route information, not autonomous navigation.

On the 90 mirrored dev levels, a 1.0 m/s cap and 60-second budget produced 90/90 successes with no collisions or timeouts. Mean completion times were 21.92 seconds for corners, 19.75 seconds for rooms, and 21.87 seconds for vertical routes. With a 1.5 m/s cap and 20-second budget, 69/90 levels succeeded and 21 timed out; there were no collisions. All 30 corners and all 30 rooms passed within 20 seconds. Nine of 30 vertical routes passed; the other 21 timed out.

Recreate the mirrored bank and both checks with:

```sh
python3 challenge_bank.py --seed 20261001 --distance 8 --per-split 30 --families 14,15,16 --mirror-y --out evidence/inputs/challenge-bank-mirrored-v1.jsonl
./build/metal_nav_guided bank-witness evidence/inputs/challenge-bank-mirrored-v1.jsonl dev results/bank-witness-1mps.csv 1.0 1200
./build/metal_nav_guided bank-witness evidence/inputs/challenge-bank-mirrored-v1.jsonl dev results/bank-witness-1.5mps.csv 1.5 400
```

The 60-second 1.0 m/s result is not directly comparable to the 20-second policy score. Both logs and CSVs are packaged as evidence inputs.

Separate uniform and priority PPO experiments have started on mirrored levels. Their development results are preliminary and are not included in this report yet. The mirrored final split remains unevaluated.

## Fixed-weight inference ablations

`evidence/inputs/threat-joint-ablations.csv` compares modes 17, 18 and 19 with the same threat-joint checkpoint, seed 800001 and 128 episodes per row. No mode was retrained. Mode 18 duplicates the previous-depth actor channel with the current frame and limits geometry-memory guidance to the newest frame; geometry guidance stays enabled. Mode 19 rescales each nonzero navigation command to the requested cap.

| Scene and condition | Mode 17 | Mode 18 | Mode 19 |
|---|---:|---:|---:|
| Table/counter, clean | 91.4% | 83.6% | 68.0% |
| Table/counter, combined stress | 90.6% | 75.0% | 57.8% |
| Held doors, clean | 91.4% | 81.3% | 73.4% |
| Held doors, combined stress | 90.6% | 94.5% | 73.4% |

History helps on these table cases, but not on every held-door condition. The full matrix also covers mixed scenes and moving threats; this table reports only table/counter and held doors.

```sh
python3 evaluation.py --ablations --checkpoint assets/checkpoints/guided-threat-joint.bin.best --output results/threat-joint-ablations.csv
```

## Fixed-policy command-cap sweep

`evidence/inputs/static-speed.csv` contains 25 mode-17 comparisons from one guided checkpoint. It tests requested caps of 0.75, 1, 1.5, 2 and 3 m/s on clean table, mixed and held-door scenes, plus combined stress on table and held doors. Each case uses seed 800001, 128 episodes and a 10-second episode limit. We kept the weights fixed; this is not speed-curriculum training.

At a 1.5 m/s cap, clean success is 92.2% on tables, 94.5% on mixed scenes and 94.5% on held doors. At 3 m/s it falls to 31.3%, 26.6% and 25.0%. In the combined-stress table/door cases at 3 m/s it falls to 16.4%/9.4%; most other episodes collide. At a 0.75 m/s cap, many cases time out at the 10-second limit.

The requested cap is not a physical speed bound. At the 3 m/s cap, the maximum observed speed is 7.34 m/s on tables, 6.88 m/s on mixed scenes and 5.61 m/s on held doors. Mean path speed includes failures; peak speed is the largest observation across 128 episodes and can be an outlier. These measurements do not establish a safe cruising speed.

![Fixed-policy requested-cap sweep and observed speeds](../artifacts/static-policy-speed-cap-sweep.png)

```sh
python3 evaluation.py --speed-sweep --checkpoint assets/checkpoints/guided-clean-stress.bin.best --output results/static-speed.csv
```

## Reproducible evidence package

`evidence/inputs/` contains the compact CSV, TSV, JSONL and paired trace files for all figures and outcomes. Selected checkpoint snapshots are in `assets/checkpoints/`. `artifacts/manifest.json` records input hashes, commands, metric definitions and limits. Rebuild the figures without running Metal:

```sh
python3 evidence.py --out artifacts
```

The 3D crossing panel and GIF use recorded simulator poses and saved obstacle geometry. They are not camera footage. Pose paths use 20 Hz samples; terminal contact/end markers use exact JSON metadata. The original and threat-joint runs share the same world seed and obstacle path; the original policy collides and the later candidate succeeds.
