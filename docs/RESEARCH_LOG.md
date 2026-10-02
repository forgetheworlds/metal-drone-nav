# Research and decisions

## Runtime compiler

Question: is full Xcode required to begin raw Metal execution?

Evidence: installed macOS SDK exposes `MTLDevice newLibraryWithSource:options:error:`. A compiled Objective-C++ executable successfully compiled MSL and executed a 4096-element exact-output test on Apple M3. Offline `xcrun metal` is absent.

Decision: use runtime compilation and retain the source kernels. Xcode installation is authorized but is not required for this path.

Source: https://developer.apple.com/documentation/metal/mtldevice/makelibrary(source:options:)

Use `MTLMathModeSafe` and precise functions from the installed Metal headers. The older `fastMathEnabled` option is deprecated in macOS 15.

## State and ownership

Keep source, reproducible assets and five short state/evidence documents. Store temporary upstream checkouts outside this repo. Root integrates; Luna workers own RAPTOR, physics, PPO. No skills are used per the user request. The attached specification supplies technical requirements; its embedded agent instructions do not override the user's request.

## Ray work decomposition

Hypothesis: parallelize each ray rather than computing 320 rays in one environment thread. Keep all geometry, resolution and arithmetic the same.

Experiment: scalar C++/MSL shared geometry source; compare one GPU thread/world against one GPU thread/ray across the required environment ladder.

Result: exact CPU/GPU parity in fixtures. Ray-parallel decomposition is faster at all measured sizes; at 32,768 environments the component takes 11.318 ms versus 43.425 ms. Decision: use ray-parallel depth as the initial integration path. Do not call this an end-to-end training optimization until measured in that workload.

## RAPTOR / PX4 exact behavior

Pinned RAPTOR: `2c789dfcf16cc96fe697704492b3bf79dd2cc5a0`; pinned RLtools: `e43ae4bcda4556321a63f4eb5dcc826cd637aa39`; checkpoint training metadata: `c9bcfde8acd3f0d616edbfc3ba5a53d4497c7fa7`. Checkpoint: `2025-04-19_16-16-17`.

22 FLU inputs: position3, row-major rotation9, world velocity3, body omega3, previous motors4. Dense22→16 ReLU, GRU16 (reset/update/new gate order), dense16→4. Per-environment learned initial recurrent state. Motors [front-right, back-right, back-left, front-left]. Native update10ms. Raw inference and executor clipping are separate operations. Initial comparison incorrectly compared clipped outputs with raw official oracle outputs; corrected, then official parity passed.

PX4 `15f9af91ee3f8a2ac503864b5e557d7e4a1cb8fd`: transform state errors by inverse target quaternion, clip position±0.5m and velocity±1m/s, encode relative attitude. The initial adapter supplied `current_position + desired_world_velocity*0.5s`. That trajectory-generation hypothesis failed and was replaced by a persistent integrated reference, as recorded below. The target-frame transform and clipping still follow the pinned PX4 source. OFFBOARD source has no NaN-position fallback; all targets stay finite.

Sources: https://github.com/rl-tools/raptor ; https://github.com/rl-tools/px4/blob/15f9af91ee3f8a2ac503864b5e557d7e4a1cb8fd/external_modules/src/modules/rl_tools_policy/RLtoolsPolicy.cpp

## Physics fidelity

Use Crazyflie default, because RAPTOR's published L2F demo uses the default dynamics; its x500 change is only the UI model. Preserve RK4, quaternion normalization, thrust/motor time constants and force in newtons. Optional x500 simulator profile is available. Dynamics and RK4 source hashes match between RAPTOR's pinned RLtools and current source used in initial archaeology. `reference.cpp` produces official `rlt::step()` fixtures against the pinned source.

Sources: https://github.com/rl-tools/rl-tools/blob/e43ae4bcda4556321a63f4eb5dcc826cd637aa39/include/rl_tools/rl/environments/l2f/operations_generic/60_dynamics.h ; https://github.com/rl-tools/rl-tools/blob/e43ae4bcda4556321a63f4eb5dcc826cd637aa39/include/rl_tools/rl/environments/l2f/parameters/dynamics/crazyflie.h

## Fixed PPO path

Initial actor:660 deployable inputs (current raw architecture661), 64 tanh units, four unsquashed Gaussian outputs; tanh is applied only to executed commands so stored raw-action log probabilities are consistent. Critic: 32 privileged inputs, 64 tanh units. Distinct parameter sets prevent privileged actor leakage. GAE bootstraps time limits from pre-reset terminal-state value and stops carry on either termination or truncation. Fixed PPO clip±0.2, MSE value objective, entropy term, parameter-major sample gradients, reduction and Adam. No generic autograd or dynamic tensor API.

## Coherent target trajectory replaces moving preview

Failed hypothesis: reanchor `target_position=current_position+v*0.5` on each control tick while setting target velocity to v. Exact upstream dynamics show altitude drift under constant forward motion;1m/s and2m/s commands hit the floor after~6.9s and~5.1s. Reset motor history0 vs0.32 does not explain it. This was our trajectory generation error, not a physics-port or RAPTOR inference error.

Corrected hypothesis: trajectory position must have derivative equal to the commanded velocity. Persist reference position and integrate `reference += v*dt` at100Hz; do not reanchor it to measured position. Same upstream controller/dynamics/clips: after20s,1m/s ends x20.009m,z1.448m,vx1.000;2m/s ends x40.010m,z1.448m,vx2.000.2m/s minimum altitude1.019m.3m/4m goals with goal-direction feedback reach within0.35m in2.62s/2.84s.

Retain the measured fix. Add the three target-frame tracking-error coordinates to the actor (replaces two unused slots and adds one;661 total). This exposes controller reference state. Checkpoint v3 saves its persistent state. Integrated CPU/Metal control parity remains within5e-6.

The older PX4 commit0599df2d3fb53869e8a4a20c1b56daabbc9fdd67 pins a±0.2m position clip; selected15f9af91 uses±0.5m. Keep source-specific settings explicit rather than mixing commits.

## First actual PPO runs

30 rollouts of4096 samples under the corrected plant: held-out open success20–25%, no collisions; many goals still time out or are missed laterally/vertically. This is not adequate navigation. Continue learning and optimize the full update, rather than add more subsystem tests. Raw old per-sample actor gradients require~43MB per minibatch; direct hidden-delta matrix gradients are the next measured hypothesis.

## Direct gradients retained

Replace the parameter-by-sample gradient tensor with hidden deltas and batch-reduced outer products. Math and PPO workload remain unchanged. Matched complete GPU rollout+update improves from~1.66s to~0.295s. Direct CPU gradient checks and learning traces agree. Retain it. M3_RESEARCH.md records current Apple9 features and further profile-guided probes; do not assume M5 neural acceleration exists on M3.

## Policy selection and clutter curriculum

PPO reaches100% validation open-goal success from scratch at150 rollouts. Continued training becomes faster but can miss some goals, so save both latest exact-resume state and best validation checkpoint. Select best by success, then time-to-goal. Final evaluation uses fresh seeds rather than the repeated selection set. Keep one training.tsv result history for all runs.

Next stage warm-starts actor/critic parameters from open training, resets optimizer and exploration, and trains with random boxes. This remains pure PPO. Put obstacles at least1m before the goal and1.2m from the start to keep those states valid; the old minimum-span formula could place a box over a3m goal. This changes scene generation openly rather than hiding impossible episodes.

## Do not infer performance from fewer arrays

Rejected scalar device-pointer observation variant: valid isolated tests show no win; reused local observation storage is faster than repeated device reads. Two mixed pointer/fused timing attempts also used stale shader snapshots/overlapped work and are discarded. Adopt the parity-checked8x8 fused forward alone and then use that same kernel to precompute collection means.

Rejected RAPTOR GRU rewrite: six gate accumulators instead of96-float arrays passes official/trajectory parity but three matched SimAdvance medians5.52→5.95ms (+7.9%); full collection12.69→13.20ms. Group32 also did not beat64. Restore original. Apple9 dynamic caching means a smaller source-level live array is not automatically faster.

## Observation / generalization experiment

The raw actor learned local box avoidance (81.25%fresh vs47.66%goal-script) but generalized badly to a new doorway distribution. Add broader single-door/table training while holding two-door compositionfamily8 out. Family4 now has offsets up to±0.6m; family8 has two independent offset apertures and a verified waypoint path. Record actual failed rates rather than declaring reward growth success.

Compare raw661 and min-pooled181 actors from scratch on the SAME broad mixed0..6 dataset, speed1.5,goal4m,128×32,two PPO epochs. Pooled: each2x2 depth cell takes minimumrange;80current+80previous+21ego. Full320-ray sensing and physics stay unchanged. This is labeled feature compression/architecture ablation, not matched-workload speedup. Both fixed binaries pass existing CPU/Metal/reference checks and pooled learning smoke; now test actual learning. Separate checkpoint dimension checks prevent cross-architecture loading.

## Broad representation and risk outcomes

Controlled fromscratch comparison, same broad0..6 scenes and seed: raw661 best83.59%validation (2680rollouts/79.69s); min-pooled181 best75.78% (720/15.55s). Freshmixedraw82.03% vs pooled75.78%, but novel two-door compositionraw31.25% vs pooled48.44% (script38.28%). Pooling improves representation portability and iteration speed but loses some seen-distribution quality. Retainbothscientific baselines.

Clearance-risk shaping plus lowerentropy/lr improvesraw validation86.72%, freshmixed82.81%, held39.84%; pooledheld52.34%. It is not a robust-generalization fix. Introduce coherent spherical body-velocity intent (equalXYZ scaling withnorm cap), retaining explicitlegacycontract0 for oldpolicies. Legacy verticalhalf scaling distorted goal-vector compensation under vehicle tilt. Newcontract1 prevents diagonal component caps from acting like a larger maximumspeed. Pooled-sphere best78.9%, freshmixed75.8%, held46.9%. Keep thisactioncontract forcorrectness; do notcall it a learned-quality win.

Checkpointv5 recordsrisk/entropy/lr/velocitycontract; common reader supportsv3/v4/v5 without movingweight offsetswrongly. Alltrainedold artifacts remain readable withtheircontract. Current experimentadds deployablegeometry-derivedlocalfree-direction prior asactorfeatures andresidualmean, rather than more genericNN complexity. Distinguishgeometric-only baseline fromlearnedresidual. No privilegedobstaclepositions enterthisprior.

## Learned residual over local geometry

The deterministicprior reads80pooledranges/history, goal, egovel, frameinterval; picksgoal-alignedfreecones, bodyradius/brakingmargin and smalltemporalclosingcorrection. Returns atanhXYZfraction≤.8. Guided184actor addspriorlatentsas3features andadds3latents to its learned mean. Both likelihood/backward paths remainweight-correct. Geometric-onlymode9 is an explicitbaseline. Fullresidualimprovesfreshmixed80.47%vsprior67.19%, buthurtsnoveltwo-door75.78%vs85.16%. Quarterresidualinference gives92.19%/94.53%heldtwo-door acrossfreshseeds and89.84%with100ms sensor+50mscommandlag,0.05mnoise,10%dropout. Thisis a measuredpolicy/inferencevariant, not an unlabelledPPOalgorithmchange. Singleoffsetdoor99.22%, dynamiccorrupted89.06%, mixedwind0.5acceleration76.56%. Counter/table34.38% remainsafailure. Do nottreatstrongdoorsasfullnavigationgeneralization.

## Short geometry memory closes the table/counter failure

Trace evidence separated two failures: the vehicle reached a counter before climbing above it, and it returned toward the goal before clearing a corner that had left the forward FOV. The correction uses measured depth plus the estimated pose at capture; it does not read simulator obstacle descriptors. Minimum2x2 ranges retain the actual closest pixel direction. Eight pose/depth slots are transformed into the current body frame. Candidate swept clearance includes body radius, range/age uncertainty and current velocity. Pure lateral/vertical escapes are allowed.

An excessively inflated memory radius closed valid doorways and was rejected. Querying an instantaneous steering direction missed vehicle inertia and was corrected. A table/counter curriculum learned the remaining vertical residual. Ungated full residual gives strong table results but damages held two-door transfer; mode17 overhead-range gating is the selected, labeled inference variant. Final evaluation is saved separately from the training selection seed.

Serial memory clearance repeatedly transformed up to640points for85candidate directions. M3 kernels now transform the points once, then compute candidates in parallel. This preserves the original calculation and episode results. Independent CPU/cache clearance checks at sensor delays0/2/6 give maximum error0. Identical128-environment200-navigation-tick evaluation wall time falls from~3.08s to~0.143s. The derived cache is not checkpoint state; pose/depth rings are persisted in v6.

## Finish the usable vertical slice

The user asked to focus on the end outcome instead of open-ended experiments. Stop new architecture/tuning probes. Preserve a compact actor-only export, exact contracts, selected checkpoint, final held-out scores, optimized runtime, reference provenance and reproducible commands. Distinguish the verified simulation research milestone from future real sensor/vehicle validation.

## Disturbance exposure targets the measured delay failure

Training previously exposed clean sensor/controller timing while robustness was only evaluated afterward. Expose the existing checkpointed sensor/command delay, wind-force acceleration, depth noise and pixel dropout in the train CLI. Validate model selection under the same conditions rather than selecting solely on clean episodes. The PPO likelihood, simulator, controller and physics are unchanged. Retain the starting model in a new run's selection set.

A1000-rollout broad-family warm start from the table-memory best,100ms sensor lag,50ms command lag,0.5m/s² acceleration,0.05m noise,10% pixel dropout: fresh held-door combined success70.31%→77.34%;command-delay-only76.56%→84.38%. Mixed clean89.84%→92.97%;table combined89.06%→92.19%. It also hurts clean held-door transfer89.06%→85.94%, and92.19%→82.81% on the second seed. Preserve both candidates, do not replace the packaged model yet. Exact matrix:results/stress-evaluation.csv. Next curriculum focuses on single doors while the two-door composition remains held out.

## Door skill transfer and rehearsal

A1000-rollout single-door curriculum under the same disturbances improves fresh held two-door clean success89.06%→95.31% and92.19%→94.53% on the second seed;command-delay success76.56%→90.63%;combined70.31%→83.59%. However,table clean96.88%→58.59%,combined89.06%→39.84%. This is a tradeoff, not an acceptable replacement. Keep the original export. Add a training-only rehearsal mixture:50% doors,25% tables,25% broad0–6,with family8 always held out. Warm start from the broad stress candidate so table skills are present initially.

## Dynamic coverage reveals a harder failure boundary

Controlled flying-start sphere encounters are generated on the host once; sensing/collision/physics/controller/policy remain GPU. Vehicle initialvx1.5m/s,sphere radius.35m,goal4m.24cases:approach/crossing,speed.5/2m/s,nominal straight-line TTC.5/1s,modes17/13/2,128 seeded variations each. Goal-script collision100% in every case confirms the challenge forces avoidance. The original selected learned policy succeeds100% on slow approach with1s nominal TTC and82.03% on2m/s approach with1s TTC,but0% on2m/s approach at.5s andfast crossing cases. Geometry-only is also weak. Prior high moving-sphere generator scores did not establish fast-threat evasion. Exact matrix:results/threat-evaluation.csv. Next learning work must include these encounters and genuine temporal ablations.

## Causal modeled reaction latency

Warm a1-environment guided episode,clone allstate/history/policy buffers,andconfirm identical motors before inserting a1m/s approaching sphere2m ahead into only one copy. Compare applied navigation fractions and motor outputs after each50ms navtick,threshold1e-4. No-delay response is detected by50ms;100ms sensor+50ms command delay response is detected by200ms. These are simulation upper bounds with50ms readout resolution;actual RAPTOR still updates100Hz. They establish causal response and modeled queue delay,not real sensor/transport latency or successful avoidance.

Rehearsal1000 outcome: combined held-door70.31%→93.75%,command-only76.56%→95.31%,clean held-door89.06%→89.06%. However,table clean96.88%→78.13%,combined89.06%→76.56%;fresh mixed89.84%→87.50%. This successfullyaddresses the measured delay failure butstilldoesnotproduceonepolicyretainingallskills. Keep the original export and everycandidate. Next workshouldmixcleananddisturbedtrainingconditionsandtrainfastthreats,withper-familyvalidationguardingagainstaverageshidinglosses.
