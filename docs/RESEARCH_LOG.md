# Research and decisions

This is a chronological record. Older entries describe their own source and
execution snapshots. Read [STATUS](../STATUS.md), [the document index](README.md)
and [combined progress](COMBINED_POLICY_PROGRESS.md) for current results.

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

## Distribution audit review (2026-10-04)

Question: does the new capability-focused audit (`docs/LOCAL_DISTRIBUTION_AUDIT.md`)
rest on reproducible measurements?

Evidence: coordinator re-computed its cross-tabs from
`results/omp-aware-navigation/evals/baseline-fast-deva.csv` joined to the dev-a
bank: detour + goal outside FOV = 7 success / 26 contact / 1 timeout;
detour + in-FOV = 12/1; direct = 81/81. TRAIN start-velocity yaw equals goal
yaw in 1024/1024 rows (max deviation 2.4e-5 deg), confirming the accidental
coupling the audit flags. Detour + out-FOV(±45°) + witness clearance <0.20 m =
166/1024 TRAIN tasks; in-FOV count 246/1024 reproduces only at ±45° half-angle.

Decision: accept the audit as round-1 baseline. Require (a) FOV convention
stated next to every in/out-FOV count, (b) per-run runner hash — three copies
exist (`aabe46e7…` published, `ac953404…` calibrated worktree, `f3af77a5…`
learner) with no control-flow difference but diverging provenance, (c) the
learner's first experiment keeps its stated matched-arm design (teacher+BC+PPO
vs PPO, 10k rollouts each, dev-c predeclared) and files the reward-ordering
audit with the research mission before any harder task class.

Record: `results/omp-coordination/review-round1-2026-10-04.md`.

## Background-job lifecycle failure (2026-10-04)

Question: how did the first matched PPO arms die at 1700/2000 while the
control arm finished?

Evidence: research session 01a10753-8d11-7135-b9de-1e711df85f77 ended its
turn with "wait for bg_3"; the harness recorded `session_exit reason=dispose`
one second later; `shaped-s1.log` mtime equals that timestamp, its last line
is rollout=1700 with no `local_train_done`, no `.end` file, and no
`metal_nav_waypoint` process afterwards. The control arm, launched the same
way but finishing 46 s before the session exit, has a complete receipt.

Decision: never end a turn while a background training job still runs. Run
matched arms in the foreground under `run_locked.py` (blocking), or accept
session lifetime = job lifetime. The incomplete checkpoint is contract-safe to
resume (sidecar v3 binds all parameters; wrong-scale resume already refused).
Recorded as `research_run_incident` in dispatch.json with an exact
append-mode job-request in `results/omp-coordination/inbox/`.

## Background-job failure resolved (2026-10-04)

Follow-up to the lifecycle failure above: the contract mission claimed the
filed resume job-request as first claimant (locked, double-claim guarded).
The resume exposed a second trap: `local-train N` is INCREMENTAL
(`finish = completed + N`), so the job-request's literal `2000` ran the arm
to 3700 instead of 2000. The claimant declared the deviation with evidence
rather than silently patching; the coordinator then recomputed checkpoint
selection over the full history and over rows <=2000 and found the identical
winner (rollout 1950), so the matched-budget comparison survives.

Decisions now standing:
1. `local-train` resume arguments must be written as INCREMENTS (300 to reach
   2000 from 1700). Both the runner usage line and any job-request must say so.
2. Report curves from rows <=2000; keep >2000 rows as declared deviation
   evidence; `.best` = rollout 1950 is the shaped-s1 selection.
3. Remaining research work (seed 2 both arms, held-out/retention evals,
   RESEARCH_EXPERIMENT_RESULTS.md) requires a root respawn of session
   01a10753-8d11-7135-b9de-1e711df85f77; do not relaunch from a dead CLI.

## Second dispose-during-job failure (2026-10-04)

The learner mission hit the identical failure the research mission hit90
minutes earlier: its final turn was a "status while the arm trains"
message, the harness recorded `session_exit reason=dispose` at turn end,
and the background Arm C job (3250/10000) died with the session. Arm T,
which had completed6 minutes earlier under the same launch pattern, is
intact.

Decision (rule now with two independent data points, both fatal to the
affected arm):
1. Training arms run inside a BLOCKING foreground tool call. No harness
   background job (`bg_N`), no `&` launch whose script outlives the turn,
   no "status while X trains" turn end.
2. If a session is disposed anyway, the arm resumes only via an explicit
   contract-checked job-request, with the INCREMENTAL rollout argument
   computed from the sidecar (`local-train N` = completed + N), and
   `run-arm.sh`-style hardcoded targets forbidden for resumes.
3. Receipt discipline stays mandatory: `finished_utc/exit` line + history
   row count are what prove completion; absence of both at a dead CLI is
   the P0 trigger, as this round demonstrated.

## Third session-end child loss — and the resume-safe design absorbing it (2026-10-04)

The local-dynamics mission's `full-s2` chunk2 (rollouts1000→2000) was at
rollout1500 when its session ended at14:57:19: `logs/full-s2.bin.log`
stops without `dynamic_train_done`/`chunk_exit`, no stderr anywhere, all
other mission CLIs ended within the same15-minute window (learner already
down14:14). Same dispose class as the two earlier failures, but the third
data point differs in one way that matters: the mission's `run_arm.sh`
launches5×1000 foreground chunks and recomputes remaining rollouts from
the per-rollout diag, so the killed chunk costs minutes, not an arm.
No corruption, no duplicate: the next invocation resumes at the recorded
count. The earlier rule (blocking foreground call, receipt or it did not
happen, incremental-only resumes) is now supplemented by the chunked
pattern for any run longer than one turn's patience.

## Grounded matched experiment: prereg chain, v5 freeze, fail-closed preflight, root-executed arms (2026-10-04)

Timeline verified from bytes, not status prose:

1. Preregistration first: `decision.md`13:12 (before any generator code),
   then root's three pretraining corrections (13:14), contract adversarial
   review (13:20), root policy-resolution + goal-ray clarifications
   (13:23/13:25), and two user clarifications (13:37/13:40).
2. The first attempt violated the combined projected frustum (independent
   az/elev admitted72/1024 TRAIN +8+6 dev rows outside the true cone),
   had no stopped-issue bucket, malformed `labels.json`, block-concat mix
   layout, and a control arm started before inbox findings were read.
   Root aborted at preflight13:33: the arm died by SIGTERM at
   rollout≈4700/10000 (receipt exit241 = run_locked sys.exit(-15) chain),
   everything archived intact under `invalid-preflight-v1/`, no pilot
   outcome ever read. The coordinator independently reconstructed the
   signal chain from the receipt before finding root's abort record —
   both agree the ~191 s of GPU is waste, disclosed as such.
3. Corrected phase: dated `decision-revision-2026-10-04.md` incorporating
   every review item; generator reworked (yaw chosen after the route so
   goal and first bend sit in the cone by construction; four distinct
   visibility labels with a raw-ray gate and pooled bins as diagnostic
   only; velocity buckets `stopped_slow` U[0,0.2] and `independent_stress`
   U[0.2,1]; side round-robin; labels schema v2 with frozen exact
   denominators). v5 banks frozen14:06, self-validation zeros, witness
   flights on v5 retained both outcomes (918/1024,119/128,113/128 —
   coordinator re-derived from the summary JSONs).
4. Fail-closed gate: `local-train --preflight RECEIT` refuses grounded
   specs without an independent PASS receipt, mutation-tested (missing,
   FAIL verdict, mutated sha all refuse before any file write). The
   contract re-derived every hash, the passage frame and the denominators
   on the frozen bytes → PASS; role-specific receipts
   `omp-contract-preflight-{control,treatment}.json` bind C=source,
   T=grounded-mix, runner source784ca2f3; the single earlier receipt is
   marked SUPERSEDED.
5. Execution moved to root's parent (`root-run-all.sh`,15:31): four clean
   `armG2-{C,T}-s{1,2}` runs (10000 fixed, seeds20261005/6, fresh paths,
   pilot never resumed), then a75-eval matrix, analysis and plots.
   Verified: C-s1 and T-s1 exit0 with `local_train_done rollouts=10000`
   and correct role receipts; C-s2/T-s2 chained; in-run dev-g1@10000
   C118 vs T121 of128 — in-run telemetry only, not the predeclared
   endpoint (pooled blocked dev-g1+dev-g2 finals, ≥+10, retention ≥−3,
   safety ≤+3; any miss = honest negative, no sweeps).

Coordinator findings filed this round: the eval-matrix receipt template
hardcodes a `runner_sha256` matching nothing (binary3a72ec8d, source
784ca2f3, script f9bd2911 — P2, fix before the receipt lands); and the
dynamics interim's paired "best-vs-best" +16 reproduces only from the
FINAL checkpoint files (best-vs-best is +8:14/6) — basis must be labeled
and bound to decision §8 before seed2 is compared. Still unverified: all
four official outcomes, dynamics seed2 + cross-probe, collision parity
and arms, the rebuilt contract tar, the M4 analysis copy.

## Coherent outcome across four completed studies (2026-10-04 evening)

All heavy comparisons are now complete, root-verified against real
checkpoint headers and optimizer budgets, and independently reviewed by
the coordinator where gates were involved. Read together:

1. **Native256 (published3a3eb3e):** BC beats fast on reliability
   (113 vs105; blocked28/43 vs20/43; paired net+8) but is much slower
   (mean arrival6.17 vs3.61 s). Capability bought by behavior cloning
   includes hesitation; speed and reliability trade against each other
   on this exposed development bank.
2. **Grounded mixture (NEGATIVE):** the minimal grounded distribution
   genuinely teaches the new detour skill (+12 pooled blocked, both
   seeds positive, safety improved) yet loses dev-c by12 beyond the
   declared −3 floor. New-skill acquisition and prior-navigation
   retention are separate axes; the preregistered falsifier did its job.
3. **Dynamic ablation (NO ADOPTION):** real source learning gains
   (warm27 →109/112) but the selector is development, the cross-drop
   prediction went0/3, and the code shows the ablated arm retained a
   cached8-frame geometry prior — the original contrast isolates only
   the direct previous-depth channel. A corrected ablation is properly
   predeclared (`decision-corrected-history.md`) before any new run.
4. **Penalty10 vs50 (NO ADOPT):** the human hypothesis is real for
   safety (contacts55→17, success +13) and real retention gains, but
   the policy buys caution with stalling (timeouts0→25, time
   +2.04 s) — exactly the failure mode the no-stalling gate was
   written to catch. Reward strength moves the safety/progress
   frontier; it does not by itself deliver fast-reliable pathfinding.

Common methodological lessons now standing as rules: receipt+header
or it did not happen; basis labels on every paired table; selectors on
development banks are labeled as such; driver/fixture bugs are fixed
and disclosed without touching PPO/reward; ablation claims require
reading the actual memory path, not the intended one; frustum
conformance is never a universal acceptance rule (goal.md operator
clarification). Failure modes are documented with full denominators —
no study was rerun to change its verdict.

## One navigator: strategy and completed follow-ups — 2026-10-05

Astra reviewed the actual trainer and completed flight records. Its strongest
explanation was task coverage, unequal transition influence and interference
between learned skills. Longer episodes can dominate samples even when older
records remain in the bank. A dev-a-only selector cannot enforce combined
capability. Larger networks and replacement algorithms were not established
as the missing ingredient. The report is `UNIFIED_POLICY_STRATEGY.md`.

Root implemented and executed these follow-ups with the preserved actor:

- Long-goal coverage: long open 2→255/256 and hallway 0→256/256, but static
  retention failed. Four matched 10,000-rollout runs completed.
- Parameter anchoring: drift fell from about 9.3 to 1.2–1.4. Retention and
  speed gates still failed. The coefficient stayed fixed in later studies.
- Motion correction: all 384 prototype moving paths intersected scene geometry.
  Root stopped unfinished invalid runs, retained evidence, implemented bounded
  sphere motion, and checked full paths and source/native positions.
- Balanced combined training: 50% static, 25% long and 25% bounded-course
  transitions. Course success 223→243/256, contacts 33→13, but arrival regressed.
- Independent native comparison: 96 valid flights, parent 43/48 versus combined
  45/48 successes, contacts five→three. Published `4e0a9f3`, 598 hashed inputs.
- Arrival analysis: reference windup did not explain the observed timeouts.
  Late controller interventions diagnosed recoverable failures; they were not
  credited as learned navigation gains.
- One-actor teacher consolidation: two 2,000-update students, then four matched
  10,000-rollout PPO runs and 64 evaluations. Final course success 217→240/256,
  contacts 39→16, long open 253→255 and hall 255→256. Short open 254→246 and
  static B 227→220, with slower arrivals. Full gates failed; no promotion.

The actor remains 184/64/4. No deployed teacher switch, route oracle or larger
network was introduced. Checkpoints and useful failures remain preserved.

## Perception support: completed negative result — 2026-10-05

Four 10,000-rollout arms and 48 evaluations completed. Final primary success
was C202/T197 of 256, contacts 54/59; fresh dev-k 192/189, contacts 64/66.
The experiment was not adopted. Its candidate-query optimization retained
100-rollout full-checkpoint byte parity while reducing wall time from 69.89
to 8.11 s. This is a measured speed gain, not a navigation gain.
Records: `results/omp-perception-support/`; report: `PERCEPTION_SUPPORT.md`.

## Rays and navigation frequency: useful hypotheses, not fixes — 2026-10-05

The user proposed more rays, wider angles and higher navigation frequency.
Existing collision attribution found 17 of 27 contacts never raw-ray measured.
Recasting 5,120 rays at the same saved poses rescued zero of those 17; six
complete colliders stayed outside view. Ten contacts had raw and pooled hits
within the last 0.4 s. A hit alone does not establish enough warning or a
safe available action. Source: `results/root-contract-review/existing-visibility-review.json`.

More rays could help thin obstacles if angular sampling is the limitation.
They cannot reveal geometry outside the field of view or behind occlusion.
Wider sensing is a stronger distinct hypothesis, but widening at fixed ray
count reduces angular density. Test coverage and density separately, preserve
capture pose/time and unknown-space semantics, and include body-volume margins.
The Fly360 primary paper studies panoramic depth for omnidirectional motion:
https://arxiv.org/abs/2603.06573 . Its results motivate a test here; they do not
validate our sensor or policy.

At 20 Hz the maximum navigation sampling wait is 50 ms. At 50 Hz it would be
20 ms, a 30 ms reduction; at 3 m/s that is 0.09 m of travel. This calculation
excludes sensing, inference, transport and motor response. Raising inference
rate alone cannot create fresh depth or repair missing coverage.

A rate experiment must preserve elapsed physical time, sensor cadence/age,
command hold, RAPTOR at 100 Hz, reward units and the discount/GAE horizon in
seconds. For a new navigation step, use gamma_new = gamma_old^(dt_new/dt_old)
when matching the same continuous discount horizon. Treat action smoothing,
previous-action history and observation contracts explicitly; merely changing
rollout step count would confound the result. Compare frozen-rate sensitivity
first, then matched learning on the same multi-capability bank if justified.
No rate or wider-FOV learning outcome is claimed yet.

## Stress harness review and cleanup — 2026-10-05

Luna independently found a runtime guard that blocked dynamics evaluation and
a matrix launcher missing its preflight gate. Root fixed evaluation constructor
ordering, retained the training guard, added full sampled-plant validation and
exclusive finite grading, and published a readable receipt-gated runner.
All four preflights now pass, including nominal scored-flight parity. The
120-evaluation frozen matrix is a diagnostic, not actor tuning. Ego remains
ideal and wind is zero; declared plant bounds are not hardware identification.

The README, document index, code map and handoff now distinguish completed,
invalid and live work. Stale perception status and historical fidelity claims
were corrected. Optional research build targets use a separate directory;
frozen binaries and unpublished drafts were preserved. Changes follow the
operator's explicit-data-flow and measured-performance code standard.

## Completed frozen stress outcomes — 2026-10-05

The corrected harness completed all 120 evaluations / 15,360 source flights.
Root checked every flight and plant hash and exclusive outcome. On bounded
courses, the consolidated-PPO actor achieves 240/256 nominal versus 217 for
its matched control. It falls to 227 with sensor delay, 218 with command delay,
211 with both, 232 with plant variation and 191 with combined stress. Contacts
rise from 16 nominal to 45 with both delays and 65 combined. Matched control
combined success is 190/256: nominal gains almost vanish under joint stress.

This is evidence of a robustness gap, not proof of one unique cause. Next
analysis must inspect actual command history, sensing/warning time and sampled
plant for paired failures before changing training. Check second-order effects
on braking, arrival, static retention, physical horizons and inference cost.
The user's standing rule is now explicit: diagnose the cause, check the fix and
its effects on other behaviors, then rerun. Do not replace diagnosis with a
parameter sweep. Full records: `results/root-stress-matrix/`; current results:
`COMBINED_POLICY_PROGRESS.md`.

## Actual delayed traces and capture-age repair — 2026-10-05

Root added read-only postflight trace capture and replayed six frozen panels.
All grades match the original matrix exactly. Both delays produce 37 contacts
on nominal successes: all 37 show some measured depth below 1 m in the final
0.4 s, and 14 have latest requested speed at least 0.1 m/s below the applied
queued speed. Combined stress has 57 lost successes, 55 with nearby depth and
22 with that command gap. Nearby geometry is not contact-object attribution.

The actual memory radius age omitted transport delay of the latest usable
capture. Root added a shared opt-in age calculation, propagated it through
CPU, Metal and the native frontend, and preserved default behavior. The
host/Metal check verifies the 100 ms uncertainty delta of 0.015 m. Six nominal
panels reproduce every scored row. Ten frozen correction evaluations show
both-delay success 211→202/256 but combined 191→197, with one new timeout.
No adoption: the second-order delay losses outweigh treating code correctness
as capability. The correction remains disabled; no coefficient sweep or
higher-frequency training was launched. The next decision must distinguish
command anticipation, sensing coverage and insufficient mixed-stress exposure.

Code, 57 hashed input files and a public standard-library reviewer preserve
all outcomes: `evidence/inputs/stress-failure/records.tar.gz`,
`navigation_stress_failure_review.py`, `COMBINED_POLICY_PROGRESS.md`.
The matrix source snapshot is `c3585ed`; later diagnostic changes do not
rewrite its freeze record. Only matching baseline/profile data are compared.

## Matched delay learning — implementation and launch, 2026-10-05

The joint trainer's actual configuration used sensor_delay=0 and command_delay=0.
The new comparison tests this missing exposure rather than another reward,
network or geometry coefficient. Control and treatment use identical combined
TRAIN banks and corresponding consolidated-PPO warmstarts, with the same actor,
rich critic, reward, anchor, fresh optimizer, physics and navigation timing.
Treatment has both100ms delays in half the lanes, with clean rehearsal in half.

Root caught a second-order assignment error before launching: even/odd lanes
would make every long lane clean and every moving-course lane delayed because
the capability groups repeat every four environments. Alternating four-lane
blocks gives both delay strata32static/16long/16combined environments. Actor
inputs contain no group identity. Actual exposure is logged per rollout.

Preflight passed default100 full-checkpoint byte parity, delay8 versus4+4 resume
byte parity, wrong-delay rejection before mutation, actual sensor-age exposure,
and stratified bank allocation. Sidecarv3 binds delay settings and rehearsal;
checkpoint/network ABI remains unchanged. Capture-age correction stays off.
Full comparison: four10000rollout runs on seeds20261180/81, then48 evaluation
panels. Final checkpoints are primary; source retention, contacts, timeouts,
common-success timing and absolute floors remain independent gates. No training
outcome is claimed before saved headers and full evaluations. Protocol:
`DELAY_LEARNING.md`, `results/root-delay-learning/decision.md`.

## Terminal contact objects and execution recovery — 2026-10-05

The interrupted tool runtime ended the delay-learning producer and evaluator.
Root verified that the real prior PIDs were absent, all frozen input hashes
matched, control seed1 had 10,000/320,000, and treatment seed1 had 6,850/219,200.
The new persistent parent resumed treatment with 3,150 incremental rollouts,
retained original receipts and logs, then continued seed2. No completed arm
was repeated. This is execution recovery, not a changed training budget.

Offline terminal-pose attribution uses the exact shared collision SDF and
bounded-motion source. All 126 contacts from the old nominal/both/combined
matrix reproduce nonpositive terminal clearance. The 37 contacts on nominal
successes under both delays consist of 28 boxes, 5 cylinders and 4 bounded movers.
Combined stress's 57 new contacts consist of 31 boxes, 11 cylinders, 5 movers and
10 room-boundary contacts. Thus a moving-threat-only interpretation would miss
most of this failure. Nearby depth is still not contact-object warning proof.

Root added a read-only dump of the actual retained terminal depth/pose ring,
with explicit frame count, sensor period/delay and physical navigation period.
Only actor-available frames are graded; newer undelivered frames are excluded.
The offline visibility grader distinguishes geometric target rays, valid measured
rays and selected pooled points, and attributes boundary contacts to their actual
face. Its dense 5,120-ray alternative is an offline geometry counterfactual at
the same pose/FOV, not measured sensing or a new flight. Sensing replay and
scored parity must complete before drawing visibility conclusions.

## Retained camera clock validation — 2026-10-05

A pre-publication range check caught a diagnostic time error. Using frame times
nominal navigation period disagreed with the shared accumulated float32 physics
clock. One grazing ray changed first surface, with a 0.865721 m range discrepancy.
Replaying native 0.01 s accumulation removes it: all 1,024 retained nominal
captures /327,680 rays agree within 2.5034e-6 m, and 1,024 delayed captures agree
within 6.4373e-6 m. Training and recorded flight grades were unaffected.

The visibility grader now uses that validated capture clock and excludes
undelivered sensor frames. This is a clock replay, not a claim that the v1 ring
format records independent timestamp measurements. Combined noise/dropout may
change measured ranges by design; only the clean captures support this direct
range parity check. All six sensing receipts are still required before the
contact-object visibility verdict. The new delay reviewer similarly requires
four real saved10,000/320,000 headers and all48 evaluated panels, checks paired
plant bytes and stable arrivals, and separates absolute floors from delay gains.
Interrupted exposure journals retain abandoned rows; last occurrence per saved
rollout belongs to the resumed continuation and avoids double-counted budgets.

## Complete delay learning and background handoff — 2026-10-05

Four final10,000/320,000 headers and48 evaluated panels complete. Both-delay
success203→231/256, contacts53→25, shared-success time−0.602s. Combined stress
198→226, contacts58→30. Nominal course240→246, longopen241→256, hall255→256.
StaticC214→201 and shortopen237→233 fail retention; absolute floors also fail.
No promotion. The214-input archive includes banks, warmstarts, source and raw
flights; `navigation_delay_review.py` recomputes every gate.

Full retained sensing review: contactobject measured in raw and pooled depth
for28/37 newbothdelay contacts and48/57 combined contacts. Dense5120 samepose/FOV
geometry rays rescue2/9 bothdelay misses and0/9 combined misses. This is a retained
window and offline counterfactual, not a wholeepisode visibility or safety proof.

Fresh native evaluation froze four final actors and48 new nominal courses,
including reflected travel/height geometry. The user reported recurring macOS
focus stealing despite batch/minimize/no-rendering. Root stopped only the owned
parent/child, preserving112/192 valid flights and an unscored interrupted run.
The nonrecording child environment now disables Qt foreground transformation;
installed Cocoa plugin support verified. One valid duplicate check kept Chrome
foreground in20 samples, with original9.4s arrival/minrange/hold unchanged.
This is background Cocoa/OpenGL, not trueheadless. Official Webots headless
setup uses Linux/Xvfb or Docker: https://cyberbotics.com/doc/guide/installation-procedure .
Future resumes must record launcher-only supplemental provenance and preserve
actors/selection/raw receipts. No remaining80 flights were launched for this
handoff. `HANDOFF.md` records exact state and the next evidence-producing steps.

## Experience-scale continuation and Astra review — October 7

The operator explicitly keeps the original navigation goal and now prioritizes
experience amount/diversity/ordering, competence-based curricula, direct-mixed
PPO controls and capacity as a separate axis. Larger teachers may be distilled
and quantized for an ESP32-S3-class target, but retained behavior and complete
onboard latency/memory costs must be tested. Webots is deferred. The local
RL VM was deleted on request; it was never Cloud Codex.

Root enabled explicit-bank learner counts without tying fixed128 development
evaluation to learner N. Default100 FULLcheckpoint byte parity and N512/fixed128
smoke passed. Frozen corpus tiling does not add unique worlds. Phase balance
avoids large-N first-slot-only exposure; the schedule is explicit and fixed.

Four single-seed pilots consumed4,194,304 samples/32,768 Adam steps each.
Throughput N128/512/2048/8192:98k/167k/198k/180k samples per total job wall second.
Astra correctly identifies this as practical PPO collection/batch regimes, not
hardware alone: policy refresh, normalization and stale updates differ.
N2048 is fastest but longopen falls128→102. N512 open/long/hall all128, staticC108
and clutter114, but course119 versus warm123. N8192 better retains difficult
courses/delays while improving staticC110 and clutter112; open125,long127.
This is not multiseed evidence or a network limit. Full60 evaluated panels and
raw telemetry are in `results/training-scale/`.

Next root run compares N512 and8192 on both T parents,41,943,040 samples/327,680
updates each (167,772,160 total), with all other axes fixed. Cheap inference-only
snapshots support retrospective capability/time curves without duplicating full
environment state. Original-record exposure now counts actual rotated entries;
logical slots alone cannot prove curriculum coverage. Counter/snapshot changes
retain full-checkpoint byte parity. No policy promotion from FPS or reward.

The one requested Astra report is `TRAINING_SCALE_REVIEW.md`. Fixed64 simulator
scratch arrays, shared actor/critic widths and exact-count warm loaders block
safe capacity experiments. Root will correct those only when the capacity axis
begins, preserving critic size and initial represented behavior. No architecture
cleanup or simultaneous curriculum/reward/network change is authorized by this
phase. Archived unpublished drafts and rejected media retain exact hashes.

## October 7 — more experience and fresh task checks

Four matched PPO regimes completed 167,772,160 transitions, with 480 scored
development panels. N512 improved short/static retention; N8192 retained long
and course skills better. All 2048 payloads were actually visited, with exact
lane and clean/delay transition shares. Missing level coverage alone does not
explain the tradeoff. No final policy promotion.

Root generated 896 fresh seeded tasks after checkpoint freeze and scored six
policies across 90 panels (11,520 flights). Clutter pooled warm205/256 becomes
225 at N512 and211 at N8192. Reflected course combined stress172 becomes159/184.
The skill tradeoff persists beyond the old development records. Same procedural
families, not sealed FINAL or independent-simulator proof.

Results, curves, retained contacts/timeouts and reproduction are in
[EXPERIENCE_SCALE_RESULTS](EXPERIENCE_SCALE_RESULTS.md). Adaptive sampling is
next, within the fixed three capability lanes. Root found the helper's normalized
weight formula did not implement the documented exact25% uniform mixture;
requested correction plus a numerical heterogeneous-count test before training.
Episode-boundary selection and outcome attribution remain under implementation.

## October 7 — controlled actor-capacity mission

User explicitly requested OMP DeepSeek to implement and run the larger-network
experiment, then root to wait/analyze. Root supplied function-preserving warm
starts for widths64/256/768/2560/5120, about12k/50k/150k/500k/1M parameters,
CPU prediction parity and unchanged critic bytes. OMP implemented isolated
actor/critic dimensions, wide Metal forward/gradient kernels and bounded gates.
Root verified49 frozen inputs and full41.94M width64 seed1 checkpoint byte
parity against the published N512 control.

Initial OMP ended while secondcontrol background tool was incomplete at1280.
Root confirmed original processes terminal and launched persistent parent-owned
recovery from actual header, then wide arms/full grading. Root corrected eval
filename sample labels to rollout*512*32. No wide-policy capability claim yet.
[Experiment direction](CAPACITY_EXPERIMENT.md); actual receipts and reporting
under results/omp-capacity-scale. Luna's sampler integration stopped on quota
and remains uncommitted/unverified; it is separate from capacity training.

## October 8 — capacity outcomes and a corrected gradient check

All10capacity arms completed419,430,400transitions and1530panels/195840 source
flights. Wider actors lose overall retention under the fixed recipe. Nominal
course success220/256 at12k becomes149/256 at1M; contacts35->105. The1M
long-open244 vs240 gain is outweighed by hallway256->159 with97contacts.
No promotion. Both small-actor full checkpoints match priorN512 controls exactly.

Root found original capacity-parity copied zero advantages, making actor-gradient
checks mostly entropy-only. New nonzero derivative probes on real observations
and all10trained models validate each layer and the added W1 rows. Raw code and
receipts retained. This correction concerns verification, not a changed learner.

One-rollout/128Adam fixed-observation probe: latentGaussian KL .00691 at64,
.04267at2560 and .09066at5120. Larger functional steps are a concrete hypothesis
for a targeted actor-step experiment, not proof that capacity itself is useless.
Critic/reward/task changes will remain separate. Root corrected delegated-report
overclaims and a misleading normalized-success-per-hour metric.

[Capacity report](ACTOR_CAPACITY_RESULTS.md) has all sizes, costs, fresh task
results, failures, corrected verification and reproduction. Original goal active.

## October 8 — actor-step full outcomes

Two full factor0.4 treatments completed41.94M transitions/327680steps each;
92paired panels at pilot and full endpoints pass matched input/stable arrival
checks. Actor-only LR.00004, criticLR.0001, all other settings fixed. Controls
reused; fresh pilot controls actually executed. No policy promotion.

Same-size full longopen55->219/256, hallway222->256 with contacts34->0,
clutter181->208, nominalcourse169->176. Pilot nominalcourse248 erodes to176
and contacts8->79 after more exposure. Fresh reflectedcombined135->118 with
contacts120->137. Narrow same-budget control remains stronger overall.

Preserved pilot fullweights and entrycounts before incremental2304rollout
extension; combined exposure exactly41,943,040 perseed. Root approved unchanged
code/input/CLI after correcting stale latefreeze/expandedparity request metadata.
Default1narrow/wide100byteparity, nonzero gradients and critic Adam isolation pass.
KLprobe .04267->.00612 supports a cause but not a complete learning fix.

Next SAME-session OMP mission investigates learning erosion using actual
reward/likelihood/GAE/data/control paths and matched failure flights. Read-only
learners; no new training or coefficient sweep before a cause is established.
[Results](ACTOR_STEP_RESULTS.md) preserve both seeds and the failure story.

## October 8 — corrected erosion audit and root review

Eight frozen stochastic source runs completed twelve consecutive 32-step windows
each. Root replayed every CSV reward/value/mask and normalization independently;
96 windows pass with maximum double replay error 0.000018 (GAE), 0.00011 (normalized).
The four low-clearance loss tables reproduce. The old hard-horizon explanation
was withdrawn. No new learner or anchor coefficient is accepted from this evidence.

Important labels: action columns are sampled latent actions; clearance is outside
the 0.18 m collision sphere before the action, excluding physics substeps. Equal
bank tasks need not retain equal random draws after unequal episode lengths.
Low-margin loss association does not establish optimizer causality. Prior anchor
results already show reduced drift with incomplete retention and slower arrivals.

Root wrote an independent replay and decision in results/omp-erosion-contract-repair.
A new same-session bounded update audit now inspects cloned saved early/full Adam
and simulator states, with parity before instrumentation. It must identify a real
update mechanism and its second-order costs before another long learning change.

## October 8 — output intervention and one composed actor

Root reviewed the optimizer audit and rejected another advantage-cap trial.
The probe used incorrect policy-active log-ratio thresholds; the learner uses
correct probability thresholds. The clipping, noise and gradient-share causal
claims are not established. The diagnostic remains useful update evidence.

Early yaw failed both registered course gates. Early XYZ with late yaw rescued
67 of 73 lost early-solved courses with two new losses. Course success 176 -> 244
of 256, contacts 79 -> 12; long open 219 -> 256; short open 225 -> 250; static B
214 -> 228; reflected combined stress 118 -> 170. Stress remains weak.
Root joined hidden neurons and output heads into ONE 967,688-parameter MLP.
All 12 single-actor panels, 1,536 actual flights, match the dual-model CSV files
exactly. No new learning or independent transfer is claimed.

[Output recovery](OUTPUT_CHANNEL_RECOVERY.md) records the intervention, exact
composition and replayable 135-record bundle. The next OMP mission owns only
results/omp-functional-retention: training-only physical XYZ teacher loss,
derivative/Metal/resume/default parity and a frozen two-seed pilot request.
Root reviews before heavy execution. The teacher is removed at inference;
shared-feature yaw and exploration effects still need checks. This larger
candidate is a research teacher, not an assumed microcontroller deployment.

## October 8 — root repairs and executes the retention gates

The OMP implementation ended before its derivative, calibration and full parity
procedures were complete. Its coefficient 0.05 request had no calibration and
was rejected. Root preserved it and completed actual nonzero derivative tests.
Root fixed the always-failing descent fixture, batch buffer alias, sidecar path
and parameter provenance, and bound the new loss source into enabled resumes.

Actual Metal derivative relative L2 is 3.6e-7; shared hidden derivatives are
nonzero while the direct yaw head remains zero. Enabled six versus three-plus-
three FULL checkpoints match on both seeds. Nine wrong-resume cases reject
without writes; disabled compatibility passes. One TRAIN-only quarter-gradient
calibration gives coefficient 4.134632354, not a development-selected value.

Root persistent validation now finishes default 100-rollout parity. A separate
root pilot parent verifies those real receipts before accepting four matched
256-rollout arms and 138 evaluation panels. No new loss outcome is claimed yet.
[Experiment](PHYSICAL_RETENTION_EXPERIMENT.md) records the intervention and gates.

Root validation subsequently completed both full 100-rollout byte comparisons.
The persistent pilot parent accepted the actual calibration and immutable files,
then launched seed 1 control. PID 14287 has real rollout history past 75; the
other three arms and 138 panels follow serially. No navigation improvement from
the loss is claimed before the complete paired comparison.

## October 8 — completed physical-retention learning results

All four 256-rollout runs and 138 panels are complete: 16.78M transitions,
17,664 scored flights. Independent root review verifies headers, exposure,
identities, stable arrivals and every gate. Course C204/T245, contacts49/11;
both-delay185/236; long-open234/256; reflected combined147/181. Parent reflected
170 improves in both seeds. Treatment costs21.5% more training wall time.

Five gates fail: staticC -5, A/clutter +4contacts, old/freshopen249/250 below253.
No promotion/full extension. Root repaired the evaluator CLI after training,
proved nominal/combined parent parity and graded without repeating learners.
[Results](PHYSICAL_RETENTION_RESULTS.md) and754hashed archive inputs replay exactly.

Three frozen arrival traces preserve all scored flights. Teacher closing velocity
is weak/negative near goal while ordinary PPO predicts positive closure on the
same student states; teacher can preserve a weakness. Arrival-aware teacher loss
is one new masked-approach hypothesis, with avoidance risk near goals explicitly
gated. The user also requested an independent OpenCode free MiMo critic/training
audit; it is actually responding. Both own isolated result folders and cannot
change the frozen source or begin pilots before root code/evidence review.

## October 8 — critic audit premise correction

Root checked the frozen ppo.metal directly: old_values is unused and the critic
uses plain mean-squared value error, with gradient value_coef*(V-return)/batch.
There is no absolute value clipping at0.2; that parameter clips the actor's
probability ratio. The audit must not recommend disabling a nonexistent value
clip. Motor state named rpm is normalized0..1, so its name does not demonstrate
input saturation. MiMo's log analysis identified the same value-clip correction.
Actual predictive-quality and input-alias probes are still in progress; mean
ratio or aggregate value loss alone does not settle critic quality.

## October 8 — arrival teacher-release attempt rejected

Root independently reviewed two masked256-rollout headers/exposures and184
comparison cells (46new,138reused). The targeted open failures worsen249->238
and freshopen250->245; longopen256->232 and reflected combined181->168. Both
seeds regress on oldopen and longopen. Nominalcourse245->246 is insufficient.
Shared long arrivals slow1.58s. No promotion, radius sweep or full extension.
[Result](ARRIVAL_RETENTION_RESULTS.md) retains the original hypothesis/failure
and narrows the earlier counterfactual-command diagnosis. Critic audit is still
working with actual on-policy data; its findings must be reviewed before a fix.

The rejected arrival-release attempt now has a666-input archive with final
checkpoints, all retained/new comparison flights, source and actual gates.
Archive-only grading is identical to live review, including every failed check.
The figure and standalone replay are published with the negative outcome; no
failure was removed to present a stronger policy.
