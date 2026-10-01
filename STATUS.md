# State — 2026-10-01, overnight run

Goal is active. No skills; only gpt-6-luna workers. User is asleep and requested continued work without questions. Xcode installation is allowed but runtime MSL compilation works without it. Host: Apple M3,10 GPU cores,16GB. ARM SSH oracle-vps available but unused. Git identity: Forge the World /178587964+forgetheworlds@users.noreply.github.com. Make clear verified milestone commits.

## What works

Actual RAPTOR weights with official16-step oracle parity (~6e-7), per-env GRU state at100Hz; L2F RK4/motor-lag physics with512 official fixtures (~1e-6); independent CPU/MSL ray/collision primitives. Navigation uses16x20 ray range/current+previous history, goal/ego/previous action/sensor age/reference error; privileged critic separate. Integrated CPU/GPU160-control-tick trajectory parity <5e-6. Fixed PPO: GAE boundary semantics, Gaussian raw-action likelihood, clipped objective, gradients, Adam. Existing test command covers required correctness; user requested focus on real learning, not more tests.

Full GPU rollout/training buffers are allocated at setup. No per-step CPU wait. NAV20Hz, depth20Hz default, actual RAPTOR/physics100Hz. Persistent reference position integrates commanded world velocity; reanchoring p+v*0.5 was a measured bad adapter and is isolated in results/open-old-adapter.bin. Delay rings8frames; first delayed commands holdzero until arrival. Nonfinite dynamics fail/reset without contaminating PPO.

## Current evidence

Raw actor661 inputs,64hidden,4latent actions. 128envs×32rollout,two PPO epochs,minibatch256. Direct gradient scratch128KiB vs43.8MiB; full update1.66s→0.295s. Hardware timestamps identified forward and serial gradient norm; SIMD reductions +8x8 fused actor forward in both collection/training now ~26ms/update, same sensor/physics/PPO work. Scalar actor remains via METAL_NAV_SCALAR_ACTOR=1. Optional profile uses real M3 encoder-boundary timestamps (dispatch-boundary sampling unsupported). gpu-bench N R FAMILY and cpu-bench R N FAMILY compare complete workloads.

CPU reference uses Accelerate SGEMM/GCD. Root fixed workspace sized onlyfor256 samples before N>256 benchmark; discard older larger-N CPU numbers. Valid box workload: N512CPU3rollouts0.310s; N2048CPU1.230s vsGPUwall0.675s; N8192CPU5.016s vsGPUwall2.707s. Small N128 is roughly equal. This is a matched optimized CPU reference, not scalar microbench.

Pure PPO open training reached100% validation at150rollouts. Box policy results/boxes.bin.best freshseed800001:81.25%success vsgoal-script47.66%, random0%. Mixed7 policy trained1500rollouts in52.2s after box warmstart; best validation81.25% at380rollouts/13.55s. Fresh mixed test75.78% vsgoal-script65.63%. Blind-depth box test47.66% supports actual perception use. Held generator failure: doorway family4 only6.25%, table/counter5 only44.53%; generalization is not good enough. Do not mark complete.

Checkpoints v3 save actor/critic/moments plus physics/worlds/RNG/GRU/reference/depth+command rings/config. CHECKPOINT.best selects validation success then time. One results/training.tsv logs allruns. Latest: results/open.bin390; results/boxes.bin600; results/mixed.bin1500. Their .best files exist. eval CKPT MODE FAMILY SEED SPEED DIST SENSOR_DELAY WIND_ACCEL NOISE DROPOUT COMMAND_DELAY; modes4learned mean,2goal script,1random. train R FAMILY CKPT WARMSTART SPEED DIST.

## Active work / next steps

All Luna worker changes are returned. Both raw661 and pooled181 binaries build and pass existing checks. Pooled preserves320-ray sensor and uses80minimum-range cells/frame plus21context. RuntimeMSL compilemacrobridge matches C++ architecture. Family4 now has±0.6m doorway offsets; heldfamily8 has two sequential offsetdoors with verified waypoint routes. Mixed7 now trains on ALLfamilies0..6. No worker currently owns root files.

Active exec session54912: build both, then `metal_nav_pooled train3000 7 results/pooled-broad.bin '' 1.5 4`, then raw `metal_nav train3000 7 results/raw-broad.bin '' 1.5 4`. Same seed/config/population; compare architecture fromscratch, not cross-load weights. Both checkpointcounts protect wrong architecture. Wait/poll before launching GPU benchmarks.

Next: evaluate both `.best` on freshseed800001 across families0..6 and held8, plus script/random. Checkblinddepth, sensor100ms delay/noise/dropout andwind. Ifweak, add measuredcollision-risk shaping or curriculum; do not claim fullrobustness on aggregate reward. Raw legacy mixedpolicy75.8%fresh buthelddoor6% indicatesrepresentation/generalizationgap. Savequalityresults andtime-to-threshold in one training.tsv. Exportbestdeployableactoronlywhenqualityverified.

RAPTOR reduced-array and group32probes LOST (paritypassed): SimAdvancemedian5.52→5.95ms; originalraptor.metal restored. Keepgroup64. Sourcecleanupremoved unadopted pointer/two-stage forward; selected8x8 fusedforward stays. Optionaltiledgradientprototype notusedyet, profile first.

App continuation heartbeat overnight-metal-navigation-build every15min, same chat. Caffeinate own exec session93901, caffeinate -i -s -t14400, onAC100%, holds awake through~08:00local; stop it on actual completion. Keep concise state here for continuation.

## Latest continuation detail (after 9df5931)

Broad fromscratch3000rollout ablation completed: pooledbest75.78%validation at720rollouts/15.55s; freshmixed75.78%, heldtwo-door48.44%. Rawbest83.59%validation at2680/79.69s; freshmixed82.03%, heldtwo-door31.25%. Goal-script heldtwo-door38.28%. Raw giveshigher in-distribution quality; pooling givesbetter compositiongeneralization and~16ms vs~23ms perupdate. Do not discard either evidence.

Risk/low-entropy finetune3000 completed forboth: pooledriskbest74.22%validation, held52.34%; rawriskbest86.72%validation, freshmixed82.81%, held39.84%. Clearance alone doesnotfixrepresentationgeneralization. Addedvelocitycontract1: equalXYZ body scaling plus spherical desiredspeedcap; legacycontract0 wascomponent caps with vertical×.5, which also distorted scriptedgoal compensation whiletilted. Pooled-sphere3000 frombroadbest:best78.9%validation; freshmixed75.8%, held46.9%. Stillnotrobust.

Checkpoint writer nowversion5,128byteheader (SimConfig80bytes,SimRun184). Commonheaderreader acceptsversions3/4/5: legacy3header112bytes; version4risk/entropy/lr fields withlegacyvelocitycontract0; version5sphericalcontractflag. Oldactors loadfor eval withtheir recordedcontract. Newtrainargs afterSPEED/DIST: RISK ENTROPY LR VELOCITY_CONTRACT. Newrisk/lr fields persisted/validated onresume. Currentdefaultcontract1. CPUreference v5 rewardrisk branch stillneedsupdate ifbenchmarkingnonzero-risk (defaultmatchedbenchmarkrisk0 isvalid).

CURRENT WIP: rootintegratedguided184actor (80cur+80prev+21ego+3depth/goalprior features); residualhead adds3priorlatents toXYZmeans, yaw residualnormal. ThirdCMake targetmetal_nav_guided. main/runtimeMSL includesguidance.hpp; thisfileisbeingcreatedbyLuna physics(workercurrentlyrunning). BUILD WILL FAIL untilguidance.hpp exists. Rootmustwaitforworker orimplementminimalapprovednav_guidance signature. OwnedworkerONLYguidance.hpp; rootchangedotherfilesforintegration. CPUreferencecontextdepthfeaturesderived184→80; baseline181/661preserved. Oncefilearrives build3targets, guidedtestrequirednewmeanparity, geometric-onlymode9 vslearnedmeanmode4 realevaluation/train. Preservecurrentmeanlikelihood consistency. No otherworkeractive. Sessions30137risk and7331sphere and54912broad areFINISHED. GPUidle. CaffeinatePID83796/session93901 stillactiveuntil~08:00.

## Guided outcome / current next step (latest)

Build nowWORKS: guidance.hpp arrived, rootrenamedapprovednav_guidance. Three binaries build; guided184 full existingtests pass (newprioraddedto mean consistentlyCPU/Metal). Guided actor:80current+80previous+21ego+3atanhXYZprior; no privilegedgeometry in prior. ResidualPPO updates arecorrect becauseprior isweight-independent. Guideprior usesdepthcones/goal/egovel/time; no low-levelcontrollerchange. Guided3000 complete(results/guided.bin,bestat90rollouts):validation86.72%. Freshmixedmode4=80.47% vsgeometric-onlymode9=67.19%; heldtwo-doorlearned75.78% vsgeometric85.16% (full residualtooaggressive).

NewINFERENCE ablationmode10=prior+0.25learnedresidual (yawalso×.25); mode11=.5; mode12depth-gatedresidual. Mode10freshmixed78.13%, heldtwo-door92.19%seed800001 and94.53%seed900001. Heldwith100ms sensorlag+50ms cmdlag+0.05mnoise+10%dropout89.84%. Dynamic-spherewiththosecorruptions89.06%. Singleoffsetdoor99.22%. Constantwind0.5m/s²mixed76.56%. FAILtable/counterfamily5only34.38% (49%collision,16%timeout). Thisvertical-clearancefamily isnext mainqualitygap. Do NOT markfullgoalcomplete.

Guidance geometric-only vslearnedresidual distinct: mode9noresidual,4full,10quarter. Preserve guide184checkpointsdimensions and v5velocitycontract1; usecurrentbinary foroldv3/4checkpoints toloadlegacycontracts correctly. Next: tracefailedtableworld todiagnosegeometryview/clearanceprior vsRAPTORtracking; improveprior ortrain curriculumfamily5, thenevalbroadranges. CurrentMVP scores3x3mincones, choosesgoal-alignedfree-ray, bounded.8atanhhint; maymissverticalclearthin tableplanes/stop tooearly. Extend fullsensorbodyclearance/waypointmemory ifneededwithoutprivilegedactor. Exportpolicy/evaluationdataafterqualitywork. No GPUjobs currentlyactive. AllLunaagentsdone. Goalapp and15minheartbeatremainactive. No questions needed.
