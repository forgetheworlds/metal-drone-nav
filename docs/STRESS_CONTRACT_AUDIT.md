# Navigation stress evaluation contract audit

Audit scope: source and the current `results/root-stress-matrix` runner. This is a code review. It does not claim completed dynamic or combined flights.

## Resolution — October 5

Root corrected the constructor order without relaxing the nominal training
guard. Evaluation installs tasks first, then enables runtime plant variation
and resets. Both dynamics and combined profiles now complete actual flights.
The public matrix runner requires all four successful 128-task preflights,
nominal parity, frozen input hashes, completed training headers and 64 PPO
evaluation receipts before starting.

The evaluator checks all 88 plant floats, declared mass/inertia/thrust/lag
ranges, inertia inverse and thrust-to-weight. Raw plants are saved and hashed
in each receipt. Every task must have exactly one finite terminal outcome.
Nominal, noisy, dynamics and combined preflights passed. The earlier snapshot
and failed attempts remain preserved; the original findings below describe
that earlier code, not current unresolved blockers.

## Original blocking findings

1. **Dynamics and combined profiles currently fail before flight.** `navigation_stress_eval.mm` enables `runtime.enabled` and calls `sim.reset()` for both profiles (lines 24–27), then calls `waypoint::make_local_run` (line 30). `make_local_run` calls `enable_task_control`, which requires `runtime.enabled==0` (`navigation_critic_training.mm:500–507`). Thus the required runtime domain path is rejected before the evaluator writes plant parameters or collects actions. Let evaluation opt into runtime domain variation while retaining a nominal-only guard for training, or add an explicit evaluation-only constructor path. Then rerun both profiles and confirm that changed parameters reach `sim_advance`.

2. **The matrix runner does not require successful preflight.** `run-after.py` waits only for 64 successful matched PPO receipts (`lines 4–10`), then starts the matrix. It does not read `preflight-receipts.json` or require `STRESS_PREFLIGHT_COMPLETE`. At audit time the receipt file contained only nominal (passed), while `preflight.py` was running depth-noise; the matrix process was already waiting. Add a hard gate for all required preflight profiles, complete CSV counts, and nominal parity before launching any matrix job. Make failure stop the runner before it creates evaluation outputs.

## Verified source behavior

- **Exact tasks and actor:** the driver reads a hashed bank with period 1, uses `mode=17`, one seeded setup, a 400-step budget, and loads the supplied checkpoint. `waypoint_task_apply` installs the bank world's geometry and goal and zeros its wind. `verify_reset` checks each start, goal, yaw, clearance, and 400-step final-hold task. This preserves the bank geometry, task, time budget, and frozen actor across profiles.
- **Corruptions and delays:** evaluation mode is excluded from `sim_clean_training_env`, so `.03 m` Gaussian noise and `.05` per-ray max-range dropout are active. The sensor delay of two frames is 100 ms at the simulator's 50 ms sensor period. The command delay of two navigation steps is 100 ms at the 50 ms navigation period. The delay code selects old ring entries and marks unavailable startup depth as max range.
- **Capture pose and time:** each depth capture writes the sensor origin and rotation into the same modulo-8 frame slot as its depth image (`sim.metal:140–152`). Observation delay selects that same frame index for depth and for the geometry-memory call, and the actor receives elapsed time since that frame (`sim.metal:177–221`). Current position, attitude, linear velocity, and angular velocity remain current simulator state. The evaluator labels ego state as ideal, which matches the code. No estimator error, noisy ego state, or state latency is tested.
- **Actor inputs:** the actor receives current ego state, goal direction and distance, previous navigation command, depth history, and a geometry-memory hint built from depth-derived points. The privileged critic state is written to a separate `critic_obs` buffer and is not used by the actor-forward call in `local_tick`. I found no direct world obstacle geometry input to the actor. The score uses world geometry, as expected for grading.
- **Domain sampler wiring, if the blocking guard is removed:** reset samples mass, inertia, rotor thrust coefficients, and rising/falling rotor time constants from a deterministic seed derived from domain seed, simulator seed, environment, and episode (`sim.metal:45–64`; `physics_domain.hpp:99–153`). The sampled struct is passed into `rl_physics_step` (`sim.metal:282–294`), so all sampled parameters affect the plant. Initial rotor RPM is set from the varied plant's hover solution. The parameter struct is exactly 88 floats (`physics.hpp:41`) and the evaluator writes its raw bytes. Current CSV logging reports only mass and inertia diagonal, and current checks validate only finite positive mass; the preflight checks only mass range. Add finite/positive and range checks for the full sampled struct, including inertia, thrust coefficients, lag constants, and thrust-to-weight, plus a per-environment provenance/hash for the raw parameter record.
- **Nominal parity:** preflight compares nominal output columns to the prior bank CSV, excluding only the `split` field. The comparison uses `zip` without asserting equal lengths, and does not assert the baseline has 128 rows. Assert both lengths and row identity before treating parity as proven.
- **Finite grades:** the evaluator checks `episodes==1`, but does not check that success, collision, and timeout are mutually exclusive and sum to one, nor that recorded metrics are finite. The summary counters can therefore hide an ungraded episode. Require exactly one terminal outcome and finite flight/task metrics per environment before writing a successful receipt.
- **Wind and claim boundary:** the bank installer sets wind to zero and the stress driver leaves `cfg.wind` at its default zero. No wind stress is active. The evaluation does not model estimator faults, non-ideal ego state, or hardware effects. Keep these limits explicit in claims.

## Evidence status

At this audit snapshot, nominal preflight passed for 128 rows with 92 successes and 36 contacts. Depth-noise preflight was still running. The dynamic and combined profiles are not proven: the runtime guard above prevents them from reaching collection. The matrix must remain diagnostic and must not support a final robustness claim until the guard and grading checks are corrected and all receipts complete.
