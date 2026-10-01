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

PX4 `15f9af91ee3f8a2ac503864b5e557d7e4a1cb8fd`: transform state errors by inverse target quaternion, clip position±0.5m and velocity±1m/s, encode relative attitude. The velocity-like nav adapter supplies a finite target position `current_position + desired_world_velocity*0.5s`, desired velocity and yaw-only target. The 0.5s preview is our explicit navigation contract; it is not a PX4 default. OFFBOARD source has no NaN-position fallback; all targets stay finite.

Sources: https://github.com/rl-tools/raptor ; https://github.com/rl-tools/px4/blob/15f9af91ee3f8a2ac503864b5e557d7e4a1cb8fd/external_modules/src/modules/rl_tools_policy/RLtoolsPolicy.cpp

## Physics fidelity

Use Crazyflie default, because RAPTOR's published L2F demo uses the default dynamics; its x500 change is only the UI model. Preserve RK4, quaternion normalization, thrust/motor time constants and force in newtons. Optional x500 simulator profile is available. Dynamics and RK4 source hashes match between RAPTOR's pinned RLtools and current source used in initial archaeology. `reference.cpp` produces official `rlt::step()` fixtures against the pinned source.

Sources: https://github.com/rl-tools/rl-tools/blob/e43ae4bcda4556321a63f4eb5dcc826cd637aa39/include/rl_tools/rl/environments/l2f/operations_generic/60_dynamics.h ; https://github.com/rl-tools/rl-tools/blob/e43ae4bcda4556321a63f4eb5dcc826cd637aa39/include/rl_tools/rl/environments/l2f/parameters/dynamics/crazyflie.h

## Fixed PPO path

Actor: 660 deployable inputs, 64 tanh units, four unsquashed Gaussian outputs; tanh is applied only to executed commands so stored raw-action log probabilities are consistent. Critic: 32 privileged inputs, 64 tanh units. Distinct parameter sets prevent privileged actor leakage. GAE bootstraps time limits from pre-reset terminal-state value and stops carry on either termination or truncation. Fixed PPO clip±0.2, MSE value objective, entropy term, parameter-major sample gradients, reduction and Adam. No generic autograd or dynamic tensor API.

## Coherent target trajectory replaces moving preview

Failed hypothesis: reanchor `target_position=current_position+v*0.5` on each control tick while setting target velocity to v. Exact upstream dynamics show altitude drift under constant forward motion;1m/s and2m/s commands hit the floor after~6.9s and~5.1s. Reset motor history0 vs0.32 does not explain it. This was our trajectory generation error, not a physics-port or RAPTOR inference error.

Corrected hypothesis: trajectory position must have derivative equal to the commanded velocity. Persist reference position and integrate `reference += v*dt` at100Hz; do not reanchor it to measured position. Same upstream controller/dynamics/clips: after20s,1m/s ends x20.009m,z1.448m,vx1.000;2m/s ends x40.010m,z1.448m,vx2.000.2m/s minimum altitude1.019m.3m/4m goals with goal-direction feedback reach within0.35m in2.62s/2.84s.

Retain the measured fix. Add the three target-frame tracking-error coordinates to the actor (replaces two unused slots and adds one;661 total). This exposes controller reference state. Checkpoint v3 saves its persistent state. Integrated CPU/Metal control parity remains within5e-6.

The older PX4 commit0599df2d3fb53869e8a4a20c1b56daabbc9fdd67 pins a±0.2m position clip; selected15f9af91 uses±0.5m. Keep source-specific settings explicit rather than mixing commits.

## First actual PPO runs

30 rollouts of4096 samples under the corrected plant: held-out open success20–25%, no collisions; many goals still time out or are missed laterally/vertically. This is not adequate navigation. Continue learning and optimize the full update, rather than add more subsystem tests. Raw old per-sample actor gradients require~43MB per minibatch; direct hidden-delta matrix gradients are the next measured hypothesis.
