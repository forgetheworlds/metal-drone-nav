# Frame controller portability

Bounded future-portability research. Not a scope change: current priority stays
navigation learning. This answers "do future drone frames need RAPTOR-like
policies we train?" with what the code and primary sources actually support.

## Verdict

A new frame needs a measured or sourced plant profile and a validated controller.
Frozen RAPTOR may already work on that frame; this must be tested. The current
repository trains navigation policies, not low-level RAPTOR controllers. If the
controller fails validation, investigate identification, timing and integration
before deciding whether adaptation or upstream retraining is needed.

## Reusable vs frame-specific

Reusable, frame-agnostic:

- RAPTOR inference: `raptor.hpp:52` forward, `raptor.hpp:121` 22-dim FLU
  observation pack (position error clipped 0.5 m, velocity error 1.0 m),
  `raptor.metal` GPU twin, weights `assets/raptor.bin` via `export_raptor.py`.
- Navigation actor and sensor/geometry frontend: `deployment.hpp:79`
  (baseline184→64→4 MLP, 50 ms navigation, 10 ms native, `deployment.hpp:47-48`).
- PPO trainer for the **navigation** actor: `ppo.metal`, `ppo.hpp`,
  `navigation_training.hpp:101`.
- Task generator, potential field, reward, and separate mechanical-contact audit:
  `navigation_tasks.hpp`, `training_potential.hpp`, `reward_audit.hpp`,
  `vehicle_geometry.hpp:241`.
- Domain-randomization mechanism: `physics_domain.hpp:99`.
- Corruption knobs and gates (delay/noise/wind/dropout): `main.mm:589`.

Frame-specific:

- `RLPhysicsParams`, 88 floats (`physics.hpp:20-38`): rotor positions and
  directions, quadratic thrust coefficients, torque constants, rising/falling
  lag, inertia and inverse, mass, action min/max (RPM normalization), `dt`,
  state limits.
- `RLPhysicsDomainRange` (`physics_domain.hpp:17`): Crazyflie-only bounds —
  ±10 % mass/inertia/thrust, ±25–35 % lag, min thrust-to-weight 1.15
  (`physics_domain.hpp:38-39`). **Rotor geometry is not randomized.**
- Separate mechanical-body audit: seven shapes in `vehicle_geometry.hpp:281`,
  `VG_DEFAULT_SAFETY_MARGIN_M 0.04` (`vehicle_geometry.hpp:40`), potential-grid
  `kBodyRadiusM 0.18` (`training_potential.hpp:133`).
- Timing: delay rings ≤6 sensor frames, ≤7 command ticks (`main.mm:589`).
- Reference feasibility: `cfg.speed` cap and `velocity_contract` projection
  (`sim.metal:225-226`).

**Minimal frame profile contract** = `{nominal RLPhysicsParams,
RLPhysicsDomainRange, vehicle shape set + safety margin, potential body
radius}`. Four fields; everything else is shared. No other abstraction.

## Can we train a new low-level controller? No

- `sim.metal:5` binds RAPTOR as `device const float* weights`; `sim.metal:257`
  reads it. No gradient path exists.
- `fixed_ppo` is a 2-layer MLP (184→64→4) plus diagonal `log_std`
  (`ppo.hpp:26-35,60`). RAPTOR is Dense(22,16)-ReLU → GRU(16) → Dense(16,4)
  with recurrent state (`raptor.hpp:9-21`). Shapes, recurrence and the action
  clamp (`raptor.hpp:95`) do not match; there is no recurrent PPO and no
  clamp-aware log-prob.
- The only trainer is `navigation_training::train`, which
  `require`s `actor_obs_dim==184` (`navigation_training.hpp:103`).
- **PPO kernel availability does not prove a controller trainer exists.**
- Upstream training samples 1000 quadrotor dynamics parameter files, trains one
  RL teacher each, then distills a single student — Docker + MKL, long runtime
  (rl-tools/raptor README, "Training"). That pipeline is not in this repository.

Consequence: the realistic options for a new frame are *reuse frozen RAPTOR* or
*run upstream RAPTOR training outside this repo*. Neither is navigation training.

## Plant distinction and x500

`rl_physics_x500_sim()` (`physics.hpp:223`) has **zero call sites** — defined and
unused, i.e. UI/preset-only in this repo. Its comment (`physics.hpp:221-222`)
states it is a distinct simplified model from the measured `x500::real`
registry profile; do not treat it as a hardware-validated plant. All six live
construction sites hardcode `rl_physics_crazyflie_default()`:
`main.mm:470,512,601,651,896` and
`webots/controllers/raptor_webots/raptor_webots.cpp:286`; `reference.cpp:96`
separately asserts equality with it.

## Real generalization range (primary sources)

- RAPTOR: "a single, end-to-end neural-network policy to control a wide variety
  of quadrotors"; tested on **10 real quadrotors from 32 g to 2.4 kg**, varying
  motor type (brushed/brushless), frame (soft/rigid), propeller (2/3/4-blade)
  and flight controller (PX4/Betaflight/Crazyflie/M5StampFly); 2084 parameters;
  trained by Meta-Imitation Learning over 1000 sampled quadrotors. — arXiv
  [2509.11481](https://arxiv.org/abs/2509.11481).
- Interface: FLU axes, motor order [front-right, back-right, back-left,
  front-left], `rpm(action)=(max_rpm-min_rpm)(action+1)/2+min_rpm`, trained at
  **100 Hz**. — [rl-tools/raptor](https://github.com/rl-tools/raptor) README.
- Warning: oscillations appear when `LinearVelocityDelayed(2)` is combined with
  thrust-to-weight **> 2** (any preset except the Crazyflie), paper §2.4.1. —
  [raptor.rl.tools](https://raptor.rl.tools).

So: adaptive within a trained envelope, **not universal**, and delay-sensitive.
No target frame has been identified, and our own profiles are declared, not
measured: `vehicle_geometry.hpp:4-6` ("not a verified hardware CAD model"),
`physics_domain.hpp:3-4` ("experiment settings; not estimates of hardware
uncertainty").

## Reuse / adapt / retrain criteria

| Frame relationship | Action |
|---|---|
| Inside RAPTOR envelope (mass/prop/motor/TWR/delay), profile known | **Test reuse** of frozen RAPTOR with the new plant, sensing and motor interface |
| Known profile, marginal out-of-envelope (TWR, lag, size) | **Stress-test** the new profile and uncertainty; widening randomization does not adapt frozen RAPTOR |
| Outside envelope, or gate 1–4 fails | Investigate plant identification and integration; consider upstream adaptation or retraining if necessary |
| Profile source unknown | Obtain a sourced profile; frame-specific claims remain unverified |

## Five-step new frame path

1. **Profile.** Transcribe mass, inertia, rotor geometry, thrust/RPM mapping and
   lag into `rl_physics_<frame>_default()` beside `physics.hpp:184`. Cite the
   datasheet/registry source. Do not invent numbers.
2. **Plant parity.** Build a per-frame fixture through `reference.cpp` (it
   asserts the Crazyflie profile matches upstream to 1e-6, `reference.cpp:96`,
   so `assets/physics.bin` cannot be reused) and run `physics_tests`
   (`main.mm:120`).
3. **Geometry + domain.** Fill the remaining contract fields: shapes and safety
   margin (`vehicle_geometry.hpp:281,40`), potential body radius
   (`training_potential.hpp:133`), domain range with min TWR ≥ 1.15.
4. **Gates 1–4** below with frozen RAPTOR. A failed gate requires diagnosis; it does not by itself prove the frame is
   outside the trained envelope.
5. **Navigation.** Only then run `train` / `eval` on the new plant and the
   paired Metal/Webots DEV matrix.

## Validation gates

Existing harnesses are parity checks, not frame gates. Gates 2–4 are proposed
acceptance thresholds, not measured results.

1. **Parity** — `raptor_tests` official 16-step fixture < 2e-5 (`main.mm:117`),
   `raptor_px4_adapter_tests` < 2e-6 (`main.mm:202`), `physics_tests` < 2e-5
   (`main.mm:155`), `closed_loop_tests` CPU/GPU < 5e-4 (`main.mm:666`,
   function at `main.mm:647`).
2. **Tracking** *(proposed)* — on the new plant, frozen RAPTOR tracks a ±1 m/s
   square-wave velocity reference for 320 native steps at 100 Hz: position RMSE
   ≤ 0.15 m, yaw error ≤ 0.2 rad, no state-limit clip
   (`physics.hpp:167-178`).
3. **Stability** *(proposed)* — hover from ±0.3 m / ±0.5 rad offsets for 10 s:
   no divergence, attitude ≤ 15°, and
   `rl_physics_domain_thrust_to_weight ≥ 1.15` at `action_max`
   (`physics_domain.hpp:83-94`).
4. **Latency** *(proposed)* — `reaction-latency CHECKPOINT [SENSOR] [COMMAND]`
   (`main.mm:2160`) at 0/1/2 command ticks (50 ms each): no oscillation onset
   and ≤ 10-point success drop vs 0-delay. Explicitly reproduce the upstream
   high-TWR + delayed-velocity case before adopting.
5. **Transfer** — existing `eval` delay/noise/wind/dropout sweep, then paired
   Metal/Webots DEV. Adopt nothing below the current bar: 25/30 Metal DEV,
   18/30 Webots DEV (`STATUS.md:60`).

## Smallest future code seam

One frame selector. Replace the six hardcoded `rl_physics_crazyflie_default()`
sites with a single profile value carrying the four contract fields, passed into
`Sim`, `cpu_reference::Trainer` and the Webots controller. The GPU side already
accepts a per-environment `RLPhysicsParams` (`sim.metal:59,246`), so no new
struct, registry or loader is needed — only the nominal seed and the domain
range. Add nothing before a second real frame exists
(`docs/CODE_DIRECTION.md`: generalize only when two real cases prove it).

## Sources

- https://github.com/rl-tools/raptor (README: interface, 100 Hz, training)
- https://raptor.rl.tools (presets, delay/TWR §2.4.1 reproduction)
- https://arxiv.org/abs/2509.11481 (generalization envelope)
- https://arxiv.org/abs/2311.13081 ("Learning to Fly in Seconds", RA-L)
- Code citations above; `STATUS.md:5,60` for the 25/30 and 18/30 figures.
