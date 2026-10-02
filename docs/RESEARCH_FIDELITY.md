# Fidelity audit: physics, vehicle, sensing, reward, task

**Date:** 2026-10-01
**Question:** Is the training environment good enough that success in it means the
policy learned navigation — and is the physics/vehicle model accurate?

**Short answer:** The physics is *faithful to its reference* (L2F/RLtools, parity
~1e-7) but that reference is a mid-fidelity rigid-body model with no aerodynamic
drag, no velocity-dependent thrust, and no turbulence. The reward is a five-term
handful whose dense term (Euclidean goal progress) actively punishes detours. The
task set is short point-to-point hops with perfect sensing and perfect ego state.
None of these are fatal, but they mean in-simulator scores currently overstate
capability. This note records what the code actually contains, what the
literature says matters, and the ranked fixes.

---

## 1. What our physics model actually contains (verified from source)

From `physics.hpp` / `physics.metal`:

- 6-DOF rigid body, quaternion orientation, **RK4 at 100 Hz**;
- per-rotor thrust polynomial coefficients `[constant, linear, quadratic]` in
  rotor speed, per-rotor torque constants;
- **first-order motor lag** with separate rising/falling time constants;
- **rigid-body gyroscopic** torque term (`omega × J omega`); rotor inertia/gyroscopic coupling is not modeled;
- gravity; wind enters as a **constant world-frame force vector**;
- RAPTOR motor commands in [-1, 1], L2F rotor ordering.

`grep` for `drag|damping|turbulence|ground_effect` over `physics.hpp` and
`physics.metal`: **no matches**. The body is inviscid.

Validation status: 9.54e-7 vs official L2F fixtures, 5.96e-7 vs official RAPTOR
oracle. So the model is *correct with respect to L2F*. The question is whether
L2F is the right target.

## 2. What is missing, and does it matter?

| Missing effect | When it matters | Evidence |
|---|---|---|
| **Aerodynamic drag (linear + quadratic)** | Braking, terminal speed, wind response — grows with speed. We push 1.5–7.34 m/s measured | Huang et al., ICRA 2009 (https://ai.stanford.edu/~gabeh/papers/ICRA09_AeroEffects.pdf) — blade flapping and thrust variation with forward velocity; NeuroBEM hybrid aero model cuts model error ~50% vs rigid-body (https://kelia.github.io/publication/neuro-bem/) |
| **Velocity-dependent thrust** (thrust drops/tilts in forward flight) | Any fast traversal; systematically changes climb at speed | Same sources; Webots supports `T = t1·|ω|·ω − t2·|ω|·V`, but this project sets every `t2` to zero. Both configured plants currently omit that effect; framework support is not evidence of an enabled physical model |
| **Ground / wall effect** (rotor proximity enhancement) | Low altitude, near tables, through gaps | Standard quadrotor aero; unmodelled here |
| **Turbulence / gusts / prop-wash** | Wind is a constant force today; no gusts, no relative-airflow coupling | Our own evaluation already labels wind a "force-equivalent disturbance", not a velocity |
| **Battery sag / ESC saturation** | Long runs, aggressive climbs | Battery-aware RL, arXiv 2609.37316 (2026): identified load-transient battery model + firmware saturation needed for aggressive flight |
| **Actuator saturation** | Normalized motor actions and rotor state are clamped; physical battery-dependent RPM/thrust authority and ESC limits are not identified | Webots uses a large kinematic `maxTorque` to suppress an extra numerical ramp. Its shaft speed is a force-equivalent proxy, not measured hardware RPM |

**Verdict on physics:** For ≤1.5 m/s indoor flight, missing drag is the main
defect but the biggest sim-to-real gap is probably *not* dynamics — it is
sensing and state (below). At the 3–7 m/s regime we already measure, aero terms
stop being second-order. Rather than build a high-fidelity aero model from
scratch, the literature-supported move is **system identification + selective
domain randomization**: randomize mass, inertia, motor time constants and thrust
coefficients — plus a drag coefficient sweep — within plausible ranges, instead
of making the model more complicated (SimpleFlight factor 4, arXiv 2412.11764).

## 3. Sensing and state: the larger gap

- The actor receives **exact simulator ego state** (position-derived velocity,
  attitude, rates) and **clean depth** by default. `docs/NEXT_PHASE.md` and
  `docs/RESEARCH_NEXT_PHASE.md` already admit state-estimation error and
  long-route structure are out of distribution.
- Depth Transfer (RA-L 2025, arXiv 2505.12428) reports a **large degradation
  when perfect simulated depth is swapped for real stereo depth**, and recovers
  it only with learned domain adaptation. That is direct evidence that our
  default clean-depth training overstates transfer.
- Corruption knobs exist (noise, dropout, delay, wind, command delay) but they
  are **training flags, not a randomized distribution**, and no temporal
  correlated position/yaw bias is implemented yet.

## 4. Reward: confirmed crude (code, not speculation)

`sim.metal` line 208 is the entire learning signal:

```text
reward = 2 × (Euclidean goal distance closed this tick)
         − 0.01
         − risk_coef × clamp((0.6 − clearance)/0.6, 0, 1)
         +10 success − 10 collision
```

Problems:

1. **Straight-line progress punishes detours.** Every step around a partition
   is negative reward. This predicts the measured corner result (0/128,
   99.994% collision): there is no gradient through the wall, only into it.
2. **Terminals dwarf the dense term** (~0.15/tick vs ±10) → "don't die" is the
   dominant strategy; hover/brake behavior and timeout mass follow.
3. **Clearance risk is context-blind** (same meaning in open hall and doorway)
   and was bolted on to fix a measured failure — the research log records it
   helping one metric and hurting others.
4. **No path-efficiency, no subgoal credit, timeouts return nothing.**

Research-backed corrections, in order:

- **Potential-based shaping with geodesic distance**: `gamma * Phi(s_next) - Phi(s)`, with the same gamma as PPO and terminal Phi=0. Under the theorem assumptions it preserves optimal policies; it does not guarantee better exploration. `training_potential.hpp` already computes
  `Phi = −min(geodesic_distance, cap)/cap` grids — wire it into the reward.
- **Action-smoothness regularization** — identified as critical for
  zero-shot real flight in SimpleFlight (arXiv 2412.11764, factor 3);
  we have no jerk/action-change term at all.
- **Time-scaled success bonus** instead of +10 plus a −0.01 trickle.
- **TTC-based risk instead of raw clearance**, normalized per family.
- Keep evaluation fully independent of shaping terms (project rule, AvoidBench).

## 5. Task and environment crudeness

- **Geometry:** fixed-capacity analytic boxes, spheres and vertical cylinders; richer rooms use combinations of those primitives. Real tasks need connected routes,
  occlusion, forced detours — the challenge bank exists for exactly this and the
  learned policies score 0/30 on two of its three families.
- **Difficulty is not a continuum:** gap width, clearance, TTC and route length
  are not exposed as sampling axes, so there is no learning frontier to
  curriculum against.
- **No feasibility gate:** witness at training cap passes 69/90 dev levels —
  up to 21 levels may be unsolvable at the training budget, which no reward can
  fix. Every training level should pass a witness check before entering a bank.
- **Short horizons:** 3–4 m hops, 10–20 s budgets vs multi-room routes; 32-step
  (1.6 s) rollouts mean long-route credit must cross many boundaries.
- **Speed adaptation** (slow in clutter, fast in open) is required by MAVRL
  (RA-L 2025) but our reward barely encodes it and measured success collapses
  at 3 m/s.

## 6. Ranked plan (cheapest highest-value first)

1. **Feasibility-gate every level** (witness at training settings); quarantine
   unsolvable levels.
2. **Reward rewrite:** geodesic potential shaping (wire
   `training_potential.hpp`), time-scaled success bonus, smoothness term,
   TTC-based risk. Ablate each term with matched budgets; falsification test =
   corner control run.
3. **Sensing distribution:** correlated pose/yaw bias + depth corruption as
   randomized training factors, not flags.
4. **Selective domain randomization** of vehicle parameters (mass, inertia,
   motor constants, thrust coefficients, drag coefficient sweep) — SimpleFlight
   factor 4 — instead of a more complex aero model.
5. **Drag + velocity-dependent thrust term**, so Metal and Webots stop disagreeing
   by construction at speed; measure the delta both ways.
6. **Task expansion:** long connected routes, occlusion, difficulty continuum
   (FlightBench TO/VO/AOL-style metadata as local descriptors).
7. **Independent verification:** frozen policy through calibrated Webots with
   baselines — the test that separates "engine verified" from "capability
   verified".

## Primary sources

- Huang et al., Aerodynamics and Control of Autonomous Quadrotor Helicopters,
  ICRA 2009: https://ai.stanford.edu/~gabeh/papers/ICRA09_AeroEffects.pdf
- NeuroBEM hybrid aerodynamic model: https://kelia.github.io/publication/neuro-bem/
- SimpleFlight / What Matters in Zero-Shot Sim-to-Real, arXiv 2412.11764:
  https://arxiv.org/html/2412.11764v2
- Depth Transfer, RA-L 2025, arXiv 2505.12428: https://arxiv.org/html/2505.12428v1
- Learning to Fly in Seconds (L2F/RLtools model definition), arXiv 2311.13081
- Flightmare (sim fidelity vs speed trade-offs), CoRL 2020:
  https://rpg.ifi.uzh.ch/docs/CoRL20_Yunlong.pdf
- Battery-aware RL for aggressive flight, arXiv 2609.37316 (2026)
- MAVRL (speed adaptation), RA-L 2025 — see docs/RESEARCH_NEXT_PHASE.md
- FlightBench task descriptors (TO/VO/AOL) — see docs/RESEARCH_NEXT_PHASE.md


## Vehicle geometry and model agreement — required before further claims

The L2F mass/inertia/rotor-arm profile describes a30.6g Crazyflie-scale vehicle. The visible Webots airframe extends roughly5.3cm laterally from its center, but both navigation simulators use an18cm-radius sphere as the collision body. That sphere is a conservative safety envelope, not a measured airframe. Physical contact and safety-margin violations must be separated; shrinking the benchmark envelope must not be reported as learned improvement.

Webots shaft velocity is computed from an arbitrary force-equivalent thrust constant, preserving the chosen force/torque curve but not identified physical motor RPM. The current motor lag is continuous in L2F RK4 and sampled analytically by the Webots controller before ODE advancement. Startup, motor timing and solver differences need a common-input trajectory comparison. Small upstream fixture error proves implementation agreement with L2F, not agreement with hardware or independent simulator dynamics.

Immediate priority is replaying the same motor commands through both plants with aligned initial conditions and recording pose, velocity, attitude and rate errors. Then compare the same geometry and sensor poses, and finally closed-loop navigation. This separates dynamics, sensing and learned-control failures before adding arbitrary aerodynamic coefficients or changing rewards. The selected policy remains a baseline with unresolved transfer failures.

[Bitcraze system-identification measurements](https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/functional-areas/pwm-to-thrust/) provide actuator evidence; [Crazyflow identification](https://learnsyslab.github.io/crazyflow/user-guide/dynamics/system-identification/) documents fitting and testing against distinct validation trajectories. These are references for calibration, not evidence that this project has calibrated hardware.


## Common-command calibration result (2026-10-02)

A free-running CPU L2F replay and independent Webots ODE plant now receive the same recorded motor commands after a2s hover warm-up. The original end-of-interval force sampling creates up to0.223rad/s roll-rate and0.233rad/s pitch-rate disagreement in the pulse cases. Analytically averaging the continuous actuator/thrust curve over each10ms command interval reduces those discrepancies to about0.00010/0.00107rad/s at10ms ODE steps. Reducing ODE steps to1ms while retaining100Hz RAPTOR and20Hz navigation gives maximum position errors0.183/0.175mm and maximum rate errors0.000037/0.000087rad/s in the roll/pitch cases.

These are limited common-input model comparisons, not hardware identification or reliable learned navigation. They isolate a real actuator-sampling defect and show solver convergence. Neither configured model currently includes identified aerodynamics. Future navigation comparisons use the calibrated protocol explicitly and retain the old protocol as a baseline.

Reproduce with `python3 webots/physics_audit.py --motor-sampling average --physics-step-ms 1 --profiles roll,pitch`. Rootcode uses Supervisor state only inside this diagnostic; navigation continues to consume sensor APIs. [Webots GPS implementation](https://github.com/cyberbotics/webots/blob/R2025a/src/webots/nodes/WbGps.cpp) reads instantaneous rigid-body velocity when a physics body exists; it is not a finite-difference estimate in this setup.


## Unknown final-goal routing: test credit assignment before adding memory (2026-10-02)

The current results separate local goal arrival from global route choice. The current guided policy scores 0/30 on rich-bank corner routes and 0/30 on rich-bank connected-room routes, while it scores 25/30 on vertical-choice routes (`results/arrival-rich-dev.csv`). A separate learned-policy waypoint diagnostic supplies privileged witness waypoints but keeps RAPTOR, physics, and collision scoring live: it completes 9/30 corner routes, 30/30 room routes, and 27/30 vertical routes at a 1.5 m/s cap and 60 s budget. Only 2/30 corner routes finish within the required 20 s; all room routes finish within 15.85 s (`results/arrival-waypoint-diagnostic.csv`). This probe does not test global route discovery because the witness supplies the route. Webots stable-arrival at 17/18 likewise measures a different short-goal task, not the rich-bank global-routing capability (`results/arrival-webots.csv`).

The guided actor is a feedforward 184→64→4 network. Its input includes current and previous pooled depth, ego state, goal direction/distance, and a short geometric prior. The eight-frame sensor-pose cloud used by that prior spans about 0.4 s. It is not an episode-long map or visited-route record. `SimRun.hidden[16]` belongs to RAPTOR’s motor controller, not the navigation actor. The bank reward still gives progress in straight-line Euclidean goal distance (`sim.metal::sim_advance`); this makes an initial detour away from the final goal costly. The new `training_potential.hpp` builds a bounded, 0.2 m grid geodesic potential for static families 14–16, but no kernel consumes it yet. It is computed from training geometry and must stay out of the actor observation and deployment path.

**Next test: one matched reward-only geodesic-potential ablation on the static corner bank.** Keep the feedforward actor, its observations, RAPTOR, mode-22 training/deployment action map, task distribution, initialization, optimizer, transitions, and evaluator fixed. Compare the current Euclidean-progress reward with `F(s,s') = γΦ(s') − Φ(s)`, using the PPO discount, zero terminal potential, and the existing train-only geometry fields. Do not add a witness-following reward or waypoint input to this test. Use held-out corner seeds/topology for selection and final evaluation, and keep room and vertical results as regression checks. Record initial distance progress, route progress, success, collision, timeout, path length, and mission time. A useful result is improved held-out final-goal success and detour progress without reducing held-out room/vertical success beyond the existing five-point tolerance. Falsify the reward hypothesis if it only improves training-bank return or waypoint reach while final-goal corner success remains unchanged. Potential-based shaping can preserve the optimal policy under its assumptions; that theorem does not guarantee that PPO will explore more effectively. [Ng, Harada, and Russell (1999)](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf)

This is the smallest first test because the failure is measurable in the current reward and a static geodesic field changes no policy input or deployed state. It is not the final answer to moving, noisy, unknown 3D routing. If the held-out test falsifies reward-only credit assignment, compare two explicit memory branches next: (1) a recurrent actor trained with contiguous sequence minibatches, episode-reset masks, and recurrent-state-aware PPO; (2) a sensor-built 3D free/occupied/unknown map with frontier selection and backtracking that feeds the existing local-goal channels. Keep the map out of world geometry and witness data. It must use depth and an estimated pose, and must treat unknown space as unknown; dynamic obstacles need a separate expiring layer. The map branch is more inspectable for visited-space and dead-end recovery, but it adds pose-drift and update costs. Recurrent PPO keeps the input interface smaller but requires a substantial change to the current independent-sample PPO path and may encode revisits unreliably.

The Berkeley RA-L paper is a useful precedent, not a drop-in recipe. It uses a 64-state LSTM after depth encoding. It gives trajectory-proximity reward during simple pillar pretraining, then removes it for its harder curriculum because many routes are valid and single-route proximity is a poor signal. Its later simulation discussion reports that the policy still struggles to backtrack from dense dead ends at higher speeds and suggests stronger memory. That evidence supports testing reward shape and memory separately; it does not show that an LSTM or map will solve this project’s route bank. [Dutta et al., “Vision-Guided Outdoor Flight and Obstacle Evasion via Reinforcement Learning,” RA-L preprint (2025)](https://www-video.eecs.berkeley.edu/papers/RAL2025.pdf)

For a map branch, preserve the distinction between free, occupied, and unknown cells. Frontier exploration uses the border of known free space and unexplored space to choose where to extend the map. These sources support that representation and selection rule; they do not establish its flight performance here. [Yamauchi, “A Frontier-Based Approach for Autonomous Exploration” (1997)](https://www.robotfrontier.com/papers/cira97.pdf) · [OctoMap, 3D probabilistic occupancy mapping (2013)](https://doi.org/10.1007/s10514-012-9321-0)
