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
- rotor **gyroscopic** torque term;
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
| **Velocity-dependent thrust** (thrust drops/tilts in forward flight) | Any fast traversal; systematically changes climb at speed | Same sources; note **Webots' propeller model already has this**: `T = t1·|ω|·ω − t2·|ω|·V` (see `webots/README.md`). Our L2F polynomial depends on ω only → the two simulators will disagree at speed by construction |
| **Ground / wall effect** (rotor proximity enhancement) | Low altitude, near tables, through gaps | Standard quadrotor aero; unmodelled here |
| **Turbulence / gusts / prop-wash** | Wind is a constant force today; no gusts, no relative-airflow coupling | Our own evaluation already labels wind a "force-equivalent disturbance", not a velocity |
| **Battery sag / ESC saturation** | Long runs, aggressive climbs | Battery-aware RL, arXiv 2609.37316 (2026): identified load-transient battery model + firmware saturation needed for aggressive flight |
| **Actuator saturation** | Motor time constants exist but no explicit torque/RPM ceiling | Webots plant debug found `maxTorque` behaves as an acceleration cap — the two models impose different limits |

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

- **Potential-based shaping with geodesic distance**: Φ(s) − Φ(s′),
  Φ = −geodesic_dist, policy-invariant (Ng et al. 1999) and removes the
  anti-detour bias. `training_potential.hpp` already computes
  `Phi = −min(geodesic_distance, cap)/cap` grids — wire it into the reward.
- **Action-smoothness regularization** — identified as critical for
  zero-shot real flight in SimpleFlight (arXiv 2412.11764, factor 3);
  we have no jerk/action-change term at all.
- **Time-scaled success bonus** instead of +10 plus a −0.01 trickle.
- **TTC-based risk instead of raw clearance**, normalized per family.
- Keep evaluation fully independent of shaping terms (project rule, AvoidBench).

## 5. Task and environment crudeness

- **Geometry:** AABB boxes/doors/tables only. Real tasks need connected routes,
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
