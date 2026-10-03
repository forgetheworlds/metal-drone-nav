# Navigation reward and credit audit

The current source objective can rank successful tight-space flights below
hovering until timeout. This is a measured objective problem, not proof that a
discount change will solve navigation. The selected policies remain unchanged.

## Actual flights and objective

We ran the 30 original-start TRAIN corner worlds with four controls. Each used
the same world, final goal, 20 s limit, 1.5 m/s command cap, 0.18 m collision
sphere, clean depth, 20 Hz navigation and actual 100 Hz RAPTOR/L2F plant. The
teacher follows privileged witness routes; it is a physical and reward
diagnostic, never learned-policy or generalization evidence. The scan condition
first commands world velocity `[0, .15, 0]` and normalized yaw `.9` for 40 ticks,
then uses that teacher. It does not establish a learned sensing behavior.

The base reward in `sim.metal` is:

```text
2 × (goal distance before − goal distance after)
− .01 per navigation tick
− .1 × clamp((.6 − minimum step clearance) / .6, 0, 1)
+ 10 on goal entry, − 10 on contact
```

No task override, domain randomization, potential shaping, auxiliary loss or
new learned weights were enabled. Goal entry is within .35 m, with no hold
requirement. Risk clearance is measured after subtracting the collision body;
a feasible narrow gap can still incur a sustained penalty. The risk component
in the receipt is recovered from the native total reward and the other exact
terms. It is not a separate approximation of obstacle clearance.

| Control | Success | Contact | Timeout |
|---|---:|---:|---:|
| Frozen selected actor | 0/30 | 30 | 0 |
| Privileged route teacher | 24/30 | 6 | 0 |
| Two-second scan, then teacher | 24/30 | 6 | 0 |
| Zero commanded velocity through RAPTOR | 0/30 | 0 | 30 |

The 120 real episodes contain 27,222 navigation transitions. Independent CSV
replay reconstructs every native reward within 5e-9 and every discounted
episode return within 1e-7. The six failed teacher routes remain in the data.

## Quantified rank inversion

With `gamma=.99`, 2 of 24 successful flights score below matched 20 s hovering
in each teacher condition. For `f14-train-0005`, the route teacher reaches the
goal at 13.00 s with return **−1.2416**, compared with **−.9690** for hovering.
The discounted components are progress **+2.9291**, time **−.9267**, clearance
risk **−3.9845**, and successful goal entry **+.7405**. With a prior sensing
scan, completion takes 15.15 s and return is **−1.0979**.

These are finite recorded reward sums. PPO handles timeout as truncation and
bootstraps from the next-state critic while stopping GAE at the episode reset.
Adding that frozen critic's discounted timeout value changes the comparison:
one route-teacher inversion remains, by only about .0094; no scan-teacher
inversion remains. The frozen critic is an estimate, not the true continuing
value. Do not present finite trace sums as the complete PPO objective.

Reward discount has a **3.45 s half-life** at 20 Hz. Successful route flights
take 11.30–13.45 s; their nominal +10 goal bonus contributes only **+.676 to
+1.042** at the start. Scan-plus-teacher flights take 13.40–15.45 s and receive
only **+.453 to +.683** of discounted terminal bonus.

GAE uses `lambda=.95`; its residual weight `gamma×lambda=.9405` has a **.565 s
half-life**. The 32-tick rollout is 1.6 s, shorter than the 2 s sensing action.
Long-range credit therefore relies on the critic. It is not mathematically
absent beyond a rollout: value bootstrapping carries information across
updates. [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
describes the value-based estimator and its bias/variance tradeoff.

Bootstrapping a training cut is appropriate for a continuing task. A deadline
that is itself part of the task needs a corresponding finite-horizon model and
remaining-time information. Our benchmark imposes a 20 s pass/fail budget while
training currently uses continuation semantics. This is a task-design question,
not evidence of a broken GAE kernel. See
[Time Limits in Reinforcement Learning](https://arxiv.org/abs/1712.00378).

![Recorded reward comparison and credit weights](../artifacts/reward-credit-audit.png)

The left panel uses actual recorded successful flights, sorted independently
per condition. The right panel shows formulas from the actual parameters;
beyond the rollout boundary, its GAE curve is a hypothetical continued trace.
Neither panel is a learning curve.

## Controlled next step

Reweighting these same transitions with `.995` removes the observed finite
return inversions. This is an offline calculation, not a new policy result.
It justifies testing one longer reward horizon against `.99`, with the same
source worlds, raw-depth warm start, seed, action mapping, reward terms,
lambda, rollout size, optimizer and training budget. Keep potential shaping
disabled because its current contract is tied to `.99`. Record gamma in the
experimental provenance and reject cross-gamma resumes.

An eligible policy must improve original-start corner DEV completion while
retaining at least 25/30 rooms and 29/30 vertical tasks. More favorable replayed
returns or lower value loss do not pass this gate. Source training and logic
must freeze before independent target evaluation; FINAL remains untouched.
Observability, exploration and memory are still unresolved. This audit does
not explain all corner failures or prove that PPO needs replacement.

## Reproduction

Build the public cold runner on macOS:

```sh
clang++ -std=c++17 -O3 -fobjc-arc \
  -DSOURCE_DIR=\"$(pwd)\" -DFIXED_PPO_ACTOR_OBS_DIM=184 \
  navigation_reward_audit.mm -framework Foundation -framework Metal \
  -framework Accelerate -o build/reward-audit
```

Hold an exclusive `fcntl.flock` on `results/metal-training.lock` while running:

```sh
build/reward-audit evidence/inputs/challenge-bank-mirrored-v1.jsonl \
  assets/checkpoints/rooms-focused-experimental.bin.best results/reward-audit-repeat
```

Rebuild the analysis and figure without simulation:

```sh
python3 reward_audit.py
```

The [proof manifest](../evidence/inputs/reward-audit/proof.json), episode table
and compressed transition CSV preserve configuration, input/source hashes,
failure IDs, term/truncation flags, reward decomposition and per-state values.
No Webots, blind final evaluation or policy promotion occurred in this audit.
