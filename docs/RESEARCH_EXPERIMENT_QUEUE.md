# 23. Research-backed strategies worth actively testing

The following are not requirements and should not be implemented wholesale.

They are promising hypotheses identified from external RL and aerial-navigation research that map closely onto failures already measured in this repository.

Use them as a prioritized research queue. For each, first confirm that the corresponding bottleneck still exists, then run the smallest controlled experiment capable of rejecting the idea.

## A. Improve the task sampler before changing PPO

The current failure-weighted challenge sampler is closely related to **Prioritized Level Replay (PLR)**.

PLR prioritizes previously encountered levels using a measure of learning potential based on TD/GAE-like error together with staleness. This is particularly relevant because the current repository already records mean absolute rollout GAE per challenge.

Do not assume the current implementation is therefore optimal.

Test several task-selection strategies against the same task bank, PPO settings, transition budget and evaluation splits.

At minimum compare:

```text id="8eq5hk"
uniform sampling

current failure-weighted / |GAE| sampling

success-frontier sampling

learning-progress sampling

PLR-style score + staleness

PLR⊥-style curated replay
```

A particularly interesting result from follow-up PLR research is **PLR⊥**: randomly generated levels are first used to measure their usefulness but do not immediately update the policy; policy updates are concentrated on curated replay levels. This counterintuitive "train on less data" strategy improved robustness and zero-shot transfer in the reported experiments.

This maps naturally onto Metal's throughput.

For example:

```text id="58d7ss"
generate 100,000 cheap candidate tasks
        ↓
probe policy
        ↓
score learning potential
        ↓
retain informative subset
        ↓
spend PPO updates there
```

Because simulation is extremely cheap, task selection may now be more important than generating even more undifferentiated transitions.

One controlled experiment should ask:

> With the exact same number of PPO update transitions, does spending those updates on intelligently selected tasks improve unseen DEV navigation and retention?

Do not compare methods using different effective training compute without reporting it.

## B. Test learning progress, not merely absolute difficulty

A high-error task is not necessarily a useful task.

It could be:

```text id="gk0zqn"
usefully difficult
```

or:

```text id="l5fawh"
unobservable
impossible
badly rewarded
extremely stochastic
far beyond current ability
```

Automatic curriculum work such as **ALP-GMM** instead focuses on regions where the learner is making large absolute learning progress. More recent curriculum work continues to use learning progress rather than raw return as the signal for allocating training.

This is highly relevant because the new task representation can expose continuous parameters such as:

```text id="m5om45"
gap width
goal distance
detour ratio
clearance
initial speed
turn angle
TTC
occlusion duration
sensor delay
```

Consider maintaining competence estimates across these regions and prioritizing places where performance is changing fastest.

Conceptually:

```text id="adk85d"
98% → 99%
little learning value

5% → 5%
possibly too hard / impossible

48% → 65%
strong learning progress
→ allocate more training
```

Do not hard-code 50–80% success as the definition of the frontier. Measure whether learning progress predicts later improvement better than success rate, absolute GAE or collision rate.

## C. Let difficult tasks evolve from useful tasks

**ACCEL** evolves curricula by editing environments that are already near the learner's capability frontier instead of repeatedly sampling completely unrelated random environments.

This is a strong fit for our procedural geometry.

Instead of:

```text id="1pyjhw"
sample completely new random world
```

consider:

```text id="v9018t"
take informative task

↓ mutate one meaningful parameter

narrow gap
move obstacle
increase detour
change approach angle
raise initial speed
shorten TTC
add occlusion
mirror route
change vertical choice

↓ validate

↓ test policy

↓ retain mutation if informative
```

This could produce a much smoother learning frontier than manually adding new obstacle families.

Every mutation must still pass:

```text id="cdb41x"
geometric feasibility
dynamic feasibility
observability
deployment-information constraints
episode budget
```

Do not allow adversarial curriculum generation to discover "difficult" tasks that are difficult only because they are impossible.

## D. Use the long-running coding agent itself as an offline curriculum designer

There is now direct robotics evidence for using LLMs to generate environment code as an automatic curriculum.

**Eurekaverse** used an LLM to generate progressively more challenging and diverse simulation environments for robot learning; its automatically generated curriculum outperformed manually designed training courses in its quadrupedal parkour setting and transferred to hardware.

This is especially relevant because this project already has a continuously operating coding/research agent.

Do NOT put an LLM inside the Metal training hot loop.

Instead, consider an outer research loop:

```text id="ij8d0e"
policy evaluation
        ↓
failure database
        ↓
Codex analyzes failure clusters
        ↓
proposes generator changes / task mutations
        ↓
strict deterministic validators
        ↓
large candidate task set
        ↓
automatic training experiment
        ↓
DEV + retention results
        ↓
research log
        ↓
repeat
```

The LLM should propose hypotheses and generator code.

Deterministic geometry, observability and feasibility checks should decide whether generated tasks are admissible.

Measured policy improvement should decide whether the curriculum idea survives.

This uses the agent where it is strong without letting it fabricate the ground truth of the RL environment.

## E. Replace straight-line progress with a better privileged training signal

One of the most directly relevant recent quadrotor results is **Quadrotor Navigation using Reinforcement Learning with Privileged Information**.

That work explicitly targets a failure similar to ours: reactive navigation works around small obstacles but struggles when large obstacles, corners or concave geometry block the direct goal direction.

Their solution uses a training-only **time-of-arrival (ToA) field** computed from known simulator geometry. The field gives a shortest-time direction toward the goal while accounting for obstacles. The deployed policy does not receive the map. They also modify travel cost near obstacles so that the resulting route prefers safer clearance instead of simply taking the geometrically shortest opening.

This is extremely close to infrastructure we already possess.

`training_potential.hpp` currently computes geometric goal potentials.

A high-priority experiment should compare:

```text id="3zfb56"
Euclidean progress

current geodesic potential

clearance-aware geodesic distance

clearance-aware time-to-arrival potential
```

A useful formulation could make travel through open space cheap and movement near obstacles more expensive:

```text id="nc7zwp"
large clearance
→ high allowed propagation speed

small clearance
→ low propagation speed

obstacle
→ blocked
```

The resulting field then represents something closer to:

> What is the fastest reasonably safe route to the goal?

rather than merely:

> What is the shortest geometric route?

Investigate using it as:

```text id="sbm3we"
potential-based shaping
and/or
privileged critic input
and/or
auxiliary direction target
```

but keep it out of deployed actor observations.

This experiment should come before assuming that long detour failure requires a substantially larger network.

## F. Explicitly model scenario difficulty like FlightBench

**FlightBench** does not merely group environments by object type. It develops scenario-difficulty criteria and shows that learned quadrotor navigation performance changes systematically with difficulty. It specifically identifies sharp corners and view occlusion as difficult cases for learned ego-vision navigation.

This aligns strongly with failures already found here.

Use FlightBench and its open-source implementation as a reference when designing our task metadata.

Do not copy its metrics blindly. Determine which ones transfer to this local-goal architecture.

The goal should be to replace statements like:

```text id="33lzl9"
family 15 is hard
```

with:

```text id="rx1th6"
path ratio = 1.7
minimum clearance = 0.31 m
required heading change = 92°
goal occlusion = 1.4 s
sensor warning = 0.65 s
start speed = 1.2 m/s
```

Then evaluate policy capability surfaces instead of obstacle-family averages.

This should also make adaptive curriculum sampling substantially more meaningful.

## G. Create a dedicated "reactive local minima" benchmark

The paper **Pushing the Limits of Reactive Planning: Learning to Escape Local Minima** directly asks when reactive navigation needs memory or mapping.

It trains feed-forward and recurrent learned additions to a reactive sensor-based planner on procedurally generated clutter and reports zero-shot transfer to real 3-D man-made environments.

This is almost exactly the architectural question facing Metal-nav.

Before adding a large recurrent policy or explicit map, create a small benchmark specifically containing:

```text id="ugfi4r"
U-shaped obstruction
large wall with visible/temporarily-visible escape
corner
short dead-end
temporarily disappearing passage
```

Then test:

```text id="1w3n92"
current actor + explicit geometry memory

same actor with longer explicit memory

small GRU actor

geometry-only prior

privileged route oracle
```

This would tell us whether the missing capability is:

```text id="v86eak"
training
vs
observation
vs
memory
vs
policy capacity
```

Do not add recurrence until this benchmark shows that state aliasing over time is a material bottleneck after reward/task issues are corrected.

## H. For moving obstacles, consider selective temporal resolution rather than simply making everything larger

The 2026 **D-Nav** system attacks dynamic UAV navigation with a spatio-temporal depth representation plus a higher-resolution refinement mechanism focused on dynamically important regions.

The key idea is more relevant than its specific architecture:

> not every part of the depth field deserves equal representational bandwidth.

It preserves broad scene structure while spending extra resolution on areas containing small or fast dynamic threats.

This may be especially relevant here because prior experiments already found that min-pooling can erase useful geometric information.

Before replacing the entire 184-input actor, consider experiments such as:

```text id="ojfvm4"
pooled global depth
+
small raw-depth patches around low-TTC / high temporal-change regions
```

or:

```text id="nu1lt7"
coarse 80-cell geometry
+
selected high-resolution dynamic cells
```

The selection mechanism can initially be deterministic and based on measured depth change.

Only learn it if deterministic selection proves insufficient.

Test this specifically on fast moving threats and narrow obstacles, not the entire benchmark.

## I. Treat recurrence as an evidence-triggered option, not a philosophical choice

Several strong recent aerial-navigation systems use recurrence.

The 2025 privileged-ToA quadrotor system uses a GRU to combine current and previous observations, and other recent end-to-end flight work reports benefits from temporal architectures.

That does not imply Metal-nav should immediately become recurrent.

Our explicit depth history and geometric memory may already encode enough temporal structure more cheaply and transparently.

The experiment should be:

```text id="2mh6fi"
same training distribution
same deployment inputs
same PPO budget
similar parameter budget

feed-forward + explicit memory
vs
small GRU
```

Evaluate specifically on cases that require information from the past.

If recurrence only improves tasks that were actually unobservable, reject that interpretation.

If it consistently improves valid partially observable tasks without damaging transfer or throughput materially, then increasing recurrent capacity becomes justified.

## J. Consider curriculum structure rather than a flat task pool

Recent 2026 work on **Active Curriculum Refinement (PATH)** treats related environments as a graph of prerequisite/difficulty relationships, first expanding across diverse curriculum paths and then concentrating training on unmastered regions.

Our tasks naturally have this structure:

```text id="k2wa01"
clear goal
   ↓
single obstacle
   ↓
blocked direct path
   ↓
larger detour
   ↓
narrow detour
   ↓
temporary occlusion
```

and independently:

```text id="p5ujzw"
static obstacle
   ↓
slow mover
   ↓
crossing mover
   ↓
short TTC
   ↓
multiple movers
```

Instead of one manually ordered curriculum, represent these as related parameterized regions.

A learner may be advanced along one dimension while weak on another.

This could allow the curriculum to become multidimensional rather than:

```text id="pdbpgv"
stage 1
→ stage 2
→ stage 3
→ stage 4
```

Do not implement a large generic curriculum framework first.

Test whether this representation improves task selection using the existing generator.

# Suggested order of investigation

Based on the repo's current evidence and the external work above, I would currently prioritize:

```text id="sm48vy"
1. reward / ToA-potential experiment

2. task metadata + measurable difficulty

3. sampler comparison:
   uniform vs current vs learning-progress vs PLR⊥

4. ACCEL-style task mutation around the learning frontier

5. observability-aware local-minima benchmark

6. dynamic-threat representation experiments

7. recurrence / additional model capacity only if still justified

8. larger architectural changes only after these
```

This ordering is a hypothesis, not a mandate.

Change it when evidence says another bottleneck is dominant.

# One especially promising combined training loop

A longer-term version worth attempting if the simpler components validate is:

```text id="q482cu"
large procedural task generator
        ↓
feasibility + observability filters
        ↓
policy probes many tasks cheaply
        ↓
estimate learning value
        ↓
retain frontier / high-progress tasks
        ↓
mutate informative tasks
        ↓
add broad rehearsal sample
        ↓
PPO update
        ↓
evaluate capability matrix
        ↓
adjust curriculum
```

The key idea is:

> Use Metal's exceptional simulation throughput not merely to produce more transitions, but to search a much larger space of possible learning experiences and spend gradient updates on the experiences that actually improve general local navigation.

That may be a more valuable use of the simulator than another order-of-magnitude increase in raw FPS.