# Next Phase: General Local Drone Navigation

## Current acceptance priorities — 2026-10-03

The operator prioritizes two outcomes: fast, reliable navigation through complicated tight spaces; and transfer with frozen weights and navigation logic into unseen simulators and environments without additional target-side training. All source training and robustness adaptation precede the evaluation freeze. Development feedback is separate from blind final evaluation. Any secondary target-simulator training is exploratory and cannot satisfy the zero-shot acceptance requirement. Report success, contact, actual speed, clearance and travel efficiency together.


## Why this phase exists

The current project has already answered several important questions.

We have evidence that:

- RAPTOR can provide the low-level flight-control layer.
- The Metal simulator can run the navigation training loop at high throughput.
- The navigation policy can learn useful local obstacle-avoidance behavior.
- The current observation/action interface is workable.
- The policy can be exported independently of the trainer.
- Procedural geometry, depth sensing, temporal information and learned navigation can work together.

That means the project should no longer focus mainly on proving that the basic architecture can work.

The more important question is now:

> Can this become a fast, robust and general local navigation system that continues working in environments that were not constructed specifically for its training?

That is the purpose of this phase.

---

# Target capability

The intended system is approximately:

```text
stereo cameras
      ↓
depth / local geometric representation
      ↓
temporal geometry
+ vehicle state
+ local goal
      ↓
navigation policy
      ↓
desired local velocity + yaw
      ↓
RAPTOR
      ↓
motors
```

A higher-level system should eventually be able to provide a local destination or waypoint without having to solve the detailed flight path itself.

The local navigation system should determine how to move there.

The policy should eventually be capable of navigating unfamiliar three-dimensional environments containing static and dynamic obstacles while maintaining useful speed and safety.

---

# What the policy needs to learn

The problem is broader than simple collision avoidance.

The policy should learn behavior such as:

- move rapidly through open space;
- slow when geometry becomes constrained;
- navigate through doorways and gaps;
- choose between going above, below or around obstacles;
- handle tables, shelves, counters, poles and overhangs;
- avoid getting trapped by locally attractive but poor routes;
- detect when an obstacle is moving;
- reason about future collision rather than only current distance;
- respond to objects crossing its future trajectory;
- maintain sufficient clearance for imperfect sensing and control;
- recover from disturbances;
- continue toward the goal instead of merely avoiding obstacles forever.

The desired outcome is therefore approximately:

```text
reach goal
+
avoid collision
+
minimize unnecessary time
+
avoid unnecessary path length
+
maintain useful safety margin
+
remain robust to imperfect sensing and dynamics
```

The exact learning objective should remain an experimental question.

---

# Why training is now a major concern

The current policy has learned from a relatively constrained procedural world distribution.

That was appropriate while validating the architecture.

It may now be the limiting factor.

A policy can perform extremely well on:

```text
boxes
poles
doors
tables
simple moving spheres
```

without having learned a broadly useful representation of local navigation.

The next training system should expose the policy to a substantially wider range of spatial problems.

The important distinction is:

```text
more training
≠
better training
```

Running additional rollouts against essentially the same experience distribution may produce little improvement or even regression.

The training distribution itself should become a research object.

---

# Environment diversity

Future environments should increasingly resemble actual navigation problems rather than isolated obstacle primitives.

Useful categories may include:

```text
rooms
hallways
corners
intersections
doorways
multiple connected spaces
tables
counters
shelves
poles
beams
overhangs
hanging obstacles
vertical gaps
dense clutter
sparse clutter
forest-like geometry
irregular combinations of geometry
```

Dynamic environments should eventually include:

```text
crossing objects
approaching objects
swinging objects
vertically moving objects
accelerating objects
multiple moving obstacles
closing gaps
other drone-like objects
```

The goal is not to reproduce the visual appearance of the real world.

For a depth-based navigation policy, the important forms of realism are:

```text
spatial realism
task realism
dynamics realism
sensor realism
timing realism
```

---

# Difficulty should form a continuum

Many navigation problems naturally vary in difficulty.

Examples:

```text
wide doorway → narrow doorway

sparse clutter → dense clutter

slow obstacle → fast obstacle

large clearance → small clearance

short route → long route

low speed → high speed

clean depth → noisy/delayed depth
```

This makes it possible to think about a **learning frontier**.

The policy should ideally spend substantial training effort on tasks that are difficult enough to teach something but not so difficult that nearly every rollout fails.

How this frontier is discovered and sampled should be researched rather than assumed.

---

# Failure-driven learning

Failures should become useful data.

If evaluation reveals repeated failure around:

```text
door frames
table edges
overhangs
moving crossings
blind corners
high-speed braking
vertical gaps
```

the system should make it easy to:

1. reproduce the failure;
2. understand the cause;
3. generate related situations;
4. train against them;
5. retest the original failure;
6. confirm improvement on broader held-out cases.

This is preferable to manually inventing the next curriculum stage without evidence.

---

# Speed is part of the navigation problem

The system is intended to navigate quickly.

Speed should therefore become part of both training and evaluation.

The important question is not:

> What is the fastest successful run?

It is closer to:

> How fast can the system operate while maintaining a useful reliability level?

Useful evaluation may therefore examine relationships such as:

```text
speed vs success

speed vs collision rate

speed vs minimum clearance

speed vs path efficiency

speed vs environment difficulty
```

A capable policy should naturally use different speeds in different circumstances rather than relying on one fixed conservative velocity.

---

# Robustness

The real system will not receive perfect information.

Relevant variation eventually includes:

```text
depth noise
missing depth
sensor delay
frame jitter
dropped frames
command delay
state-estimation error
wind
vehicle-parameter variation
motor-response variation
disturbances
```

These should increasingly become part of the training distribution when appropriate.

The goal is not randomization for its own sake.

Randomization should represent plausible deployment variation and should measurably improve held-out robustness.

---

# Learning algorithm research

PPO is currently the verified training baseline.

It should remain available as a control condition.

Alternative methods are interesting only if they improve the system-level outcome.

In particular, recent group-relative continuous-control approaches may be worth investigating because this simulator can cheaply generate multiple attempts at the same navigation problem.

This naturally allows questions like:

> Given several trajectories through the same challenge, which behaviors were relatively better?

Any alternative learning algorithm should be evaluated against PPO under controlled conditions.

The relevant outcome is not novelty.

The relevant outcome is:

```text
better navigation
better stability
better generalization
and/or
less wall-clock training cost
```

---

# Webots

Webots should become an important independent environment.

The purpose is not simply to make a nicer demo.

It can provide:

- a second physics implementation;
- independent sensor simulation;
- actual simulated motors/propellers;
- richer scene construction;
- a procedural navigation benchmark;
- RAPTOR integration validation;
- failure discovery outside the Metal simulator;
- potentially a secondary training environment.

A basic Webots task can remain extremely simple:

```text
start at A

target = B

unknown obstacles exist between them

reach B without collision
```

The world around that task can vary substantially.

---

# RAPTOR in Webots

The Webots integration should test the actual low-level control stack.

A vehicle should be created or configured whose physical parameters make the RAPTOR integration meaningful.

Before testing navigation, RAPTOR itself should be validated in Webots.

The important question is:

> If RAPTOR receives valid reference commands, can it reliably fly this simulated vehicle?

Only after that is established does it become meaningful to test:

```text
navigation policy
      ↓
velocity/yaw reference
      ↓
RAPTOR
      ↓
Webots motors
```

This separation makes failures diagnosable.

---

# Webots as an independent benchmark

Webots should eventually support reproducible procedural evaluation.

An episode can conceptually be:

```text
generate world from seed

place vehicle

choose target

run frozen policy

terminate on:
- success
- collision
- timeout

record metrics
```

Potential metrics include:

```text
success rate
collision rate
time to goal
path length
path efficiency
average speed
peak speed
minimum clearance
RAPTOR tracking error
```

Different benchmark suites can exercise different capabilities.

Some Webots worlds should remain genuinely held out.

If Webots becomes part of development or training, separate development worlds from final evaluation worlds.

---

# Metal and Webots should complement each other

The current Metal environment has an important advantage:

```text
extremely cheap parallel experience
```

Webots has a different advantage:

```text
richer and more independent simulation
```

A potentially strong research loop is:

```text
train quickly in Metal
       ↓
evaluate in Webots
       ↓
discover failure
       ↓
understand missing capability
       ↓
improve training
       ↓
retrain
       ↓
evaluate again
```

Whenever possible, reproduce important Webots failures in the high-throughput simulator.

When a failure depends on phenomena that cannot be represented adequately there, Webots itself may become useful for additional training or adaptation.

Whether that improves results should be measured.

---

# External evaluation

Eventually the project should also be tested against external navigation benchmarks.

Candidates currently include:

```text
DodgeDrone
AvoidBench
FlightBench
```

These are examples rather than mandatory dependencies.

The project should continually look for evaluation environments that genuinely test the capability we care about.

External benchmarks should be used to increase confidence that the system has learned navigation rather than peculiarities of our simulators.

---

# Evidence matters

This repository should increasingly make its claims visible.

Results should not exist only as console output or comments in a research log.

Useful artifacts include:

```text
training curves
evaluation curves
success over training
success vs speed
collision vs speed
robustness curves
PPO vs alternative algorithm comparisons
wall-clock training comparisons
rendered environments
screenshots
short videos
GIFs
failure examples
before/after comparisons
Webots runs
```

These should come from real reproducible experiments.

A useful repository should allow someone to look at the README and quickly understand:

```text
what was built
what actually works
how well it works
how training changed it
where it fails
what evidence supports those statements
```

Videos and rendered examples are especially useful for navigation because many important behaviors are difficult to understand from a scalar metric.

For example, showing:

```text
early policy → crashes

later policy → avoids obstacle but takes poor path

improved policy → navigates cleanly at higher speed
```

can communicate something that a success percentage alone cannot.

Failures are also worth showing when they explain the next research direction.

---

# Research discipline

Do not optimize for producing features.

Optimize for reducing uncertainty about the final capability.

For every major change, ask:

```text
What problem are we trying to solve?

What evidence says this is currently a problem?

What is the smallest experiment that could test the idea?

What result would disprove the hypothesis?

Did the system-level outcome improve?

What became the next bottleneck?
```

Keep successful baselines.

Keep useful failed experiments.

Do not protect an implementation simply because substantial work went into it.

Likewise, do not replace a working component merely because a newer technique exists.

---

# Current central question

The project has progressed beyond:

> Can we train a drone navigation policy?

The more useful question is now:

> What combination of training environments, learning process, sensing representation, control integration and evaluation is required to produce a fast, robust local navigation policy that continues working outside the simulator that trained it?

The purpose of this phase is to answer that question with experiments and evidence.

The eventual repository should make the answer increasingly visible through:

```text
working code
benchmarks
plots
rendered environments
videos
failure analysis
and repeatable experiments
```