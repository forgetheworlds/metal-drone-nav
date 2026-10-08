# Goal

## Current research phase — October 7

Focus on experience scale, competence-driven curriculum and adaptive task
sampling, with direct-mixed PPO controls and policy capacity as a separate axis.
Optimize held-out capability gained per training hour, not raw FPS or reward.
The exact operator brief is [TRAINING_SCALE_PHASE.md](docs/TRAINING_SCALE_PHASE.md).
Webots runs are permitted when needed for independent validation. Use batch
mode and minimized windows, keep runs isolated on RL port23456, and avoid
foreground recording unless the user requests it. The dedicated local VM was
deleted; training runs directly on the Mac. The broader transfer goal remains intact. Larger models can be research teachers;
we may distill and quantize their behavior for an ESP32-S3-class target. Delivery
still requires one small policy with measured control quality, latency, RAM and
flash use, including geometry guidance. Compression is not assumed successful.

## Operator clarification — 2026-10-04

The navigator receives a geometric goal. Training goals outside the current
sensor view can be useful when they teach the required behaviours: inspecting,
braking, short memory, detours and safe goal-directed motion. Do not reject a
task merely because its goal is outside the camera frustum. Judge training by
learned navigation capability and transfer. Keep visibility labels accurate,
and check physical feasibility and whether needed geometry can be observed in
time. A visible-goal training mixture is an experiment, not a universal task
acceptance rule.

## Combined-policy acceptance — 2026-10-05

The intended outcome is one navigator that handles static obstacles, moving
obstacles, tight spaces and longer destinations together. Train reusable
behaviors across varied tasks and combinations, preserve strong existing
skills, and verify the frozen system in independent environments. Separate
specialist successes are useful milestones, not completion of this outcome.

## North star

The eventual system is a drone capable of carrying out open-ended tasks in unfamiliar environments.
A larger cloud multimodal model performs slow semantic reasoning. It receives camera observations approximately every few seconds, or after a selected local destination has been reached, and decides what visually meaningful place the drone should reach next based on the user's task and its current view.
Example mission:
Exit a train, search this side of the platform for stairs, move around to inspect the opposite side, return to the user, and then guide the user toward the desired destination.
The cloud model is responsible for reasoning such as:
- what the user is asking;
- recognizing stairs, corridors, exits and other semantic objects;
- deciding what area should be inspected next;
- maintaining mission-level progress;
- selecting the next visually grounded local destination.
It is NOT responsible for micromanaging flight around chairs, walls, poles, tables, people, doorway edges, overhangs, or other local geometry.
A visual target selected by the cloud is converted by an onboard geometric/perception system into a verified local 3-D goal or goal region.
Assume that upstream goal estimation will eventually be made sufficiently reliable. This repository does not need to solve semantic goal selection.
The job of metal-drone-nav is:
Given a valid local 3-D goal plus deployable onboard sensing and vehicle state, reach that goal quickly, safely and reliably through previously unseen local 3-D geometry.
Conceptually:
```text
verified local goal
+
local depth / geometry
+
temporal perception
+
ego state
        ↓
navigation policy
        ↓
desired body velocity + yaw rate
        ↓
persistent trajectory adapter
        ↓
frozen RAPTOR
        ↓
motors
```
The navigation policy and local geometric planner are not assumed to be separate systems.
The navigation policy may itself learn the short-horizon planning required to get around obstacles on the way to a provided local goal.
Do NOT optimize this repository for understanding the whole human mission or constructing a semantic/global route through an entire building. That belongs above this layer unless experiments demonstrate that some additional onboard memory is necessary for LOCAL navigation.
The cloud operates in stages.
A typical episode of the deployed local navigator should resemble:
```text
cloud selects a visually grounded nearby destination
        ↓
navigator reaches it through local geometry
        ↓
cloud receives a new view and chooses another destination
        ↓
navigator reaches that one
        ↓
repeat
```
Therefore the core capability is not:
Navigate from one side of an unknown building to a final coordinate 50 metres away.
It is closer to:
Reach the next sensible locally supplied target despite whatever local geometry lies between the vehicle and that target.
The local goal will normally be visually grounded at the time the higher-level model selects it, or immediately beyond a visible affordance such as a doorway/opening.
This distinction must influence the training distribution.
Long hidden mazes and globally ambiguous routes may be useful diagnostics, but they should not dominate training simply because they are harder.
Improve the TRAINING DISTRIBUTION and TASK GENERATION so that training efficiently produces a general local navigation policy.
Do not begin by making the neural network larger or by replacing PPO.
First make sure the policy is being trained on the right problem.
Your goal is to answer experimentally:
What distribution of local start-goal navigation problems best teaches the policy the reusable geometric skills required between successive cloud-selected visual subgoals?
Treat the task generator itself as a research object.
Think in terms of NAVIGATION CAPABILITIES, not named simulator obstacle families.
Existing boxes, poles, doors, tables, hallways, rooms, barriers, spheres, etc. are primitives for producing spatial problems. They are not themselves the final taxonomy.
Useful capabilities include:
- accelerate toward a free goal;
- decelerate appropriately;
- arrive without overshoot;
- recover when starting with nonzero velocity;
- handle arbitrary starting yaw.
- obstacle lies between vehicle and goal;
- choose left/right/above/below;
- return efficiently toward the goal afterward.
- doorway;
- gap;
- corridor;
- furniture spacing;
- vertical opening;
- constrained clearance.
- direct segment is blocked;
- reaching the goal requires temporary lateral or vertical motion;
- Euclidean goal distance may temporarily increase;
- the detour should still be LOCAL and realistically solvable from onboard information.
- multiple traversable gaps;
- one is shorter, wider, safer or dynamically preferable;
- avoid hard-coding one detour direction.
- recently observed geometry leaves the FOV;
- safe continuation may require remembering it briefly;
- distinguish genuine memory requirements from impossible observation setups.
- crossing obstacle;
- approaching obstacle;
- closing gap;
- different relative velocities and time-to-collision;
- temporal information must actually be necessary.
Separately vary:
- initial velocity;
- braking requirement;
- wind/disturbance;
- vehicle parameters;
- sensor delay;
- command delay;
- depth noise/dropout.
Do not conflate geometric/navigation difficulty with control/sensor difficulty. They should be independently measurable axes.
Obstacle/task generation must be driven by the intended behavioral problem.
Bad generator philosophy:
Add more random boxes until success drops.
Better philosophy:
Construct diverse situations in which a particular navigation decision is necessary, vary all irrelevant geometry, and verify that multiple solutions or geometric variations exist.
Avoid teaching accidental shortcuts such as:
- always pass doors on one side;
- obstacle type determines correct action;
- fixed goal direction;
- fixed start pose;
- fixed obstacle ordering;
- family-specific coordinates;
- repeated exact room composition;
- constant distance or speed;
- a particular training seed being predictive of the solution.
Use mirroring, rotation where valid, dimensional variation, placement variation, composition variation, and independent train/dev seeds.
Prefer distributions with continuous variation over a small catalogue of memorisable templates.
Every generated task used for learning must be known to be physically/geometrically feasible under the declared task contract.
Use privileged geometry ONLY on the training/evaluation infrastructure side to:
- reject impossible starts/goals;
- ensure adequate endpoint clearance;
- establish that a collision-free route exists;
- estimate route/path difficulty;
- compute training-only diagnostics or potential shaping;
- classify the problem.
A witness route is a GENERATOR/TEACHER diagnostic.
It must never become an actor input unless running an explicitly labelled oracle experiment.
The actor should continue receiving only information that can exist at deployment.
Do not interpret a geometric witness as proof that the actual RAPTOR-controlled vehicle can complete the task inside the episode budget. Where necessary, separately verify dynamic/time feasibility.
Do not train heavily on tasks whose correct solution fundamentally depends on information the deployed policy could never have.
The current local-waypoint visibility audit found contacts where the colliding geometry had never appeared in the sensor, including obstacles entirely outside the FOV.
When generating tasks, explicitly measure questions such as:
- Was the relevant obstacle ever observable before the decision became unavoidable?
- How much warning time did the sensor provide?
- Was the target itself visually/local-geometrically grounded?
- Does solving this require short-term memory, or impossible knowledge?
- Is a failure due to policy quality, sensor FOV/resolution, occlusion, control lag, or task construction?
Impossible or severely aliased cases can remain useful diagnostics, but label them. Do not silently mix them into the main learning distribution and then ask PPO to solve missing information.
Do not use a single vague difficulty scalar unless its meaning is explicit.
Useful measurable difficulty variables include:
- direct route clear vs blocked;
- shortest feasible path / Euclidean distance ratio;
- required detour distance;
- minimum route clearance;
- gap width relative to vehicle envelope;
- vertical clearance;
- number of meaningful route choices;
- obstacle density;
- required heading change;
- required braking;
- start speed;
- goal distance;
- relative obstacle speed;
- time to collision;
- sensor warning time;
- occlusion duration;
- sensor/command delay;
- noise/dropout;
- dynamics variation.
Keep these attributes in task metadata so results can be stratified later.
The generator should make it possible to answer:
What specifically makes this episode difficult?
Extremely easy episodes provide little new information.
Nearly impossible episodes provide little useful successful experience.
Prefer substantial training mass near the current policy's learning frontier, while retaining enough easy/rehearsal tasks to preserve existing capabilities.
Do not assume a fixed success percentage is universally optimal, but investigate adaptive sampling based on measured competence.
The repository already contains failure-weighted challenge sampling. Understand what it actually prioritizes before reusing it. Do not assume mean absolute GAE is automatically the best notion of task difficulty.
Potential alternatives include sampling based on some combination of:
- recent success;
- collision;
- timeout;
- improvement rate;
- staleness;
- task class;
- difficulty stratum;
- uncertainty.
Any adaptive sampler must retain a nonzero broad-distribution floor so the policy does not forget solved skills.
The repository has repeatedly observed capability tradeoffs:
- improving doors while degrading tables;
- improving disturbance robustness while degrading clean transfer;
- continued training after a strong checkpoint collapsing performance.
New training should therefore be cumulative.
When introducing a new skill, retain rehearsal from previously learned capabilities.
Do not train sequentially on isolated families and assume earlier behavior will survive.
Measure retention explicitly.
Do not create tasks and assume the existing reward is automatically appropriate.
The reward audit already demonstrated successful long detours that can rank below hovering under the current discounted objective.
For every major new task class:
1. construct a valid successful reference/witness trajectory;
2. compare its reward/return against obvious undesirable behaviors:
   - hovering;
   - blindly moving straight toward the goal;
   - unnecessary detours;
   - unsafe shortcuts;
   - collision;
3. verify that the objective prefers the behavior we actually want.
For local blocked-goal tasks, investigate training-only route/geodesic progress rather than relying only on Euclidean distance progress.
The repository already contains potential-field infrastructure. Privileged world knowledge may be used to COMPUTE TRAINING REWARD without becoming an actor input, provided the shaping is mathematically and experimentally justified.
Do not claim a reward improvement because offline returns look better. The acceptance test is learned navigation performance.
The current system has:
- 20 Hz navigation;
- 100 Hz RAPTOR/physics;
- current episode budgets around 20 s;
- gamma/GAE choices with finite effective credit horizons.
The long-course audit showed that some privileged witness routes required 21.85–34.80 s despite a 20 s policy budget.
A generator must therefore match task horizon to the actual deployment/local-goal contract.
Long routes are useful only when deliberately testing composition or long-horizon behavior.
Do not call a task “hard local navigation” when it is actually impossible within the declared time/control limits.
The existing local-waypoint experiment uses 1–3 m goals.
Do not blindly treat 1–3 m as the final answer.
The deployment architecture suggests the cloud will choose successive visual subgoals. Determine experimentally what local goal horizon is appropriate.
Possible ranges should be evaluated based on:
- camera visibility;
- local geometry;
- expected cloud reassessment rate;
- policy reliability;
- route complexity;
- training credit horizon;
- speed.
The final system may use adaptive subgoal distances rather than one fixed distance range.
Distinguish:
The vehicle only needs to enter/reach the target region and continue smoothly toward the next goal.
The vehicle should arrive slowly enough to stabilize while the higher-level system reassesses or the mission requires a stop.
The code already has waypoint/final-hold concepts.
Do not force every intermediate navigation target to have exactly the same terminal behavior without considering the deployment contract.
Evaluate both when appropriate.
The final navigator must eventually react to moving threats.
The existing generic navigation-task generator currently rejects dynamic clutter, while separate threat/course experiments exist.
Unify this thoughtfully rather than simply turning dynamics on everywhere.
Start with controlled local dynamic problems where:
- the threat is observable;
- relative motion matters;
- collision would occur without avoidance;
- speed/TTC are measurable;
- success remains physically feasible.
Progress from slow/easy temporal cases to fast crossings/approaches.
Keep static-navigation competence in the rehearsal mixture.
The deterministic geometry prior is strong and sometimes outperforms the learned residual.
Always compare:
- geometry-only;
- learned policy behavior;
- geometry + learned residual where applicable.
Do not celebrate learning simply because PPO loss changes.
A learned policy should demonstrate that it adds useful behavior beyond the deterministic prior on the target capability.
If the learned residual makes a solved geometric case worse, treat that as a real regression.
Maintain strict separation between:
May affect weights.
May be inspected repeatedly and used for architecture/curriculum/checkpoint decisions.
Must not influence training, curriculum design, thresholds or checkpoint selection.
The old FINAL geometry has already been exposed by research. When the system is ready for a serious final claim, generate a fresh sealed suite after freezing:
- policy weights;
- observation contract;
- action mapping;
- geometry logic;
- task-generation logic relevant to evaluation;
- selection criteria.
Independent Webots transfer remains particularly important.
Do not optimize against Webots test outcomes and then call the result zero-shot transfer.
Do not reduce navigation quality to success rate alone.
Track at minimum where relevant:
- success;
- collision/contact;
- timeout;
- time to goal;
- actual mean and peak speed;
- minimum clearance;
- path length;
- path efficiency / shortest-feasible-path ratio;
- RAPTOR tracking error;
- goal arrival speed;
- sensor warning time for relevant failures;
- results by task class and difficulty.
The intended result is:
fast navigation at a high reliability level,
not merely one fast successful trajectory or one conservative collision-free policy.
Do not make large speculative rewrites.
For each proposed change:
1. State the specific observed bottleneck.
2. State the hypothesis.
3. Identify the smallest controlled experiment.
4. Define what outcome would falsify the hypothesis.
5. Preserve the current baseline.
6. Run matched evaluations.
7. Record regressions as well as gains.
8. Only retain the change if the system-level outcome improves.
Do not add model complexity, architectural layers, reward terms or generator complexity without evidence that they address an actual limitation.
Prefer simple explicit code and fixed data flows, consistent with docs/CODE_DIRECTION.md.
Start by auditing the CURRENT local training distribution against the deployment contract above.
Do not immediately code.
First produce a concise written analysis containing:
1. What current navigation_waypoint_training.mm and navigation_tasks.hpp actually train.
2. Which parts align well with the staged cloud → local-goal deployment architecture.
3. Which parts are mismatched or underrepresented.
4. A capability-based taxonomy for the next task generator.
5. Which variables should define task difficulty.
6. Which existing obstacle/world generators can be reused.
7. Which new generators are genuinely necessary.
8. How to guarantee feasibility and observability without leaking privileged information to the actor.
9. A proposed TRAIN/DEV/future-sealed split.
10. A curriculum/sampling strategy that preserves prior skills.
11. Any reward changes required BEFORE making tasks harder.
12. The smallest first experiment that would provide the most information.
Then implement only the minimum infrastructure required for that first experiment.
Keep this statement in mind whenever creating an obstacle, task family, curriculum stage, reward term or evaluation:
The point of training is not to make the drone good at our simulator's obstacle catalogue. The point is to teach a reusable local navigation capability: given a valid nearby goal selected by a higher-level visual planner, use deployable onboard information to get there quickly, safely and robustly through unfamiliar geometry.
Every generated task should have a clear answer to:
What reusable local-navigation skill does this episode teach or measure that the deployed system will actually need?
If the answer is unclear, do not add the task merely because it is difficult.
