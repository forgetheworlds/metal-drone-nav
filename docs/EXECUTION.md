# Heavy-job execution

Training, evaluation and native Webots flights share one exclusive lock:
`results/metal-training.lock` in the primary repository. Worktree jobs must use
the primary repository launcher, not a separate lock in their worktree.

```sh
python3 /Users/muadhsambul/RL/run_locked.py -- COMMAND ARGUMENTS
```

The launcher waits for the lock, then replaces itself with the command while
retaining the open lock descriptor. Its PID and exit status now belong to that
command. Cancelling the launcher therefore cancels the actual command rather
than leaving it running behind a terminated wrapper. The lock releases when
the command exits. The old saved-command path under
`results/mimo-learning-next/run_locked.py` forwards to this public launcher.

A command that creates separate process groups must still own their cleanup.
For a Webots batch, launch each simulator in a new process group, keep the real
process handle, bound its runtime, terminate that group on failure, wait for
the actual exit and save a per-flight receipt. Include load errors, missing
robot or policy, invalid sensor/motor initialization and absent episode result
as execution failures. A mover driver continuing after a failed vehicle
controller is not a completed navigation flight.

Use persistent parent-owned execution for full training and benchmark batches.
An external agent ending while awaiting a background callback has repeatedly
disposed its jobs. A CLI being alive is not evidence that training started.
Inspect the actual child, checkpoint header, exit status and complete outcomes.

The October 5 cancellation check used a private temporary lock. It verified
that the actual command PID equals the launcher PID, a second command waits
during execution, cancellation ends the first command, and the waiting command
then finishes. It did not acquire or alter the shared production lock.

Normal RL Webots runs use port 23456, `--minimize --batch`. The shared Webots
installation, other-project files/settings and other-project processes remain
outside this repository's execution ownership. Visible recording is used only
for actual scene movies and is disclosed with their evidence.

## Code map

The shared data path is depth capture → observation → actor → body command →
reference adapter → RAPTOR → physics. These files own that path:

| Owning module | Responsibility |
|---|---|
| `sensor_profile.hpp`, `sim.metal` | Camera projection, capture pose/time, depth rendering and simulator steps |
| `guidance.hpp`, `sim.metal` | Depth-derived geometry guidance and retained observations |
| `deployment.hpp`, `navigation_training.hpp` | Observation/action contract, checkpoint and actor loading |
| `sim.metal`, `raptor.hpp` | Persistent references and frozen low-level flight control |
| `physics.hpp`, `physics_domain.hpp`, `world.hpp` | L2F plant, declared variation and collision geometry/motion |
| `ppo.hpp`, `ppo.metal`, `main.mm` | Rollouts, GAE, PPO, optimizer and Metal orchestration |
| `navigation_critic_training.mm` | Explicit task-bank trainer and controller-state critic experiment |
| `navigation_training_mix.py` | Fixed environment lanes and measured capability exposure |
| `navigation_challenge_tasks.py`, `navigation_geometry_check.cpp` | Procedural tasks and offline clearance checks |
| `navigation_consolidation.mm` | Source-flight collection and one-actor balanced teacher fitting |
| `navigation_stress_eval.mm`, `navigation_stress_matrix.py` | Frozen stress flights and receipt-gated diagnostics |
| `obstacle_motion.py`, `webots/bounded_motion_scene.py` | Shared bounded motion and isolated native scene export |

Small research entry points deliberately reuse the existing simulator and
trainer. They do not introduce a second physics implementation or a new RL
framework. Shader layouts and task-bank ABI are explicit; numerical parity
and complete workloads justify the specialized hot paths.

The unused recurrent, exploration and active-sensing source drafts remain
local research artifacts. Their presence is not evidence of a tested feature.
The old local CMake recurrent target was removed from the default build because
its source has not been published. No draft source or failed result was deleted.

## Optional research builds

Build into a separate directory to preserve active experiment binaries:

```sh
cmake -S . -B build-research -DCMAKE_BUILD_TYPE=Release -DBUILD_NAVIGATION_EXPERIMENTS=ON
cmake --build build-research --target navigation_stress_eval navigation_consolidation navigation_geometry_check -j 4
```

These targets use the current source. They do not reconstruct an older
experiment merely by having the same actor dimensions. In particular,
`build/metal_nav_motion_train` for the consolidated PPO comparison was compiled
before the diagnostic action hooks were added. Recover its runner from commit
`9aba037` and retain the saved source/binary hashes. Current runner hashes in a
sidecar identify runtime contract metadata, not necessarily compiled source.

## October 5 handoff

Astra's report is [UNIFIED_POLICY_STRATEGY.md](UNIFIED_POLICY_STRATEGY.md).
Coverage, transition influence and interference have driven the subsequent
anchor, bounded-course and consolidation studies. Keep final-checkpoint
results primary; the 1,000-update consolidation midpoint is secondary.

| Local records | State and next action |
|---|---|
| `results/root-distance-learning/` | Complete: four 10,000-rollout runs; long skills gained, static retention failed |
| `results/root-anchor-retention/` | Complete: four 10,000-rollout runs; useful drift reduction, combined gates failed |
| `results/root-motion-fidelity/` | Corrected bounded movers and preserved invalid v1 paths; do not resume v1 |
| `results/root-balanced-valid-motion/` | Complete: four 10,000-rollout runs and 64 evaluations; course gains, arrival loss |
| `results/root-cross-factor/` | Complete: 28 evaluations on length/direction/height recombinations |
| `results/root-valid-motion-native/` | Complete: 96 native flights; public 598-input archive and reviewer |
| `results/root-arrival-recovery/` | Complete: interventions and reference diagnostics; not learned-policy gains |
| `results/root-arrival-consolidation/` | Complete: two 2,000-update students and 48 evaluations; no promotion |
| `results/root-consolidated-ppo/` | Four 10,000-rollout runs and 64 evaluations complete; review final gates before adoption |
| `results/root-stress-matrix/` | Complete: four preflights and 120 evaluations / 15,360 flights; inspect receipts, never restart by default |

The consolidated comparison changes only actor warmstart: unchanged combined
parent versus the final consolidated student. Both train on the same bank,
reward, actor, critic, anchor, physics and PPO settings, with fresh optimizers.
Seeds are 20261142/43. `provenance/jobs.json` holds actual exit status and saved
headers; `provenance/eval-receipts.json` holds all 64 evaluator exits. No result
from the stress matrix selects or trains the actor.

Stress reproduction uses the existing local checkpoint and bank paths:

```sh
python3 navigation_stress_matrix.py preflight
python3 navigation_stress_matrix.py matrix
```

The runner requires a saved `freeze.json`, all four successful preflights,
complete PPO headers and all 64 evaluation receipts. It rejects overwriting
existing flight tables. Each receipt binds actor, bank, output and raw plant
hashes. Failed preflight v1 is retained separately. Declared stress bounds are
mass/thrust ±10%, inertia axes ±10% in addition to mass scaling, motor lag
0.75–1.35, depth noise 0.03 m, 5% missing rays and 100 ms delays. Ego is ideal
and wind is zero. These are diagnostic assumptions, not identified hardware
uncertainty.

Before another experiment, read `decision.md`, saved headers, bank provenance
and actual processes. A timeout in observation is not permission to restart.
Keep defaults and controls unchanged until all capability, safety, speed and
retention gates pass. A fresh sealed set comes after the full policy and
selection rule are frozen.
