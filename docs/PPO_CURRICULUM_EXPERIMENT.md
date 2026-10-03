# Source PPO corner curriculum

Date: 2026-10-03

## Decision

Do not promote a checkpoint or extend this PPO run. The best all-90 original-start DEV result was 54/90 at stage 3: 0/30 corners, 24/30 rooms, and 30/30 vertical. The preserved rooms-focused control is also 54/90, with 0/30 corners, 25/30 rooms, and 29/30 vertical. No checkpoint met the gate of a corner gain plus at least 25/30 rooms and 29/30 vertical.

The run used 400 PPO rollouts and 1,638,400 transitions. It changed the start of family-14 TRAIN scenes in three stages and kept family-15 and family-16 TRAIN worlds in each balanced batch. Stage 4 returned all corner starts to the original position. Stage 4 fell to 19/90 DEV (0/30 corners, 14/30 rooms, 5/30 vertical). This is evidence that success from shifted reset states did not compose into success from the original start. Stage 1-3 reset-task episode rates below include all three families; they are not corner completion rates.

## Curriculum and physical checks

The bank was `evidence/inputs/challenge-bank-mirrored-v1.jsonl`, SHA-256 `e2170fbe8e8c2d074ffb08de4269175a6de0bdcd410c3b60bd699ee7b5fc491e`. The warm start was the raw-depth lift `results/raw-depth-guidance/rooms-focused-rawlift.bin`, SHA-256 `9e2923ed05e30a02e55e9d74feb3c9104409b407a5dfb94ec4e9c2e2f79e21c0`. Its unchanged control score was 54/90: 0/30, 25/30, 29/30.

For a selected family-14 TRAIN witness node, the curriculum translated the static world and final goal together so the node reached the simulator's normal `[0, 0, 1.5]` spawn. Quarter turns preserve the axis-aligned wall boxes exactly. The code checks the 18 cm vehicle clearance along each remaining witness segment and at the final goal. It selects a later route node when a requested transform would place the route outside the fixed room bounds. Six corner levels used node 3 instead of the requested node 2 or 1 in stages 2 and 3. The final stage uses the original start and world.

At every stage reset, source checks confirmed the selected TRAIN family slots were 43/43/42, the start and RAPTOR reference were `[0, 0, 1.5]`, the vehicle quaternion was identity, rotor RPM was initialized, and the depth ring contained its native 12 m startup values. PPO optimizer moments continued across stages. Stage 2-4 resumed from the full stage-1 PPO checkpoint, including actor, critic, optimizer, and simulator state, then reset the simulator with the next stage's checked world table.

The route node and world geometry only controlled TRAIN starts. The actor kept the 824-feature goal/depth/ego/history observation. No waypoint or hidden route feature entered the actor. The original physics, RAPTOR path, spherical collision body, mode-22 training action map, mode-17 evaluation path, 1.5 m/s cap, and 400-step/20-second DEV budget stayed fixed.

## PPO objective actually used

The trainer used 128 environments, 32 steps per rollout, two epochs, batch size 256, PPO clip 0.2, GAE `gamma=0.99` and `lambda=0.95`, learning rate `0.0001`, entropy coefficient `0.0005`, and gradient norm cap `0.5`. The warm start reset optimizer moments.

Potential shaping was disabled. The reward in `sim.metal::sim_advance` remained `2 × (previous goal distance − next goal distance) − 0.01 − 0.1 × clamp((0.6 − clearance)/0.6, 0, 1) + 10 on success − 10 on collision`. Success used the existing 0.35 m radius. Thus the task gave straight-line final-goal progress, clearance risk, and terminal feedback; it gave no route-node reward. The actor sampled a Gaussian action in mode 22 and executed the existing observation-dependent guidance blend. The reward and PPO discount were not changed.

## Results

| Stage | Corner start | Reset-task outcomes across all sampled families | Original-start DEV corners / rooms / vertical | Total DEV |
|---|---|---:|---:|---:|
| 1 | after second corner, route node 3 | 2,638 / 3,318 success | 0 / 21 / 30 | 51/90 |
| 2 | mid-hall, requested node 2; 6 levels fell forward to node 3 | 1,829 / 2,645 success | 0 / 19 / 30 | 49/90 |
| 3 | after first turn, requested node 1; 6 levels fell forward to node 3 | 2,332 / 2,951 success | 0 / 24 / 30 | 54/90 |
| 4 | original start, route node 0 | 1,810 / 4,824 success | 0 / 14 / 5 | 19/90 |

The stage reset outcome is useful for training diagnostics only. Stage 4 is the original-start curriculum task, but its TRAIN aggregate also includes rooms and vertical scenes. The DEV CSVs each contain exactly 90 original-start episodes. FINAL was not loaded.

## Failure phase from frozen DEV traces

I traced two adjacent mirrored DEV corner cases with both the preserved control and stage-3 checkpoint. The custom trace path was checked against the 90-case evaluator: success, collision, elapsed time, path length, and minimum clearance match for these scenes.

Both policies begin by moving forward at about 1.4 m/s with near-zero yaw. In `f14-dev-0000`, the first opening is north at about `y=4.35 m`; the goal is farther east. The stage-3 policy remains near the first partition at `x≈1.2–1.7 m` and switches lateral direction several times. It reaches `y≈-1.0 m` by 3.5 s, never reaches the north opening, and contacts the wall at 3.77 s. The control contacts at 2.80 s. In the mirrored `f14-dev-0001`, the stage-3 policy contacts at 2.22 s near `x=1.70 m, y=0.14 m`; the control contacts at 2.94 s near `x=1.70 m, y=-1.51 m`.

These traces show the unresolved first-turn phase. The actor does not scan before it reaches the wall. Near the wall, it changes lateral commands without committing to the opening. The teacher's earlier TRAIN diagnosis showed that its successful corner routes used a 2 s sensing prefix and that the privileged mirrored waypoint was off-camera at the branch. These results point to a requirement for a reliable pre-contact sensing action and a way to keep the discovered opening useful while turning. The current 0.4 s geometry history does not prove whether active sensing alone is enough or whether longer memory or a sensor-built map is needed. This result does not prove a mathematical limit of raw depth or feed-forward PPO.

## Reproduction

All three commands hold `fcntl.flock` on `results/metal-training.lock` through `results/mimo-learning-next/run_locked.py`.

```sh
python3 results/mimo-learning-next/run_locked.py -- /usr/bin/clang++ -std=c++17 -O3 -fobjc-arc -framework Foundation -framework Metal -framework Accelerate -DFIXED_PPO_ACTOR_OBS_DIM=824 -DSOURCE_DIR=\"/Users/muadhsambul/RL\" navigation_curriculum.mm -o build/ppo-curriculum
python3 results/mimo-learning-next/run_locked.py -- build/ppo-curriculum --train evidence/inputs/challenge-bank-mirrored-v1.jsonl results/raw-depth-guidance/rooms-focused-rawlift.bin results/ppo-curriculum 100
python3 results/mimo-learning-next/run_locked.py -- build/ppo-curriculum --resume evidence/inputs/challenge-bank-mirrored-v1.jsonl results/ppo-curriculum/stage-1.bin results/ppo-curriculum 100 1
```

The recorded execution completed stage 1, caught an infeasible stage-2 transform during preflight, then resumed stage 2-4 with per-level clear-route selection. The compiled source now preserves that stage-1 transform and the corrected later-stage selection.

## Receipts

- `stages.csv` and `rollouts.csv`: stage metrics and per-rollout PPO diagnostics.
- `run.log` and `resume-run.log`: locked training output, including reset checks and stage-end DEV results.
- `stage-1-dev.csv` through `stage-4-dev.csv`: all-90 original-start DEV results.
- `stage-1.bin` through `stage-4.bin`: full PPO checkpoints. The stage-3 checkpoint is the best total DEV result; it is not promotable.
- `control-traces/` and `stage-3-traces/`: two mirrored frozen DEV trajectories per checkpoint with per-trajectory hashes.
- `build.log`: successful Objective-C++/Metal host build.

Source SHA-256: `332e6a5203070d75994d7a22d8d443183c41d1e9c6e298d9700e2197c9aa15e2`. Compiled binary SHA-256: `4b7c44d09bb3f571ac6d4d6f0ec281d5afb393fbdbab57282f5d095f5bfe14c0`.

No selected asset changed. No Webots run, final-split read, target-simulator training, or external training service was used. No PPO or trace process remains active.

## Next source experiment

Do not increase the PPO budget for the same curriculum; it produced no measurable corner DEV gain. The next source experiment should test a defined pre-contact scan or a memory/map mechanism that keeps the first opening available while the vehicle turns. Keep the existing control and all-90 DEV gate. Do not promote anything without a corner gain and at least 25/30 room and 29/30 vertical retention. Freeze source training before any independent simulator evaluation.

## Public receipt

[Complete stage and original-start DEV proof](../evidence/inputs/ppo-curriculum/proof.json). The reset-task aggregate includes rooms and vertical scenes. Shifted-start success must not be reported as original-start corner success.
