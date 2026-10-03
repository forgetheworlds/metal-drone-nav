# Corner imitation: failed experiment and measured perception limit

The source-only imitation experiment did not improve navigation. None of its candidates replaces the selected PPO room policy.

![Recorded training and evaluation](../artifacts/imitation-learning-failure.png)

## What ran

The teacher followed privileged TRAIN routes through actual RAPTOR motors and Metal physics. The student received the final goal, real depth/history and ego features. Intermediate waypoints were never substituted for its goal. The teacher command limit was 1.49 m/s inside the deployed 1.5 m/s cap; collision scoring used the existing 0.18 m sphere and a 20 s budget.

Twenty-four of 30 TRAIN flights succeeded, yielding 5,844 observations and body-command labels. One thousand actor updates minimized squared error in the deployed gated, bounded command. Critic and actor standard deviations were retained. These are parameter warmstarts, not completed PPO optimization or resumable imitation optimizer state.

A second collection used a 50/50 teacher/student world-velocity mixture with teacher yaw zero. None of its 30 flights succeeded. Its 3,157 pre-contact corrective labels were combined with the first dataset for another 1,000 updates. Failed mixture flights are not expert successes. No DEV or FINAL labels were used for training.

| Measurement | Result |
|---|---:|
| Initial source policy, all 90 DEV cases | 39 successes |
| Corners across both stages | 0/30 throughout |
| All DEV cases after 500 updates | 0/90 |
| First-stage recorded batch loss | 0.571 → 0.044 |
| First command, normalized four-axis RMSE | 0.526 |
| Correct initial lateral sign | 13/24 |
| Second-stage final batch loss | 0.018 |

The selected room policy remains the separate, preserved 54/90 control (25/30 rooms). This failed experiment started from an earlier source policy. These scores must not be merged into one progression.

## What the audit established

A controlled probe made 15 pairs from TRAIN geometry, mirrored each pair about Y, and set both final goals' Y coordinates to zero. It used the actual Metal mode-17 sensor, memory and observation kernels. All 184 actor inputs matched exactly at spawn, while the privileged routes required opposite first turns.

These are matched counterfactuals, not original DEV rows. They establish that an immediate route-specific teacher command can require information absent from the observation. They do not prove that the worlds remain indistinguishable after sensing, or that this alone explains every failed original episode. The original fit also gives little weight to the first decision: 24 starting observations form about 0.4% of the dataset.

The independent code audit found no obvious final-goal leak, body/world sign error or command-loss Jacobian defect. A targeted GPU finite-difference check covered translation, guidance gates, yaw and the spherical cap; maximum error was 1.31e-5. The collector supplies teacher command/history before native motor simulation and is limited to the tested zero-delay contract.

The next step is to expose useful sensory information before teaching a route choice, retain broad-family examples, and check held-out completion. Falling training loss is insufficient.

## Reproduce

- [Public source warmstart](../assets/checkpoints/corner-imitation-source-warmstart.bin)
- [Complete progression](../evidence/inputs/navigation-imitation/progression.csv)
- [Proof, raw evaluation receipts and controlled-probe source](../evidence/inputs/navigation-imitation/proof.json)
- [Implementation](../navigation_imitation.mm)

```sh
cmake -S . -B build/imitation -DCMAKE_BUILD_TYPE=Release
cmake --build build/imitation --target metal_nav_imitation -j2
build/imitation/metal_nav_imitation --check results/imitation-gradient-check.json
```

Run heavy simulation under an exclusive fcntl.flock on results/metal-training.lock. With the existing local lock wrapper:

```sh
python3 results/mimo-learning-next/run_locked.py --   build/imitation/metal_nav_imitation   evidence/inputs/challenge-bank-mirrored-v1.jsonl   assets/checkpoints/corner-imitation-source-warmstart.bin   results/imitation-reproduction 1000

python3 results/mimo-learning-next/run_locked.py --   build/imitation/metal_nav_imitation   evidence/inputs/challenge-bank-mirrored-v1.jsonl   results/imitation-reproduction/candidate-1000.bin   results/imitation-aggregate-reproduction 1000   results/imitation-reproduction/demonstrations.bin

python3 evidence.py --imitation-only
```

The lock wrapper is a local helper; the public proof does not depend on an external RL framework. The proof embeds the controlled-probe source and runner for extraction into results/imitation-review/. FINAL tasks, target-simulator training and hardware flight were not used.
