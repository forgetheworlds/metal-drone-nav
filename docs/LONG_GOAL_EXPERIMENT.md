# Learning to reach longer supplied goals

A locally supplied destination can be at the far end of a visible hallway.
The earlier 1–3 m training banks did not test that behavior. I built two
12–14 m source probes: an open room and a straight hallway with 2 m interior
width. Starts retain the old bank's velocity and yaw. The geometric goal is
the actual destination throughout the flight.

The BC baseline reached 128/128 open goals and 111/128 hallway goals. The fast
PPO policy reached 47/128 and 32/128. The moving-trained policy reached
112/128 and 44/128. This gap exists even on direct routes, before adding
complicated detours or moving obstacles.

I tested whether the actor's distance scalar was one cause. Its usual feature
is distance divided by 10, capped at 1.5. The
[diagnostic](../navigation_distance_probe.mm) caps only that feature at 0.3.
Actual goal position, direction, geometric guidance, reward and arrival tests
still use the full destination. A Metal fixture checks that other features,
ticks and environments remain byte-identical; disabling the transform
reproduced all 128 reference flights exactly.

| Policy | Open success, original → capped | Hallway success, original → capped |
|---|---:|---:|
| Fast PPO | 47 → 69 | 32 → 42 |
| BC | 128 → 128 | 111 → 114 |
| Moving-trained | 112 → 122 | 44 → 28 |
| Static richer-critic final seed 1 | 7 → 49 | 0 → 18 |

These are fixed-weight source diagnostics, 128 flights per cell. They use known
room geometry with new destinations and corridor walls, not a blind final set.
The cap removed the fast policy's 70 open-room contacts but left 59 timeouts.
It also increased the moving policy's hallway contacts from 84 to 98. I did
not adopt it. Distance representation matters, but this change does not teach
reliable long navigation.

The next experiment changes training coverage. Both arms use the same BC
warmstart, actor, rich controller-state critic, PPO and reward. The control
keeps the canonical eight source tasks per environment. The treatment keeps
all eight records byte-identical and adds two long open and two long hallway
tasks. It uses the full distance input. Two matched seeds receive 10,000
rollouts each; final checkpoints are primary.

Fresh long probes are excluded from checkpoint selection. Evaluation also
includes all five static retention banks. Gates require a long-task gain,
retention of the BC long capability and old task skills, an open-task floor,
and arrival speed without extra contacts or timeouts. The new build's default
path reproduced an entire 100-rollout checkpoint byte for byte.

The [bank generator](../navigation_distance_tasks.py) reproduces both original
probe banks exactly. Straight segments prove geometric clearance, not dynamic
feasibility. Actual BC preflights reached all 128 open goals in both the new
TRAIN and development panels, and 112/128 and 110/128 hallway goals. Failed
preflights remain in the experiment records.

All four matched runs completed 10,000 rollouts and 320,000 optimizer steps,
followed by 61 evaluations. Across two independently trained policies on each
128-task panel, the long-trained arm reached 255/256 open goals and 256/256
hallway goals; the short-task controls reached 2/256 and 0/256. Mean successful
arrival was 10.64 s and 10.69 s for the treatment. The long-task safety,
BC-retention and arrival gates passed.

Static retention failed: development B fell 226→215/256 and clutter fell
222→216/256. A was 206→205, C213→210, and short open256→253. The long candidate
is preserved as a useful learned behavior, but it is not promoted as the
combined navigator. Source-selected best checkpoints also fail retention.
The next [anchor comparison](ANCHOR_RETENTION_EXPERIMENT.md) tests preserving
behavior during these updates; [composite challenges](COMPOSITE_CHALLENGES.md)
expose additional combinations of skills.

The [completed-run archive](../evidence/inputs/distance-learning/records.tar.gz)
contains 213 hashed inputs: checkpoints, histories, diagnostics, all flights,
banks, job receipts and original frozen source. The source was recovered from
commit72f19c5 and checked against the launch hashes before packaging. Recompute
the [full review](../evidence/inputs/distance-learning/review.json) with:

```sh
python3 navigation_distance_review.py evidence/inputs/distance-learning/records.tar.gz
```

Defaults and selected policies are preserved. These direct-route tasks test
distance coverage. Complex tight routes, robust sensing and frozen independent
transfer remain separate evidence requirements.

```sh
clang++ -std=c++17 -O3 -fobjc-arc -DSOURCE_DIR="\"$(pwd)\"" \
  -DFIXED_PPO_ACTOR_OBS_DIM=184 navigation_distance_probe.mm \
  -framework Metal -framework Foundation -framework Accelerate \
  -o build/navigation_distance_probe
python3 navigation_distance_tasks.py OPEN_TEMPLATE.bin OUT_DIRECTORY --seed 20261120
build/navigation_distance_probe CHECKPOINT BANK.bin OUT.csv 0
build/navigation_distance_probe CHECKPOINT BANK.bin OUT.csv 1
```

The [diagnostic archive](../evidence/inputs/distance-probe/records.tar.gz)
contains the banks, manifests, all original/capped flight records, tested
checkpoints, invocation receipts and the source used for this probe. Each
input is bound in its SHA256.json manifest. The generator's seed 20261120
reproduces the original bank bytes exactly.
