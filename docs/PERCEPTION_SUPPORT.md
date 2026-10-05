# Experimental perception support

This profile changes the three geometry-prior features used by the 184-input
navigation actor. It is an experiment, not a promoted policy or a safety
guarantee. The original profile and policy assets remain unchanged.

The old hit-only clearance query returns 3 m when it has no retained obstacle
hit. That does not establish that a rear or lateral direction was observed.
The new module keeps unknown space distinct from occupied space. It derives a
sampled free-range bound from usable depth and its capture pose, checks near
and far projections in that camera, and uses a conservative pixel neighborhood.
Retained obstacle hits still bound motion. Invalid depth or an unsupported
direction remains unknown; a selected unknown direction allows slow exploration.
This sampled bound does not certify an entire swept vehicle volume.

The geometric goal may be outside the current view. Those tasks are retained.
The old out-of-view goal speed cap already existed; this experiment does not
claim to introduce it. Its new contrast includes an unknown lateral or vertical
alternative when an obstacle blocks an in-view goal.

Both the host and Metal use [perception_support.hpp](../perception_support.hpp).
The explicit experimental [runner](../perception_support_runner.mm) uses the
existing simulator, RAPTOR and PPO implementation through `main.mm`. The flag
changes the prior tail before actor sampling. PPO stores that same observation
and raw Gaussian log probability. Evaluation passes the same profile flag.
Resume metadata binds the profile and compiled module/kernel hash.

## Build and run

```sh
clang++ -x objective-c++ -std=c++17 -O3 -fobjc-arc \
  -framework Foundation -framework Metal -framework Accelerate \
  -DSOURCE_DIR="\"$PWD\"" -DFIXED_PPO_ACTOR_OBS_DIM=184 \
  -DNAV_SENSOR_PROFILE=1 perception_support_runner.mm -o build/metal_nav_ps
python3 run_locked.py -- build/metal_nav_ps ps-test
python3 run_locked.py -- build/metal_nav_ps local-train 10000 /tmp/ps.bin \
  assets/checkpoints/local-waypoint-fast-experimental.bin.best \
  --spec source --seed 20261101 --time-cost 1 \
  --eval-spec dev-a --eval-every 50 --perception-support 1
```

`local-train` counts are incremental on resume. Use the saved header and strict
sidecar rather than subtracting a history-row count. `--perception-support 0`
is the matched control. The portable function accepts camera mount explicitly;
the current training comparison uses the legacy camera profile.

## Verified before full training

Root corrected the first untrained patch's unknown-clamping and unrelated-ray
errors, an undersized pose array, impossible directional test, wrong test-ray
index and Metal constant address space. Failed code and logs are retained.

The executed checks cover front/rear/side classification, delayed capture yaw,
missing depth, frame expiry, allowed unknown motion, host/Metal prior parity,
unchanged non-prior observation components and startup sensor delay. Maximum
host/Metal prior difference was below 1e-7. The default profile produced
byte-identical 128-flight CSVs. A matched 100-rollout control also retained
identical actor/critic optimizer parameter hashes. A support-on one-rollout
smoke saved a real header with one rollout and 32 optimizer steps.

## Learning comparison and limits

Four fresh runs use control/treatment across seeds 20261101 and 20261102,
10,000 rollouts each, identical fast warmstart, task bank, reward, physics,
network, PPO and action transform. Final checkpoints are primary; selection on
dev-a is disclosed. Fresh nonselector dev-k and old retention sets are evaluated.
The fixed gates require fewer contacts without material loss of success, old
skills, timeout rate or speed. A missed gate means no adoption at this budget.
There is no coefficient sweep or larger network in this experiment.

Full training and evaluation are still pending. Lower loss or prior correctness
alone is not evidence of better navigation. Exporting weights alone does not
deploy this profile: a native frontend must reproduce the same prior from its
range/pose history, frame timing and sensor semantics before a frozen transfer
claim. Generic production `eval`/`export` do not apply the experimental profile.

## Candidate-query optimization

The first implementation queried all 85 candidate directions serially in one
thread per environment. Treatment collection took about 0.58 s per rollout,
versus 0.028 s for the control. Root preserved that pilot and split the
independent queries across GPU threads. A small per-environment table holds
their occupancy and free-range evidence; candidate scoring then uses that table.
The scalar reference is retained in the same portable module.

In a matched 100-rollout run, wall time fell from 69.8904 s to 8.11273 s
(8.61×). The full saved checkpoints were byte-identical, including optimizer
and simulator state. This is a performance gain, not a navigation gain. The
four full comparison arms were restarted fresh with this faster implementation;
the original complete control and partial treatment remain as pilot evidence.
