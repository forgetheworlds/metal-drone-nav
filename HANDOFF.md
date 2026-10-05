# Navigation handoff — October 5

The goal is active and incomplete. Build one fast, reliable local 3-D navigator
for supplied geometric goals, with static/moving avoidance, tight/long routes,
robustness and frozen independent transfer. Start with `goal.md`,
`docs/CODE_DIRECTION.md`, `docs/WRITING_STANDARD.md` and `STATUS.md`.

The user has about 1% usage left and asked for a handoff. Automatic continuation
is paused for this handoff; the goal remains active. All current jobs are
stopped or complete. Do not revive old missions or launch a fresh batch blindly.
Only Luna new native agents; external only authorized OMP
`opencode-go/mimo-v2.6-flash`. Root code work and cause-first diagnosis remain
preferred. Preserve default assets, controls, failed runs and actual videos.

## Latest completed learning

Four matched PPO runs completed 10,000 rollouts / 320,000 optimizer steps each;
all 48 evaluation panels completed. Actor 184/64/4 and rich critic 64 remain
unchanged. Control trains nominal; treatment mixes clean and both 100 ms delays
across static/long/bounded courses. Both use the same bank, warmstart, optimizer,
reward and anchor. Alternate FOUR-env blocks prevent task-family/delay coupling.

| Panel | Control success/contact/timeout | Mixed-delay success/contact/timeout |
|---|---:|---:|
| Long open | 241/0/15 | 256/0/0 |
| Long hallway | 255/1/0 | 256/0/0 |
| Static A | 205/32/19 | 206/30/20 |
| Static B | 216/25/15 | 219/25/12 |
| Static C | 214/32/10 | 201/36/19 |
| Short open | 237/0/19 | 233/0/23 |
| Clutter | 210/21/25 | 209/24/23 |
| Nominal course | 240/16/0 | 246/10/0 |
| Sensor delay | 232/24/0 | 239/17/0 |
| Command delay | 226/30/0 | 242/14/0 |
| Both delays | 203/53/0 | 231/25/0 |
| Combined stress | 198/58/0 | 226/30/0 |

Each cell pools two trained policies on the same 128 tasks. Treatment improves
both-delay success in both seeds and is 0.602 s faster on common successes.
However, static C and short-open retention plus absolute B/clutter/open floors
fail. **Not promoted.** No source or native result completes the full goal.

Recompute the complete source verdict and figure:

```sh
python3 navigation_delay_review.py evidence/inputs/delay-learning/records.tar.gz
python3 navigation_delay_results.py evidence/inputs/delay-learning/records.tar.gz
```

The archive has 214 hashed inputs, including final checkpoints, warmstarts,
banks, source, histories, exposure, resume provenance and flight/plant records.
Local records: `results/root-delay-learning/`. The interrupted treatment seed 1
resumed 3,150 incremental rollouts from its saved 6,850 header. All four final
headers are verified. Wall times include long suspension; they are not a clean
throughput comparison. Sidecar v3 pins delays/rehearsal and source; older v2
contracts require their original runner. The checkpoint ABI did not change.

## Failure diagnosis and retained sensing

Old stress failures were replayed without changing grades. Of 37 new contacts
with both delays, 28 hit boxes, five cylinders and four movers. Actual retained
rays/pooling measured the contact object in 28 cases. Nine lacked a geometric
320-ray hit in the retained window; 5,120 same-pose/FOV geometric rays recover
two. Under combined stress, 48 of 57 contact objects were measured and dense
rays recover none of the nine missing cases. This is only the retained final
eight captures, with undelivered frames excluded—not full-episode visibility
or proof of sufficient warning.

`results/root-contact-attribution/` holds raw sensing, contact attribution and
`sensing-review.json`. `review-sensing.py` requires all six replay receipts.
Capture time must replay native float32 0.01 s accumulation. Frame multiplied
by nominal period gave one 0.866 m grazing-ray error. Correct replay matches
655,360 nominal/delayed rays within 7 micrometres. The v1 sensing file contains
images/poses, not independent capture timestamps. Do not overstate that scope.

The opt-in capture-age uncertainty correction remains OFF: it improved combined
stress but worsened both-delay outcomes. Public earlier replay evidence is
`evidence/inputs/stress-failure/records.tar.gz`. Do not repeat it by default.

## Fresh Webots benchmark: 112/192 preserved, deliberately stopped

`results/root-delay-native/` freezes the four final actors and NAV files, a
predeclared selection of 48 fresh courses and the comparison protocol. Two new
seeds (20261210/11), 6–12.5 m routes, six kinds, both difficulties, and reflected
horizontal/vertical geometry. No policy training or source selection used these
new tasks. Nominal native sensing/timing only; this does not test native 100 ms
latency. Both seeds and roles must finish all cases before any headline result.

The partial raw benchmark is also preserved remotely in
`evidence/inputs/delay-native-partial/records.tar.gz`, with a manifest marking
112 complete valid flights and 80 remaining. It includes actors, selection,
banks, world files, flight telemetry and receipts.

The user stopped the intrusive launch behavior. Parent 89792 was terminated;
its owned Webots child was disposed. `receipts.json` preserves 112 valid scored
flights. An interrupted unscored directory is separate. Do not count it as a
crash. Do NOT rerun `run.py`: it rejects existing NAV files, and repeating it
would duplicate completed flights. Write a resume parent from the fixed
`selection.json`, `actors.json`, `receipts.json` and per-flight manifests.
Skip only completed valid (bank/index/actor) tuples; retain all valid failures.
Use the existing NAV files. Verify every actor/bank/source hash first.

### Background launch is now verified

The old launcher already used `--batch --minimize --no-rendering`; macOS still
promoted each Qt instance and stole focus. Non-recording launches now set
`QT_MAC_DISABLE_FOREGROUND_APPLICATION_TRANSFORM=1` in the child environment.
The installed Cocoa plugin supports it. A valid duplicate check kept Chrome
foreground in all 20 samples, with exactly the same success, 9.4 s arrival,
0.419725 m minimum range and stable hold as its original scored flight.
`background-check.json` is a launcher check, NOT an extra benchmark trial.

This is background Cocoa/OpenGL, not true headless. Official Webots headless
instructions use Linux/Xvfb or Docker. Do not set Qt offscreen blindly on this
Mac: cameras need a working OpenGL backend. Do not edit the shared Webots app,
preferences or other project's processes. If focus stealing recurs, stop the
RL batch and move it to a validated headless setup. Never automate GUI flights
again without checking this launcher behavior first.

The helper changed after the original native freeze, solely for child receipts,
cancellation and foreground suppression. Preserve `freeze.json`; record a
supplemental launcher provenance receipt rather than silently replacing hashes.
Native physics, controller binary, actor, sensors and scene generator stay fixed.
Synthetic timeout/parent-signal checks proved owned child disposal. They are
process tests, not navigation evidence. Default recording is a visible exception
and should not run automatically during this handoff.

## Next work

1. Finish the remaining 80 native flights using the verified background launcher,
   exact frozen actors/selection and persistent ownership. Audit loads, world/NAV
   hashes, stable hold, reset velocity, actual mover poses and all failures.
2. Publish the full paired result. Do not promote based on the partial batch.
3. Diagnose static C and short-arrival losses before new training. Missing delay
   exposure is now supported as one cause; broader retention remains unresolved.
   Consider behavior rehearsal during PPO only after inspecting actual failures.
   A larger network, wider view or higher Hz remains a hypothesis, not a fix.
4. Preserve physical horizons, reward units, sensing age and RAPTOR timing in any
   rate comparison. A fresh sealed final set follows complete system freeze.

All heavy work uses `/Users/muadhsambul/RL/results/metal-training.lock` through
`run_locked.py`. Resume rollout counts are incremental. Real PIDs, saved headers,
receipts and raw flights—not a CLI label or loss curve—prove execution.
