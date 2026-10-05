# One frozen navigator on longer combined courses

The combined-task policies ran through actual Webots physics, sensors, RAPTOR,
motors and collision grading. Each policy stayed frozen throughout its flights.
The first 24 bounded-course development records were selected in file order
before any target result: four per capability, split across two difficulties.
Both parent policies and both combined policies ran all 24 tasks.

![Native combined transfer](../artifacts/plots/native-combined-transfer.png)

| Policy | Success / contact / timeout | Mean successful arrival |
|---|---:|---:|
| Parent seed 1 | 23 / 1 / 0 | 10.033 s |
| Combined seed 1 | 23 / 1 / 0 | 9.800 s |
| Parent seed 2 | 20 / 4 / 0 | 9.995 s |
| Combined seed 2 | 22 / 2 / 0 | 10.120 s |

Each row contains 24 actual flights. Seed 1 retains every outcome and is
0.233 s faster on common successes. Seed 2 has three wins and one loss, with
common successes 0.018 s faster. Across the two seeds, contacts fall from
five to three and success rises from 43 to 45 of 48 trials.

These are independent implementation results on inspected development
geometry, not a sealed final test or physical flight. Source arrival and full
retention gates still fail, so these policies are not promoted as the complete
combined navigator. Sensor noise, delay, estimator error and dynamics stress
are separate tests.

All 96 run receipts show process exit zero, navigation and RAPTOR loaded,
shared hold grading, unique world and policy hashes, and actual trajectory
records. Every moving case has measured obstacle poses checked against the
declared bounded motion within 1 cm. Successful flights hold inside 0.35 m,
at no more than 0.5 m/s, for 0.2 s. Invalid and failed attempts from earlier
smoke work remain separate.

The [598-input archive](../evidence/inputs/native-combined/records.tar.gz)
contains all worlds, manifests, native logs, receipts, trajectory records,
frozen NAV files, selection protocol and controller/motion source. Recompute
the [audit](../evidence/inputs/native-combined/review.json):

```sh
python3 native_combined_review.py evidence/inputs/native-combined/records.tar.gz
```

The numerical reviewer needs only the Python standard library. Normal reruns
use the isolated RL project with port 23456, batch and minimized flags, and the
absolute shared Metal lock. The shared Webots installation is read-only.
