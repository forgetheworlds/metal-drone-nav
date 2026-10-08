# Releasing the teacher near goals did not fix arrival

The goal-distance release experiment failed its intended arrival test and weakened
other skills. I kept the stronger unmasked-retention candidate. No masked policy
was promoted and no further coefficient or radius trial follows automatically.

The two masked treatments completed 256 rollouts and 4,194,304 transitions each,
with 32,768 optimizer steps. They used the same composed parents, seeds, source
bank, inherited exploration parameters, fresh Adam, reward, physics, PPO and
coefficient as the completed unmasked treatments. Only the training teacher-loss
weight changed: zero inside the 0.35 m goal radius, smoothly increasing to one
at three radii. There is no mask or teacher at inference.

Root reviewed the source and actual default-off checkpoint parity, active resume
and derivative gates before execution. Prior parents and controls were reused;
46 new scored panels plus 138 retained panels form the full 184-cell comparison.
Every outcome remains in the denominator. The evaluator and task banks are fixed.

| Source success/contact/timeout, pooled /256 | Unmasked retention | Goal-distance release |
|---|---:|---:|
| Short open | 249/0/7 | 238/0/18 |
| Fresh short open | 250/0/6 | 245/0/11 |
| Long open | 256/0/0 | 232/0/24 |
| Fresh long open | 256/0/0 | 242/0/14 |
| Nominal course | 245/11/0 | 246/10/0 |
| Course, both delays | 236/20/0 | 230/26/0 |
| Static B | 232/16/8 | 221/19/16 |
| Static C | 217/28/11 | 219/30/7 |
| Reflected course, combined stress | 181/75/0 | 168/88/0 |

Both seeds lose old open-space arrivals: 121 -> 112/128 and 128 -> 126/128.
Both also lose long-open arrivals: 128 -> 113 and 128 -> 119. The nominal course
is nearly unchanged. That small gain cannot compensate for regression elsewhere.
Common successful long-open flights are 1.58 s slower, and fresh long-open flights
are 1.54 s slower. The registered review rejects 38 checks, including duplicate
checks expressing the same floor; this is not 38 independent failed task types.

The earlier diagnosis remains valid in its scope: the teacher can command weak
or negative closing motion on specific near-goal timeout states. However,
removing its loss near goals did not teach a better endpoint policy. The frozen
counterfactual commands were not a proof that this learning intervention would
work. Shared network parameters, sampling and critic estimates can affect other
parts of a trajectory; this comparison does not isolate which caused the loss.

The next decision awaits the independent critic audit and measured failure
trajectories. A larger critic, a new reward or another radius must have evidence
before another full run. The original unmasked result still has five failed
gates, so it also remains a research candidate rather than the finished navigator.

Raw outcomes, endpoint checkpoints, actual process receipts and the independent
review are retained in `results/omp-arrival-retention/`. Publication of a full
archive is pending. The previous experiment's published
[replayable evidence](PHYSICAL_RETENTION_RESULTS.md) remains the stronger control.
