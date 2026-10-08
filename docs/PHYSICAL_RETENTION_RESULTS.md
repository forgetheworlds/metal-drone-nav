# Retention preserves useful navigation during PPO

The physical XYZ reference loss substantially reduced forgetting. Ordinary PPO
fell from the composed parent's 244 to 204 course successes out of 256; the
retention treatment finished at 245. Course contacts were 49 for ordinary PPO
and 11 for treatment. Both training seeds improved in the paired comparison.

The four matched runs and all 138 evaluations are complete: 16,777,216 training
transitions and 17,664 scored source flights. Each arm trained for 256 rollouts,
4,194,304 transitions and 32,768 optimizer steps. Actors have 184 inputs, 5,120
hidden units and four outputs; the critic remains 64/64/1. Each candidate is one
navigation policy above the existing command adapter and frozen RAPTOR.

Both arms used the same composed parent per seed, inherited exploration
parameters, fresh Adam, the same 2,048 source task records and the same mixed
clean/delayed allocation. Per-entry counters match the full budget. Reward,
physics, sensing, PPO and parameter anchor stayed fixed. Treatment adds only
the training-only physical XYZ mean loss. Its teacher is removed at inference.
The coefficient 4.134632354 came from the registered TRAIN gradient-norm rule;
development outcomes did not select it. [Implementation and gates](PHYSICAL_RETENTION_EXPERIMENT.md).

![Paired navigation outcomes](../artifacts/plots/physical-retention.png)

Every cell below pools two separately trained policies on the same 128 task
instances per panel. Parent and endpoint policies are graded without selecting
an earlier peak. All failures remain in the denominator.

| Source panel, success/contact/timeout | Starting parent | Ordinary PPO | PPO + retention |
|---|---:|---:|---:|
| Nominal course | 244/12/0 | 204/49/3 | **245/11/0** |
| Course, both 100 ms delays | 233/23/0 | 185/71/0 | **236/20/0** |
| Course, combined stress | 227/28/1 | 195/59/2 | **228/28/0** |
| Long open | 256/0/0 | 234/0/22 | **256/0/0** |
| Long hallway | 256/0/0 | 256/0/0 | **256/0/0** |
| Short open | 250/0/6 | 240/0/16 | **249/0/7** |
| Static A | 220/24/12 | 207/33/16 | **217/28/11** |
| Static B | 228/17/11 | 219/24/13 | **232/16/8** |
| Static C | 222/26/8 | 206/36/14 | **217/28/11** |
| Clutter | 224/15/17 | 218/18/20 | **227/19/10** |
| Fresh course, nominal | 246/10/0 | 216/40/0 | **246/10/0** |
| Fresh reflected course, combined stress | 170/86/0 | 147/109/0 | **181/75/0** |

The fresh reflected stress result improves over the parent in both seeds:
79 -> 83/128 and 91 -> 98/128. Against ordinary PPO the gains are 11 and 23.
This is evidence of some improvement beyond preservation. It is still source
development evidence from existing procedural families, with ideal ego state
and no wind; it is not blind final or independent simulator transfer.

On common successful flights, treatment reaches long-open goals 0.44 s faster
than the parent and fresh long-open goals 0.55 s faster. Reflected combined
stress adds 0.04 s. Training took about 35.5 minutes for all four arms; treatment
cost about 21.5% more than control. These are shared-laptop wall times, not a
hardware-only benchmark. Both inference policies have the same architecture.

## Five gates still fail

The result is **not adopted as the finished navigator**. The primary course
success/contact gates pass, but static C loses five successes from the parent,
static A and clutter add four contacts each, and old/fresh short-open success
is 249/250 rather than the required 253 out of 256. The original gates remain
unchanged. A parent that already misses an absolute floor does not excuse it.

The arrival diagnosis reveals a specific teacher weakness. Seed 1 parent solves
122/128 old open tasks; ordinary PPO solves 128; retention solves 121. Timeout
flights finish slowly outside the goal region. On those same student states,
read-only replay of the parent predicts weak or negative closing velocity while
the ordinary PPO control predicts positive closing velocity. Reference-position
error is about 0.05–0.06 m. All three diagnostic runs reproduce their scored
flight CSVs exactly. Counterfactual means do not prove a whole alternative
trajectory, but they justify testing whether the teacher obstructs final approach.

The next isolated hypothesis releases the teacher penalty smoothly near the
supplied goal while retaining it farther away. The risk is losing avoidance
near doorways or other close-goal obstacles. The independent MiMo audit also
examines critic information, fitting and update correctness. No coefficient
sweep or full extension follows from this result alone.

## Execution repairs and reproduction

The delegated implementation initially supplied unrun gate procedures and an
uncalibrated coefficient 0.05 request. Root preserved and rejected that request,
repaired derivative/descent/batch fixtures and sidecar source binding, then ran
actual math, full default parity, enabled resume and mismatch gates. The original
grader compiled the wrong CLI. Root built the stress evaluator, verified exact
nominal and combined-stress parent-flight parity, and completed grading without
repeating training. Failed logs and original request/build records are retained.

The [754-input archive](../evidence/inputs/physical-retention/records.tar.gz)
contains full final learner checkpoints, composed parents, banks, raw flight
and physics tables, exposure counters, histories, source, gate receipts and
residual diagnostics. It reproduces the independent review and figure:

```sh
python3 navigation_physical_retention_results.py \
  evidence/inputs/physical-retention/records.tar.gz \
  --output evidence/inputs/physical-retention/review.json \
  --plot artifacts/plots/physical-retention.png
```

Archive-only grading matches the live result exactly, including all five failed
gates. The original navigation goal remains active and incomplete.
