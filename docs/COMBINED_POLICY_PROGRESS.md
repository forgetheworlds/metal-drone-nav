# What combined-task training improved

Four matched source runs each completed 10,000 rollouts/320,000 optimizer steps,
followed by 64 evaluations. The new distribution reserves 50% of transitions for
static tasks,25% for long goals and 25% for valid bounded-motion courses. The
control reserves 50% static and 50% long. Network, reward, physics and anchor
settings are identical. Final checkpoints are primary.

| Source panel | Control success/contact/timeout | Combined success/contact/timeout |
|---|---:|---:|
| Long open | 256/0/0 | 226/0/30 |
| Long hallway | 254/1/1 | 256/0/0 |
| Static A | 206/37/13 | 216/35/5 |
| Static B | 211/35/10 | 222/25/9 |
| Static C | 209/33/14 | 222/27/7 |
| Short open | 240/0/16 | 235/0/21 |
| Clutter | 202/31/23 | 219/23/14 |
| Bounded composite | 223/33/0 | 243/13/0 |

Each cell pools two trained policies on 128 task instances. New courses improve
with 25 paired wins and 5 losses. Static obstacle sets also improve, but free-space
arrival loses reliability. The combined navigator is not promoted.

Separate factor probes vary length 5.5–12.5 m, reverse travel, swap floor/ceiling
layouts, and combine those changes. Combined models reach 117/118,115/120,
110/115 and 107/113 successes per 128 respectively. Their frozen parents reach
119/108,113/107,100/98 and 96/85. These are development recombinations, not a
sealed final set; they were neither trained nor used for checkpoint selection.

An independent 96-flight Webots benchmark compares both frozen parents with
both combined models on the first 24 bounded-course records, in file order.
Parent success is 23/24 and 20/24; combined success 23/24 and 22/24. Contacts fall
from 5 to 3 pooled over 48 trials; all runs are valid, and no target training occurs.
Seed 1's common successful flights are 0.233 s faster; seed 2's 0.018 s faster.
This is real full-stack development transfer, not physical flight or completion.

## Diagnosing the remaining arrival failure

The open-room timeouts keep moving and accumulate 9–20 m paths. A 15 s late
reference controller recovers 20 of 21 short-open failures but introduces one
contact; it recovers 17 of 30 long-open failures. It is an intervention diagnostic,
not the deployed policy. Frozen BC recovery from the same learner states helps
only modestly. These late rollouts cannot be treated as ordinary BC states.

A full 400-tick capture also checked reference-error observability. In the last
five seconds of the failed second-seed flights, no actor reference component
was clipped. Median tracking error is about 0.070 m for short-open and 0.079 m for
long-open; maximum long-open error 0.162 m. Reference windup does not explain
these failures. I did not add an anti-windup fix.

## Consolidating behavior into one actor

BC completed 1,482/1,536 new source arrival tasks with varied heights, directions,
distances and initial motion. Successful flights yield 243,058 observation/action
pairs;3 contacts and 51 timeouts remain recorded. Separate successful static,
hallway and bounded-course flights come from the stronger combined policy.
One student uses equal capability groups, then equal successful episodes, then
frames. Group IDs never enter the 184-input actor.

Two students completed 2,000 supervised updates and 48 closed-loop evaluations.
At the 1,000-update midpoint, all three open/hallway panels reach 256/256. Final
2,000-update counts are short open 251,long open 255,hallway 256, static A 203,
B 209, C 215, clutter 214, composite 238. Neither stage passes full retention. Lower
imitation loss is not claimed as navigation progress by itself.

The matched PPO comparison used the prescribed final 2,000-update student
versus its unchanged combined parent, with the same balanced source bank and
PPO/anchor/reward settings. Both are single actors. No deployed teacher switch,
network-size change or reward coefficient sweep is introduced. All earlier
checkpoints, failures and native recordings remain preserved.

## PPO after consolidation: completed

All four runs completed 10,000 rollouts and 320,000 optimizer steps; all
64 final/selected evaluations completed. Root checked checkpoint headers,
128-row exclusive outcomes, stable arrivals and identical paired task geometry.
The primary final results pool two seeds on the same 128 tasks per panel:

| Panel | Parent warmstart: success/contact/timeout | Consolidated warmstart: success/contact/timeout |
|---|---:|---:|
| Long open | 253/0/3 | 255/0/1 |
| Long hallway | 255/1/0 | 256/0/0 |
| Static A | 212/39/5 | 209/38/9 |
| Static B | 227/29/0 | 220/28/8 |
| Static C | 212/40/4 | 217/31/8 |
| Short open | 254/0/2 | 246/0/10 |
| Clutter | 220/29/7 | 218/25/13 |
| Bounded composite | 217/39/0 | 240/16/0 |

Combined courses gain 31 paired wins with eight losses. Common successful
course flights take 0.249 s longer. Long open gains two successes but takes
0.746 s longer on common successes. Short open loses eight successes and
static B loses seven. Several other panels also miss the 0.5 s speed gate.
The full acceptance gates fail; the treatment is not promoted. Static A and
open failures are stronger in seed 1, so pooled course gains do not establish
consistent improvement across all skills.

Records, per-seed counts and paired timing are in
`results/root-consolidated-ppo/root-summary.json`; the complete jobs,
checkpoints and 64 flight tables remain in that directory. These local records
have not yet been packaged as a public raw archive. The frozen stress matrix
is a separate diagnostic; no stress outcome selects or trains these policies.

## Factor-separated stress: completed

The frozen diagnostic completed 120 evaluations and 15,360 source flights.
Root checked every receipt, flight-table hash, raw-plant hash and exclusive
terminal outcome. Course outcomes for the two PPO seeds are:

| Profile | Parent warmstart: success/contact/timeout | Consolidated warmstart: success/contact/timeout |
|---|---:|---:|
| Nominal | 217/39/0 | 240/16/0 |
| Depth noise, 0.03 m | 220/36/0 | 242/14/0 |
| Missing rays, 5% | 223/33/0 | 244/11/1 |
| Sensor delay, 100 ms | 213/43/0 | 227/29/0 |
| Command delay, 100 ms | 209/47/0 | 218/38/0 |
| Both delays | 188/68/0 | 211/45/0 |
| Declared plant variation | 212/44/0 | 232/23/1 |
| All declared stress | 190/66/0 | 191/65/0 |

The nominal course gain survives each separate stress axis, but nearly
vanishes under their combination. Adding separate drops is not a valid model
of combined degradation. Noise and dropout improvements in this seeded
sample are not evidence that corruptions generally help navigation.

Short-open treatment success is 246/256 nominal and 239/256 under combined
stress. Ego state is ideal and wind is zero. This diagnostic identifies delay
and combined-stress failures for further causal analysis; it does not validate
hardware robustness. All panel/seed records remain in
`results/root-stress-matrix/`, with summaries in `root-summary.json`.

## Delayed-flight diagnosis and a rejected isolated repair

Six 400-step trace replays reproduced every scored result from the frozen
matrix. Among nominal successes lost to contact, both delays cause 37 new
contacts; all have some measured geometry below 1 m within the final 0.4 s.
In 14 cases the latest requested speed is at least 0.1 m/s below the speed
still being applied. Combined stress causes 57 new contacts: 55 have nearby
measured geometry, and 22 show that late-command speed gap. This measures
command history and nearby depth, not whether the contact object was visible
or whether sufficient braking room existed.

Source review found a timing omission in geometry memory. Its uncertainty
radius included history lookback age but not the age of the newest delayed
capture. The opt-in correction adds that capture age to the existing age term,
with no new coefficient. At 100 ms it adds 0.015 m to point uncertainty.
Host/Metal radius parity passed. Six nominal panels remained exactly identical.

Frozen corrected policies then completed ten matched 128-flight evaluations.
On combined courses, both-delay success changes 211→202/256 (seven wins,
16 losses); combined-stress success changes 191→197 (14 wins, eight losses).
Both-delay contacts rise 45→54; combined contacts fall 65→58 with one new
timeout. This is a real timing defect but its isolated correction is not a
navigation improvement across conditions. It stays disabled by default.

The [57-input archive](../evidence/inputs/stress-failure/records.tar.gz) includes
all six replay traces, grades, age-profile evaluations, source and receipts.
Recompute the paired diagnosis and nominal parity without Metal:

```sh
python3 navigation_stress_failure_review.py evidence/inputs/stress-failure/records.tar.gz
```

Trace source/binary hashes and provenance have different roles. The trace
receipts pin executed binary and data; the archived `age-source` is the later
age-correction snapshot, not a claim that it compiled the earlier trace binary.
The shared logic remains experimental. Training/export must bind its enabled
profile and shader contract explicitly before treating any weights as portable.
