# What combined-task training improved

Four matched source runs each completed10,000 rollouts/320,000 optimizer steps,
followed by64 evaluations. The new distribution reserves50% of transitions for
static tasks,25% for long goals and25% for valid bounded-motion courses. The
control reserves50% static and50% long. Network, reward, physics and anchor
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

Each cell pools two trained policies on128 task instances. New courses improve
with25 paired wins and5 losses. Static obstacle sets also improve, but free-space
arrival loses reliability. The combined navigator is not promoted.

Separate factor probes vary length5.5–12.5m, reverse travel, swap floor/ceiling
layouts, and combine those changes. Combined models reach117/118,115/120,
110/115 and107/113 successes per128 respectively. Their frozen parents reach
119/108,113/107,100/98 and96/85. These are development recombinations, not a
sealed final set; they were neither trained nor used for checkpoint selection.

An independent96-flight Webots benchmark compares both frozen parents with
both combined models on the first24 bounded-course records, in file order.
Parent success is23/24 and20/24; combined success23/24 and22/24. Contacts fall
from5 to3 pooled over48 trials; all runs are valid, and no target training occurs.
Seed1's common successful flights are0.233s faster; seed2's0.018s faster.
This is real full-stack development transfer, not physical flight or completion.

## Diagnosing the remaining arrival failure

The open-room timeouts keep moving and accumulate9–20m paths. A15s late
reference controller recovers20 of21 short-open failures but introduces one
contact; it recovers17 of30 long-open failures. It is an intervention diagnostic,
not the deployed policy. Frozen BC recovery from the same learner states helps
only modestly. These late rollouts cannot be treated as ordinary BC states.

A full400-tick capture also checked reference-error observability. In the last
five seconds of the failed second-seed flights, no actor reference component
was clipped. Median tracking error is about0.070m for short-open and0.079m for
long-open; maximum long-open error0.162m. Reference windup does not explain
these failures. I did not add an anti-windup fix.

## Consolidating behavior into one actor

BC completed1,482/1,536 new source arrival tasks with varied heights, directions,
distances and initial motion. Successful flights yield243,058 observation/action
pairs;3 contacts and51 timeouts remain recorded. Separate successful static,
hallway and bounded-course flights come from the stronger combined policy.
One student uses equal capability groups, then equal successful episodes, then
frames. Group IDs never enter the184-input actor.

Two students completed2,000 supervised updates and48 closed-loop evaluations.
At the1,000-update midpoint, all three open/hallway panels reach256/256. Final
2,000-update counts are short open251,long open255,hallway256, staticA203,
B209,C215,clutter214,composite238. Neither stage passes full retention. Lower
imitation loss is not claimed as navigation progress by itself.

The next matched PPO comparison uses the prescribed final2,000-update student
versus its unchanged combined parent, with the same balanced source bank and
PPO/anchor/reward settings. Both are single actors. No deployed teacher switch,
network-size change or reward coefficient sweep is introduced. All earlier
checkpoints, failures and native recordings remain preserved.
