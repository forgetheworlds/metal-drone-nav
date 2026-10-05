# Controller state in the PPO critic

The navigation actor does not need to command motors. RAPTOR does that work.
The training value function does need to predict what that controller will do
next. The old 32-input critic does not see motor RPM, RAPTOR's hidden state,
the yaw-reference error, the previous navigation action or held velocity.

I constructed two states with the same body pose, motion and world but different
motor/controller state. On Metal, their original 32 critic features and all
184 actor input bytes were identical. The new controller features differed by
0.717356. The old critic cannot distinguish that pair, although their next
physical response can differ. That establishes an input alias, not its effect
on navigation performance.

The experiment gives both arms a 64-input, 64-hidden-unit critic. The control's
extra inputs are zero; treatment receives controller state and explicit motion
features. The navigation actor remains unchanged: 184 inputs, 64 hidden units
and four outputs. None of the privileged critic features enter deployment.

The extra features contain RAPTOR hidden state, motor RPM, sine/cosine of yaw
reference error, previous navigation action, desired world velocity, radial
progress rate, goal distance and actual speed. These are populated for both
the current value and post-step bootstrap value. The default 32-input build
remains the existing path.

Old value weights are explicitly mapped into the first 32 columns. Extra
weights start at zero in both arms, preserving the initial value function.
This is a parameter warmstart; full checkpoint resume still requires the
matching architecture and saved training contract.

## Comparison

Four fresh source runs compare control/treatment across seeds 20261014 and
20261015, 5,000 rollouts each. They start from the same BC checkpoint and use
the same actor, source bank, PPO, physics, time cost 1, arrival bonus 10 and
collision penalty 50. No reward coefficient changes between arms.

The final checkpoint is primary; dev-a selection and selected checkpoints are
reported separately. Nonselector dev-r2 must gain at least five successes and
remove at least eight timeouts over the two seeds, with contacts no more than
control plus three. Old task success must stay within three of control, open
success must be at least 253/256, and successful-arrival time must not increase
more than 0.5 s. A missed gate means no adoption at this budget.

Pre-clip actor and critic gradient norms are recorded. More input information
can increase gradient magnitude, so clipping is part of the explanation to
check if the result fails.

## Checks before training

The executed feature test kept actor bytes and basic critic features identical
while distinguishing the extra state. Both one-rollout smokes saved real
headers with one rollout, 32 optimizer steps, 12,104 actor parameters and
4,225 critic parameters. The default 32-input build produced a byte-identical
100-rollout checkpoint to the preserved baseline, including simulator and
optimizer state. Gradient instrumentation only reads the pre-clip gradients.

Build `navigation_critic_training.mm` with actor dimension 184 and critic
dimension 64. `NAV_CRITIC_CONTROL_STATE=0` selects the control and `=1` the
treatment. The default build retains 32 critic inputs. Full comparison results
are below; no policy asset has been replaced.

## Full run results

All four arms completed 5,000 rollouts and 160,000 optimizer steps. The
48 evaluations completed after two launcher corrections: unsupported reward
flags were removed, and dev-c was supplied through its original frozen bank.
The three completed evaluations were retained when the second correction was
made. Training, checkpoint selection and grading did not change.

Final checkpoints, pooled over two seeds:

| Tasks | Control success / contact / timeout | Controller-state critic |
|---|---:|---:|
| Nonselector dev-r2 | 209 / 19 / 28 | 233 / 14 / 9 |
| Selector dev-a | 198 / 32 / 26 | 208 / 40 / 8 |
| Existing dev-b | 201 / 28 / 27 | 218 / 30 / 8 |
| Existing dev-c | 202 / 34 / 20 | 214 / 35 / 7 |
| Open | 207 / 0 / 49 | 251 / 0 / 5 |
| Clutter | 201 / 27 / 28 | 217 / 26 / 13 |

Each row contains 256 flights per arm. On dev-r2, successful-arrival mean
fell from 4.76 s to 4.61 s. The primary success, timeout, contact and speed
gates passed, as did relative retention. The absolute open-space floor did
not: 251/256 is below 253. **Not adopted at this budget.**

The dev-a-selected checkpoints tell a different story. Both arms reach
225/256 on dev-r2; control reaches 256/256 open tasks and treatment 255/256.
The final-checkpoint improvement therefore does not establish superiority
over the best selected control on every task. Selection and training stability
remain part of the problem.

[Raw records and source snapshot](../evidence/inputs/critic-state/records.tar.gz)
contain 80 hashed files, including all flight CSVs, histories, gradient
diagnostics, actual budget receipts, code, checks and failed launcher attempts.
The [computed gate review](../evidence/inputs/critic-state/review.json) retains
the failed open-space gate. The next diagnosis concerns those five remaining
open-space timeouts and retaining the strongest behaviors across task classes.
