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
are pending; no policy asset has been replaced.
