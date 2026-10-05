# Preserving useful behavior during PPO

Broader task coverage taught the long-route behavior, but keeping old tasks in
the bank did not preserve every old skill. I am testing the existing actor
parameter anchor before adding another training method.

Both arms use the same static-plus-long source bank, BC warmstart, 184/64/4
actor, controller-state critic, full goal distance and original observation and
action paths. Both use time cost 1, contact penalty 50 and arrival bonus 10.
The only learning intervention is an anchor coefficient of 0 versus 0.01.
The anchor adds `lambda * (weights - original_reference)` to the actor gradient
before clipping. Its radius is zero, and it includes exploration parameters.
It is a parameter penalty, not a behavioral guarantee or distillation loss.

The [runner](../navigation_critic_training.mm) now exposes `--anchor` and
`--anchor-radius`. It saves the original launch reference in the existing
hash-bound safeguard format. Resume restores that reference rather than
anchoring to the resumed policy. A coefficient mismatch or corrupt reference
is rejected before checkpoint writes.

Read-only diagnostics record PPO-gradient norm, anchor-gradient norm and the
combined norm separately. Per-slot counters record actual training transitions.
In the first control run, the four long slots account for about 59% of observed
transitions despite being one-third of the episode cycle. This is an early
execution measurement, not a final learning result.

The default path reproduced an entire 100-rollout checkpoint byte for byte.
Active anchoring produced identical full checkpoints for six uninterrupted
rollouts and three plus three resumed rollouts. The original references also
matched. Per-slot counts sum to exactly 4,096 transitions per rollout.

Four fresh matched runs use two seeds and 10,000 rollouts per arm. Final
checkpoints are primary. The same old dev-a selector is retained to keep the
anchor comparison separate from a selection change. Evaluation covers both
long panels, all five static retention panels and the new composite diagnostic.

Gates require at least 253/256 successes on each long panel and short open
flight, retention within three successes of matched control on each static
panel, and recovery to at least 223/256 on static B and 219/256 on clutter.
Contacts and timeouts must each stay within three of control, with no more
than 0.5 s added on common successful flights. Individual seed results remain
visible. The stronger static floors prevent declaring success merely because
the policy retains a weaker BC reference.

The runs are active. If the anchor blocks new learning or only restores BC
behavior, I will keep the result and move to consolidation from successful
source trajectories of the static, long and moving policies. There will be no
automatic coefficient sweep or promotion from a single improved panel.
