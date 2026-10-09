# Critic geometry experiment

This experiment tests whether a critic with more complete world geometry helps
PPO preserve and improve the combined navigation policy. The actor still uses
depth, ego state and a geometric goal. It receives no new world information.

The [audit](CRITIC_AUDIT_RESULTS.md) found that the original critic receives only
three obstacle centers, without object shape or motion parameters. That is a
structural gap; it does not establish the cause of the policy failures.

| Component | Control | Treatment |
|---|---|---|
| Actor | 184 inputs, 5,120 hidden units | Same |
| Critic | 226 inputs, 64 hidden units | Same |
| Critic parameters | 14,593 | Same |
| Original critic inputs | Original 64 | Same |
| Additional inputs | 162 zeros | Count, elapsed time, geometry and motion for all 16 objects |
| PPO, teacher, reward and source tasks | Preserved physical-retention recipe | Same |

The additional features are training-only. The critic and its extra information
are absent from deployed navigation inference. Both critics start with the
existing value weights, biases and output head. New input weights start at zero.
The control's extra weights remain inactive; equal nominal parameter count does
not separate information from every possible change in function class.

The [shared feature builder](../navigation_critic_geometry.hpp) uses stored
object centers, size fields and velocity parameters. For bounded moving spheres,
the third size field is phase in radians. It supplies the motion parameters and
actual time rather than calling peak velocity instantaneous velocity. The same
builder is used for current and next-state value predictions. Sensor history and
pending commands remain outside this addition.

## Verification before training

Root executed these gates:

- CPU and Metal feature values match exactly. Actor observations and the original
  64 critic inputs have identical hashes between arms.
- New critic input gradients are nonzero; CPU/Metal raw gradient difference is
  at most 0.000000254 on the checked batch.
- Initial value weights, biases and output head are preserved. A strict CPU
  fixture initially differed by 0.00000787 under FMA contraction. Disabling
  contraction in the reference test restored exact equality. Training compiler
  flags remain unchanged, and the original failed fixture is retained.
- With nonzero extra weights, bootstrap predictions agree with the post-step
  value check within 0.0000129. This prevents zero initialization from concealing
  a missing next-state feature path.
- Both seeds, both arms: six rollouts versus three plus three resumed rollouts
  produce identical full checkpoint bytes. Wrong feature and width contracts
  are refused without overwriting saved weights.
- The actor is byte-identical after the first complete PPO rollout. The existing
  inference frontend then produces identical scored CSVs on 128 paired tasks.

These are engineering checks, not navigation improvement results.

## Learning comparison

Four runs use 512 environments and 256 rollouts each: 16.78 million transitions
in total. Both arms use the two original composed parents, fresh Adam, the same
2,048 unique source tasks, half clean and half 100 ms sensor/command delay lanes,
physical-action teacher coefficient 4.134632354, actor step factor 0.4, and speed
cap 1.5 m/s. Each final model receives the same 23 development/stress panels.
There is no target-side training or blind FINAL evaluation.

The decision is frozen before those outcomes. It requires improved static-C
success and contacts, reliable old/fresh open and long arrivals, retained course
success, and bounded success/contact/timeout losses across every panel. Shared
successful arrival times must remain within the declared limits. Training and
evaluation costs are reported separately. Target-fit metrics cannot promote a
policy. A failed recipe will be retained rather than followed by a feature sweep.

The frozen source, gates, decision and actual parent-owned jobs are in
`results/root-critic-geometry`. Default assets are unchanged. All four producers and 92 evaluations are complete. The treatment fails eight
checks; see [results](CRITIC_GEOMETRY_RESULTS.md).
