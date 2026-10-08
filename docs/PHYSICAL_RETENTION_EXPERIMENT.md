# Preserving translation while PPO learns

The composed actor recovered useful XYZ navigation and kept the later turn
behavior. The next experiment adds a training-only loss on physical XYZ velocity
means. A frozen copy of the composed parent receives the same observations as
the student. Both means pass through geometry guidance, its gate, tanh and the
spherical speed cap before comparison. The teacher is absent at inference.

PPO, the critic, tasks, reward, sensors and dynamics stay fixed. The actor has
5,120 hidden units and the critic has 64. Both control and treatment start from
the same composed weights, inherit their exploration parameters and use fresh
Adam. Shared actor features can still change yaw and exploration; the evaluation
must check those effects.

Root review found the delegated delivery had prepared gate procedures rather
than completed them. Its coefficient 0.05 request had no calibration receipt and
was rejected. Root made the derivative probe nonzero through a real ordinary PPO
update, fixed an impossible descent assertion and an aliased batch-scaling
check, and bound the new loss implementation into enabled resume sidecars.
Original delivery and failed gate logs are retained.

Actual checks now show:

- Metal derivative relative L2 error 3.6e-7 against float64, with nonzero
  derivatives through both hidden halves and no direct yaw-head derivative.
- Exact enabled-loss checkpoint parity for six rollouts versus three plus three
  on both seeds; teacher and anchor references are preserved.
- Nine mismatched resume cases reject before checkpoint writes. Coefficient-zero
  resume without a teacher sidecar also passes.

The TRAIN-only calibration uses the first nonzero discrepancy after one ordinary
PPO rollout. PPO gradient norm is 7.1121 and unit retention-gradient norm is
0.4300. The registered quarter-norm rule gives coefficient 4.134632354 for both
seeds. No development outcomes chose that value. Later gradient ratios can differ.

Full 100-rollout default checkpoint parity now passes on both seeds. The root
parent accepted the calibrated, hash-bound request and started the actual pilot:
control and treatment on two seeds, 4.19 million transitions each. It then
evaluates the parents and learners on 138 source panels.
Course success, contact counts, static/open/long retention, stress and shared
arrival time decide whether to keep it. There is no selected-best substitution.

Current commands, immutable source hashes, processes and raw gates are in
`results/omp-functional-retention/`. This is an implementation and controlled
experiment; a navigation improvement from the new loss has not yet been measured.
