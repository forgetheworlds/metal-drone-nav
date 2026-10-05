# Learning delayed navigation while retaining other skills

The earlier combined trainer used zero sensor and command delay. Frozen stress
flights showed a substantial delay failure, while a capture-age correction
alone improved combined stress and worsened the two-delay condition. The next
comparison tests whether explicit delay exposure teaches useful anticipation.

Both arms train one actor on the same static, long and bounded-moving bank.
Control has no delay. Treatment has 100 ms sensor and 100 ms command delay in
half the environments, with clean rehearsal in the other half. Every stratum
retains 50% static, 25% long and 25% combined-course transitions. Four-environment
blocks alternate, so delay assignment cannot accidentally become task-family
assignment. Capability IDs never enter the actor.

The network, critic, reward, anchor, sensor projection, navigation rate, plant
and RAPTOR are unchanged. The rejected capture-age profile stays off. Each arm
starts from the same corresponding seed's consolidated-PPO final actor and
critic, with fresh optimizer state and exploration standard deviation.

## Implementation and checks

`navigation_critic_training.mm` now accepts `--sensor-delay` and
`--command-delay`. Compile the explicit experiment with `NAV_DELAY_REHEARSAL=1`.
The host and runtime Metal compiler receive the same flag. Sidecar version 3
pins both delays and the clean rehearsal flag; mismatches stop before writes.
The checkpoint/network ABI is unchanged. Previous version 2 contracts remain
historical and require their preserved runner to resume.

Preflight completed these measured checks:

- A fresh 100-rollout nominal run produces a byte-identical full checkpoint
  to the preserved motion trainer.
- Eight delayed rollouts uninterrupted match four plus four resumed, including
  optimizer and simulator state.
- A resume with the wrong delay is rejected without changing the checkpoint.
- Each 32-step rollout contains 2,048 clean-lane and 2,048 delayed-lane rows.
  Clean rows have zero depth age; actual delayed rows reach 100 ms.
- Each clean/delayed capability stratum contains 32 static, 16 long and
  16 moving-course environments.

## Full comparison

Four 10,000-rollout runs use two matched seeds, 20261180/81, with 320,000
optimizer steps each. Final checkpoints are primary. Development selection
on dev-a is secondary. All eight nominal panels and four composite stress
profiles test both seeds and arms. No target-side or final-set training occurs.

The delay gate requires at least ten more pooled successes, ten fewer contacts
and a success gain in both seeds. Old capability success can fall by at most
three per panel; contacts and timeouts have separate bounds. Common successful
arrival can slow by at most 0.5 s. Absolute open/hall/static floors still apply
to adoption. A mechanism gain cannot excuse losing another navigation skill.

The fixed decision, preflight proof, real process receipts, source freeze and
checkpoints are in `results/root-delay-learning/`. Run counts on resume are
incremental; inspect the saved header. Training is active until all four headers
and the full evaluation receipts exist. No completed learning result is claimed
by the preflight.
