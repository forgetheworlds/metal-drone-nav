# Actor capacity experiment

The next capacity comparison keeps the mixed PPO recipe fixed and widens only
the actor. Target hidden widths are 64, 256, 768, 2,560 and 5,120: about 12k,
50k, 150k, 500k and 1M actor parameters. The 64-input, 64-unit critic stays fixed.

`navigation_capacity_checkpoint.py` prepares warm-start-only parameter files.
Existing neurons retain their weights. Added neurons receive independent input
weights and zero output weights, preserving initial predictions while allowing
new features to learn. Output biases, action standard deviations and critic
bytes remain unchanged. Sixteen CPU probes match exactly at all five widths.
Metal parity and actual training remain to be verified.

The current kernels share actor and critic width and contain a fixed 64-unit
SIMD path. Those limits must be corrected before larger models can train safely.
The implementation runs in the isolated `navigation-actor-capacity` worktree;
competence sampling remains a separate experiment in the main checkout.

The delegated OMP mission first completes matched 64-versus-256 training on both
seeds, then extends the other widths where practical. Each arm targets 41,943,040
transitions with the same corpus, reward, PPO settings, delay rehearsal and warm
policy. Final held-out capability, retention and useful progress per hour decide
whether capacity helps. Larger teachers do not establish ESP32-S3 deployment;
compression and retained onboard behavior require later measurements.

The complete mission, runtime status, raw experiments and report live under
`results/omp-capacity-scale/`. No policy is promoted before root reviews the code
and navigation outcomes.
