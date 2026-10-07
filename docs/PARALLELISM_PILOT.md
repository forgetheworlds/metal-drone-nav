# PPO parallelism pilot

The first scaling pilot compares four practical collection/batch regimes with
one frozen warm policy and the same 4,194,304 samples and 32,768 Adam updates.
Network, corpus, reward, PPO settings and delay mixture are fixed. Larger N
also changes policy refresh and data age; this is not a pure hardware result.

| Environments | Policy refreshes | End-to-end samples/s | Peak Metal buffers | Peak process RSS |
|---:|---:|---:|---:|---:|
| 128 | 1,024 | 98,209 | 14.6 MB | 33.0 MB |
| 512 | 256 | 166,755 | 50.2 MB | 78.9 MB |
| 2,048 | 64 | 198,460 | 192.7 MB | 248.1 MB |
| 8,192 | 16 | 180,069 | 762.6 MB | 918.4 MB |

These wall times include startup, eight fixed development checks and checkpoint
I/O. Device utilization includes other laptop work. The actor is unchanged at
184/64/4, with a 64-input critic. Phase-balanced starts spread the same 2,048
unique records; they do not create new world diversity.

| Final success / 128 | Warm | N128 | N512 | N2048 | N8192 |
|---|---:|---:|---:|---:|---:|
| Short open | 123 | 126 | 128 | 121 | 125 |
| Static C | 100 | 108 | 108 | 97 | 110 |
| Clutter | 107 | 110 | 114 | 104 | 112 |
| Long open | 128 | 124 | 128 | 102 | 127 |
| Long hallway | 128 | 128 | 128 | 128 | 128 |
| Combined course | 123 | 100 | 119 | 122 | 120 |
| Both 100 ms delays | 117 | 97 | 116 | 115 | 119 |
| Combined stress | 111 | 94 | 107 | 109 | 112 |

All 60 warm/final evaluation panels completed. N2048 is fastest but loses long
arrival; N128 loses course skills. N512 restores open/long completion and helps
clutter. N8192 better retains difficult courses and delay performance. Neither
is promoted from this single seed. The follow-up compares these two regimes
on both source parents with 41,943,040 samples each, all other axes fixed.

The longer comparison records exact rotated-entry exposure and cheap
inference-only model snapshots. Those snapshots cannot resume training or prove
budgets; full checkpoint headers and actual process exits do. The follow-up
scores all skills at equal sample checkpoints and reports grading cost separately.

Raw pilot records, telemetry and the fixed decision are under
`results/training-scale/`. [Astra's review](TRAINING_SCALE_REVIEW.md) explains
what the comparison can establish and which capacity controls are still missing.
