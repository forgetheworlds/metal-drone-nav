| command group | logs | reset probe PASS | observation probe PASS | observation probe skipped | evaluation rows | errors |
|---|---|---|---|---|---|---|
| local-eval, delay sweep | 18 | 18 | 0 | 18 | 18 | 0 |
| local-eval, legacy camera geometry | 90 | 90 | 90 | 0 | 90 | 0 |
| local-eval, native camera geometry | 12 | 12 | 12 | 0 | 12 | 0 |
| local-test (production parity gates + probes on all five banks) | 2 | 2 | 2 | 0 | 0 | 0 |
| local-trace | 6 | 6 | 6 | 0 | 0 | 0 |

Training arms (one probe per training start and one per periodic DEV evaluation):

| arm | command | reset probe PASS lines | observation probe PASS lines | source |
|---|---|---|---|---|
| candidate-local | local-train (this runner) | 41 | 41 | `runs/candidate-local.log` |
| candidate-fast | local-train (this runner) | 41 | 41 | `runs/candidate-fast.log` |
| control-source | train-tasks (unchanged production command) | 0 | 0 | `runs/control-source.log` |

Resume checks: file = checks/split-part2.log, resume_contract_pass = 1, resume_bank_file_pass = 1, reset_pass_lines = 2, observation_pass_lines = 2; negative cases refused before load = 13, refused after running a probe = 0 (the second number must stay 0).

`local-bank` and `local-contract` are host-only and run no simulation, so they produce no probe line and make no success claim. The control arm is produced by the unchanged production `train-tasks` command, which has its own validation (`navigation_training::validate_rollout`) and does not run this runner's probes; its behaviour on the same banks is measured only by the `local-eval` rows above.
