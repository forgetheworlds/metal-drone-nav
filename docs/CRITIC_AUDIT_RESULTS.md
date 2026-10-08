# Critic and training audit

The numerical critic update passed the checks we ran. The critic receives much
less geometry and sensing context than the actor. That is a candidate training
limit, but this audit does not establish that it causes the policy failures.

The user requested an independent OpenCode MiMo audit after larger actors lost
useful skills during continued PPO training. MiMo inspected the frozen source
and collected trajectories from the ordinary-PPO control of the physical
retention experiment. Root reviewed its code, corrected errors in the analysis,
and added an independent check of the actual Metal value-loss kernel.

## What was checked

The actor has 184 inputs and 5,120 hidden units. The critic has 64 inputs and
64 hidden units, with 4,225 parameters. The actor's inputs include current and
previous depth and geometry guidance. The critic receives vehicle and controller
state, goal information, nearest clearance, and the centers of the first three
obstacles. It does not receive all obstacle shapes and motion, sensing history,
or queued commands.

The source bank contains 2,048 unique task payloads. In its rotated entries,
moving obstacles never occupy those first three obstacle slots. Constructed
world pairs can therefore have identical critic inputs and different depth.
This proves that some information is absent. It does not prove how much that
absence affects expected policy return or learning.

On a real, nonzero critic minibatch, root checked the actual Metal sample-loss
derivative, critic backpropagation, gradient reduction and clipping, and a fresh
Adam step against the CPU reference:

| Check | Maximum absolute difference |
|---|---:|
| Value-loss derivative versus its direct formula | 0 |
| Critic value prediction | 0.00000811 |
| Raw critic gradient | 0.00000155 |
| Scaled critic gradient | 0.0000000522 |
| Adam parameter update | 0.000000119 |

All checked tolerances passed. Gradients were nonzero on 4,097 of 4,225 critic
parameters and on every varying input dimension in that batch. This is one
batch with fresh Adam moments. It does not verify every inherited optimizer
state, establish adequate capacity, or prove useful value predictions.

There is no critic value clipping in this learner. The 0.2 clip parameter belongs
to the actor's probability-ratio objective. Motor inputs are already normalized.

## Prediction quality

The collector produced 1,015,808 main transitions through RAPTOR. With its
determinism and smoke checks, total diagnostic collection was 1,097,728
transitions. Six rows were overwritten by a collector header error and are
excluded. The original file is preserved.

Two measurements answer different questions:

| Reference for the critic prediction | Rows | RMSE | Explained variance |
|---|---:|---:|---:|
| Its 32-step GAE training targets | 1,015,802 | 3.349 | 0.847 |
| Observed discounted rewards to actual success or contact | 928,445 | 8.135 | 0.462 |
| Observed rewards with at least 32 future steps | 698,277 | 6.885 | 0.296 |

The GAE targets contain the critic's own bootstrap prediction. A good fit to
those targets is not independent proof of future prediction quality. The
observed rewards come from one stochastic policy and correlated trajectories;
they are also not noiseless conditional-value targets. Timeout and unfinished
episodes are excluded from the observed-return measurement. None of these
numbers is an information or capacity ceiling.

## Root's decision

No policy or training change is adopted from this audit. MiMo's proposed
count-and-nearest-center extension remains an untested feature hypothesis.
It does not supply full shape, motion, sensing or command history. A failed
version would not refute every missing-information explanation.

A useful next comparison must preserve the initial value function, use a
consistent feature builder for current and bootstrap states, match critic
parameter count and training exposure, and keep actor, source tasks, reward
and teacher fixed. Navigation success, contact, arrival time and broad skill
retention must decide the result. Target-fit improvements alone are insufficient.

Root also rejected claims that approximately stable clipping factors rule out
optimizer effects, that existing critic biases can be zeroed during a
function-preserving expansion, or that a changed critic has no indirect effect
on exploration and forgetting.

## Reproduce the evidence

From the repository root, with NumPy installed:

```sh
python3 navigation_critic_audit_results.py
```

The [replay](../navigation_critic_audit_results.py) checks the hashes of all 72
retained inputs in [the archive](../evidence/inputs/critic-audit/records.tar.gz),
then recomputes the GAE-target and observed-future-return measurements. It runs
offline and does not launch training or a simulator.

The archive also contains the raw collector, frozen control checkpoint and bank,
probe sources, frozen compiler inputs, root's loss-kernel source and receipt,
geometry correction, and root's review of MiMo's first report. The corrected report and correction ledger are also retained. The first
report includes rejected conclusions; `root-review/review.md` records the
corrections. This package records an audit, not a new navigation learning result.
