# Open-goal mean versus sampling

The frozen policy can reach seven open goals with sampled actions that it fails
to reach with its deployment mean. Its weights, task geometry and physics are
unchanged. This identifies a concrete behavior gap before another training run.

The test uses the prior physical-retention actor, seed 1, on all 128 nominal
open development tasks. The seven failures were selected mechanically from
the existing scored mean-policy CSV, before inspecting any sampled outcome.

| Frozen execution | Success | Contact | Timeout |
|---|---:|---:|---:|
| Deployment mean, mode17 | 121/128 | 0 | 7 |
| Training action map, mode22, sampler disabled | 121/128 | 0 | 7 |
| Learned Gaussian sampling, eight seeds | 1023/1024 | 1 | 0 |

All seven mean failures succeed in every sampled seed: **56/56** paired
rescues, with no contact on those seven tasks. The one sampled contact occurs
on a task the mean policy completes. All 1,280 episodes are retained.

## What was held fixed

The actor is the same 184-input, 5,120-hidden, four-output MLP. Neither its mean
weights nor its learned global standard deviations are updated. RAPTOR,
nominal physics, sensing, speed cap, 400 navigation ticks and the stable-arrival
contract are unchanged. Only the sampler changes.

The normal mean replay matches the original scored flights exactly. Mode22
with its sampler disabled matches every physical result field of mode17. This
rules out a different deterministic guidance map as the explanation for this
comparison. The original actor parameters are hashed before and after each run.

## Interpretation

On these tasks, random action perturbations can break behavior that the mean
policy does not resolve. High stochastic training success therefore does not
guarantee reliable mean-policy arrival. This is more specific than saying the
network needs more capacity or that PPO needs more samples.

This intervention does not prove how training created the gap or that a
particular variance schedule will fix it. The teacher's weak near-goal commands
and controller response remain possible interacting causes. Adding random
commands to deployment is not the proposed remedy: it introduces a new contact
even in the open-task diagnostic.

A next training experiment should make reliable mean arrival part of what is
learned while preserving broad behavior. If sampling variance becomes
state-dependent, sampling, stored likelihood, PPO ratios, entropy and gradients
must all use the same effective variance. Scaling sampled noise after the fact
while scoring the old Gaussian would invalidate that comparison. Near-goal
gradient size, functional policy drift and retention must be checked before
full learning. No coefficient or network sweep is justified by this result.

## Reproduce

```sh
python3 navigation_open_mean_results.py \
  --archive evidence/inputs/open-mean-diagnostic/records.tar.gz
```

The offline replay verifies hashes, all episode identities, stable arrivals,
mode parity, the predeclared failure set and every sampled outcome. The archive
retains the frozen actor, source bank, code, source inputs, execution receipts
and all scored CSVs. It launches no simulator and performs no learning.
