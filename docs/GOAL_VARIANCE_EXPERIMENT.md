# Goal-dependent training variance

The frozen-policy diagnosis found seven open goals that sampled actions reach
and the deployment mean does not. This experiment tests lower training variance
near the goal while keeping the mean actor, reward, teacher and task bank fixed.
It is a hypothesis, not a new successful policy or a deployment noise fallback.

The fixed scale is 0.25 inside the 0.35 m hold region, rises smoothly to 1.0 by
three goal radii, and stays 1.0 farther away. The scale uses the supplied goal
distance, not visibility or an obstacle oracle. The mean inference map is unchanged.

The effective distribution is Gaussian with
`log_std_effective = log_std_global + log(scale(observation))`. Both the sampler
and stored old likelihood use it. The current PPO likelihood, ratio, entropy,
mean derivative and global-log-standard-deviation derivative use the same value.
The scale is fixed with respect to actor parameters. No action is shifted using
a parameter-dependent mean after it has been sampled and scored.

Source changes are isolated under `results/root-goal-variance`; the old engine
and original experiments remain intact. The source contract binds the compiled
profile, so a saved run cannot silently resume under a different distribution.
The actor remains 184/5120/4 with four global standard-deviation parameters.

## Checks before learning

The first nonzero CPU/Metal loss check covers near and far observations, positive
and negative advantages, and clipped and unclipped ratios. Maximum gradient
difference is 0.000000418; loss/ratio/entropy difference 0.000000477. Central
finite differences agree within 0.000213. Both near and far gradients are nonzero.

Default-off full100-rollout parity against the preserved engine, enabled6 versus
3+3 full resume, and wrong-distribution refusal are executing. A separate bounded
real-update probe checks actual sampled likelihoods and effective-distribution
KL on near/far states with the original teacher loss. Those results must be read
before a full pilot. No full training comparison has been launched.

Lower variance can increase mean-gradient size and functional drift, reduce
useful exploration, or preserve the teacher's weak approach behavior. Matching
a formula is not proof that the policy improves. Broad success, contacts,
timeouts, common-success arrival speed and stress retention will remain the
outcome checks if the engineering and update-size gates justify learning.
