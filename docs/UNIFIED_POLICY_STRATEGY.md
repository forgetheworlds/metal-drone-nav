# One reusable navigation policy

The strongest next step is to preserve useful behavior while learning from a cumulative task bank. Keep the current actor, geometry guidance, short geometry memory and RAPTOR controller. The evidence does not yet justify a larger network or a new RL algorithm. No existing checkpoint meets the full combined objective.

This review used the current owning code and the final and selected-best flight CSVs in `results/root-distance-learning/evals/`. `STATUS.md` and `LONG_GOAL_EXPERIMENT.md` still describe active runs; the actual completed records supersede that execution status. No policy is promoted by this review.

## What the failures establish

The long-task experiment establishes a large coverage effect with the same actor and critic. At final checkpoints, pooled over two seeds, adding four long slots changes open success from 2 to 255 of 256 and hallway success from 0 to 256. It also raises static B contacts from 24 to 39 and clutter contacts from 28 to 37. B success falls 226 to 215; clutter falls 222 to 216. These fail the frozen retention gates. This is not merely an arrival-speed issue or a failure to learn the new tasks.

The old eight records remain byte-identical, but their training influence does not. `waypoint_task_apply` indexes `environment * period + episodes % period`. Each environment cycles episodes, not equal numbers of transitions. Longer flights occupy more optimizer samples. Early collision ends an episode; a stalled flight consumes the full budget. Global advantage normalization in `ppo.hpp` then couples those samples. Thus cumulative records are necessary but do not guarantee balanced rehearsal or retention. The exact amount of sample displacement must be measured, not inferred from slot counts.

The selected-best results do not repair the tradeoff. Long treatment reaches 255/256 on each long panel, but old C falls 214 to 208 and clutter 220 to 216 against selected control. `navigation_critic_training.mm` selects on dev-a success, then contacts, then time. That rule has no long, moving or cross-panel retention condition. A broad selector is necessary for a reusable policy, but selection alone has not been shown to produce one.

The controller-state critic repairs a real value-input alias. Static final dev-r2 improves 209 to 233, yet open success misses its floor at 251/256. In the mixed moving trainer, the rich critic improves fresh moving success 232 to 241 while worsening static avoidance. Better value estimates can improve the behavior favored by the current samples and reward without preserving another capability. This is evidence against treating the critic as the whole retention solution.

Strong contact cost can favor waiting; arrival reward can restore motion while increasing contacts. These are observed tradeoffs. They do not establish that another scalar reward sweep will recover all skills. The timeout implementation bootstraps value but cuts the GAE chain; contact and success terminate without bootstrap. Keep this distinction when diagnosing long stalled episodes. Do not describe all timeouts as terminal collision equivalents.

The best supported explanation is coverage plus optimization interference and an incomplete selection objective. Representation effects also exist: the distance-input cap changes fixed-weight behavior, but does not reliably solve it. Insufficient network capacity, mandatory new memory and a broken PPO algorithm remain unproven hypotheses.

## Starting points and knowledge to preserve

Use `results/omp-local-capability/runs/bc.bin` as the next controlled warmstart. It preserves short open flight and already reaches 128/128 long open and 110/128 fresh long hallway goals. It is not a random policy trained only by cloning: its provenance is fast-PPO checkpoint 7a followed by 1,000 cloning updates on successful source flights from the fast policy plus the brake/turn rule.

BC is not a complete static teacher. In the new retention CSVs it reaches A102, B105, C106, open128 and clutter104, each of128. Long-control final policies have stronger pooled B and clutter scores than duplicated BC. An anchor to BC can therefore pass a BC-retention test while failing to preserve the strongest behavior already learned.

Preserve three complementary sources of behavior: the static long-control checkpoints, the long-treatment checkpoints, and the previously transferred moving specialist. Keep the moving specialist's reported native challenge30/32 versus fast13/32 as evidence of a useful skill, with its static failures attached. Neither rich joint training nor long training has demonstrated that moving transfer survives. Do not choose one seed retrospectively and call it the universal starting policy; keep both seed results and their provenance.

Do not average their weights. If explicit combination becomes necessary, collect successful TRAIN trajectories from each teacher under the same deployable observation/action contract, balance them by capability, and train one student. Teacher choice can use training-task metadata; the deployed student must receive no teacher ID, family label or oracle route. Conflicting teacher actions at similar observations need inspection rather than averaging. Use learner-state teacher queries only after checking that the teacher can recover from those states. Successful teacher trajectories alone do not prove safe labels on arbitrary student states.

## Smallest controlled next experiment

First test the existing parameter anchor on the already understood static-plus-long distribution. Do not add moving tasks in the same causal comparison.

Both arms should use the exact 12-slot long treatment bank, BC warmstart, rich critic, actor184/64/4, reward1/50/10, current PPO settings and unchanged observation/action path. Use two fresh matched seeds and10,000 rollouts per run: four runs,320,000 optimizer steps each. Control anchor is zero. Treatment uses one frozen coefficient, proposed lambda0.01 and radius0. This is an explicit mechanism test, not an empirically optimal coefficient. Do not run an automatic sweep if it fails.

`PPOTrainer` already stores a reference actor and adds `lambda * (weights-reference)` before actor gradient clipping. A radius enables a parameter-distance hinge, refreshed once per rollout. This is parameter anchoring, not behavioral KL, replay or distillation. It currently covers all actor parameters, including log standard deviation. With radius zero the force is exact for its declared quadratic penalty. An anchor can reduce exploration as well as mean-action drift, and can resist needed improvements.

`local-train` recaptures its warmstart but does not enable the anchor. Expose the existing implementation through that runner with default zero. Reuse the existing safeguard contract, saved `.anchorref` and hash checks from `train_challenge_bank`; resume must restore the original reference, not anchor to the resumed weights. Require default-off full-checkpoint parity and an interrupted/resumed active-anchor equivalence check. Log separate pre-anchor PPO norm, anchor norm and combined clipping; existing combined norms cannot identify the cause of a smaller update. Record actual transitions and completed outcomes per capability.

Final checkpoints remain primary. Freeze all thresholds before launch. Require both long panels at least253/256, short open at least253/256, and each old static panel no more than three successes below its matched unanchored control. Require B and clutter to recover to within three of the completed original long-control final scores: at least223 and219. Require contacts and timeouts separately no worse than matched control plus three per panel, and no more than0.5s slower on common successful flights. Report paired wins/losses and each seed; aggregate gains must not hide one seed's collapse. A failure means no promotion at this budget.

For secondary checkpoint selection, construct a new dedicated selector with static, long and moving strata. Use hard retention floors first, then worst-stratum success and contact counts, then common-success time. Apply the same rule to both arms. Keep current nonselector panels as exposed development evidence; do not silently turn them into a fresh sealed test. Final-checkpoint inference avoids crediting a change to selection as an anchor effect.

## Path to the combined outcome

Build the cumulative challenge bank now, separately from the fixed anchor comparison. Include longer staggered gaps, offset openings, over/under choices, poles with overhangs, and observable crossing/approaching obstacles. Compose these capabilities within episodes, not only in separate family slots. Vary dimensions, goal distance, start velocity/yaw, threat speed and irrelevant geometry. Keep long free flight, arrival/braking and existing short static tasks as rehearsal. Geometric witnesses establish clearance; retain separate actual controller feasibility checks and warning-time labels. Outside-view goals are valid, and neither FOV nor short-goal distance is a universal acceptance gate. Keep noise/delay and plant changes as independently recorded axes so complex geometry does not hide their effect.

If anchoring preserves static gains while retaining long competence, carry the tested mechanism into that cumulative static+long+moving bank. Keep fixed nonzero coverage and measure transition shares. Test dynamic addition against the frozen successful static+long parent. Do not assume its present high contact penalty transfers unchanged to moving encounters.

If the anchor mainly returns behavior toward BC or blocks learning, the next justified step is behavioral consolidation from successful static, long and moving source flights, followed by balanced on-policy PPO and learner-state corrections. This targets behavior directly; a parameter anchor cannot encode three different teachers. A train-only label-fit check followed by closed-loop development flights can test whether the existing student can represent the combined behavior before any capacity change.

After a candidate meets the combined source gates, freeze weights, geometry logic, sensing, action mapping and selection. Test new independent transfer geometry, then separate latency/noise, dynamics/disturbance and complex tight/long-route axes. Existing inspected Webots challenges are development evidence now. No source-bank result establishes unfamiliar physical-flight reliability or completes the goal.
