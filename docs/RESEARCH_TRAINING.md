# Research: next training experiments (post-peak capability collapse) — revision 2

Date: 2026-10-02. Scope: delegated mission `results/mimo-delegation/next-training-research.md` — explain why the
Metal+RAPTOR PPO pipeline reaches useful capability then collapses, and rank concrete next experiments using
primary sources plus actual local failures. This file plus `results/mimo-training-research/**` are the only
paths written. No training run, no eval/benchmark invocation, no git, no engine/source change; every number
below comes from artifacts already on disk.

Line references pin `main.mm` to its working-tree state at 2026-10-02 (the file grew 1598 → 1617+ lines while
the sibling learning mission added `--epochs`, `--adv-clip`, `--value-coef`, `--anchor`).

## Revision notes (what parent review corrected)

1. **Citation attribution.** arXiv:2405.00662 is Moalla, Miele, Pyatko, Pascanu, Gulcehre — not Curi/Berariu.
2. **Three controls are not one.** PPO likelihood-ratio clipping, gradient-norm clipping, and KL early stopping
   are distinct mechanisms; norm clipping is *not* "the only trust region", and saturation does *not* mean it
   does nothing (§2).
3. **KL early-stop heuristic** is cited to OpenAI Spinning Up's PPO page, not attributed to the 2017 paper.
4. **Gradient stats recomputed.** Correct metric is the fraction `actor_scale < 1` (raw norm > 0.5), and a mean
   norm must not be obtained by inverting the mean scale (§4.3).
5. **Weight norms/cosine do not measure feature rank** and therefore do not rule out representation collapse
   (§4.4, §3).
6. **"Per-step reward flat ⇒ objective flat" withdrawn** and reconciled against the probe return 10.68 → 7.88;
   directionality is labelled **unresolved** (§4.2).
7. **Stale recommendation removed.** The learning mission is terminal (7999/8000 rollouts, explicit tested
   arms); `--epochs` / `--adv-clip` must not be re-proposed — they are tested-and-refuted/neutral, and the
   flags are uncommitted working-tree additions (§4.5, §5).
8. **Hypotheses are unproved candidates**; no experiment in this mission isolates an independent cause (§6).

## 1. Lead finding

The post-peak decline is a **real behavioral regression of the policy under continued on-policy optimization** —
not evaluation noise and not train-split memorization. From the identical peak checkpoint
(`results/rooms-focused.bin.best`, sha `7b14b5de…`, dev 54/90), 300 further rollouts with unchanged settings
(`results/mimo-learning-next/ctrl-r52.*`) leave dev at **0.100 (9/90) with zero newly solved levels** — 45 lost,
0 gained; collisions 36 → 76; mean path 6.64 → 8.12 m; mean progress 0.680 → 0.520.

Two earlier claims are corrected:

- **The training objective does degrade with dev.** In-training `ret_mean` falls 8.30 (r301–400) → 5.93
  (r401–500) → 4.59 (r501–600), `rew_mean` 0.184 → 0.155 → 0.129, `ep_success` 0.565 → 0.362 → 0.283; a
  pre-update probe run gives return 10.68 (dev 0.589) at the peak checkpoint vs 7.88 (dev 0.133) at the final one
  (`results/mimo-learning-next/probe-peak.bin.diag.csv`, `probe-final.bin.diag.csv`). Per-step reward being flat
  over the *single* window r201–500 says nothing about episodic discounted return. **Whether the return decline
  causes or follows the dev decline is unresolved** (§5 R3 proposes the measurement that settles it).
- **Weight evidence rules out only parameter-norm explosion, not representation collapse** (§4.4). Feature rank
  has never been measured here; under Moalla et al. it is the missing test.

What is *not* instrumented is divergence: `grep -ci kl` = 0, and the diag schema (now
`main.mm:1499`) still contains no approx-KL, no PPO clip fraction, and no explained-variance column — the only
telemetry added by the sibling mission is `actor_drift_l2`.

## 2. Three distinct controls: present vs absent

| mechanism | what it constrains | in this pipeline |
|---|---|---|
| **Likelihood-ratio clipping** (PPO surrogate, ratio clipped to `[1−ε, 1+ε]`, ε=0.2) | the *objective* each update maximizes; limits how far a single update can exploit a ratio change | implemented (`ppo.metal:256-310`, `policy_active` at `ppo.metal:290`); **clip fraction never logged**, so how often the clip binds is unknown |
| **Gradient-norm clipping** (global L2 rescale to 0.5) | the *parameter-step magnitude* of each update — it is active and functioning: every minibatch's raw actor gradient is rescaled to ‖g‖ ≤ 0.5 before Adam (`main.mm:665`, `main.mm:921`) | implemented; **binding on 98.5% of rollouts** (§4.3). Saturation means it is doing its job on almost every rollout *and* that it no longer differentiates between rollouts — a constant step-size cap, which cannot oppose accumulated parameter drift. It is **not** a divergence measure and it is **not** the only control present |
| **KL early stopping** (halt policy updates when KL(old‖new) exceeds a target) | divergence in *policy space* accumulated across updates/minibatches | **absent** — no KL is computed anywhere. The heuristic of stopping at a target KL is an implementation detail documented in OpenAI Spinning Up's PPO page: https://spinningup.openai.com/en/latest/algorithms/ppo.html |

Any earlier phrasing of the form "global-norm clipping is the only trust region / saturation means it does
nothing" is retracted: the pipeline runs ratio clipping *and* norm clipping as live controls, and simply lacks
the telemetry to observe divergence plus the KL stop to act on it.

## 3. State of knowledge

| claim | state | falsifier / how to settle it |
|---|---|---|
| Dev decline is real, sustained, beyond sampling noise | **known** (80 dev evals; peak 25/30 f15 → 1/30; ctrl 0.600 → 0.100; Wilson 95% width ≈ ±0.10–0.20 at n=90, ±0.26 at n=30) | paired re-eval of the same checkpoints reproducing ≈peak rates — the sibling's `baseline-verify-dev.csv` already reproduces 25/30 exactly |
| Not train/dev split overfitting | **known** (train and dev fall together: sibling zero-rollout check train f15 20/30 → 10/30, dev 25/30 → 10/30) | fresh train-split eval of peak vs final showing train flat while dev falls |
| Not parameter-norm explosion | **known** (peak W1 Frobenius 7.73, row norms ≤1.53; final 10.10, row norms ≤2.70, 0 rows >5; init 0.115) | — |
| Representation/feature-rank collapse | **unknown — never measured** | single obs-trace forward pass per checkpoint; participation ratio / singular-value spectrum of penultimate features at peak vs final (§5 R4) |
| Objective signal declines with dev | **known** (diag windows §4.2; probe 10.68 → 7.88) | — |
| Return decline *causes* dev decline (vs. consequence) | **unresolved** | time-ordered probe cadence through the collapse window (§5 R3) |
| Not curriculum drift in the sampler | **known** (stationary uniform draw within fixed quotas, `challenge_training.hpp:320-366`; `rehearsal_envs=0`) | logged active-level histograms drifting over rollouts |
| Single-knob hyperparameter fixes | **known refuted/neutral for the tested set** (§4.5) | a *different* untested mechanism |
| Monotone unbounded actor drift drives the loss (sibling working model) | **supported by intervention** (anchor λ3 caps drift 2.08 → 0.07 and holds dev 0.550 vs control 0.378, 2 seeds, 11/0 and 20/0 per-level wins, zero losses) but still a candidate, not a settled cause of the *original* from-scratch collapse | an intervention that changes drift without changing dev, or vice versa |
| Likelihood-ratio divergence accumulates post-peak | **hypothesis H1** (unmeasured) | KL <0.01 and clip fraction <0.1 while dev falls |
| Terminal credit swamped by dense/shaped terms | **hypothesis H2** (unmeasured) | terminal share of discounted return does not fall post-peak |
| Entropy/std contraction is the direct cause | **unlikely** (axes 0–2 log_std −1.06 → −1.52 over 4000 rollouts; only −0.05…−0.16 inside the collapse window; eval is deterministic `mode=17`, σ unused at test) | holding log_std fixed and observing the same collapse |

## 4. Evidence

### 4.1 Collapse and strict-subset loss

| run | peak | final | shape |
|---|---|---|---|
| `results/rooms-focused.bin*` (4000 rollouts) | 54/90 (r300; f15 25/30) | 30/90 (r4000; f15 10/30) | lost 25, gained 1; collisions 36 → 60 |
| `results/mimo-learning-next/ctrl-r52.bin*` (600 rollouts, warmstart from the peak) | 0.600 (r300) | **0.100 (r600)** | lost 45, gained **0**; collisions 36 → 76; timeouts 0 → 5 |
| `results/room-seed-replication/seed54.bin*` | 45/90 (r200) | 45/90 (r400) but 7 lost / 7 gained | churn at constant aggregate |
| `results/mimo-learning-next/ctrl-r54.bin*` (sibling's second control seed) | 0.533 (r350) | 0.456 (r600) | **no collapse — collapse magnitude is seed-dependent** |
| `results/mimo-room-learning/trajectories.csv` arms A/B/C | 25/30, 18/30, 19/30 | all ≤ baseline | warmstart arm never beats its own rollout-0 checkpoint |

Consequence: any "fixes collapse" claim needs ≥2 seeds and a horizon beyond r600 (learning mission §8.3).

### 4.2 The objective signal declines — reconciled, directionality unresolved

`results/mimo-learning-next/ctrl-r52.bin.diag.csv`, window means:

| window | rew_mean | ret_mean | ep_success |
|---|---|---|---|
| r201–300 | 0.159 | 6.91 | 0.505 |
| r301–400 | 0.184 | 8.30 | 0.565 |
| r401–500 | 0.155 | 5.93 | 0.362 |
| r501–600 | 0.129 | 4.59 | 0.283 |

Pre-update probes (1 rollout, seed 52, from `probe-*.bin.diag.csv` / `.history.csv`):

| probe | return (ret_mean) | dev success | collision |
|---|---|---|---|
| `probe-r0` (warmstart) | 4.005 | 0.4333 | 0.567 |
| `probe-peak` | 10.680 | 0.5889 | 0.411 |
| `probe-final` | 7.879 | 0.1333 | 0.722 |

Reading: episodic return falls −26% from peak to final (and −45% from its r301–400 in-training maximum), moving
*with* dev, not against it. The earlier "objective does not collapse / reward flat while dev falls" statement
was an artifact of averaging per-step reward over a single mid-training window; per-step means are insensitive
to episode success, episode length, and terminal events. **Unresolved:** whether the return decline precedes,
coincides with, or follows the dev decline (§5 R3). Also unresolved: whether post-peak updates are reward
hacking (the sibling's probe argues against it — the policy gets worse on its own objective).

### 4.3 Update-control telemetry (corrected numbers)

`actor_scale = min(1, 0.5/‖g‖)` is written once per minibatch, so a row holds the **last minibatch's** scale of
that rollout (`main.mm:1059`, kernel `main.mm:665`, `critic_max_norm = 0.5` at `main.mm:921`). Over all 600
rollouts of `ctrl-r52`:

- **`actor_scale < 1` (raw norm > 0.5): 591/600 = 98.5%** — the clip binds on nearly every rollout.
- `actor_scale < 0.5` (raw norm > 1.0): 85.5%.
- Scale distribution: min 0.133, median 0.312, mean 0.359, max 1.000.
- Implied raw norms, computed **per row** as `0.5/scale` and *then* summarized: median **1.60**, p10 0.87, p90
  2.43, max 3.75. Inverting the mean (0.5/0.359 = 1.39) would understate the median by ~15% and is **not** how
  the figure is derived.
- Trend: r1–300 `frac(scale<1)` = 97.3%, median implied norm 1.47; r301–600 = 99.7%, median 1.72 — step size is
  pinned at the cap before *and* after the peak.
- Critic: `critic_scale < 1` in 78.0% of rollouts; per-row implied norms median 0.69, p90 1.22, max 2.73.

Interpretation (corrected): norm clipping is functioning as a **constant step-size bound**; because it is
binding almost everywhere it stops discriminating between rollouts, so it cannot by itself prevent monotone
parameter drift — but it is not "doing nothing", and it is separate from ratio clipping (whose binding rate is
unlogged) and from KL stopping (absent).

Sibling-side measurements in the same file: `ratio` ≈ 0.999–1.002 (mean over the 32 updates *before* each
update, so not a divergence measure, `main.mm:1027`), `pol_loss` ≈ 0, `val_loss` 0.89–1.31 flat, `adv_gt3_frac`
0.08–0.13, `actor_drift_l2` 0.61 (r100) → 1.43 (r300) → 2.08 (r484) — monotone, no restoring force.

### 4.4 What weight norms show, and what they do not

Peak vs final actor `W1`: Frobenius 7.73 → 10.10, cosine 0.824 (rooms) / 0.982 (ctrl), row norms ≤1.53 → ≤2.70,
0 rows above 5. This rules out **weight-norm explosion and gross weight reorganization**. It says nothing about
the **rank or participation ratio of the feature matrix** — the quantity Moalla et al. show deteriorates in PPO
and links to trust-region degradation. No observation trace has been collected, so representation collapse is
**unknown**, with an explicit falsifier in §3/§5 R4.

### 4.5 What the terminal learning mission already tested (do not re-run)

`results/mimo-learning-next/report.md`: **7999 / 8000** rollouts, ~35 checkpoints, exclusive lock. Status:
**terminal**. Flags `--epochs`, `--adv-clip`, `--value-coef`, `--anchor`, `--anchor-radius` are
**uncommitted working-tree additions to `main.mm`** (line refs shift between snapshots).

| mechanism | flag | verdict (sibling §4) |
|---|---|---|
| Advantage winsorization ±3σ | `--adv-clip 3` | **refuted — harmful** (0.422 → 0.189) |
| Value coefficient 0.5 → 1.0 | `--value-coef 1` | **refuted** (0.422 → 0.322) |
| PPO epochs 2 → 1 | `--epochs 1` | **neutral** — not adopted |
| Failure-weighted sampling | `priority` | neutral — not adopted |
| Weak plain anchors | `--anchor 0.1 / 0.3` | refuted (0.417 mean → 0.339 / 0.378) |
| Plain anchor λ1 from scratch | `--anchor 1` | refuted — blocks learning (never above warmstart 0.389) |
| **Actor parameter anchor (L2-SP) λ3, retention** | `--anchor 3` | **accepted** — 0.378 → 0.550 mean, 11/0 and 20/0 wins, drift capped 2.08 → 0.07 |
| Hinge anchor, from scratch | `--anchor 0.3 --anchor-radius 1.6` | mixed (+0.200 s52 / −0.145 s54) — not claimed |

Also terminal from that mission: **54/90 dev ceiling unbroken** across all ~8000 rollouts; `f14` is 0/30 in
every run including the peak; `.best` selection always reproduces the peak; the divergence-class levers this
mission cares about (KL stop, clip-fraction/EV telemetry, reward decomposition, probe cadence) were **never
run**.

## 5. Ranked next experiments (revision 2)

Concrete, unrefuted, and each tied to a specific unknown. Costs use the sibling's measured throughput (~20–30 s
per 600-rollout arm; probe/eval ≈ 0.3 s + eval time).

| # | experiment | settles | cost |
|---|---|---|---|
| **R1** | **Divergence + objective telemetry**: add per-rollout approx-KL (old vs new policy over the 4096 rows), PPO clip fraction (already computable from `policy_active`), critic explained variance, and a reward decomposition (progress / time / risk / terminal / potential) to the diag schema at `main.mm:1499` | makes H1/H2 falsifiable; nothing in the schema covers these (only `actor_drift_l2` was added). Must pass the sibling's default-off bitwise regression (10/5-rollout, 0 mismatches except `gpu_s`) | additive edit + one 600-rollout control replay ≈30 s |
| **R2** | **KL early-stop arm** (`--kl-stop ≈0.02`, Spinning Up heuristic) in the sibling's paired retention protocol: continuation from `ctrl-r52.bin.best`, 2 seeds, to r484, control = same without the flag | whether *policy-space* divergence control adds anything beyond grad-norm clipping and the accepted λ3 anchor. Reports dev retention, `actor_drift_l2`, KL and clip-fraction time series together | 2 arms × 484 rollouts ≈1 min wall; never tested (§4.5) |
| **R3** | **Time-order the objective decline**: reuse the existing probe protocol (`probe-*.bin`) at every 50-rollout cadence through a control collapse window, recording return *and* dev together | the **unresolved** cause-vs-consequence question of §4.2 with a direct time series instead of three snapshots | ~13 probes × (0.3 s + eval); no new mechanism |
| **R4** | **Feature-rank measurement** from a single observation trace: participation ratio / singular spectrum of penultimate activations at peak (r300), mid (r450) and final checkpoints | the **unknown** representation-collapse claim (§3, §4.4) — offline, no training | one forward pass per checkpoint + analysis script |
| **R5** | **Eval-protocol hardening**: Wilson CIs on every dev number, paired re-eval of candidate vs current best on identical levels/seeds, stop reporting best-of-80 as the peak | quantifies the ~0.18 best-of-80 selection inflation (simulated at true p=0.75, n=30) and the blind `worst_family_success` metric (`f14` ≡ 0, `main.mm:1266-1272`) | protocol only |

**Do not re-test** (tested and rejected, §4.5): `--adv-clip`, `--value-coef 1`, `--epochs 1`, `priority`,
weak/plain anchors λ0.1–λ0.3, from-scratch plain λ1. `--anchor 3` is already the accepted retention mechanism —
not a new proposal here.

## 6. Top hypotheses — candidates, causes not distinguished

No experiment performed for this mission isolates one cause; the three below are **independent, unproved
candidates** that can be separated only by R1–R4.

**H1 — policy-space divergence beyond what ratio/norm clipping controls.** Ratio clipping bounds each update's
objective and norm clipping bounds each update's step size, but neither observes accumulated KL; with no KL stop
the policy may drift monotonically (as `actor_drift_l2` 0.61 → 2.08 shows) past the region dev measures.
- *Prediction:* KL and clip fraction rise into/through r300→r550; a KL stop at ~0.02 holds dev near peak at r484.
- *Falsifier:* KL <0.01 and clip fraction <0.1 while dev falls (then the cause is not divergence).
- *Control:* `ctrl-r52` (identical warmstart/settings/seed, reproduces 0.600 → 0.100).
- *Cost:* R1 + R2 (≈1 min wall for the paired arm).

**H2 — terminal credit swamped by dense/shaped terms.** Dev measures terminal success; per-step progress plus
scale-16 shaping dominate the per-step signal, and the ±10 terminal term reaches advantages only through the
critic across ~12 truncation hops (400 steps ÷ 32-step rollouts).
- *Prediction:* decomposition shows terminal share of discounted return falling post-peak; raising the success
  bonus 10 → 20–30 at fixed shaping restores dev.
- *Falsifier:* terminal share flat or dominant across the collapse, and a terminal-weight arm leaves the dev
  trajectory unchanged. Note §4.2 *weakens* this hypothesis: return now tracks dev, so a pure metric-misalignment
  story is less likely than before this revision.
- *Control:* `ctrl-r52`; sibling's `--potential-scale 8` arm is the shaping-only contrast already measured (worse,
  19/30) — move terminal, not shaping, alone.
- *Cost:* R1 decomposition + 1 arm × 600 rollouts.

**H3 — representation-rank deterioration degrading the trust region.** Moalla et al. show PPO feature rank
collapses alongside performance and that representation collapse and trust-region degradation mutually
reinforce; weight norms here (§4.4) cannot detect it.
- *Prediction:* participation ratio of penultimate features declines from peak (r300) to final, and the decline
  precedes the dev drop.
- *Falsifier:* stable rank/spectrum across peak → final checkpoints.
- *Control:* the three existing checkpoints `ctrl-r52` peak/final + `rooms-focused.bin.best` (no new training).
- *Cost:* R4 (one trace per checkpoint).

Adjacent candidate with the most support so far: **H4 monotone parameter drift under a saturated constant
step-size cap** (sibling's working model, with intervention evidence from `--anchor 3`: drift 2.08 → 0.07 and
retention +0.172, 2 seeds, zero losses). It is still a candidate for the original collapse, and H1–H3 must be
distinguished from it rather than assumed redundant.

## 7. GR2PO: primary source verified, verdict = no controlled pilot

- **Exact paper:** *GR2PO: Group Relative Return Policy Optimization for Continuous Robot Control*, Pengqin Wang,
  Qiming Zhang, Shaojie Shen, Jun Ma, **arXiv:2609.19850v1 [cs.RO], submitted 17 Sep 2026**. Group-normalized
  discounted returns at each rollout time index; **zero-tail at the rollout boundary with no value
  bootstrapping**; no critic; entropy regularization + KL-based early stopping; MuJoCo v5, 5 seeds.
- **Official code: none** (promised after acceptance). Only unrelated LM-GRPO repositories exist — the rebrand
  trap this mission was warned about; do not substitute them.
- **Its own tables:** PPO higher on Walker2d (5964.93 vs 5023.95) and Humanoid (9484.00 vs 8330.68).
- **Fit: poor.** Episodes are 400 steps = 12.5 rollouts of 32, so zero-tailing discards γ³² = 0.725 per boundary;
  group normalization across our 128 parallel envs mixes bank levels of different difficulty.
- **Verdict:** no GR2PO-vs-PPO pilot. Take its two cheap components — **KL early stopping** (which this pipeline
  lacks anyway, §2) and entropy regularization — into R2. Revisit only if official code lands, or if R1 falsifies
  H1 (KL ≈0 through collapse) *and* R4 shows stable feature rank.

## 8. Pitfalls and traps

1. **Best-of-many selection.** `.best` is picked from 80 evals (4000/50) on the same dev split; at true p=0.75,
   n=30 the median maximum of 80 draws is 0.93. Peaks are optimistic; sustained post-peak floors are not.
2. **Small-n.** n=30 Wilson 95% width ≈ ±0.26 at 0.83 — 25/30 vs 22/30 across seeds is noise; 25/30 vs 1/30 is
   not.
3. **`ep_success` in `.diag.csv` is a ~10-episode estimate** (episodes are 400 steps ≈ 12.5 rollouts) — use the
   90-episode dev eval.
4. **Per-step reward means are not episodic return.** A flat `rew_mean` window does not license a claim about
   the objective (§4.2).
5. **Inverting a mean scale is not a gradient norm.** Report per-row `0.5/scale` summaries, and state that the
   column is the rollout's *last* minibatch (§4.3).
6. **Weight norms/cosine ≠ feature rank.** Never write "no representation collapse" without a trace (§4.4).
7. **Do not conflate the three controls** (§2), and do not attribute the KL-stop heuristic to the 2017 paper.
8. **`worst_family_success` is blind** (`f14` ≡ 0/30), so unqualified selection silently degrades to overall
   success (`main.mm:1266-1272`).
9. **Do not infer GR2PO from LM-GRPO material** (primary: arXiv:2609.19850, no official code).
10. **Moving code.** `main.mm` changed during this analysis; re-check line numbers before citing them in a patch,
    and treat `--epochs` / `--adv-clip` / `--value-coef` / `--anchor` as uncommitted working-tree additions.
11. **`.best` has no replay sidecar**; `train-bank` requires `checkpoint` + `checkpoint.replay.state` together
    (`main.mm:1252-1253`) — do not resume from `.best` in place.
12. **Seed-dependence.** `ctrl-r54` did not collapse within 600 rollouts; collapse claims need ≥2 seeds.

## 9. Citations mapped to claims

| # | source | supports |
|---|---|---|
| 1 | Schulman, Wolski, Dhariwal, Radford, Klimov, *Proximal Policy Optimization Algorithms*, arXiv:1707.06347 (2017) | the clipped likelihood-ratio surrogate only — ε, ratio-space control (§2) |
| 2 | OpenAI Spinning Up, *PPO* algorithm page — https://spinningup.openai.com/en/latest/algorithms/ppo.html | the **KL early-stopping heuristic** (target-KL stop on policy updates) cited for R2/H1 — **not** attributed to the 2017 paper |
| 3 | Schulman, Dhariwal, Radford, Klimov, *High-Dimensional Continuous Control Using GAE*, arXiv:1506.02438 (2015) | GAE λ/γ bootstrap structure → why terminal credit crosses ~12 truncation hops (H2) |
| 4 | Ng, Harada, Russell, *Policy Invariance Under Reward Transformations*, ICML 1999 | potential shaping is policy-invariant only in the discounted infinite-horizon limit; our Φ-term is inert for optimality but large per-step (H2) |
| 5 | **Moalla, Miele, Pyatko, Pascanu, Gulcehre**, *No Representation, No Trust: Connecting Representation, Collapse, and Trust Issues in PPO*, arXiv:2405.00662, NeurIPS 2024 | feature-rank deterioration in PPO, its two-way link with trust-region degradation, and why weight norms cannot test it (H3, §4.4) — **attribution corrected in revision 2** |
| 6 | Andrychowicz et al., *What Matters In On-Policy Reinforcement Learning?*, arXiv:2006.05990 (2020) | LR is the strongest knob; linear LR decay helped 4/5 tasks — ordering LR/drift controls above entropy arms |
| 7 | Patterson, Neumann, White, White, *Empirical Design in Reinforcement Learning*, arXiv:2304.01315, JMLR 2024 | CIs, best-of-many selection bias, paired difference evaluation (R5, pitfalls 1–2) |
| 8 | Henderson et al., *Deep Reinforcement Learning That Matters*, arXiv:1709.06560 (2018) | seed-to-seed variance (pitfall 12) |
| 9 | Dohare, Hernandez-Garcia, Lan, Rahman, Mahmood, Sutton, *Loss of plasticity in deep continual learning*, Nature 632:768–774 (2024) | sustained decline under continued training; plasticity-oriented framing of the collapse |
| 10 | Abbas, Modayil, White, Machado, *Loss of Plasticity in Continual Deep RL*, arXiv:2303.07507, CoLLAs 2023 | plasticity loss vs catastrophic forgetting → why the strict-subset loss pattern matters |
| 11 | Wang, Zhang, Shen, Ma, *GR2PO*, arXiv:2609.19850v1 (17 Sep 2026) | GR2PO verdict (§7) |
| 12 | de Oliveira et al., *Learning Without Critics? Revisiting GRPO in Classical RL Environments*, arXiv:2511.03527 (2025) | critic-free methods lose to PPO on long-horizon classical tasks |

Revision-2 attribution note: row 5 previously named Curi/Berariu; the correct authorship is Moalla, Miele,
Pyatko, Pascanu, Gulcehre (verified against the arXiv abstract listing), and no claim in this document rests on
the incorrect attribution.

## 10. Current state of this mission

- Deliverables: this file (revision 2), `results/mimo-training-research/report.md` (revised),
  `results/mimo-training-research/handoff.json` (revised).
- Revisions applied per parent review: citation attribution, three-control distinction, Spinning-Up KL citation,
  recomputed gradient statistics, weight-vs-rank scoping, objective/reconcile with probe and unresolved label,
  removal of the refuted epochs/adv-clip recommendation, labeling of working-tree flags, and hypothesis
  status as unproved candidates.
- No training, eval, Webots/Blender, git, or source/engine change; `results/mimo-training-research/**` contains
  only documents produced by this analysis. The full literature survey was not repeated — only the cited set was
  corrected and re-mapped.
- Not owned here (reported for the core learning agent): implementing R1–R4, running R2, and any change to
  `main.mm`, `ppo.metal`, `sim.metal`, or reward code.
