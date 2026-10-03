# Geodesic auxiliary source review

Date: 2026-10-03

## Decision

The direction auxiliary is training-only and remains opt-in. Its pre-action labels use the active bank level and current physical pose; the sign and body-frame conversion are correct. The tested coefficient did not produce a family-14 corner success in either 600-rollout seed. Do not adopt it as a default or increase its strength based on this evidence.

I found and fixed one resume compatibility defect in `main.mm`: a safeguard-v1 file has no direction-field hash. The loader used to compare the missing hash with the current field hash, which rejected a valid legacy run with potential shaping enabled and the direction auxiliary disabled. The loader now skips that comparison only for v1 files when the auxiliary coefficient is zero. It still rejects v1 resume when the auxiliary coefficient is nonzero. Potential-shaping fields remain bound by the v9 PPO checkpoint.

## Source checks

- `Sim::collect` writes auxiliary labels after observation/action creation but before `sim_advance`. `sim_act` changes command and navigation-reference fields; it does not change physical pose. The target kernel therefore uses the same pre-action pose as the actor observation.
- Each label uses the old `bank_active_ids` value for that transition. `sim_advance` may replace the ID only after it processes the transition. Rehearsal IDs and invalid field directions produce an invalid label.
- The target is the normalized direction of increasing `Phi=-distance/cap`. CPU/Metal label parity passed. The world-to-body conversion uses `Rᵀ`; the PPO gradient sign points toward the target.
- The auxiliary differentiates the deployed translation gate and `tanh` map. It adds gradients to XYZ means only. It leaves yaw, the observation vector, PPO log probability, deployment code, and reward unchanged.
- Default coefficient zero skips target allocation/collection and the auxiliary-gradient dispatch. The auxiliary is not part of evaluation or deployment.
- Safeguard v2 stores the auxiliary coefficient and direction-field hash. V1 resume is allowed only when the coefficient is zero. Active anchors still require their saved reference and matching hash.

## Evidence and checks

The mirrored-bank report records zero family-14 successes for control and auxiliary runs at seeds 42 and 43 after 600 rollouts. Both methods selected the same development checkpoint for each seed. The prior 30-level TRAIN preflight found valid directions for all levels, exact CPU/Metal direction parity, mean cosine 0.789 to the first witness segment, and minimum clearance 0.974 m over the local 0.6 m target segment. An isolated auxiliary-only actor update reduced direction loss from 0.38226 to 0.37879. These checks confirm the label and gradient; they do not establish route completion.

The legacy-resume probe used a copy of the mirrored seed-42 control checkpoint and replay state, plus a v1 safeguard header, with potential scale 16 and direction auxiliary coefficient 0. It resumed at rollout 600, completed rollout 601, and wrote a v2 safeguard sidecar. The source checkpoint and replay state were not changed.

The exact build and resume output is in `legacy-v1-shaped-resume.log`. The verified source hashes are in `handoff.json`. I did not run another long training, a full test suite, or Webots. I did not change simulation, physics, geometry, assets, or deployment code. No files were staged or committed.

The auxiliary only reports its PPO losses before its added gradient. The rollout diagnostics show the coefficient and valid target count, not a combined PPO-plus-auxiliary loss. Treat the logged PPO policy loss as the PPO component only.

## Limits found after the experiment

27 of 30 initial TRAIN labels had an absolute vertical component above 0.2 (mean 0.466), while the provided first witness segments were horizontal. These are local gradients, not complete physically executed expert routes. Logged post-clipping actor scales combine PPO and auxiliary gradients; they do not identify relative contributions. Earlier anchor-disabled drift diagnostics compared against random initialization and cannot establish update drift. The corrected diagnostic does not change optimization.

Portable evidence: [proof](../evidence/inputs/geodesic-direction-aux/proof.json). Selected policies remain unchanged.
