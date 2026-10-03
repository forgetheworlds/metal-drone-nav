# Raw-depth guided actor: version and lift check

## Result

The v2 actor input preserves the selected 184-feature guided policy and appends both unpooled 16×20 depth frames. The old model still uses the v1 asset reader. The raw model uses a separate v2 file magic and metadata. A v1 reader rejects a v2 file, and a v2 reader rejects a v1 file.

The selected 184-feature rooms checkpoint lifts to 824 features by copying the old first-layer weights for features 0–180, moving the three prior skip weights to new features 821–823, and setting all raw-depth weights to zero. Old context and pooled-depth feature indices stay unchanged. The new actor therefore behaves exactly like the old actor until training changes raw-channel weights.

On the mirrored 90-level DEV bank, the original checkpoint and lifted checkpoint produced identical results for every level. Both scored 54/90: family 14 0/30, family 15 25/30, family 16 29/30. Per-level success, collision, timeout, progress, path, elapsed time, peak speed, and minimum clearance matched exactly. This is a representation test, not a learned improvement. The lifted checkpoint is not selected for deployment.

## Motivation and limit

The saved active-sensing CSV reports that one common safe parallax prefix separates raw current depth in 15/15 mirrored TRAIN pairs at 2.0 seconds, while pooled actor depth separates 3/15. The 824-feature actor keeps the old 181 non-hint features, then appends current raw ranges at 181–500, previous raw ranges at 501–820, and the unchanged geometry hint at 821–823. This provides an opt-in way to test raw cues without changing the old policy's function or action map.

The first saved active-sensing probe was sampled at 2Hz and is superseded for timing claims. The corrected 20Hz probe reports raw depth distinctions in 15/15 mirrored pairs at 1.5s, 2.0s, and 2.5s. Pooled depth/history distinguishes 12/15, 5/15, and 12/15 at those times; 14/15 pairs separate at one or more recorded branch points. This supports testing raw channels, but it does not prove that 184D history loses a cue needed for successful navigation or that the 824D actor improves behavior. The new raw channels use the same 320 RangeFinder rays and 12m normalization. They add no simulator geometry or route data.

## Observation and deployment contract

The MSL and CPU simulator pack the v2 fields from the current and previous raw range-ring slots. The native controller uses the same two ring slots through `pack_raw_depth_observation`. At reset, both raw frames are the first captured frame, matching the existing startup history rule. The geometry prior stays at the end of the input, so the v1-to-v2 weight lift preserves its skip connection.

A v1 asset has `NAVPOL1` magic, version 1, 184 observations, and 12,104 weights. A v2 raw asset has `NAVRAW2` magic, version 2, 824 observations, and 53,064 weights. The Webots controller defaults to `policy_version=1`; a raw v2 policy requires explicit `policy_version=2`. Wrong-version files fail to load.

## Verification

The private raw target and original guided target both build and pass the project test command. The v2 lift probe reports zero deployment-command error on 16 probes, raw pack error `5.96e-8`, and CPU/Metal actor-mean error `1.19e-7`. The raw guided build's mixed-domain CPU/Metal observation error is `1.19e-7`.

The old checkpoint and lifted checkpoint were each evaluated on the same mirrored 90-level DEV bank. All per-level episode outcomes and recorded path metrics matched exactly. The raw feature matrix lift was also checked directly: all 40,960 new first-layer raw-depth weights are zero; all copied prefix, prior, head, and log-standard-deviation weights match the old checkpoint.

The Webots controller now selects policy version 1 or 2 explicitly. I built it with the shared Webots R2025a app, then ran two frozen DEV room scenes under both policy versions. The runs used port 23456, minimized batch mode, and no rendering.

The easier `f15-dev-0000` scene succeeded under both versions in 523 steps (5.23s). The harder mirrored `f15-dev-0001` scene collided under both versions in 656 steps (6.56s). The second run reproduces the original failure; it does not show a gain. The paired Webots traces differ by at most `1.3e-5m` in sampled position, `8.6e-5m/s` in measured velocity, and `6.0e-5` in logged target velocity or goal error. This is a two-scene wiring check, not transfer or generalization evidence.

No imitation training, PPO training, final split, or full Webots matrix was run. Keep this version opt-in while learning remains unverified. If that probe shows the deployed 184 input retains the cue, do not use 824 solely on the earlier sampled difference. Do not treat representation parity or the two native smoke cases as a navigation gain.

## Native 20Hz raw-input shadow audit

A separate 5-second Webots run used the original v1 policy to control the drone. The v2 824-feature observation ran in shadow only; it did not affect commands. The run recorded 100 navigation observations at 20Hz, including both 320-ray raw frames, their capture times and camera poses, and all 824 packed features.

The live pack audit found zero declared layout error across all rows. Independently recomputing the 2x2-min pooled channels from the raw 16x20 ranges gave a maximum feature error of `2.44e-8`; recomputing raw range normalization at indices 181–500 and 501–820 gave the same maximum error. The previous frame's capture time and 12-value pose matched the preceding row's current frame exactly. Current sensor capture time matched the navigation sample time on all 100 rows. The largest observed same-ray current/previous difference was 11.575m (step 69, ray 148; 12.0m versus 0.425m). The first observation uses the same captured frame for current and previous history.

To check that both raw ranges can affect the actor, a nonzero CPU/Metal canary used that native sample. Its first layer read the selected current ray into one hidden unit and the previous ray into another; separate head weights mapped them to different action means. Replacing only current or only previous changed its assigned mean by `0.726179` and had zero cross-effect. CPU and MSL means differed by at most `5.96e-8`. This checks field ordering, channel offsets, and actor access to both inputs. It does not show that a trained actor uses these channels or performs better.

The shadow-enabled v1 trajectory matched the saved v1 run exactly for all 84 common samples through 4.0s: position, measured velocity, target velocity, and goal error all had zero difference. This confirms the shadow collector did not change the v1 action path for the compared prefix. The shadow run itself ended at its 5s limit without success or collision. Source and raw data hashes are in `native-shadow/manifest.json`.

## Parent verification and current sensing evidence

The corrected 20 Hz paired experiment records a common physical yaw/parallax prefix with all sensor and memory updates. At 2.0 s, raw channels separate all 15 mirrored TRAIN pairs; the legacy pooled/history input separates 5. This establishes retained sensory information at that sample, not learned decisions or universal sufficiency.

The parent independently checked all 100 native shadow rows: raw and pooled normalization error is at most 2.44e-8, with 99 exact previous-pose/time chains. The raw evaluator now uses the shared complete CSV/scoring implementation. All 90 lifted DEV episodes match the control exactly on 16 outcome/metadata fields, including 54/90 successes.

- [Portable proof and native shadow CSV](../evidence/inputs/raw-depth-guidance/proof.json)
- [Prior failed imitation experiment](IMITATION_EXPERIMENT.md)

Build the opt-in target metal_nav_raw_guided, then use its lift-guided command with the preserved rooms-focused checkpoint. Exported raw policies use separate NAVRAW2 magic and require policy_version=2 in the RL Webots controller. A lift has fresh optimizer/simulation state; it is a parameter warmstart, not continuation of the earlier optimizer.
