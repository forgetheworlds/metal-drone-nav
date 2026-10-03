# Course-bank feasibility and native Webots review

## Decision

The course bank has useful geometry and at least one moving-world route works in native Webots. The current evidence does not establish that the long courses fit the learned policy's 20 s episode budget. It also does not establish the declared 0.04 m clearance floor on the privileged follower's executed path. These are limits in the claim, not proof that the course bank is impossible or broken.

The evaluated policy baseline is weak on the bank: the saved report records 8/108 for `guided-table-memory` mode 17 and 10/108 for `rooms-focused-experimental` mode 17. Those scores alone do not prove the courses are infeasible. A separate native DEV replay with `assets/navigation.bin` succeeded, but it is one scene and does not exercise contact response.

## Budget and witness timing

`bank-eval` uses 400 steps at 50 ms, or 20 s. `bank-witness` results use 1200 steps, or 60 s. Every one of the 108 DEV witness episodes completed after 20 s: elapsed time ranged from 21.85 to 34.80 s. Actual followed path length ranged from 14.40 to 26.15 m. The generated route lengths range up to 28.1 m. At the requested 1.5 m/s cap, 28.1 m has a 18.73 s ideal travel-time lower bound. This leaves little time for turns, acceleration and corrections. The measured privileged follower runs at about 0.63–0.81 m/s and does not establish 20 s completion.

Thus `108/108` means that a privileged route follower reached DEV goals within 60 s. It does not mean the routes fit the policy's 20 s budget. A 20 s privileged witness check is the missing feasibility measurement. Do not infer impossibility from the existing timing mismatch: the ideal lower bound is below 20 s, and the learned action cap differs from the witness's 1.0 m/s cap.

The course movers are constrained to stay inside the room for the first 20 s. In 27 of the 45 moving DEV records, at least one mover's recorded room-exit time is earlier than the 60 s witness episode's finish. This does not invalidate the intended 20 s policy episodes. It means the 60 s witness pass may finish after a mover leaves the room and cannot be described as a complete in-room moving-course demonstration.

## Clearance and timing limits

The static generated witness polylines pass their 0.09 m body-surface-clearance target. The executed follower path is closer to obstacles: `witness-dev-movers.csv` records clearance below the challenge's 0.04 m floor in 29 of 108 runs, with a minimum of 0.00968 m. `world.hpp::wclearance` subtracts the 0.18 m vehicle radius, so these are body-surface margins, not centerline distances. All 108 runs still reported success with no collision. The evidence supports collision-free passage in those 60 s runs; it does not support the stated minimum safety margin on the flown path.

The calibrated timing file reports a measured inverse-speed spread of 0.5921 s/m, then stores 0.40 s/m for use. `error_bound_s` multiplies the stored value by 1.3. The resulting bound is still lower than the measured spread by about 0.250 s per route metre, or 2.50 s at a 10 m arc. The observed DEV witness runs all pass, but the formal timing buffer is not conservative relative to its own pilot measurement. Treat this as an unverified mover-timing guarantee, not proof that the moving scenes fail.

## Export validation fix

`export_course_world.py::validate_record` calculated each obstacle extent for the initial room check, then reused the final z extent for all axes during future mover checks. A thin vertical cylinder with radius 0.30 m, half-height 0.10 m, center x=13.7 m, and x velocity 0.01 m/s was accepted by that stale-extent logic even though its 0.30 m radius crosses the x=14 m room boundary by t=4 s. The check now uses the proper per-axis extent at every sample. This is a general exporter validation fix; the current generator uses taller-than-wide moving cylinders and the existing course bank was not regenerated.

The corrected exporter still validates the selected saved F16 DEV export. Its static shapes and mover schedule match the bank, and the analytic 0.5 s samples have zero arithmetic error. The source bank remains unchanged.

## One native DEV integration check

Scene: `mc16-dev-0001-se5621845-wabb5b4df-mirror-y` (family 16, DEV). The Webots mover is an upright cylinder with radius 0.2809 m and half-height 0.3288 m. It moves along y at -0.223807 m/s. The diagnostic run used the saved Webots world, the existing `assets/navigation.bin` actor, the 0.18 m Raptor collision sphere, a 1 ms Webots/ODE step, 100 Hz RAPTOR, and 20 Hz depth/navigation. It did not change route geometry, physics or the time budget.

The actor reached the first goal at 17.09 s after 1709 RAPTOR steps. The receipt reports success, no contact, 341 sensor updates, and a 0.222646 m minimum pooled range. A native 20×16 RangeFinder frame was saved at 0.05 s. The Supervisor logged 18 native mover positions from 0.001 to 17 s. At each sample the actual world position was 0.000224 m behind `center + velocity * time`, equal to one 1 ms update at the mover's speed. The exporter arithmetic check is exact; the runtime controller applies its field update after the current physics step.

This confirms one full-stack pass in a moving DEV scene and confirms the moving Solid changes position in Webots. No contact occurred, so this episode does not validate the collision response against the mover. The sensor audit preserves one frame, not a time series that identifies the moving object in the learned input.

The RL Webots process exited, port 23456 is free, the temporary world was removed, and the run's shared controller files were copied to the case folder and cleaned up. No other Webots process was stopped or changed.

## Native recording

The same successful frozen-policy DEV route was recorded in a visible Webots R2025a run. The MP4 is 1280×720 at 25 fps and 17.08 s long. Webots decoded all frames; its black-frame gate found 0 s of near-black footage. Sampled frames at 1, 4, 8 and 16 s show the drone. The 4 s frame also shows the drone near the moving cylinder and the overhead slab. This is native simulator footage, not a reconstruction. The recording keeps the exact route, actor, body and physics; it changes only the Viewpoint.

## Files and checks

- `handoff.json` contains exact hashes, counts, paths and reproduction commands.
- `run_native_moving.py` performs the one native run and cleans the RL project's shared run files after copying them.
- `native-moving-dev/` preserves the source and diagnostic worlds, metadata, episode receipt, 100 Hz state log, 20 Hz actor/sensor log, first RangeFinder frame, contact log, driver position samples and manifest.
- [Native Webots MP4](native-moving-movie/native-moving-dev.mp4), its native 1 s snapshot, raw receipt/log, mover samples, black-frame verdict and manifest are in `native-moving-movie/`.

Verified: Python bytecode compilation, the synthetic stale-extent regression, saved DEV exporter parity, and the one native Webots run. No bank rebuild, broad policy evaluation, training or final-split evaluation was run.

## Public artifacts

The movie is an unmodified native Webots recording. It shows the existing frozen default navigation policy, not a newly trained replacement. The goal score is first entry within 0.35 m, with no stable-hold dwell. Actual mean path speed is 12.2246 / 17.09 = 0.715 m/s; the 1.933 m/s value is a peak, not cruise speed. This is one selected DEV example, not a reliability rate or blind-final proof.

- [Native moving-course video](../artifacts/videos/native-moving-course.mp4)
- [Portable proof and exact recorded world](../evidence/inputs/dynamic-course/proof.json)
- [Course bank](../evidence/inputs/course-bank-v1.jsonl)
- [Generator](../navigation_courses.py)
- [Mover-aware Webots exporter](../webots/export_course_scene.py)

Cold export (no simulation): python3 webots/export_course_scene.py mc16-dev-0001-se5621845-wabb5b4df-mirror-y --bank evidence/inputs/course-bank-v1.jsonl. Default output is inside the RL Webots project. Normal runs use --minimize --batch and port23456; native recording requires the documented visible realtime batch exception.

![Native frame at four seconds](../artifacts/videos/native-moving-course-preview.jpg)
