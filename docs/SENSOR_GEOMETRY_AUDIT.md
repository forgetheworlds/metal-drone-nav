# Sensor geometry audit: source ray contract vs native Webots R2025a RangeFinder

**Date:** 2026-10-03
**Owner:** results/omp-sensor-transfer (fixture, controller, analyzer, this report)
**Status:** original fixture measurement complete; no navigation result is claimed. The later optional implementation, optimized-cache correction, and frozen-policy sensitivity check are documented in [CALIBRATED_SENSOR_PROFILE.md](CALIBRATED_SENSOR_PROFILE.md).

## Verdict

The native sensor contract is **axial depth along the node +X optical axis on a horizontal-FOV pinhole
with square pixels**, and its vertical half-tangent is **0.8 = tan(45 deg) * 16/20**, i.e. the
16/20 aspect value in the lead — not 0.75.

The source model (`world.hpp::wcamera`) uses 0.75. The production adapter
(`raptor_webots.cpp::normalize_ranges`) **deliberately compensates** for that difference by resampling
native rows onto the source's 0.75 ray grid, so the 0.75/0.8 difference is not by itself an adapter
bug. Its measured consequences on the source grid are:

1. on surfaces whose depth varies with the vertical angle (floor/ceiling) the adapter feeds the policy
   **exactly 0.9375x (= 0.75/0.8) the source-model ray range**, measured ratio 0.937500 (ground) and
   0.937567 (ceiling); on axial-constant surfaces (perpendicular walls, markers, occluder) the ratio is
   1.000000 and the median residual is 0;
2. **0–9.4% of model pixels land on a different scene primitive than the corresponding native ray**
   (0/320 in the symmetric pose, 30/320 worst case), always at horizontal edges/corners; there the
   per-pixel adapter-vs-source-model difference reaches **8.16 m** (worst case: source ray sees the
   ceiling at 9.46 m, adapter reports the wall at 2.87 m).

Separately, the native sensor sits 0.08 m ahead of the body centre while the source casts depth rays
from the body centre: same-direction ranges differ by **median 0.046 m, p95 0.107 m** (that is the
uncompensated mount offset the actor's depth channels carry; the guidance-memory pose transform is
correct as written and is not implicated).

No production adapter repair is justified by these measurements; see sections 7–8.

## 1. Fixture, provenance and exact repro

`webots/worlds/sensor-geometry-audit.wbt` — static calibration rig, 10 locked axis-aligned primitives
(ground, ceiling, 4 walls, one truncated front wall to expose floor/ceiling/side walls, one vertical
pole, one 6 cm occluder blade, two asymmetric markers). The RangeFinder copies the production mount
exactly: `translation 0.08 0 0`, `rotation 0 0 1 0`, `projection "planar"`, `fieldOfView 1.570796327`,
`width 20`, `height 16`, `near 0.03`, `minRange 0.03`, `maxRange 12`, `noise 0`, `resolution 0.001`,
enabled at 50 ms (20 Hz, production cadence).

14 poses: 9 static (three frontal distances, two yaw, a 90 deg side look, a low floor pose, a
discontinuity pose) plus 5 first-refresh latency probes that teleport the rig between two poses.
Every pose records the raw 320 native floats, the measured RangeFinder and body world pose from
Supervisors, GPS/InertialUnit/Gyro, live fov/min/max/sampling settings and the step time.

Build, run and analysis (the run takes the shared GPU lock and the dedicated port; the controller
self-terminates and returns non-zero if any ray is non-finite):

```sh
cd /Users/muadhsambul/RL
WEBOTS_HOME=/Users/muadhsambul/embodied/work/Webots.app make -C webots/controllers/sensor_geometry_audit
flock results/metal-training.lock -c '/Users/muadhsambul/embodied/work/Webots.app/Contents/MacOS/webots \
  --minimize --batch --mode=fast --no-rendering --port=23456 --stdout --stderr \
  webots/worlds/sensor-geometry-audit.wbt'
python3 webots/controllers/sensor_geometry_audit/analyze.py
python3 webots/controllers/sensor_geometry_audit/parity_check.py
```

Artifact hashes (sha256):

| artifact | sha256 |
|---|---|
| `webots/worlds/sensor-geometry-audit.wbt` | `b3ff6a6ea2ed6c7aea510f151d6192b68100090d9fe20f5f16e9bf29cd40df1c` |
| `webots/controllers/sensor_geometry_audit/sensor_geometry_audit.c` | `75f7a540ae267b0ff2832b84ba7a1c86af4766040a136039419e630323655cf6` |
| `webots/controllers/sensor_geometry_audit/analyze.py` | `f149cdda9fdc8a529698413954efd7e8f9849b8407d0eb4c5b3fb8b69dbebf5b` |
| `results/omp-sensor-transfer/sensor-geometry/native-frames.jsonl` | `68c840ffe697eb7779458aa06d34c1b8308f7179034db89b9299dfd6b36d2d33` |
| `webots/controllers/raptor_webots/raptor_webots.cpp` (read-only, audited) | `f2a73544476655dae863d6909a13208885785562b241b61af62ae3130c83e2d8` |
| `world.hpp` (read-only, audited) | `5e651fdf2dcb2d252b64d688a46ab0b793849356bd10a1c205d2b7cd9fabffa2` |
| `guidance.hpp` (read-only, audited) | `19e2b0c0af38528f0dcb2686184757f9899c8cf94d0d6e9f732da0b57d735671` |

Every reported native number below is an actually loaded, finite sensor value: 14 x 320 =
**4480/4480 rays finite**, zero rays mapped to the 12 m no-hit contract. All comparisons use the
measured device world pose, never an assumed one.

## 2. Native contract, as measured

The documented contract ([RangeFinder reference](https://raw.githubusercontent.com/cyberbotics/webots/R2025a/docs/reference/rangefinder.md),
[Camera reference](https://raw.githubusercontent.com/cyberbotics/webots/R2025a/docs/reference/camera.md)) is that
`fieldOfView` is **horizontal**, that range-finder pixels are **square**, and that
`vertical FOV = 2 * atan(tan(fieldOfView/2) * height/width)`. The implementation agrees:

* [`WbWrenCamera::computeFieldOfViewY`](https://raw.githubusercontent.com/cyberbotics/webots/R2025a/src/webots/wren/WbWrenCamera.cpp)
  is `2*atan(tan(fovX*0.5)/aspectRatio)` with `aspectRatio = mWidth/mHeight` -> `tan(vFOV/2)=0.8`.
* [`resources/wren/shaders/encode_depth.vert`](https://raw.githubusercontent.com/cyberbotics/webots/R2025a/resources/wren/shaders/encode_depth.vert)
  writes `distToCamera = -vCoordTransformed.z`: **axial** view-space depth, not ray length.
  [`encode_depth.frag`](https://raw.githubusercontent.com/cyberbotics/webots/R2025a/resources/wren/shaders/encode_depth.frag)
  replaces `< minRange` or `>= maxRange` with `FLT_MAX`, and
  [`WbWrenCamera::copyContentsToMemory`](https://raw.githubusercontent.com/cyberbotics/webots/R2025a/src/webots/wren/WbWrenCamera.cpp)
  copies attachment 1, i.e. that clamped value; `wbr_*` then hands the raw float buffer to the controller
  ([`range_finder.c`](https://raw.githubusercontent.com/cyberbotics/webots/R2025a/src/controller/c/range_finder.c)).
* [`WbRangeFinder::updateOrientation`](https://raw.githubusercontent.com/cyberbotics/webots/R2025a/src/webots/nodes/WbRangeFinder.cpp)
  applies `rotateRoll(pi/2)` then `rotateYaw(-pi/2)`, i.e. the FLU axis re-orientation.
* Installed header `Webots.app/Contents/include/controller/c/webots/range_finder.h` confirms
  `image[(y) * (width) + (x)]`, rows top-to-bottom.

Measured confirmations (all against fitted ray-casts of the frozen scene from the measured pose):

| property | measurement | verdict |
|---|---|---|
| extrinsics: axis +X, image left +Y, up +Z | mapping-hypothesis median residual 2.53e-4 m for `left_+Y_up_+Z`; 1.07e-3 / 0.196 / 0.925 m for the three alternatives | confirmed |
| horizontal tangent | 50 column probes, `max|measured − |2(x+0.5)/20−1|| = 1.88e-4`, mean 8.46e-5 | `tan(hFOV/2) = 1.0` |
| vertical tangent | 61 row probes, `max|measured − 0.8·g| = 1.95e-4`, mean 7.20e-5; `max|measured − 0.75·g| = 4.71e-2`, mean 3.67e-2; least-squares fit **0.799974** | **0.8**, not 0.75 |
| axial (not ray range) | perpendicular-wall pixels: median `|m − axial|` 0–3.84e-4 m (max 2.29e-3) vs median `|m − ray_range|` 0.428–1.100 m | axial, confirmed |
| pixel centres | the 1.0 horizontal fit above is only consistent with `(i+0.5)/w` and no half-pixel offset | confirmed |
| min/maxRange | not exercised by any pose (measured values span 0.71–10.00 m, no clipping) | not tested |

`g = 1 − 2(r+0.5)/16` is the row-centre coordinate; the row probes recover the tangent magnitude
hypothesis-free as `cam_z / measured_axial` on ground/ceiling pixels and `|cam_y − wall_y| / measured_axial`
on side-wall pixels (+X-facing, level poses only).

## 3. The 0.75 vs 0.8 lead

* The native vertical half-tangent is 0.8, exactly `tan(90deg/2) * 16/20`. The source's 0.75 is
  `0.8 * 15/16`, i.e. the native **outermost row-centre** tangent for a 16-row sensor; it equals the
  native image extent at row centres, not at image edges. Whatever the intent, it is not the
  aspect-correct value for a 20x16 square-pixel sensor.
* The adapter does compensate: `normalize_ranges` (`raptor_webots.cpp:117-138`) maps each source-grid
  tangent `v = 0.75*g(r)` to the native row `source_y = (1 − v/tan_v_native)*8 − 0.5`, reads the
  **minimum** of the two straddling native rows, and rescales axial to ray range with the source ray's
  norm `sqrt(1+u^2+v^2)`. The measured result is that this min always selects native row `r`
  (`source_y = r + 0.46875 − 0.0625r`, so `r` is the larger-`|v|` row on a monotone surface), whose
  tangent is `0.8*g(r)` instead of the source's `0.75*g(r)`.

## 4. Measured consequences on the source grid

Replication is not a re-implementation: `parity_check.py` extracts `clampf`, `normalize_ranges` and
`pool_ranges` **verbatim from the production source text**, compiles them standalone, and runs them on
the measured frames against the analyzer's port — max difference over 4480 pixels is
`8.6e-7 m` (ranges) and `6.6e-7 m` (pooled). Status PASS.

| comparison (adapter output vs analytic source-model ray) | median | p95 | max |
|---|---|---|---|
| all model rays, origin = camera | 5.08e-4 m | 0.4636 m | 8.1635 m |
| all model rays, origin = body centre | 8.65e-2 m | 0.4987 m | 8.1635 m |
| interior pixels (native 8-neighbourhood of one primitive) | 2.95e-4 m | 0.3952 m | 0.5750 m |
| boundary pixels (native 8-neighbourhood crosses a primitive) | 7.77e-4 m | 1.5267 m | 8.1635 m |

Signed residual (adapter − source-model ray) and ratio, by the primitive the source ray hits:

| model-ray hit | n | signed median | median ratio adapter/model |
|---|---|---|---|
| GROUND | 450 | −0.198523 m | **0.937500** |
| CEILING | 208 | −0.449860 m | **0.937567** |
| FRONT_WALL | 575 | +0.000000 m | 1.000000 |
| LEFT_WALL / RIGHT_WALL | 639 / 439 | +0.000000 / +0.000152 m | 1.000000 |
| OCCLUDER | 98 | +0.000000 m | 1.000000 |
| POLE | 133 | +0.000583 m | 1.000365 |
| MARKERS | 18 | ~0 | 1.000000 |

The exact 15/16 ratio on floor/ceiling and the exact 1.0 ratio on axial-constant surfaces identify the
mechanism unambiguously: resampling cannot recover the source ray's value on surfaces whose depth
varies with the vertical angle, because the native image contains no sample at the source's tangent.
This is an inherent property of training a 0.75-slope grid on a 0.8-slope sensor, not a coding error in
the resampler. The bias direction is conservative (obstacles appear closer), except at edges.

Structural disagreement per frame (source model ray vs native ray, same pixel index): 6/320, 30/320,
5/320, 9/320, 9/320, **0/320**, 21/320, 13/320 → 0–9.4%, and every disagreement sits on a horizontal
edge or room corner (e.g. `nat=CEILING model=FRONT_WALL` at the truncated front wall's top edge).
Worst per-pixel cases: model ray sees ceiling at 9.46/11.83 m while the adapter reports 2.87/3.90 m
(wall/occluder), i.e. `-8.16 m`.

Pooled 8x10 channel (what the v1 actor consumes) inherits this: per-pose median cell difference
≤1.5e-3 m but maxima 0.46–4.96 m on boundary cells.

## 5. Mount origin (2 cm-free statement: 8 cm)

* Source: `sim.metal:144` and `cpu_reference.hpp:281` cast `wray` from `s.position` (body centre);
  `cpu_reference.hpp:273` also stores body-centre poses.
* Native: the sensor is at body + 0.08 m along body +X (`NavigationQuadrotor.proto` / this fixture),
  and the controller's `current_ranges`/`pooled` come straight from it.
* Same-direction range difference (model ray from camera vs from body centre): median **0.0456 m**,
  p95 **0.1071 m**, max 6.97 m (edge pixels). For a frontal plane the difference is the full 0.08 m.
* Pooled cells differing by >5 mm because of the mount alone: 42, 58, 22, 67, 63, 70, 37, 25 of 80
  across the eight static poses; per-cell median 0.069–0.097 m.

So the actor's depth channels are systematically short by the forward mount offset by construction.
This is a physical sensor-placement difference between the training model and the native rig; the
adapter cannot undo it without knowing surface normals, and the source model would have to place its
virtual sensor 0.08 m forward (a source-representation change).

## 6. Pose, rotation, quaternion and time alignment

* Device pose from Supervisor matches the commanded rig pose to numeric precision (e.g. rig `[0.6,0,1.5]`
  yaw 0.5 -> device `[0.670207, 0.038354, 1.5]` = body + R(0.08,0,0)); GPS equals rig position;
  IMU roll/pitch/yaw = (0,0,yaw) with level rig.
* Rotation: the 9-element row-major orientation is local->world, consistent with
  `rotate_body_to_world` in the production controller and with the fitted mapping in section 2.
* Latency: for the five first-refresh probes (rig teleported between two poses 0.6 m / 0.5 rad apart),
  frames match **their own** pose (median residual 1.05e-4 m and 2.85e-4 m) and not the previous pose
  (1.45e-3 m and 1.11 m). No stale frame, no measurable 50 ms capture lag.
* `guidance.hpp:112-117` builds each memory hit as `old_pose + range*ray` in world and then subtracts
  `current_pose` and rotates into the current frame. Camera-origin historical poses with a body-origin
  current pose are therefore **required and correct**; the source uses body-origin poses because its
  ranges are body-origin. No inconsistency and no change is proposed here.

## 7. Owning layer for each observed mismatch

| observation | owning layer | evidence |
|---|---|---|
| native vertical half-tangent 0.8 vs source 0.75 | source representation, `world.hpp:89-92` (`wcamera`, `-z*0.75f`) | row probes; fit 0.799974 |
| −6.25% range scale on floor/ceiling after resampling | interaction: source 0.75 grid sampled on a 0.8 sensor; the adapter's min-of-two-native-rows picks native row `r` | ratio exactly 0.937500 |
| 0–9.4% of pixels on a different primitive; up to 8.16 m per-pixel error at edges | same interaction, surfacing in the adapter's resampling | mismatch table; worst-pixel table |
| depth channels short by 0.08 m | adapter sensor mount vs source body-centre ray origin (physical placement) | mount comparison; pooled cells |
| axial/radial conversion | adapter is correct: it converts axial→ray range with the source ray's norm, and matches 1.000000 on axial-constant surfaces | by-hit ratios |
| pose cloud origins / rotation / time | correct, no defect found | section 6 |

## 8. Adapter repair: none justified (no patch proposed)

The measurements do not support a production adapter patch:

* The mixed pose origins in the guidance-memory call are correct (`guidance.hpp:112-117`), so the
  obvious-looking "make `current_pose` a camera origin" change would be a bug.
* The 0.9375 scale and the edge disagreements are caused by the source evaluating rays at
  0.75·g(r) that the native sensor never samples. Any adapter-side change would have to either
  distort measured data or re-define the target grid; both change the actor input distribution.
* The 8 cm mount offset cannot be removed from a single range image without surface normals.

Options for the parent, in order of increasing cost, none executed here:

1. **Leave as is.** The residual is bounded, zero on axial-constant geometry, and conservative
   (nearer) on floor/ceiling; it costs at most a few boundary pixels per frame.
2. **Change the source's `tan_v` to 0.8** (`world.hpp::wcamera`). This removes the 15/16 scale and the
   vertical-edge disagreement by construction, and makes the adapter resampling an identity in tangent.
   It changes the observation distribution of every existing 184/824 policy, so it requires retraining
   and then held-out zero-shot validation before adoption.
3. **Move the source's virtual sensor origin to body + 0.08 X** to match the physical mount, likewise a
   source-representation change requiring retraining.

Per the project contract, the source-policy representation must stay explicit and any change must be
followed by held-out zero-shot validation; no such change, training run or weight edit was made or
proposed as ready.

## 9. Limits of this audit

* One static room of axis-aligned boxes plus one cylinder; no motion blur, no noise (both configured off
  in production), no thin geometry beyond the 6 cm blade, no moving obstacles, no production room
  geometry. The 14 poses are calibration cases, not flights.
* `minRange`/`maxRange`/`resolution` clipping paths are **not** exercised (measured values span
  0.71–10.00 m); the no-hit path was exercised in an earlier run and mapped explicitly to the source's
  12 m contract, and the final run has zero such rays.
* Adapter statements use the exact source text (parity PASS at 8.6e-7 m), not a re-implementation; the
  source-model ray values are analytic, and are labelled as such, while every native value is measured.
* Nothing here establishes navigation success, transfer, generalization or hardware validity, and
  nothing here was fitted to any navigation metric.

## 10. Artifacts

| path | content |
|---|---|
| `results/omp-sensor-transfer/sensor-geometry/native-frames.jsonl` | 14 raw 320-pixel frames + measured poses/settings/times |
| `results/omp-sensor-transfer/sensor-geometry/summary.json` | all fits, hypothesis tables, residual statistics, mismatch counts, latency, hashes |
| `results/omp-sensor-transfer/sensor-geometry/pixels_<case>_<n>.csv` | per-pixel maps: measured axial, native-projection axial prediction, residual, source-model ray range from camera and from body, adapter range |
| `results/omp-sensor-transfer/sensor-geometry/adapter-pixels.csv` | pooled adapter replication rows |
| `results/omp-sensor-transfer/sensor-geometry/discontinuity.json` | row-by-row discontinuity profile |
| `results/omp-sensor-transfer/sensor-geometry/parity/` | extracted production source text + compiled harness + PASS receipt |
| `results/omp-sensor-transfer/sensor-geometry/analyzer-stdout.txt` | full analyzer console output |
