# RAPTOR Webots transfer check

This is an independent Webots R2025a simulation. Webots ODE advances the vehicle and resolves contacts. The controller runs the frozen RAPTOR policy at 100 Hz and the exported navigation policy at 20 Hz. The navigation policy reads the RangeFinder image, ego sensors, and its saved local depth history. A Supervisor contact query scores collisions; obstacle positions do not enter the actor input.

The vehicle uses the pinned Crazyflie mass and diagonal inertia from `physics.hpp`. Its normalized actuator command goes through the L2F thrust polynomial and rise/fall motor lag. The controller converts requested thrust to signed propeller angular speed. Webots uses `T = t1 * |omega| * omega - t2 * |omega| * V`; its reaction torque is `-Q` about the shaft axis. The PROTO sets alternating thrust signs and negative `q1` so both thrust and reaction torque match the L2F rotor directions. The large motor `maxTorque` avoids Webots adding a second motor acceleration ramp. It is a kinematic limit; the controller retains the measured L2F motor lag.

The L2F reference initializes rotor state at computed hover throttle. Webots starts its internal propeller velocity at zero. The first motor command starts the rotor spin-up, so the transfer runs include a startup transient. The six-route results below use this state. They were not changed to hide the transient.

The 20×16 RangeFinder reports axial depth. Plane and marker tests set its optical axis to body +X, image left to body +Y, and image up to body +Z. Its origin is 8 cm ahead of the body center to clear the body collision shape. The pose history stores this camera origin; the current navigation query stays at the body center. The controller resamples the native vertical slope 0.8 to the policy slope 0.75, then applies the 2×2 min pool. Two center-pixel plane tests matched measured depth within 0.5 mm after accounting for the camera pose and the off-center pixel ray. No near-clip offset is applied. The calibration readings are in `results/range-calibration.json`.

On the first 50 ms capture, the current and previous range features both use that same frame. This matches the reference reset state. Later captures keep the preceding frame in the history slot.

The collision body is a conservative 0.18 m sphere. It matches the policy simulator's swept collision footprint and does not change the visible airframe or explicit mass/inertia. The FLU camera axis, world position, body rotation, velocity, and rates are recorded from Webots sensors. GPS, InertialUnit, and Gyro use ideal Webots outputs in this first transfer check. They are not noisy flight estimates.

Build the native controller with the installed Webots R2025a toolchain:

```sh
WEBOTS_HOME=/Users/muadhsambul/embodied/work/Webots.app \
  make -C webots/controllers/raptor_webots
```

Run one route in batch mode:

```sh
/Users/muadhsambul/embodied/work/Webots.app/Contents/MacOS/webots \
  --minimize --batch --mode=fast --no-rendering --stdout --stderr \
  webots/worlds/a_to_b.wbt
```

Run a reproducible matrix and preserve per-run logs, episode JSON, and pose traces:

```sh
python3 webots/benchmark.py \
  --worlds a_to_b,a_to_b_offset_box \
  --seeds 1,2,3 \
  --policies ../assets/navigation.bin
```

Generate and run the independent 3D challenge set with both the original and static policies:

```sh
python3 webots/challenge_generator.py --run --port 23456
```

This writes 18 scenes: six offset doorways, six low table overhangs, and six mixed pole/box layouts. A separate 20 cm voxel search checks that a sphere with the 0.18 m body radius plus 0.04 m planning margin has a valid start-to-goal path. The witness is saved as metadata for evaluation only. It is not sent to the actor. Each episode saves the exact Webots route, its geometry metadata, actor/controller/PROTO/world hashes, the contact result, and a raw pose trace. The challenge runner uses a separate Webots port so it cannot attach to another project's live Webots app. It also writes a top-down GPS-trace reconstruction SVG. The SVG is labeled as a reconstruction, not a simulator screenshot.

The runner writes `webots/results/benchmark.csv` and one folder per world, policy, and seed. The controller flushes `last-run.json`, `last-run-exit.marker`, and `last-run-trace.csv` before it asks Webots to stop. The marker files are authoritative because batch shutdown can cut off the last controller stdout lines.

`a_to_b.wbt` is the empty-room baseline. `a_to_b_offset_box.wbt` has a real static ODE box across the straight route. The benchmark assigns a deterministic lateral offset to this box for each seed. The uncalibrated camera runs are preserved under `results/pre-camera-axis-calibration/` and do not measure policy transfer.

After camera and collision-envelope calibration, `navigation.bin` completed all six saved routes: three empty-room seeds and three offset-box placements. The empty-room routes took 2.79 s. The box offsets were −0.402, +0.502, and −0.288 m. Those routes took 2.92–5.98 s. All six had no Supervisor contact and final error below 0.35 m. `benchmark.csv` records the metrics. Each per-run folder retains the exact `route.wbt`, episode JSON, contact marker, raw pose trace, and Webots log. This is a small transfer check across one obstacle family. It does not establish broad generalization.

The frozen RAPTOR hover diagnostic ran for 8 s with no contact. It settled at 1.448 m from a 1.5 m start and reached 0.759 m/s during startup. The 1 m/s world-velocity reference reached the 4 m goal in 3.65 s. Its whole-episode velocity tracking RMS was 0.243 m/s, including startup; measured forward speed was about 1 m/s by 2 s. These traces are in `results/diagnostics/`.

The first cylinder-based matrix used the wrong primitive axis and is archived under `results/pre-cylinder-axis-correction/`. It does not establish the intended vertical-pole task results. The corrected generator uses Webots local Z as the cylinder axis; Supervisor checks on a table and mixed scene confirm the actual axes are world Z. After rerunning the24 affected episodes, both policies complete6/6 tables; mixed clutter remains weak (original1/6, static2/6). Doorway results from the unaffected boxes remain5/6 and6/6. These are development cases with ideal ego sensors and a conservative18cm collision sphere. They do not establish reliable broad transfer or actual-airframe collision accuracy.

All batch launches use `--minimize --batch` and the RL-only port23456. The runtime also provides a `physics-audit` phase: it warms RAPTOR hover, applies small common motor-command pulses, and compares measured ODE states against a free-running CPU L2F reference. This calibrates numerical/model agreement; it is not hardware validation.

Primary Webots R2025a references: [Propeller](https://github.com/cyberbotics/webots/blob/R2025a/docs/reference/propeller.md), [Motor](https://github.com/cyberbotics/webots/blob/R2025a/docs/reference/motor.md), [RangeFinder](https://github.com/cyberbotics/webots/blob/R2025a/docs/reference/rangefinder.md), [Physics](https://github.com/cyberbotics/webots/blob/R2025a/docs/reference/physics.md), [Supervisor](https://github.com/cyberbotics/webots/blob/R2025a/docs/reference/supervisor.md), [InertialUnit](https://github.com/cyberbotics/webots/blob/R2025a/docs/reference/inertialunit.md).
