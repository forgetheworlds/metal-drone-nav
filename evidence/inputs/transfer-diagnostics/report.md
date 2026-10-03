# Focused Webots transfer diagnostics

The audited family-15 DEV matrix had 18/30 Webots successes and 12 contacts; all 12 contacts were in the hard band. The focused Metal evaluation had 25/30 successes. This diagnostic replays two Metal-success/Webots-contact cases and one common hard success. It also tests one explicit post-policy action scale on the two contact cases. These five runs do not explain every matrix failure.

## Decision

The selected contact points match the declared 0.18 m collision sphere against the saved scene geometry. The evidence points to navigation behavior, not a collision-envelope discrepancy. The two sampled failures have different proximate modes:

- At the first offset doorway, the policy sees a very close wall and still reaches the panel edge before it has moved far enough into the opening. Half-scale commands make the approach slower, but the route still reaches a solid part of the wall. Training needs earlier lateral alignment and enough clearance margin before forward motion.
- At the second sampled scene, the vehicle reaches the floor with a 3.10 m/s world speed and about 17.3 rad/s body-rate magnitude. A half-scale diagnostic prevents this floor impact and keeps the vehicle above 1.10 m, but it later hits the second doorway panel. The lower command scale reduces this instability; it does not solve the route.

This supports a focused training target: learn to center offset doorways before crossing, and limit motion when the RAPTOR state is already far from its target. It does not support replacing the collision model or applying a global half-speed setting as a complete fix.

## Contact evidence

For `f15-dev-0004-sd401bbed-w5e651fdf`, Webots reports a contact at step 196, 1.97 s, at `(1.24593, -0.51564, 1.17027)` m. The body center is `(1.07796, -0.57945, 1.17027)` m. The scene metadata places the wall front at `x=1.24593` m and the doorway's right panel edge at `y=-0.51564` m. The center-to-edge distance is about 0.180 m, matching the physical sphere. The allowable center interval for that opening is approximately `y=[-1.618,-0.696]` m after the 0.18 m margin; the vehicle center is at `y=-0.579` m at contact.

At the 1.95 s navigation update, two 2×2-pooled rays report a minimum range of 0.116 m. The actor requests body velocity `(0.997, -0.516, 0.125)` m/s. Contact follows 20 ms later. The minimum pooled range first drops below 0.18 m at 1.85 s, 120 ms before contact. The actor's lateral command points toward the opening, but it does not reach the required centerline before the vehicle reaches the wall.

The half-scale replay contacts the same first partition at 5.41 s. Its point `(1.24593,-0.00158,1.79134)` m lies on the solid panel. Peak speed falls from 1.497 to 1.085 m/s. This shows that slower commands alone do not provide the required lateral route.

For `f15-dev-0001-s6958c653-w5e651fdf-mirror-y`, Webots reports a floor contact at step 655, 6.56 s, at `(5.48108, 0.57308, 0)` m. The body center is at `z=0.179996` m, again matching the 0.18 m collision sphere. Body-rate magnitude first exceeds 5 rad/s at 3.27 s, then altitude falls below 0.5 m at 5.31 s. At contact, world speed is 3.098 m/s; body rates are `(1.02,-13.34,11.04)` rad/s; the difference between measured world velocity and the current target is about 3.65 m/s. The nearest 20 Hz policy frame has a minimum pooled range of 1.510 m and target body velocity `(1.235,0.173,-0.795)` m/s. This is a high-rate tracking failure, not contact with a nearby wall.

With post-policy velocity and yaw scaled to 0.5, this case does not exceed 5 rad/s until 12.20 s, does not fall below 1.10 m, and has a peak speed of 1.351 m/s. It later contacts the solid face of the second partition at 12.40 s. This intervention supports command amplitude as a cause of the severe rate/altitude excursion in this case, but the route still fails.

The common-success control `f15-dev-0014-scd18eac9-w5e651fdf` replays successfully at 10.51 s. The early startup sink is shared: each baseline replay has `z=1.49212 m` and `vz=-0.200758 m/s` at 0.05 s. That transient alone does not explain the selected failures.

## Reproduction and limits

Run the baseline diagnostics with:

```sh
python3 webots/transfer_diagnostics.py
```

Run the explicit action-scale diagnostic with:

```sh
python3 webots/transfer_diagnostics.py --roles first_partition_contact,second_partition_floor_contact --nav-scale 0.5
```

Both use the saved DEV scenes, policy `assets/navigation-rooms-experimental.bin`, 1 ms ODE step, average actuator sampling, 100 Hz RAPTOR, 20 Hz navigation, the existing 0.18 m sphere, and the original 2000-step limit. They use no Supervisor geometry or pose as actor input. The action-scale run changes only the output body-velocity and yaw-rate scale after actor inference; it is a diagnostic variant, not a candidate deployment.

The trace files store 100 Hz position, velocity, quaternion, rates, target, motor and goal state; 20 Hz current/previous pooled ranges, guidance prior, previous intent and actor action; and exact native contact points. Webots returns a node id but no DEF name for these contacts. The scene AABBs and contact coordinates identify the contacted surfaces. The result is limited to two failures, one control, and two action-scale variants. No final split was read, and no training was run.

The saved Metal CSV contains per-episode results but no matching per-step state, pooled observation and action trace. A causal Metal/Webots trajectory comparison needs a cold CLI that accepts one saved DEV scene and deployed actor, then exports synchronized 100 Hz world state/target/motor samples and 20 Hz actor observation/action samples with step number and simulated time. Do not pass Webots contact or scene geometry into the Metal actor.
