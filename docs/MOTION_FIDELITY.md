# Moving obstacles that respect the environment

I missed a validation condition in the first compositional generator. It
checked a clear path for the drone but allowed the moving spheres to pass
through corridor walls, a hanging beam, or the room boundary. An independent
audit of all 384 moving TRAIN records found an intersection in every record
within the 20 s window. I stopped that unfinished training batch and preserved
its first completed control, partial treatment and original logs.

The corrected generator uses smooth, bounded sphere trajectories and actual
side openings for crossing threats. Approaching spheres stay beyond the floor
obstruction and below the hanging obstruction. It checks each mover's full
swept sphere against the other geometry and the room for all 20 seconds,
including a bound between sampled times. The corrected 384 moving records have
a minimum certified margin of 0.111 m. Drone-path validation remains separate.

Kind 3 extends the existing world ABI without changing its size. Its fields
are radius, travel amplitude and phase, with a direction/peak-speed vector.
The center follows a sine trajectory. That gives finite acceleration and a
bounded path instead of motion that continues through a wall. Legacy shapes
and linear motion retain their old semantics.

The [shared source geometry](../world.hpp) and
[motion module](../obstacle_motion.py) implement the same trajectory in Metal,
CPU checks and the independent Webots Supervisor. CPU/Metal position, depth and
collision tests covered 1,001 times through 20 seconds; maximum difference was
4.77e-7. The default path reproduced a complete 100-rollout checkpoint byte for
byte, including optimizer and simulator state.

![Corrected bounded-motion worlds](../artifacts/plots/bounded-composite-worlds.png)

*Declared primitives at five seconds. Side openings let crossing spheres enter
the corridor. This geometry rendering contains no flown trajectory.*

A frozen anchored policy then ran the first crossing record in actual Webots.
It reached the goal and held successfully in 10.6 s, without contact. Navigation
and RAPTOR both loaded; 212 actual mover pose samples matched the declared
trajectory within 0.000431 m. This is a load, motion and full-stack flight smoke,
not a broad transfer benchmark. An earlier attempt had a Python tuple/list
field error and mismatched receipt slug; it remains invalid and preserved.

On the corrected 128-task source development panel, BC reaches 92 goals; the
two anchored parents reach 113 and 102. None were trained on this bank. The
next matched training comparison uses these parents and fixed transition
shares: control 50% static / 50% long; treatment 50% static / 25% long / 25%
compositional. Groups are interleaved across environments, and every active
environment contributes a transition each tick, so episode duration cannot
change those shares. All original and new records remain in the treatment.

The corrected source runs are active. The original prototype results are not
silently replaced. Independent transfer, harder motion, sensor errors and
dynamics stress remain subsequent tests of the combined policy.
