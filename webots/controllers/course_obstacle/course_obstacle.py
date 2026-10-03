#!/usr/bin/env python3
"""Supervisor controller that drives the moving course obstacles.

The course-world exporter writes one Robot node (DEF CourseMoverDriver,
controller "course_obstacle") whose customData lists every moving obstacle
of the bank record:

    movers=<idx>:<cx>,<cy>,<cz>:<vx>,<vy>,<vz>|<idx>:...

For each entry the target Solid (DEF ChallengeObstacle<idx>) is placed at

    translation(t) = center + velocity * t

where t is the Webots simulation time measured from world start.  This is
exactly the kinematics world.hpp wclearance() uses in the Metal sim (pure
linear motion, no wrap, no clamp; the exporter only accepts records whose
movers stay inside the room for HOLD_S = 20 s).  Static obstacles are not
listed and never move.
"""

from controller import Supervisor


def parse_movers(raw: str) -> list[tuple[str, list[float], list[float]]]:
    # Optional cold-path telemetry fields follow the mover schedule.
    raw = raw.split(";", 1)[0]
    if not raw.startswith("movers="):
        raise ValueError(f"unexpected course_mover customData: {raw[:64]!r}")
    movers = []
    for entry in raw[len("movers="):].split("|"):
        if not entry:
            continue
        idx, center_s, velocity_s = entry.split(":")
        center = [float(v) for v in center_s.split(",")]
        velocity = [float(v) for v in velocity_s.split(",")]
        if len(center) != 3 or len(velocity) != 3:
            raise ValueError(f"mover {idx} needs 3-vectors")
        movers.append((f"ChallengeObstacle{idx}", center, velocity))
    if not movers:
        raise ValueError("customData declares no movers")
    return movers


def main() -> int:
    sup = Supervisor()
    timestep = int(sup.getBasicTimeStep())
    custom_data = sup.getCustomData() or ""
    movers = parse_movers(custom_data)
    diagnostic_trace = "diagnostic_trace=1" in custom_data.split(";")[1:]
    nodes = []
    for def_name, center, velocity in movers:
        node = sup.getFromDef(def_name)
        if node is None:
            raise ValueError(f"world is missing DEF {def_name}")
        field = node.getField("translation")
        if field is None:
            raise ValueError(f"{def_name} has no translation field")
        nodes.append((def_name, node, field, center, velocity))
    # t = 0 position already equals the exporter's initial translation.
    next_sample_s = 0.0
    while sup.step(timestep) != -1:
        t = sup.getTime()
        for _, _, field, center, velocity in nodes:
            field.setSFVec3f([
                center[0] + t * velocity[0],
                center[1] + t * velocity[1],
                center[2] + t * velocity[2],
            ])
        if diagnostic_trace and t + 1e-9 >= next_sample_s:
            for def_name, node, _, center, velocity in nodes:
                actual = node.getPosition()
                expected = [center[i] + t * velocity[i] for i in range(3)]
                print(
                    "COURSE_MOVER_SAMPLE "
                    f"time_s={t:.3f} def={def_name} "
                    f"actual_xyz={actual[0]:.6f},{actual[1]:.6f},{actual[2]:.6f} "
                    f"expected_xyz={expected[0]:.6f},{expected[1]:.6f},{expected[2]:.6f}",
                    flush=True,
                )
            next_sample_s += 1.0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
