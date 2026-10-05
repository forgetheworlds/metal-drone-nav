"""Seeded compositional 3-D tasks; witnesses grade geometry, not flight feasibility."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import subprocess

from navigation_distance_tasks import read_bank, write_bank
from obstacle_motion import position

KINDS = ["staggered_gaps", "vertical_weave", "poles_overhang", "gap_crossing", "vertical_approach", "combined"]


def box(center, half_extent, velocity=(0, 0, 0)):
    return (0, center, half_extent, velocity)


def sphere(center, radius, velocity):
    return (1, center, (radius, radius, radius), velocity)


def bounded_sphere(center, radius, peak_velocity, amplitude, phase):
    return (3, center, (radius, amplitude, phase), peak_velocity)


def corridor(width):
    return [box((6, sign * (width / 2 + .06), 2.5), (8, .06, 2.5)) for sign in [-1, 1]]


def corridor_with_side_opening(width, x):
    walls = []
    for sign in [-1, 1]:
        for low, high in [(-2, x - .7), (x + .7, 14)]:
            walls.append(box(((low + high) / 2, sign * (width / 2 + .06), 2.5),
                             ((high - low) / 2, .06, 2.5)))
    return walls


def opening(x, center_y, width, corridor_width):
    walls = []
    for low, high in [(-corridor_width / 2, center_y - width / 2),
                      (center_y + width / 2, corridor_width / 2)]:
        walls.append(box((x, (low + high) / 2, 2.5), (.12, (high - low) / 2, 2.5)))
    return walls


def candidate(rng, kind, difficulty, length_range=(8.5, 11.5)):
    length = rng.uniform(*length_range)
    z = rng.uniform(1.4, 1.8)
    start, goal = (0, rng.uniform(-.2, .2), z), (length, rng.uniform(-.2, .2), z)
    width = 3.2
    crossing_x = length * .59
    obstacles = (corridor_with_side_opening(width, crossing_x)
                 if kind in ["gap_crossing", "combined"] else corridor(width))
    route = [start]
    gap_width = rng.uniform(1.15, 1.55) if difficulty == 0 else rng.uniform(.85, 1.2)
    offset = rng.uniform(.45, .65) if difficulty == 0 else rng.uniform(.65, .85)
    if kind in ["staggered_gaps", "gap_crossing", "combined"]:
        stages = [length * .32, length * .68]
        sign = rng.choice([-1, 1])
        for index, x in enumerate(stages):
            y = sign * offset * (-1 if index else 1)
            obstacles.extend(opening(x, y, gap_width, width))
            route.extend([(x - .45, y, z), (x + .45, y, z)])
    if kind in ["vertical_weave", "vertical_approach"]:
        # Alternate a floor obstruction and a hanging obstruction in one flight.
        height = rng.uniform(1.65, 1.9)
        hanging_base = rng.uniform(2.2, 2.5)
        for x, over in [(length * .32, True), (length * .68, False)]:
            if over:
                obstacles.append(box((x, 0, height / 2), (.25, width / 2, height / 2)))
                route_z = height + .55
            else:
                obstacles.append(box((x, 0, (5 + hanging_base) / 2),
                                     (.25, width / 2, (5 - hanging_base) / 2)))
                route_z = hanging_base - .55
            route.extend([(x - .6, 0, route_z), (x + .6, 0, route_z)])
    if kind == "poles_overhang":
        for index, fraction in enumerate([.25, .45, .65, .8]):
            x, y = length * fraction, rng.uniform(-.15, .15)
            radius = rng.uniform(.16, .25)
            obstacles.append((2, (x, y, 2.5), (radius, radius, 2.5), (0, 0, 0)))
            side = (-1 if index % 2 else 1) * .85
            route.extend([(x - .5, side, z), (x + .5, side, z)])
        obstacles.append(box((length * .5, 0, 3.7), (2, width / 2, .45)))
    if kind == "combined":
        # A hanging beam between offset gaps adds a vertical decision.
        x = length * .5
        obstacles.append(box((x, 0, 3.35), (.25, width / 2, 1.65)))
        route[3:3] = [(x - .6, 0, 1.05), (x + .6, 0, 1.05)]
    if kind in ["gap_crossing", "vertical_approach", "combined"]:
        speed = rng.uniform(.35, .65) if difficulty == 0 else rng.uniform(.65, 1.0)
        x = crossing_x
        phase = rng.uniform(-math.pi / 2 - .25, -math.pi / 2 + .25)
        if kind == "vertical_approach":
            # Keep the entire smooth path beyond the floor obstruction and
            # below the hanging obstruction; validate its swept sphere later.
            center_x = length * .58
            amplitude = min(rng.uniform(1.1, 1.8), center_x - length * .32 - .65)
            sphere_z = min(z, hanging_base - .38)
            obstacles.append(bounded_sphere((center_x, .55, sphere_z), .28,
                                            (-speed, 0, 0), amplitude, phase))
        else:
            direction = rng.choice([-1, 1])
            obstacles.append(bounded_sphere((x, 0, z), .28,
                                            (0, direction * speed, 0),
                                            rng.uniform(2.2, 3.4), phase))
    route.append(goal)
    assert len(obstacles) <= 16
    return start, goal, obstacles, route, gap_width


def pack_world(obstacles, goal, seed, family):
    data = bytearray(676)
    for i, (kind, center, extent, velocity) in enumerate(obstacles):
        struct.pack_into("<I9f", data, i * 40, kind, *center, *extent, *velocity)
    struct.pack_into("<3I6f", data, 640, len(obstacles), seed, family, *goal, 0, 0, 0)
    return bytes(data)


def samples(route, speed, start_wait=0.0):
    points, clock, length = [], start_wait, 0.0
    if start_wait:
        count = math.ceil(start_wait * speed / .04)
        points.extend((*route[0], start_wait * i / count) for i in range(count))
    for a, b in zip(route, route[1:]):
        distance = math.dist(a, b)
        count = max(1, math.ceil(distance / .04))
        for index in range(count):
            fraction = index / count
            points.append((*[a[j] + fraction * (b[j] - a[j]) for j in range(3)], clock + fraction * distance / speed))
        clock += distance / speed
        length += distance
    points.append((*route[-1], clock))
    return points, length, clock


def grade(checker, world, queries):
    wire = world + struct.pack("<I", len(queries)) + b"".join(struct.pack("<4f", *q) for q in queries)
    data = subprocess.run([str(checker)], input=wire, stdout=subprocess.PIPE, check=True).stdout
    assert len(data) == len(queries) * 4
    return struct.unpack("<" + "f" * len(queries), data)


def motion_position(obstacle, time):
    kind, center, size, velocity = obstacle
    return position(kind, center, size, velocity, time)


def mover_clearance(checker, obstacles, goal, seed, family):
    minimum = float("inf")
    for index, obstacle in enumerate(obstacles):
        if not any(obstacle[3]):
            continue
        assert obstacle[0] in [1, 3], "mover audit currently supports spheres"
        world = pack_world([o for i, o in enumerate(obstacles) if i != index], goal, seed, family)
        queries = [(*motion_position(obstacle, step * .02), step * .02) for step in range(1001)]
        speed = math.dist(obstacle[3], (0, 0, 0))
        other_speed = max((math.dist(o[3], (0, 0, 0)) for i, o in enumerate(obstacles) if i != index), default=0)
        # Checker subtracts drone radius .18. Replace it with mover radius and
        # bound the unsampled motion interval using its peak speed.
        bound = min(grade(checker, world, queries)) + .18 - obstacle[2][0] - (speed + other_speed) * .01
        minimum = min(minimum, bound)
    return None if minimum == float("inf") else minimum


def build(checker, seed, environments, period, length_range=(8.5, 11.5), mirror_x=False, mirror_z=False):
    rng = random.Random(seed)
    entries, labels = [], []
    for env in range(environments):
        for slot in range(period):
            kind = KINDS[(env + slot) % len(KINDS)]
            difficulty = (env // len(KINDS) + slot) % 2
            for attempt in range(100):
                start, goal, obstacles, route, gap_width = candidate(rng, kind, difficulty, length_range)
                if mirror_x or mirror_z:
                    def reflect(point):
                        return (12 - point[0] if mirror_x else point[0], point[1],
                                5 - point[2] if mirror_z else point[2])
                    def reflect_velocity(velocity):
                        return (-velocity[0] if mirror_x else velocity[0], velocity[1],
                                -velocity[2] if mirror_z else velocity[2])
                    start, goal = reflect(start), reflect(goal)
                    route = [reflect(point) for point in route]
                    obstacles = [(shape, reflect(center), size, reflect_velocity(velocity))
                                 for shape, center, size, velocity in obstacles]
                scene_seed = rng.getrandbits(32)
                family = 14 if kind == "staggered_gaps" else 16 if kind.startswith("vertical") else 1
                world = pack_world(obstacles, goal, scene_seed, family)
                own_clearance = mover_clearance(checker, obstacles, goal, scene_seed, family)
                if own_clearance is not None and own_clearance <= .02:
                    continue
                moving = any(any(o[3]) for o in obstacles)
                direct, _, _ = samples([start, goal], 1.1)
                mover_speed = max(math.dist(o[3], (0, 0, 0)) for o in obstacles)
                valid_route = False
                for start_wait in ([0, .75, 1.5, 2.5, 4] if moving else [0]):
                    witness, route_length, duration = samples(route, 1.1, start_wait)
                    queries = [(*start, 0), (*goal, duration)] + witness + direct
                    values = grade(checker, world, queries)
                    lower_bound = min(values[2:2 + len(witness)]) - .02 * (1 + mover_speed / 1.1)
                    if min(values[:2]) > .35 and lower_bound > .06 and duration < 17:
                        valid_route = True
                        break
                if valid_route:
                    break
            else:
                raise RuntimeError(f"No geometric witness for {kind} env{env} slot{slot}")
            entry = bytearray(756)
            entry[:676] = world
            velocity = tuple(rng.uniform(-.35, .35) for _ in range(3))
            yaw = rng.uniform(-math.pi, math.pi)
            distance = math.dist(start, goal)
            direct_clearance = min(values[2 + len(witness):]) - .02 * (1 + mover_speed / 1.1)
            route_class = int(direct_clearance <= .02)
            struct.pack_into("<10f6f4I", entry, 676, *start, *velocity, *goal, yaw,
                             values[0], values[1], direct_clearance, lower_bound, route_length, distance,
                             family, scene_seed, route_class, attempt + 1)
            entries.append(bytes(entry))
            labels.append({"env": env, "slot": slot, "kind": kind, "difficulty": difficulty,
                           "scene_seed": scene_seed, "initial_distance_m": distance,
                           "gap_width_m": gap_width, "obstacle_count": len(obstacles),
                           "moving_count": sum(any(o[3]) for o in obstacles),
                           "mover_own_clearance_lower_bound_m": own_clearance,
                           "witness_route": route, "witness_duration_s_at_1p1_mps": duration,
                           "witness_initial_wait_s": start_wait,
                           "witness_body_clearance_lower_bound_m": lower_bound,
                           "witness_kind": "scheduled geometric path, not RAPTOR flight",
                           "direct_body_clearance_lower_bound_m": direct_clearance,
                           "route_class": route_class, "generation_attempts": attempt + 1})
    return entries, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checker", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--period", type=int, default=1)
    parser.add_argument("--rehearsal", type=Path)
    parser.add_argument("--length-min", type=float, default=8.5)
    parser.add_argument("--length-max", type=float, default=11.5)
    parser.add_argument("--mirror-x", action="store_true")
    parser.add_argument("--mirror-z", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert 5.5 <= args.length_min <= args.length_max <= 12.5
    entries, labels = build(args.checker.resolve(), args.seed, 128, args.period,
                           (args.length_min, args.length_max), args.mirror_x, args.mirror_z)
    write_bank(args.output / "challenges.bin", args.period, entries)
    if args.rehearsal:
        old_period, old = read_bank(args.rehearsal)
        assert len(old) == 128 * old_period
        mixed = []
        for env in range(128):
            mixed.extend(old[env * old_period:(env + 1) * old_period])
            mixed.extend(entries[env * args.period:(env + 1) * args.period])
        write_bank(args.output / "mixed.bin", old_period + args.period, mixed)
    report = {"schema": "compositional-v2-bounded-motion", "seed": args.seed, "period": args.period, "count": len(entries),
              "length_range_m": [args.length_min, args.length_max],
              "reflection_x": args.mirror_x, "reflection_z": args.mirror_z,
              "entry_sha256": hashlib.sha256(b"".join(entries)).hexdigest(),
              "checker_sha256": hashlib.sha256(args.checker.read_bytes()).hexdigest(),
              "geometry_source_sha256": hashlib.sha256(Path(__file__).with_name("world.hpp").read_bytes()).hexdigest(),
              "sensor_stress": "nominal; separate axis", "dynamics_stress": "nominal; separate axis",
              "records": labels}
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"count": len(entries), "period": args.period,
                      "kinds": {kind: sum(x["kind"] == kind for x in labels) for kind in KINDS},
                      "max_attempts": max(x["generation_attempts"] for x in labels)}))


if __name__ == "__main__":
    main()
