#!/usr/bin/env python3
"""Create independent, seeded Webots challenge geometry with a checked route witness."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORLDS = ROOT / "worlds"
RESULTS = ROOT / "results"
BASE_WORLD = WORLDS / "a_to_b.wbt"
BOUNDED_WORLD = WORLDS / "a_to_b_bounded.wbt"
BODY_RADIUS_M = 0.18
PLANNER_MARGIN_M = 0.04
GRID_STEP_M = 0.2
START = (0.0, 0.0, 1.5)
GOAL = (4.0, 0.0, 1.5)


def box(center: tuple[float, float, float], size: tuple[float, float, float], name: str) -> dict:
    return {"kind": "box", "name": name, "center": list(center), "size": list(size)}


def pole(x: float, y: float, radius: float, height: float, name: str) -> dict:
    return {"kind": "cylinder", "name": name, "center": [x, y, height / 2.0], "radius": radius, "height": height}


def generate_doorway(seed: int) -> tuple[list[dict], dict]:
    rng = random.Random(seed ^ 0xD00A)
    x = round(rng.uniform(1.9, 2.2), 3)
    direction = rng.choice((-1, 1))
    cy = round(direction * rng.uniform(0.55, 0.95), 3)
    cz = round(rng.uniform(1.38, 1.62), 3)
    width = round(rng.uniform(1.2, 1.35), 3)
    height = round(rng.uniform(1.65, 1.85), 3)
    thick, y0, y1, z0, z1 = 0.28, -4.5, 4.5, 0.15, 3.8
    lo_y, hi_y = cy - width / 2, cy + width / 2
    lo_z, hi_z = cz - height / 2, cz + height / 2
    obstacles = []
    if lo_y > y0:
        obstacles.append(box((x, (y0 + lo_y) / 2, (z0 + z1) / 2), (thick, lo_y - y0, z1 - z0), "door_left"))
    if y1 > hi_y:
        obstacles.append(box((x, (hi_y + y1) / 2, (z0 + z1) / 2), (thick, y1 - hi_y, z1 - z0), "door_right"))
    if lo_z > z0:
        obstacles.append(box((x, cy, (z0 + lo_z) / 2), (thick, width, lo_z - z0), "door_sill"))
    if z1 > hi_z:
        obstacles.append(box((x, cy, (hi_z + z1) / 2), (thick, width, z1 - hi_z), "door_header"))
    return obstacles, {"wall_x_m": x, "opening_center_y_m": cy, "opening_center_z_m": cz,
                       "opening_width_m": width, "opening_height_m": height}


def generate_table(seed: int) -> tuple[list[dict], dict]:
    rng = random.Random(seed ^ 0x7AB1E)
    x, y = 2.0, round(rng.uniform(-0.35, 0.35), 3)
    bottom = round(rng.uniform(1.62, 1.72), 3)
    tabletop = box((x, y, bottom + 0.09), (1.35, 2.0, 0.18), "table_top")
    obstacles = [tabletop]
    leg_radius, leg_height = 0.12, bottom
    for i, dx in enumerate((-0.52, 0.52)):
        for j, dy in enumerate((-0.78, 0.78)):
            obstacles.append(pole(x + dx, y + dy, leg_radius, leg_height, f"table_leg_{i}_{j}"))
    return obstacles, {"table_center_x_m": x, "table_center_y_m": y,
                       "table_underside_z_m": bottom, "tabletop_size_m": [1.35, 2.0, 0.18],
                       "leg_radius_m": leg_radius}


def generate_mixed(seed: int) -> tuple[list[dict], dict]:
    rng = random.Random(seed ^ 0xC17E2)
    side = rng.choice((-1, 1))
    lateral = round(rng.uniform(0.45, 0.7), 3)
    box_offset = round(rng.uniform(1.05, 1.4), 3)
    first_y = round(rng.uniform(-0.12, 0.12), 3)
    obstacles = [
        pole(1.2, first_y, 0.2, 2.5, "pole_a"),
        pole(2.0, -side * lateral, 0.18, 2.8, "pole_b"),
        pole(2.85, side * lateral, 0.18, 2.5, "pole_c"),
        box((1.75, -side * box_offset, 0.95), (0.55, 0.7, 1.7), "box_low"),
        box((2.8, side * box_offset, 2.15), (0.65, 0.55, 1.25), "box_overhang"),
    ]
    return obstacles, {"alternating_side": side, "first_pole_y_m": first_y, "pole_lateral_m": lateral,
                       "box_lateral_m": box_offset, "pole_radii_m": [0.2, 0.18, 0.18]}


def intersects_sphere(point: tuple[float, float, float], obstacle: dict, radius: float) -> bool:
    x, y, z = point
    if obstacle["kind"] == "box":
        center = obstacle["center"]
        size = obstacle["size"]
        dx = max(abs(x - center[0]) - size[0] / 2, 0.0)
        dy = max(abs(y - center[1]) - size[1] / 2, 0.0)
        dz = max(abs(z - center[2]) - size[2] / 2, 0.0)
        return dx * dx + dy * dy + dz * dz < radius * radius
    cx, cy, cz = obstacle["center"]
    radial = max(math.hypot(x - cx, y - cy) - obstacle["radius"], 0.0)
    zmin, zmax = cz - obstacle["height"] / 2, cz + obstacle["height"] / 2
    vertical = max(zmin - z, 0.0, z - zmax)
    return radial * radial + vertical * vertical < radius * radius


def route_witness(obstacles: list[dict]) -> list[list[float]]:
    """A 20 cm 6-connected voxel search proves a collision-free center route."""
    x0, y0, z0 = -0.4, -4.0, 0.3
    nx, ny, nz = 25, 41, 21
    radius = BODY_RADIUS_M + PLANNER_MARGIN_M
    points = [
        (round(x0 + GRID_STEP_M * i, 3), round(y0 + GRID_STEP_M * j, 3), round(z0 + GRID_STEP_M * k, 3))
        for i in range(nx) for j in range(ny) for k in range(nz)
    ]
    def index(i: int, j: int, k: int) -> int:
        return (i * ny + j) * nz + k
    start = (2, 20, 6)
    goal = (22, 20, 6)
    if any(intersects_sphere(START, ob, radius) or intersects_sphere(GOAL, ob, radius) for ob in obstacles):
        raise ValueError("challenge geometry blocks the start or goal")
    free = bytearray(nx * ny * nz)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                p = points[index(i, j, k)]
                free[index(i, j, k)] = not any(intersects_sphere(p, ob, radius) for ob in obstacles)
    start_i, goal_i = index(*start), index(*goal)
    if not free[start_i] or not free[goal_i]:
        raise ValueError("voxelized start or goal is blocked")
    predecessor = {start_i: -1}
    queue = collections.deque([start_i])
    while queue and goal_i not in predecessor:
        current = queue.popleft()
        i, rem = divmod(current, ny * nz)
        j, k = divmod(rem, nz)
        for di, dj, dk in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            a, b, c = i + di, j + dj, k + dk
            if 0 <= a < nx and 0 <= b < ny and 0 <= c < nz:
                nxt = index(a, b, c)
                if free[nxt] and nxt not in predecessor:
                    predecessor[nxt] = current
                    queue.append(nxt)
    if goal_i not in predecessor:
        raise ValueError("no valid start-to-goal route for generated geometry")
    path = []
    current = goal_i
    while current >= 0:
        path.append(points[current])
        current = predecessor[current]
    path.reverse()
    return path


def world_nodes(obstacles: list[dict]) -> str:
    nodes = []
    for i, ob in enumerate(obstacles):
        name = f"ChallengeObstacle{i:02d}"
        if ob["kind"] == "box":
            cx, cy, cz = ob["center"]
            sx, sy, sz = ob["size"]
            nodes.append(
                f'DEF {name} Solid {{ name "{ob["name"]}" translation {cx} {cy} {cz} '
                f'children [ Shape {{ appearance PBRAppearance {{ baseColor 0.63 0.34 0.18 roughness 1 }} '
                f'geometry Box {{ size {sx} {sy} {sz} }} }} ] '
                f'boundingObject Box {{ size {sx} {sy} {sz} }} locked TRUE }}'
            )
        else:
            cx, cy, cz = ob["center"]
            r, h = ob["radius"], ob["height"]
            nodes.append(
                # Webots Cylinder's native symmetry axis is local Z, the ENU up axis.
                f'DEF {name} Solid {{ name "{ob["name"]}" translation {cx} {cy} {cz} '
                f'children [ Shape {{ appearance PBRAppearance {{ baseColor 0.25 0.36 0.52 roughness 1 }} '
                f'geometry Cylinder {{ radius {r} height {h} }} }} ] '
                f'boundingObject Cylinder {{ radius {r} height {h} }} locked TRUE }}'
            )
    return "\n".join(nodes)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_run_manifest(policy: str, world_name: str, seed: int, physics_step_ms: int,
                      motor_sampling: str, physics_profile: str) -> None:
    run_dir = RESULTS / Path(policy).stem / f"{world_name}-seed-{seed}"
    actor = (ROOT / policy).resolve()
    if not actor.is_file():
        raise FileNotFoundError(f"actor asset missing: {actor}")
    raptor = ROOT.parent / "assets" / "raptor.bin"
    if not raptor.is_file():
        raptor = ROOT / "assets" / "raptor.bin"
    metadata_path = RESULTS / "challenges" / f"{world_name}.json"
    files = {
        "route_world": run_dir / "route.wbt",
        "generator_metadata": metadata_path,
        "navigation_actor": actor,
        "raptor_actor": raptor,
        "controller_source": ROOT / "controllers/raptor_webots/raptor_webots.cpp",
        "controller_binary": ROOT / "controllers/raptor_webots/raptor_webots",
        "vehicle_proto": ROOT / "protos/RaptorCrazyflie.proto",
        "range_calibration": RESULTS / "range-calibration.json",
    }
    manifest = {
        "schema": "webots-challenge-run-v1",
        "webots_release": "R2025a",
        "policy": Path(policy).name,
        "family": json.loads(metadata_path.read_text())["family"],
        "seed": seed,
        "physics_basic_time_step_ms": physics_step_ms,
        "raptor_control_period_ms": 10,
        "navigation_period_ms": 50,
        "motor_sampling": motor_sampling,
        "physics_profile": physics_profile,
        "files_sha256": {name: sha256(path) for name, path in files.items()},
        "collision_radius_m": 0.18,
        "range_noise_stddev_m": 0.0,
        "ego_sensors": "ideal Webots GPS, InertialUnit, and Gyro; no estimator noise",
        "startup_condition": "The reference L2F state starts at hover rpm; Webots motor internal angular velocity starts at zero and spins up from the first command.",
        "actor_inputs": "Depth and ego state only. Obstacle geometry and route-witness metadata are not passed in Robot.customData or observations.",
        "real_flight_claim": False,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def render_trace_reconstruction(world_name: str, policy: str, seed: int) -> Path | None:
    """Draw a labeled top-down reconstruction from Webots GPS samples and world geometry."""
    run_dir = RESULTS / Path(policy).stem / f"{world_name}-seed-{seed}"
    trace_path = run_dir / "trace.csv"
    metadata_path = RESULTS / "challenges" / f"{world_name}.json"
    episode_path = run_dir / "episode.json"
    if not (trace_path.is_file() and metadata_path.is_file() and episode_path.is_file()):
        return None
    with trace_path.open(newline="") as source:
        samples = list(csv.DictReader(source))
    if len(samples) < 2:
        return None
    metadata = json.loads(metadata_path.read_text())
    episode = json.loads(episode_path.read_text())
    width, height = 760, 650
    xmin, xmax, ymin, ymax = -0.5, 4.5, -2.5, 2.5
    scale, left, top = 90.0, 155.0, 85.0
    def px(x: float) -> float:
        return left + (x - xmin) * scale
    def py(y: float) -> float:
        return top + (ymax - y) * scale
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f4f1e9"/>',
        '<text x="40" y="34" font-family="sans-serif" font-size="20" font-weight="700">Webots GPS trace reconstruction</text>',
        '<text x="40" y="58" font-family="sans-serif" font-size="12" fill="#555">Top-down reconstruction from saved GPS samples and world geometry; not a simulator screenshot.</text>',
        f'<rect x="{px(xmin)}" y="{py(ymax)}" width="{(xmax-xmin)*scale}" height="{(ymax-ymin)*scale}" fill="#e9e7df" stroke="#767676"/>',
    ]
    for obstacle in metadata["obstacles"]:
        if obstacle["kind"] == "box":
            cx, cy = obstacle["center"][:2]
            sx, sy = obstacle["size"][:2]
            x1, x2 = max(cx-sx/2, xmin), min(cx+sx/2, xmax)
            y1, y2 = max(cy-sy/2, ymin), min(cy+sy/2, ymax)
            if x2 > x1 and y2 > y1:
                parts.append(f'<rect x="{px(x1):.1f}" y="{py(y2):.1f}" width="{(x2-x1)*scale:.1f}" height="{(y2-y1)*scale:.1f}" fill="#9b6046" stroke="#563c32"/>')
        else:
            cx, cy = obstacle["center"][:2]
            radius = obstacle["radius"]
            parts.append(f'<circle cx="{px(cx):.1f}" cy="{py(cy):.1f}" r="{radius*scale:.1f}" fill="#657b9b" stroke="#283951"/>')
    points = " ".join(f"{px(float(row['x'])):.1f},{py(float(row['y'])):.1f}" for row in samples)
    parts.append(f'<polyline points="{points}" fill="none" stroke="#147d92" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>')
    start_x, start_y = float(samples[0]["x"]), float(samples[0]["y"])
    end_x, end_y = float(samples[-1]["x"]), float(samples[-1]["y"])
    parts.append(f'<circle cx="{px(start_x):.1f}" cy="{py(start_y):.1f}" r="7" fill="#2d8c57"/>')
    parts.append(f'<circle cx="{px(GOAL[0]):.1f}" cy="{py(GOAL[1]):.1f}" r="8" fill="#bd4a3b"/>')
    parts.append(f'<circle cx="{px(end_x):.1f}" cy="{py(end_y):.1f}" r="5" fill="#147d92" stroke="white"/>')
    parts.append('<text x="40" y="590" font-family="sans-serif" font-size="13">Green: start  •  Red: goal  •  Blue: recorded GPS samples  •  Brown/gray: Webots obstacle projection</text>')
    parts.append(f'<text x="40" y="615" font-family="sans-serif" font-size="13">Episode: success={episode["success"]}, time={episode["time_s"]:.2f}s, final error={episode["final_error_m"]:.3f}m. Trace shown through {float(samples[-1]["time_s"]):.2f}s.</text>')
    parts.append('</svg>')
    output = RESULTS / f"{world_name}-seed-{seed}-reconstruction.svg"
    output.write_text("\n".join(parts) + "\n")
    return output


def create_case(family: str, seed: int, write_world: bool = True, bounded_room: bool = False) -> tuple[str, dict]:
    if family == "doorway":
        obstacles, parameters = generate_doorway(seed)
    elif family == "table_overhang":
        obstacles, parameters = generate_table(seed)
    elif family == "mixed_clutter":
        obstacles, parameters = generate_mixed(seed)
    else:
        raise ValueError(f"unknown family: {family}")
    witness = route_witness(obstacles)
    name = f"challenge_{family}_{seed}{'_bounded' if bounded_room else ''}"
    source = (BOUNDED_WORLD if bounded_room else BASE_WORLD).read_text()
    source = source.replace("# WEBOTS_OBSTACLES_INSERTION_POINT", world_nodes(obstacles), 1)
    source = re.sub(r'title "[^"]*"', f'title "{name}"', source, count=1)
    source = re.sub(r'customData "[^"]*"',
                    f'customData "phase=navigation;seed={seed};policy=../assets/navigation.bin;speed=1.5;distance=4;goal=4,0,1.5;max_steps=800"',
                    source, count=1)
    world_path = WORLDS / f"{name}.wbt"
    if write_world:
        world_path.write_text(source)
    witness_length = sum(math.dist(a, b) for a, b in zip(witness, witness[1:]))
    metadata = {
        "schema": "webots-challenge-v1",
        "family": family,
        "seed": seed,
        "bounded_room": bounded_room,
        "start_xyz_m": list(START),
        "goal_xyz_m": list(GOAL),
        "body_collision_radius_m": BODY_RADIUS_M,
        "planner_clearance_m": PLANNER_MARGIN_M,
        "generator_parameters": parameters,
        "obstacles": obstacles,
        "route_witness": {"grid_step_m": GRID_STEP_M, "waypoint_count": len(witness),
                          "length_m": round(witness_length, 3), "points_xyz_m": witness},
        "actor_observation_policy": "No obstacle geometry or witness points are provided in Robot.customData or policy inputs.",
    }
    if bounded_room:
        metadata["room_bounds_m"] = {"x": [-2, 14], "y": [-5, 5], "z": [0, 5]}
    meta_path = RESULTS / "challenges" / f"{name}.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return name, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="41001,41002,41003,41004,41005,41006")
    parser.add_argument("--families", default="doorway,table_overhang,mixed_clutter")
    parser.add_argument("--policies", default="../assets/navigation.bin,../assets/navigation-static.bin")
    parser.add_argument("--run", action="store_true", help="run generated worlds with Webots")
    parser.add_argument("--summary-only", action="store_true", help="summarize saved episodes without launching Webots")
    parser.add_argument("--bounded-room", action="store_true", help="use walls at the exact Metal bounds [-2,14] x [-5,5] x [0,5]")
    parser.add_argument("--webots", type=Path, default=Path("/Users/muadhsambul/embodied/work/Webots.app/Contents/MacOS/webots"))
    parser.add_argument("--port", type=int, default=23456, help="isolated Webots controller port")
    parser.add_argument("--physics-step-ms", type=int, default=1)
    parser.add_argument("--motor-sampling", choices=("end", "average"), default="average")
    parser.add_argument("--physics-profile", default="hover")
    args = parser.parse_args()
    cases = []
    for family in args.families.split(","):
        for seed_text in args.seeds.split(","):
            seed = int(seed_text)
            name = f"challenge_{family}_{seed}{'_bounded' if args.bounded_room else ''}"
            metadata_path = RESULTS / "challenges" / f"{name}.json"
            if args.summary_only and metadata_path.is_file():
                cases.append((name, json.loads(metadata_path.read_text())))
            else:
                cases.append(create_case(family, seed, write_world=not args.summary_only,bounded_room=args.bounded_room))
    print(f"generated {len(cases)} cases with valid 3D route witnesses", flush=True)
    if not args.run and not args.summary_only:
        print("Pass --run to execute each case with the requested policies.")
        return 0
    rows = []
    if args.run:
        sys.path.insert(0, str(ROOT))
        from benchmark import run_one
        for policy in args.policies.split(","):
            for world_name, metadata in cases:
                seed = int(metadata["seed"])
                print(f"run policy={policy} family={metadata['family']} seed={seed}", flush=True)
                result = run_one(args.webots, world_name, seed, policy, 800, args.port,
                                 args.physics_step_ms, args.motor_sampling, args.physics_profile)
                save_run_manifest(policy, world_name, seed, args.physics_step_ms,
                                  args.motor_sampling, args.physics_profile)
                result.update(
                    family=metadata["family"],
                    obstacle_count=len(metadata["obstacles"]),
                    route_witness_length_m=metadata["route_witness"]["length_m"],
                    generator_parameters=json.dumps(metadata["generator_parameters"], sort_keys=True),
                )
                rows.append(result)
        # Exact worlds are retained beside each episode route.wbt. Remove only
        # the generated templates after all policies have consumed them.
        for world_name, _ in cases:
            (WORLDS / f"{world_name}.wbt").unlink(missing_ok=True)
    else:
        for policy in args.policies.split(","):
            for world_name, metadata in cases:
                seed = int(metadata["seed"])
                episode = RESULTS / Path(policy).stem / f"{world_name}-seed-{seed}" / "episode.json"
                if not episode.is_file():
                    raise FileNotFoundError(f"missing saved episode: {episode}")
                result = json.loads(episode.read_text())
                result.update(
                    policy=Path(policy).name,
                    world=world_name,
                    seed=seed,
                    family=metadata["family"],
                    obstacle_count=len(metadata["obstacles"]),
                    route_witness_length_m=metadata["route_witness"]["length_m"],
                    generator_parameters=json.dumps(metadata["generator_parameters"], sort_keys=True),
                )
                rows.append(result)
    fields = ["policy", "world", "seed", "success", "collision", "timeout", "steps", "time_s",
              "path_m", "final_error_m", "peak_speed_mps", "min_sensor_range_m", "altitude_min_m",
              "altitude_max_m", "tracking_rms_mps", "sensor_updates", "navigation_loaded", "family",
              "obstacle_count", "route_witness_length_m", "generator_parameters", "physics_step_ms",
              "motor_sampling", "physics_profile"]
    summary = RESULTS / "challenge-matrix.csv"
    with summary.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    if any(row["world"] == "challenge_doorway_41002" and row["policy"] == "navigation.bin" for row in rows):
        image = render_trace_reconstruction("challenge_doorway_41002", "../assets/navigation.bin", 41002)
        if image:
            print(f"reconstruction={image}")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
