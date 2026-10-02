#!/usr/bin/env python3
"""Export one frozen Metal challenge-bank room as an exact Webots scene."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


WEBOTS_DIR = Path(__file__).resolve().parent
ROOT = WEBOTS_DIR.parent
DEFAULT_BANK = ROOT / "results" / "challenge-bank-mirrored-v1.jsonl"
BOUNDED_TEMPLATE = WEBOTS_DIR / "worlds" / "a_to_b_bounded.wbt"
DEFAULT_WORLD_DIR = WEBOTS_DIR / "worlds"
DEFAULT_METADATA_DIR = WEBOTS_DIR / "results" / "metal_scenes"
RAPTOR_PROTO = WEBOTS_DIR / "protos" / "RaptorCrazyflie.proto"
ROOM_BOUNDS = {"x": [-2.0, 14.0], "y": [-5.0, 5.0], "z": [0.0, 5.0]}
START = [0.0, 0.0, 1.5]
BODY_RADIUS_M = 0.18


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def vector3(value: object, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{label} must have three numbers")
    return [finite_number(component, f"{label}[{axis}]") for axis, component in enumerate(value)]


def read_record(bank_path: Path, failure_id: str) -> tuple[dict, bytes, bytes]:
    bank_bytes = bank_path.read_bytes()
    matches = []
    for line_number, raw_line in enumerate(bank_bytes.splitlines(keepends=True), 1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid bank JSON on line {line_number}: {error}") from error
        if isinstance(record, dict) and record.get("failure_id") == failure_id:
            matches.append((record, raw_line))
    if len(matches) != 1:
        raise ValueError(f"expected one record for {failure_id}, found {len(matches)}")
    record, source_line = matches[0]
    if record.get("record_type") != "challenge" or record.get("schema_version") != 1:
        raise ValueError("unsupported challenge record schema")
    if record.get("split") not in ("train", "dev"):
        raise ValueError("Webots video scenes may only use train/dev bank levels; final is reserved")
    return record, bank_bytes, source_line


def validate_record(record: dict) -> list[dict]:
    family = record.get("family")
    if family not in (14, 15, 16):
        raise ValueError(f"unsupported spatial family: {family}")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", str(record.get("failure_id", ""))):
        raise ValueError("failure_id contains unsafe path characters")
    if record.get("room_bounds") != ROOM_BOUNDS:
        raise ValueError("bank room bounds do not match the Webots bounded-room template")
    if abs(finite_number(record.get("body_radius_m"), "body_radius_m") - BODY_RADIUS_M) > 1e-6:
        raise ValueError("bank collision radius differs from the current RaptorCrazyflie sphere")
    goal = vector3(record.get("goal"), "goal")
    if not (ROOM_BOUNDS["x"][0] < goal[0] < ROOM_BOUNDS["x"][1] and
            ROOM_BOUNDS["y"][0] < goal[1] < ROOM_BOUNDS["y"][1] and
            ROOM_BOUNDS["z"][0] < goal[2] < ROOM_BOUNDS["z"][1]):
        raise ValueError("goal is outside the fixed room")
    wind = vector3(record.get("wind"), "wind")
    if any(abs(component) > 1e-8 for component in wind):
        raise ValueError("this exporter supports the static zero-wind spatial bank only")
    obstacles = record.get("obstacles")
    if not isinstance(obstacles, list) or len(obstacles) > 16:
        raise ValueError("obstacles must be a list of at most 16 shapes")
    for index, obstacle in enumerate(obstacles):
        label = f"obstacles[{index}]"
        if not isinstance(obstacle, dict):
            raise ValueError(f"{label} must be an object")
        kind = obstacle.get("kind")
        if kind not in (0, 1, 2):
            raise ValueError(f"{label}.kind must be AABB(0), sphere(1), or vertical cylinder(2)")
        vector3(obstacle.get("center"), f"{label}.center")
        half = vector3(obstacle.get("half_extent"), f"{label}.half_extent")
        if any(component < 0.0 for component in half):
            raise ValueError(f"{label}.half_extent cannot be negative")
        velocity = vector3(obstacle.get("velocity"), f"{label}.velocity")
        if any(abs(component) > 1e-8 for component in velocity):
            raise ValueError(f"{label} moves; this exporter preserves static geometry only")
        dimensions = half if kind == 0 else ([half[0]] * 3 if kind == 1 else [half[0], half[0], half[2]])
        if any(component <= 0.0 for component in dimensions):
            raise ValueError(f"{label} has a non-positive collision dimension")
        center = vector3(obstacle["center"], f"{label}.center")
        for axis, name in enumerate(("x", "y", "z")):
            extent = half[axis] if kind == 0 else half[0] if kind == 1 or axis < 2 else half[2]
            if center[axis] - extent < ROOM_BOUNDS[name][0] - 1e-4 or center[axis] + extent > ROOM_BOUNDS[name][1] + 1e-4:
                raise ValueError(f"{label} extends outside the fixed room bounds")
    return obstacles


def vrml_float(value: float) -> str:
    return format(float(value), ".9g")


def obstacle_nodes(obstacles: list[dict]) -> str:
    nodes = []
    for index, obstacle in enumerate(obstacles):
        name = f"ChallengeObstacle{index:02d}"
        label = f"metal_{index:02d}_kind{obstacle['kind']}"
        x, y, z = map(vrml_float, obstacle["center"])
        if obstacle["kind"] == 0:
            hx, hy, hz = obstacle["half_extent"]
            sx, sy, sz = map(vrml_float, (2.0 * hx, 2.0 * hy, 2.0 * hz))
            nodes.append(
                f'DEF {name} Solid {{ name "{label}" translation {x} {y} {z} '
                f'children [ Shape {{ appearance PBRAppearance {{ baseColor 0.63 0.34 0.18 roughness 1 }} '
                f'geometry Box {{ size {sx} {sy} {sz} }} }} ] '
                f'boundingObject Box {{ size {sx} {sy} {sz} }} locked TRUE }}'
            )
        elif obstacle["kind"] == 1:
            radius = vrml_float(obstacle["half_extent"][0])
            nodes.append(
                f'DEF {name} Solid {{ name "{label}" translation {x} {y} {z} '
                f'children [ Shape {{ appearance PBRAppearance {{ baseColor 0.25 0.36 0.52 roughness 1 }} '
                f'geometry Sphere {{ radius {radius} }} }} ] '
                f'boundingObject Sphere {{ radius {radius} }} locked TRUE }}'
            )
        else:
            radius = vrml_float(obstacle["half_extent"][0])
            height = vrml_float(2.0 * obstacle["half_extent"][2])
            nodes.append(
                f'DEF {name} Solid {{ name "{label}" translation {x} {y} {z} '
                f'children [ Shape {{ appearance PBRAppearance {{ baseColor 0.25 0.36 0.52 roughness 1 }} '
                f'geometry Cylinder {{ radius {radius} height {height} }} }} ] '
                f'boundingObject Cylinder {{ radius {radius} height {height} }} locked TRUE }}'
            )
    return "\n".join(nodes)


def scene_world(record: dict, obstacles: list[dict], policy: str, max_steps: int) -> str:
    source = BOUNDED_TEMPLATE.read_text()
    marker = "# WEBOTS_OBSTACLES_INSERTION_POINT"
    if source.count(marker) != 1:
        raise ValueError("bounded room template must contain one obstacle insertion marker")
    source, timestep_replacements = re.subn(r"\bbasicTimeStep\s+10\b", "basicTimeStep 1", source, count=1)
    if timestep_replacements != 1:
        raise ValueError("bounded room template must specify one 10 ms physics step to replace")
    failure_id = record["failure_id"]
    goal = vector3(record["goal"], "goal")
    sim_config = record.get("sim_config")
    if not isinstance(sim_config, dict):
        raise ValueError("sim_config must be an object")
    config = {
        "phase": "navigation",
        "seed": str(record["seed"]),
        "policy": policy,
        "speed": "1.5",
        "distance": vrml_float(finite_number(record["sim_config"]["distance"], "sim_config.distance")),
        "goal": ",".join(map(vrml_float, goal)),
        "max_steps": str(max_steps),
        "motor_sampling": "average",
        "goal_objective": "entry",
    }
    custom_data = ";".join(f"{key}={value}" for key, value in config.items())
    source = source.replace(marker, obstacle_nodes(obstacles), 1)
    source = re.sub(r'title "[^"]*"', f'title "Metal challenge {failure_id}"', source, count=1)
    source = re.sub(r'customData "[^"]*"', f'customData "{custom_data}"', source, count=1)
    return source


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("failure_id", help="exact train/dev failure_id from the frozen JSONL bank")
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--policy", default="../assets/navigation.bin",
                        help="policy path relative to the Webots project root")
    parser.add_argument("--max-steps", type=int, default=2000,
                        help="100 Hz RAPTOR steps; 2000 steps is the 20 s Metal budget")
    parser.add_argument("--world-dir", type=Path, default=DEFAULT_WORLD_DIR)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    args = parser.parse_args()
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if not args.policy or any(character in args.policy for character in ';"\r\n'):
        parser.error("--policy must be a non-empty path without customData delimiters")
    record, bank_bytes, source_line = read_record(args.bank, args.failure_id)
    obstacles = validate_record(record)
    proto_text = RAPTOR_PROTO.read_text()
    if not re.search(r"boundingObject\s+Sphere\s*\{\s*radius\s+0\.18\s*\}", proto_text):
        raise ValueError("RaptorCrazyflie PROTO no longer declares the bank's 0.18 m safety sphere")
    name = f"metal_{args.failure_id}"
    world_bytes = scene_world(record, obstacles, args.policy, args.max_steps).encode("utf-8")
    world_path = args.world_dir / f"{name}.wbt"
    metadata_path = args.metadata_dir / f"{name}.json"
    metadata = {
        "schema": "webots-metal-scene-v1",
        "scene_name": name,
        "failure_id": args.failure_id,
        "split": record["split"],
        "family": record["family"],
        "family_name": record["family_name"],
        "start_xyz_m": START,
        "goal_xyz_m": record["goal"],
        "room_bounds_m": ROOM_BOUNDS,
        "body_collision_radius_m": BODY_RADIUS_M,
        "basic_time_step_ms": 1,
        "raptor_control_period_ms": 10,
        "navigation_period_ms": 50,
        "motor_sampling": "average",
        "controller_policy": args.policy,
        "actor_inputs": "Depth, ego sensors, goal and local geometry prior only; bank obstacles and witness are not sent through Robot.customData or observations.",
        "witness_usage": "Scoring metadata only; never written to the world or controller customData.",
        "source_bank": str(args.bank.resolve()),
        "source_bank_sha256": sha256(bank_bytes),
        "source_record_sha256": sha256(source_line),
        "world_sha256": sha256(world_bytes),
        "raptor_proto_sha256": sha256(proto_text.encode("utf-8")),
        "obstacles": obstacles,
        "scoring_metadata": record,
    }
    atomic_write(world_path, world_bytes)
    atomic_write(metadata_path, (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(f"exported failure_id={args.failure_id} split={record['split']} family={record['family']}")
    print(f"world={world_path}")
    print(f"metadata={metadata_path}")
    print(f"obstacles={len(obstacles)} goal={','.join(map(vrml_float, record['goal']))} room=[-2,14]x[-5,5]x[0,5]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
