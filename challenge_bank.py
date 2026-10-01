#!/usr/bin/env python3
"""Build deterministic, route-validated spatial challenge levels.

The generated simulator seed is for environment index 0. Each record includes
its vector base seed and index so the corresponding level is reproducible in
the existing simulator without changing SimConfig.

Geometry contracts:
  14: two connected AABB partitions force a northward turn, an eastward hall
      traversal, then a southward turn before the goal.
  15: two offset full-height door partitions and a table require two doorway
      crossings plus a lateral detour around intermediate furniture.
  16: a floor-mounted barrier requires flight above it, a ceiling overhang
      requires flight below it, and a final slab offers either vertical lane.

Every route is a geometric witness for a 0.18 m spherical vehicle envelope.
It is sampled at no more than 0.05 m spacing against the same AABB signed
distance and fixed room bounds as world.hpp. It proves that a collision-free
path exists; it does not prove that a controller can fly the route.

Example:
    python3 challenge_bank.py --distance 8 --per-split 128 \
        --out artifacts/challenge-bank-v1.jsonl
    python3 challenge_bank.py --distance 8 --per-split 128 --validate-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import struct
import sys
from collections import Counter
from typing import Any

MASK32 = 0xFFFFFFFF
SIM_ENV_STRIDE = 747796405
SIM_ENV_OFFSET = 2891336453
BODY_RADIUS_M = 0.18
ROOM = {"x": [-2.0, 14.0], "y": [-5.0, 5.0], "z": [0.0, 5.0]}
FAMILY_NAMES = {
    14: "bent_hallway_corner",
    15: "connected_rooms_offset_doors_furniture",
    16: "vertical_over_under_choice",
}
FAMILY_CONTRACTS = {
    14: "connected partitions force north turn, east hall, then south turn",
    15: "offset doors with intermediate table detour",
    16: "cross low barrier above, overhang below, then choose over or under slab",
}
SPLITS = ("train", "dev", "final")


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def add(a: float, b: float) -> float:
    return f32(a + b)


def sub(a: float, b: float) -> float:
    return f32(a - b)


def mul(a: float, b: float) -> float:
    return f32(a * b)


def xorshift32(state: int) -> int:
    state &= MASK32
    state ^= (state << 13) & MASK32
    state ^= state >> 17
    state ^= (state << 5) & MASK32
    return state & MASK32


class WorldRng:
    def __init__(self, seed: int):
        self.state = seed & MASK32 or 1

    def uniform(self) -> float:
        self.state = xorshift32(self.state)
        return f32((self.state >> 8) * (1.0 / 16777216.0))


def sim_world_seed(config_seed: int, env_index: int) -> int:
    """Mirror sim_reset_one -> wtraining_family -> wgenerate for 14–16."""
    state = (config_seed + env_index * SIM_ENV_STRIDE + SIM_ENV_OFFSET) & MASK32
    # wtraining_family returns families 14–16 without consuming RNG state.
    return xorshift32(state)


def box(center: tuple[float, float, float], half: tuple[float, float, float]) -> dict[str, Any]:
    return {
        "kind": 0,
        "center": [f32(v) for v in center],
        "half_extent": [f32(v) for v in half],
        "velocity": [0.0, 0.0, 0.0],
    }


def add_box(world: dict[str, Any], center: tuple[float, float, float], half: tuple[float, float, float]) -> None:
    world["obstacles"].append(box(center, half))


def add_doorway(world: dict[str, Any], wall_x: float, gap_y: float, gap_z: float,
                half_gap: float, half_gap_z: float) -> None:
    low_y = sub(gap_y, half_gap)
    high_y = add(gap_y, half_gap)
    low_z = sub(gap_z, half_gap_z)
    high_z = add(gap_z, half_gap_z)
    wall_half_x = f32(0.12)
    add_box(world, (wall_x, mul(add(-5.0, low_y), 0.5), 2.5),
            (wall_half_x, mul(add(low_y, 5.0), 0.5), 2.5))
    add_box(world, (wall_x, mul(add(high_y, 5.0), 0.5), 2.5),
            (wall_half_x, mul(sub(5.0, high_y), 0.5), 2.5))
    add_box(world, (wall_x, 0.0, mul(low_z, 0.5)),
            (wall_half_x, 5.0, mul(low_z, 0.5)))
    add_box(world, (wall_x, 0.0, mul(add(high_z, 5.0), 0.5)),
            (wall_half_x, 5.0, mul(sub(5.0, high_z), 0.5)))


def add_table(world: dict[str, Any], x: float, y: float, half_x: float,
              half_y: float, top_z: float) -> None:
    slab_half_z = f32(0.08)
    underside = sub(top_z, slab_half_z)
    add_box(world, (x, y, top_z), (half_x, half_y, slab_half_z))
    leg_x = sub(half_x, f32(0.12))
    leg_y = sub(half_y, f32(0.12))
    leg_half = f32(0.08)
    leg_center_z = mul(underside, 0.5)
    leg_half_z = mul(underside, 0.5)
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            add_box(world, (add(x, mul(sx, leg_x)), add(y, mul(sy, leg_y)), leg_center_z),
                    (leg_half, leg_half, leg_half_z))


def generate_world(config_seed: int, env_index: int, family: int, distance: float) -> dict[str, Any]:
    seed = sim_world_seed(config_seed, env_index)
    rng = WorldRng(seed)
    span = f32(min(max(distance, 3.0), 10.0))
    # The generic wgenerate goal draws happen before its family branches.
    rng.uniform()  # initial goal y, replaced by family helper
    rng.uniform()  # initial goal z, replaced by family helper
    difficulty = rng.uniform()
    world: dict[str, Any] = {
        "family": family,
        "family_name": FAMILY_NAMES[family],
        "geometry_contract": FAMILY_CONTRACTS[family],
        "seed": seed,
        "difficulty": difficulty,
        "difficulty_band": "easy" if difficulty < 1/3 else ("medium" if difficulty < 2/3 else "hard"),
        "goal": [0.0, 0.0, 0.0],
        "wind": [0.0, 0.0, 0.0],
        "obstacles": [],
        "parameters": {},
    }

    goal_scale = add(0.90, mul(0.10, rng.uniform()))
    goal_x = mul(span, goal_scale)
    goal_y = mul(sub(rng.uniform(), 0.5), 1.20)
    goal_z = add(1.10, mul(0.80, rng.uniform()))
    world["goal"] = [goal_x, goal_y, goal_z]

    if family == 14:
        wall_x = mul(goal_x, add(0.24, mul(0.04, rng.uniform())))
        gap_x = mul(goal_x, add(0.70, mul(0.08, difficulty)))
        lane_width = sub(1.80, mul(0.80, difficulty))
        lane_edge = sub(5.0, lane_width)
        wall_y = mul(add(-5.0, lane_edge), 0.5)
        wall_half_y = mul(add(lane_edge, 5.0), 0.5)
        add_box(world, (wall_x, wall_y, 2.5), (0.15, wall_half_y, 2.5))
        add_box(world, (mul(add(wall_x, gap_x), 0.5), add(lane_edge, 0.03), 2.5),
                (add(mul(sub(gap_x, wall_x), 0.5), 0.15), 0.15, 2.5))
        route_y = mul(add(lane_edge, 5.0), 0.5)
        witness = [
            [0.0, 0.0, 1.5],
            [sub(wall_x, 0.40), route_y, 1.5],
            [add(wall_x, 0.40), route_y, 1.5],
            [add(gap_x, 0.55), route_y, 1.5],
            [add(gap_x, 0.55), goal_y, 1.5],
            [goal_x, goal_y, goal_z],
        ]
        world["parameters"] = {"turn": "north-east-south", "lane_width_m": lane_width,
                               "partition_x_m": wall_x, "turn_gap_x_m": gap_x}
    elif family == 15:
        wall_1 = mul(goal_x, add(0.16, mul(0.02, rng.uniform())))
        wall_2 = mul(goal_x, 0.85)
        gap_1 = -add(0.60, mul(0.80, difficulty))
        gap_2 = add(0.60, mul(0.80, difficulty))
        gap_half = sub(0.85, mul(0.30, difficulty))
        gap_z = f32(1.50)
        gap_half_z = sub(0.85, mul(0.10, difficulty))
        add_doorway(world, wall_1, gap_1, gap_z, gap_half, gap_half_z)
        add_doorway(world, wall_2, gap_2, gap_z, gap_half, gap_half_z)

        table_x = mul(goal_x, 0.50)
        table_y = mul(sub(rng.uniform(), 0.5), 0.20)
        table_half_x = f32(min(0.45, mul(goal_x, 0.06)))
        table_half_y = add(0.60, mul(0.20, difficulty))
        table_top = add(1.45, mul(0.25, difficulty))
        add_table(world, table_x, table_y, table_half_x, table_half_y, table_top)
        add_box(world, (mul(goal_x, 0.38), 3.75, 0.55), (0.22, 0.35, 0.55))
        add_box(world, (mul(goal_x, 0.50), -3.80, 0.80), (0.25, 0.35, 0.80))
        add_box(world, (mul(goal_x, 0.62), 3.70, 0.50), (0.22, 0.40, 0.50))

        table_side_y = add(table_y, add(table_half_y, 0.50))
        witness = [
            [0.0, 0.0, 1.5],
            [sub(wall_1, 0.32), gap_1, gap_z],
            [add(wall_1, 0.32), gap_1, gap_z],
            [sub(sub(table_x, table_half_x), 0.25), table_side_y, 1.5],
            [add(add(table_x, table_half_x), 0.25), table_side_y, 1.5],
            [sub(wall_2, 0.32), gap_2, gap_z],
            [add(wall_2, 0.32), gap_2, gap_z],
            [goal_x, goal_y, goal_z],
        ]
        world["parameters"] = {"door_1_x_m": wall_1, "door_2_x_m": wall_2,
                               "door_gap_centers_y_m": [gap_1, gap_2],
                               "door_clear_width_m": mul(gap_half, 2),
                               "table_center_m": [table_x, table_y, table_top],
                               "table_half_extent_xy_m": [table_half_x, table_half_y]}
    else:
        low_x = mul(goal_x, add(0.14, mul(0.02, sub(1.0, difficulty))))
        overhead_x = mul(goal_x, add(0.46, mul(0.02, difficulty)))
        choice_x = mul(goal_x, add(0.76, mul(0.02, difficulty)))
        low_top = add(1.50, mul(0.25, difficulty))
        overhead_bottom = sub(2.80, mul(0.20, difficulty))
        choice_half_z = add(0.10, mul(0.05, difficulty))
        overhead_half_z = mul(sub(5.0, overhead_bottom), 0.5)
        add_box(world, (low_x, 0.0, mul(low_top, 0.5)), (0.12, 5.0, mul(low_top, 0.5)))
        add_box(world, (overhead_x, 0.0, add(overhead_bottom, overhead_half_z)),
                (0.12, 5.0, overhead_half_z))
        add_box(world, (choice_x, 0.0, add(2.0, choice_half_z)), (0.10, 5.0, choice_half_z))
        over_height = add(add(low_top, 0.18), 0.25)
        choose_over = difficulty >= 0.5
        choice_height = add(2.75, mul(0.10, difficulty)) if choose_over else 1.50
        witness = [
            [0.0, 0.0, 1.5],
            [sub(low_x, 0.36), 0.0, 1.5],
            [sub(low_x, 0.36), 0.0, over_height],
            [add(low_x, 0.36), 0.0, over_height],
            [sub(overhead_x, 0.36), 0.0, over_height],
            [sub(overhead_x, 0.36), 0.0, 1.5],
            [add(overhead_x, 0.36), 0.0, 1.5],
            [sub(choice_x, 0.36), 0.0, 1.5],
            [sub(choice_x, 0.36), 0.0, choice_height],
            [add(choice_x, 0.36), 0.0, choice_height],
            [goal_x, goal_y, goal_z],
        ]
        world["parameters"] = {"low_barrier_x_m": low_x, "low_barrier_top_m": low_top,
                               "overhang_x_m": overhead_x, "overhang_bottom_m": overhead_bottom,
                               "choice_slab_x_m": choice_x, "choice_slab_half_height_m": choice_half_z,
                               "witness_choice": "over" if choose_over else "under"}
    world["witness_route"] = witness
    world["direct_route_clearance_m"] = route_clearance(world, [[0.0, 0.0, 1.5], world["goal"]])[0]
    world["witness_min_clearance_m"], world["witness_sample_count"] = route_clearance(world, witness)
    return world


def point_clearance(world: dict[str, Any], p: list[float]) -> float:
    x, y, z = p
    room_clearance = min(x + 2, 14 - x, y + 5, 5 - y, z, 5 - z) - BODY_RADIUS_M
    obstacle_clearance = math.inf
    for obstacle in world["obstacles"]:
        c = obstacle["center"]
        h = obstacle["half_extent"]
        kx, ky, kz = abs(x-c[0])-h[0], abs(y-c[1])-h[1], abs(z-c[2])-h[2]
        outside = math.sqrt(max(kx, 0.0)**2 + max(ky, 0.0)**2 + max(kz, 0.0)**2)
        signed = outside + min(max(kx, ky, kz), 0.0)
        obstacle_clearance = min(obstacle_clearance, signed-BODY_RADIUS_M)
    return f32(min(room_clearance, obstacle_clearance))


def route_clearance(world: dict[str, Any], route: list[list[float]], spacing: float = 0.05) -> tuple[float, int]:
    minimum = math.inf
    samples = 0
    for a, b in zip(route, route[1:]):
        length = math.sqrt(sum((b[i]-a[i])**2 for i in range(3)))
        count = max(1, math.ceil(length/spacing))
        for j in range(count+1):
            t = j/count
            p = [a[i]+(b[i]-a[i])*t for i in range(3)]
            minimum = min(minimum, point_clearance(world, p))
            samples += 1
    if len(route) == 1:
        minimum = point_clearance(world, route[0])
        samples = 1
    return f32(minimum), samples


def validate_world(world: dict[str, Any]) -> None:
    if len(world["obstacles"]) > 16:
        raise ValueError(f"family {world['family']} uses more than 16 obstacles")
    start = [0.0, 0.0, 1.5]
    goal = world["goal"]
    route = world["witness_route"]
    if any(abs(start[i]-route[0][i]) > 1e-6 for i in range(3)):
        raise ValueError("witness route has wrong start")
    if any(abs(goal[i]-route[-1][i]) > 1e-6 for i in range(3)):
        raise ValueError("witness route has wrong goal")
    for obstacle in world["obstacles"]:
        for axis, bounds in zip("xyz", (ROOM["x"], ROOM["y"], ROOM["z"])):
            c = obstacle["center"]["xyz".index(axis)]
            h = obstacle["half_extent"]["xyz".index(axis)]
            if c-h < bounds[0]-1e-5 or c+h > bounds[1]+1e-5:
                raise ValueError(f"obstacle extends outside room on {axis}: {obstacle}")
    if point_clearance(world, start) <= 0.04:
        raise ValueError(f"unsafe start clearance: {point_clearance(world, start)}")
    if point_clearance(world, goal) <= 0.04:
        raise ValueError(f"unsafe goal clearance: {point_clearance(world, goal)}")
    witness_clearance, _ = route_clearance(world, route)
    if witness_clearance <= 0.04:
        raise ValueError(f"invalid witness clearance: {witness_clearance}")
    direct_clearance, _ = route_clearance(world, [start, goal])
    if direct_clearance >= -0.02:
        raise ValueError(f"challenge does not obstruct the direct route: {direct_clearance}")


def stable_base_seed(master_seed: int, split: str, family: int, per_split: int, distance: float) -> int:
    key = f"challenge-bank-v1|{master_seed}|{split}|{family}|{per_split}|{distance:.4f}".encode()
    candidate = int.from_bytes(hashlib.sha256(key).digest()[:4], "little")
    # Ensure all three difficulty bands appear in a split-family batch.
    for attempt in range(4096):
        seed = (candidate + attempt * 0x9E3779B9) & MASK32
        bands = Counter(generate_world(seed, index, family, distance)["difficulty_band"]
                        for index in range(per_split))
        if all(bands[band] >= max(1, per_split//10) for band in ("easy", "medium", "hard")):
            return seed
    raise RuntimeError("could not choose a balanced deterministic base seed")


def source_hash(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scene_hash(world: dict[str, Any]) -> str:
    stable = {key: world[key] for key in ("family", "seed", "goal", "wind", "obstacles")}
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_records(master_seed: int, per_split: int, distance: float,
                  families: list[int], world_hash: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for split in SPLITS:
        for family in families:
            base_seed = stable_base_seed(master_seed, split, family, per_split, distance)
            for env_index in range(per_split):
                world = generate_world(base_seed, env_index, family, distance)
                validate_world(world)
                world["record_type"] = "challenge"
                world["schema_version"] = 1
                world["split"] = split
                world["environment_index"] = env_index
                world["sim_config"] = {"family": family, "seed": base_seed,
                                        "distance": f32(min(max(distance, 3.0), 10.0)),
                                        "environment_index": env_index}
                world["room_bounds"] = ROOM
                world["body_radius_m"] = BODY_RADIUS_M
                world["witness_min_clearance_m"] = route_clearance(world, world["witness_route"])[0]
                world["direct_route_clearance_m"] = route_clearance(world, [[0.0, 0.0, 1.5], world["goal"]])[0]
                world["failure_id"] = f"f{family}-{split}-{env_index:04d}-s{world['seed']:08x}-w{world_hash[:8]}"
                if world["failure_id"] in used_ids:
                    raise ValueError(f"duplicate failure id: {world['failure_id']}")
                used_ids.add(world["failure_id"])
                world["scene_sha256"] = scene_hash(world)
                world["world_source_sha256"] = world_hash
                world["selection_policy"] = (
                    "final split is held out from training, checkpoint selection, and curriculum design"
                    if split == "final" else "development split may be used for training or checkpoint selection"
                    if split == "dev" else "training split")
                records.append(world)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("artifacts/challenge-bank-v1.jsonl"))
    parser.add_argument("--seed", type=int, default=20261001, help="deterministic master seed")
    parser.add_argument("--per-split", type=int, default=128)
    parser.add_argument("--distance", type=float, default=8.0, help="maximum goal range in metres; values are capped at 10m")
    parser.add_argument("--families", default="14,15,16")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    families = [int(value) for value in args.families.split(",") if value]
    if not families or any(family not in FAMILY_NAMES for family in families):
        parser.error("--families must be a comma-separated subset of 14,15,16")
    if args.per_split < 30:
        parser.error("--per-split must be at least 30 so all difficulty bands can be represented")
    if not math.isfinite(args.distance) or args.distance < 3.0 or args.distance > 10.0:
        parser.error("--distance must be between 3 and 10 metres")
    if args.seed < 0 or args.seed > MASK32:
        parser.error("--seed must be an unsigned 32-bit integer")

    world_path = pathlib.Path(__file__).with_name("world.hpp")
    records = build_records(args.seed, args.per_split, args.distance, families, source_hash(world_path))
    counts = Counter((record["family"], record["split"], record["difficulty_band"]) for record in records)
    print(f"validated {len(records)} levels; families={families}; per_split={args.per_split}; distance={args.distance:.1f}m")
    for family in families:
        for split in SPLITS:
            bands = ", ".join(f"{band}={counts[(family,split,band)]}" for band in ("easy","medium","hard"))
            group = [r for r in records if r["family"]==family and r["split"]==split]
            print(f"family={family} split={split} {bands} min_witness_clearance={min(r['witness_min_clearance_m'] for r in group):.3f}m")
    if args.validate_only:
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    print(f"wrote {args.out} ({len(records)} JSONL records)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError) as error:
        print(f"challenge_bank: {error}", file=sys.stderr)
        raise SystemExit(2)
