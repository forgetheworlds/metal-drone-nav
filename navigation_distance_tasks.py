"""Build reproducible long-goal source banks without changing the vehicle model.

The straight segment proves geometric clearance, not successful vehicle flight.
Starts retain the template's velocity and yaw, including nonzero motion.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import struct

HEADER_BYTES = 88
ENTRY_BYTES = 756


def read_bank(path):
    data = Path(path).read_bytes()
    magic, version, period, count, entry_bytes = struct.unpack_from("<8s4I", data)
    assert magic == b"WPBANK1\0" and version == 1 and entry_bytes == ENTRY_BYTES
    payload = data[HEADER_BYTES:]
    assert len(payload) == count * ENTRY_BYTES
    assert hashlib.sha256(payload).hexdigest().encode() == data[24:88]
    return period, [payload[i:i + ENTRY_BYTES] for i in range(0, len(payload), ENTRY_BYTES)]


def write_bank(path, period, entries):
    payload = b"".join(entries)
    digest = hashlib.sha256(payload).hexdigest().encode()
    header = struct.pack("<8s4I", b"WPBANK1\0", 1, period, len(entries), ENTRY_BYTES)
    Path(path).write_bytes(header + digest + payload)


def clearance(point, hallway):
    x, y, z = point
    room = min(x + 2, 14 - x, y + 5, 5 - y, z, 5 - z) - .18
    return min(room, 1 - abs(y) - .18) if hallway else room


def distance_entries(template_entries, seed, hallway):
    rng = random.Random(seed)
    entries, labels = [], []
    for env, template in enumerate(template_entries):
        start = [rng.uniform(-1, .5), rng.uniform(-.5, .5), rng.uniform(1.1, 2.5)]
        goal = [rng.uniform(12.5, 13.2), rng.uniform(-.5, .5), start[2] + rng.uniform(-.15, .15)]
        entry = bytearray(template)
        struct.pack_into("<I", entry, 640, 2 if hallway else 0)
        if hallway:
            for obstacle, y in enumerate([-1.05, 1.05]):
                struct.pack_into("<I9f", entry, obstacle * 40, 0, 6, y, 2.5, 8, .05, 2.5, 0, 0, 0)
        struct.pack_into("<3f", entry, 652, *goal)
        struct.pack_into("<3f", entry, 676, *start)
        struct.pack_into("<3f", entry, 700, *goal)
        distance = math.dist(start, goal)
        initial, final = clearance(start, hallway), clearance(goal, hallway)
        lower_bound = min(initial, final)
        assert lower_bound > .3 and distance < 15
        struct.pack_into("<6f", entry, 716, initial, final, lower_bound, lower_bound, distance, distance)
        struct.pack_into("<I", entry, 748, 0)
        entries.append(bytes(entry))
        labels.append({"env": env, "start": start, "goal": goal, "distance_m": distance,
                       "surface_clearance_lower_bound_m": lower_bound,
                       "start_velocity": struct.unpack_from("<3f", entry, 688),
                       "start_yaw": struct.unpack_from("<f", entry, 712)[0]})
    return entries, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rehearsal", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    period, templates = read_bank(args.template)
    assert period == 1 and len(templates) == 128
    panels = {}
    for kind in ["open", "hallway"]:
        entries, labels = distance_entries(templates, args.seed, kind == "hallway")
        panels[kind] = entries
        write_bank(args.output / (kind + ".bin"), 1, entries)
        (args.output / (kind + ".json")).write_text(json.dumps({"seed": args.seed, "records": labels}, indent=2) + "\n")
    if args.rehearsal:
        period, original = read_bank(args.rehearsal)
        assert period == 8 and len(original) == 1024
        mixed = []
        for env in range(128):
            mixed.extend(original[env * 8:(env + 1) * 8])
            # Eight broad old tasks plus four long tasks per episode cycle.
            for slot in range(4):
                mixed.append(panels["open" if slot < 2 else "hallway"][(env + 31 * slot) % 128])
        write_bank(args.output / "mixed.bin", 12, mixed)


if __name__ == "__main__":
    main()
