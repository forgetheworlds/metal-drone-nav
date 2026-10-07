"""Generate a fresh development suite for the experience-scaling comparison.

These banks are evaluation inputs, never sampler inputs. This is a new seeded
development suite, not a sealed final test or an independent simulator.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import struct
import subprocess

from navigation_distance_tasks import distance_entries, read_bank, write_bank


def generate_suite(output, generator, checker):
    output.mkdir(parents=True, exist_ok=True)
    banks = {}
    for name, spec, seed in [
        ("static", "dev-a", 20261301),
        ("clutter", "clutter", 20261302),
        ("open", "open", 20261303),
    ]:
        prefix = output / name
        command = [str(generator), "local-bank", spec, str(prefix), "--seed", str(seed)]
        with (output / f"{name}-generation.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        period, entries = read_bank(prefix.with_suffix(".bin"))
        if period != 1 or len(entries) != 128:
            raise ValueError("Expected one fresh task per evaluation environment")
        rng = random.Random(seed + 73)
        randomized = []
        for original in entries:
            entry = bytearray(original)
            # Independent small initial motion; no velocity alignment with goal.
            direction = [rng.gauss(0, 1) for _ in range(3)]
            norm = sum(value * value for value in direction) ** .5
            speed = rng.uniform(0, .3)
            struct.pack_into("<3f", entry, 688, *(speed * value / norm for value in direction))
            randomized.append(bytes(entry))
        write_bank(prefix.with_suffix(".bin"), period, randomized)
        banks[name] = {"seed": seed, "generator_spec": spec, "path": name + ".bin"}

    _, templates = read_bank(output / "open.bin")
    for name, hallway in [("long-open", False), ("long-hallway", True)]:
        entries, labels = distance_entries(templates, 20261310, hallway)
        write_bank(output / f"{name}.bin", 1, entries)
        (output / f"{name}.json").write_text(json.dumps(labels, indent=2) + "\n")
        banks[name] = {"seed": 20261310, "path": name + ".bin"}

    for name, seed, reflected in [("course", 20261311, False), ("course-reflected", 20261312, True)]:
        directory = output / name
        command = ["python3", str(Path(__file__).with_name("navigation_challenge_tasks.py")),
                   str(directory), "--checker", str(checker), "--seed", str(seed),
                   "--period", "1", "--length-min", "6", "--length-max", "12.5"]
        if reflected:
            command += ["--mirror-x", "--mirror-z"]
        with (output / f"{name}-generation.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        banks[name] = {"seed": seed, "path": name + "/challenges.bin"}

    for bank in banks.values():
        path = output / bank["path"]
        period, entries = read_bank(path)
        bank.update(period=period, count=len(entries), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    report = {"scope": "Fresh seeded development; not sealed FINAL. No training on these payloads.",
              "generator_sha256": hashlib.sha256(generator.read_bytes()).hexdigest(),
              "checker_sha256": hashlib.sha256(checker.read_bytes()).hexdigest(), "banks": banks}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Generated seven hash-bound banks, 896 fresh evaluation tasks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--checker", type=Path, required=True)
    args = parser.parse_args()
    generate_suite(args.output.resolve(), args.generator.resolve(), args.checker.resolve())
