#!/usr/bin/env python3
"""Run reproducible closed-loop Webots A-to-B episodes and preserve raw traces."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORLDS = ROOT / "worlds"
RESULTS = ROOT / "results"
DEFAULT_WEBOTS = Path("/Users/muadhsambul/embodied/work/Webots.app/Contents/MacOS/webots")


def with_custom_data(source: str, updates: dict[str, str]) -> str:
    match = re.search(r'customData\s+"([^"]*)"', source)
    if not match:
        raise ValueError("world has no Robot.customData field")
    fields = dict(
        part.split("=", 1)
        for part in match.group(1).split(";")
        if "=" in part
    )
    fields.update(updates)
    replacement = "customData \"" + ";".join(f"{key}={value}" for key, value in fields.items()) + "\""
    return source[: match.start()] + replacement + source[match.end() :]


def run_one(webots: Path, world_name: str, seed: int, policy: str, steps: int, port: int = 23456) -> dict:
    base = WORLDS / f"{world_name}.wbt"
    if not base.is_file():
        raise FileNotFoundError(base)
    world = WORLDS / f".run-{world_name}-{Path(policy).stem}-{seed}.wbt"
    output_dir = RESULTS / Path(policy).stem / f"{world_name}-seed-{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    updates = {
        "phase": "navigation",
        "seed": str(seed),
        "policy": policy,
        "max_steps": str(steps),
    }
    source = base.read_text()
    world_parameters = {}
    if world_name == "a_to_b_offset_box":
        rng = random.Random(seed)
        offset_y = rng.uniform(-0.55, 0.55)
        world_parameters["box_y_offset_m"] = round(offset_y, 5)
        source = source.replace(
            "translation 2.0 0.0 1.5",
            f"translation 2.0 {offset_y:.5f} 1.5",
            1,
        )
    world.write_text(with_custom_data(source, updates))
    shutil.copy2(world, output_dir / "route.wbt")
    try:
        for stale in (RESULTS / "last-run.json", RESULTS / "last-run-exit.marker", RESULTS / "last-run-trace.csv"):
            stale.unlink(missing_ok=True)
        process = subprocess.run(
            [str(webots), f"--port={port}", "--batch", "--mode=fast", "--no-rendering", "--stdout", "--stderr", str(world)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        (output_dir / "webots.log").write_text(process.stdout + process.stderr)
        if process.returncode != 0:
            raise RuntimeError(f"Webots exited {process.returncode}; see {output_dir / 'webots.log'}")
        result_path = RESULTS / "last-run.json"
        marker_path = RESULTS / "last-run-exit.marker"
        trace_path = RESULTS / "last-run-trace.csv"
        if not result_path.is_file() or not marker_path.is_file():
            raise RuntimeError(f"controller did not write its result marker; see {output_dir / 'webots.log'}")
        result = json.loads(result_path.read_text())
        shutil.copy2(result_path, output_dir / "episode.json")
        shutil.copy2(marker_path, output_dir / "exit.marker")
        if trace_path.is_file():
            shutil.copy2(trace_path, output_dir / "trace.csv")
        result.update(world=world_name, policy=Path(policy).name, seed=seed, **world_parameters)
        return result
    finally:
        world.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--webots", type=Path, default=DEFAULT_WEBOTS)
    parser.add_argument("--worlds", default="a_to_b,a_to_b_offset_box")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--policies", default="../assets/navigation.bin")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--port", type=int, default=23456, help="isolated Webots controller port")
    args = parser.parse_args()
    if not args.webots.is_file():
        parser.error(f"Webots executable not found: {args.webots}")
    rows = []
    for policy in args.policies.split(","):
        for world in args.worlds.split(","):
            for seed_text in args.seeds.split(","):
                seed = int(seed_text)
                print(f"run policy={policy} world={world} seed={seed}", flush=True)
                rows.append(run_one(args.webots, world, seed, policy, args.steps, args.port))
    summary = RESULTS / "benchmark.csv"
    fields = [
        "policy", "world", "seed", "success", "collision", "timeout", "steps", "time_s",
        "path_m", "final_error_m", "peak_speed_mps", "min_sensor_range_m", "altitude_min_m",
        "altitude_max_m", "tracking_rms_mps", "sensor_updates", "navigation_loaded", "box_y_offset_m",
    ]
    accumulated: dict[tuple[str, str, str], dict] = {}
    if summary.exists():
        with summary.open(newline="") as previous:
            for old in csv.DictReader(previous):
                accumulated[(old["policy"], old["world"], old["seed"])] = old
    for row in rows:
        accumulated[(row["policy"], row["world"], str(row["seed"]))] = row
    with summary.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(accumulated[key] for key in sorted(accumulated))
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
