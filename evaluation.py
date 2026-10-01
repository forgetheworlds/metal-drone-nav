#!/usr/bin/env python3
"""Run the compact, reproducible navigation evaluation matrix."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FAMILY_NAMES = {
    3: "moving_spheres",
    5: "table_counter",
    7: "mixed_training_families_0_6",
    8: "held_two_doorways",
}
MODE_NAMES = {2: "goal_script", 13: "geometry_memory", 17: "guided_memory"}
FIELDS = [
    "case_id", "group", "mode", "mode_name", "family", "family_name", "seed",
    "speed_mps", "distance_m", "sensor_delay_frames", "wind_param",
    "depth_noise_m", "dropout_probability", "command_delay_steps",
    "episodes", "success", "collision", "timeout", "progress",
    "mean_goal_time_s", "mean_speed_mps", "peak_speed_mps", "min_clearance_m",
    "sim_wall_s", "gpu_s", "elapsed_wall_s", "started_utc", "completed_utc",
    "raw_eval_line", "raw_gpu_line",
]
EVAL_RE = re.compile(
    r"^(eval family=\d+ mode=\d+ episodes=\d+ .*? min_clearance=\S+ wall_s=\S+)$",
    re.MULTILINE,
)
GPU_RE = re.compile(r"^(eval_GPU_s=\S+)$", re.MULTILINE)
METRIC_RE = re.compile(r"([a-z_]+)=([^\s]+)")


def build_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []

    def add(group: str, mode: int, family: int, seed: int, **overrides: object) -> None:
        cfg: dict[str, object] = {
            "group": group,
            "mode": mode,
            "family": family,
            "seed": seed,
            "speed_mps": 1.5,
            "distance_m": 4.0,
            "sensor_delay_frames": 0,
            "wind_param": 0.0,
            "depth_noise_m": 0.0,
            "dropout_probability": 0.0,
            "command_delay_steps": 0,
        }
        cfg.update(overrides)
        cfg["mode_name"] = MODE_NAMES[mode]
        cfg["family_name"] = FAMILY_NAMES[family]
        cfg["case_id"] = "_".join(
            [group, f"m{mode}", f"f{family}", f"s{seed}"]
            + [f"{key}={value}" for key, value in overrides.items()]
        )
        cases.append(cfg)

    # Compare all three control modes on mixed training scenes and held-out,
    # table, and moving-obstacle scenes using one fresh seed.
    for family in (7, 8, 5, 3):
        for mode in (17, 13, 2):
            add("baseline", mode, family, 800001)

    # Check a second fresh seed with the guided policy.
    for family in (7, 8, 5, 3):
        add("fresh_seed", 17, family, 900001)

    # Change one disturbance at a time. Use the two most diagnostic obstacle
    # families. Baseline-zero cases above cover each default setting.
    stress = (
        ("sensor_delay", {"sensor_delay_frames": 2}),
        ("wind", {"wind_param": 0.5}),
        ("depth_noise", {"depth_noise_m": 0.05}),
        ("dropout", {"dropout_probability": 0.1}),
        ("command_delay", {"command_delay_steps": 1}),
    )
    for label, override in stress:
        for family in (8, 3):
            add(label, 17, family, 800001, **override)

    # Combined moderate corruption checks interaction on held-out doors and tables.
    combined = {
        "sensor_delay_frames": 2,
        "wind_param": 0.5,
        "depth_noise_m": 0.05,
        "dropout_probability": 0.1,
        "command_delay_steps": 1,
    }
    for family in (8, 5):
        add("combined", 17, family, 800001, **combined)
    return cases


def parse_metrics(stdout: str) -> tuple[dict[str, str], str, str]:
    eval_match = EVAL_RE.search(stdout)
    gpu_match = GPU_RE.search(stdout)
    if not eval_match or not gpu_match:
        raise RuntimeError(f"Could not parse evaluation output:\n{stdout}")
    lines = dict(METRIC_RE.findall(eval_match.group(1)))
    lines["family"] = lines.get("family", "")
    lines["mode"] = lines.get("mode", "")
    lines["episodes"] = lines.get("episodes", "")
    return lines, eval_match.group(1), gpu_match.group(1)


def evaluate_case(cli: Path, checkpoint: Path, case: dict[str, object], timeout: float) -> dict[str, object]:
    started = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    command = [
        str(cli), "eval", str(checkpoint), str(case["mode"]), str(case["family"]),
        str(case["seed"]), str(case["speed_mps"]), str(case["distance_m"]),
        str(case["sensor_delay_frames"]), str(case["wind_param"]),
        str(case["depth_noise_m"]), str(case["dropout_probability"]),
        str(case["command_delay_steps"]),
    ]
    begin = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
    elapsed = time.perf_counter() - begin
    if result.returncode:
        raise RuntimeError(
            f"Evaluation failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    metrics, eval_line, gpu_line = parse_metrics(result.stdout)
    if metrics["mode"] != str(case["mode"]) or metrics["family"] != str(case["family"]):
        raise RuntimeError(f"Evaluation output does not match requested case {case['case_id']}: {eval_line}")
    gpu_match = re.search(r"=([^\s]+)", gpu_line)
    return {
        **case,
        "episodes": metrics["episodes"],
        "success": metrics["success"],
        "collision": metrics["collision"],
        "timeout": metrics["timeout"],
        "progress": metrics["progress"],
        "mean_goal_time_s": metrics["mean_goal_time"],
        "mean_speed_mps": metrics["mean_speed"],
        "peak_speed_mps": metrics["peak_speed"],
        "min_clearance_m": metrics["min_clearance"],
        "sim_wall_s": metrics["wall_s"],
        "gpu_s": gpu_match.group(1) if gpu_match else "",
        "elapsed_wall_s": f"{elapsed:.6f}",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "raw_eval_line": eval_line,
        "raw_gpu_line": gpu_line,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, default=ROOT / "build/metal_nav_guided")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "results/guided-table-memory.bin.best")
    parser.add_argument("--output", type=Path, default=ROOT / "results/evaluation.csv")
    parser.add_argument("--timeout", type=float, default=120.0, help="timeout per evaluation, in seconds")
    parser.add_argument("--dry-run", action="store_true", help="print the planned matrix; do not launch GPU runs")
    parser.add_argument("--resume", action="store_true", help="skip case IDs already present in the output CSV")
    args = parser.parse_args()
    cli = args.cli if args.cli.is_absolute() else ROOT / args.cli
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if not cli.is_file():
        parser.error(f"CLI binary not found: {cli}")
    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}")

    cases = build_cases()
    if args.dry_run:
        print(f"planned_cases={len(cases)} output={output}")
        for case in cases:
            print(case["case_id"])
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    complete: set[str] = set()
    if args.resume and output.exists() and output.stat().st_size:
        with output.open(newline="") as f:
            complete = {row["case_id"] for row in csv.DictReader(f)}
    remaining = [case for case in cases if case["case_id"] not in complete]
    mode = "a" if args.resume and output.exists() and output.stat().st_size else "w"
    with output.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if mode == "w":
            writer.writeheader()
        for index, case in enumerate(remaining, 1):
            print(f"[{index}/{len(remaining)}] {case['case_id']}", flush=True)
            row = evaluate_case(cli, checkpoint, case, args.timeout)
            writer.writerow(row)
            f.flush()
    print(f"wrote={output} cases={len(remaining)} skipped={len(complete)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
