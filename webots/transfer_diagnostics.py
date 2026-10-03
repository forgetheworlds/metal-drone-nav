#!/usr/bin/env python3
"""Capture detailed, opt-in telemetry for three saved room-transfer cases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / "results"
SOURCE = HERE / "results/room-webots-transfer/cases.csv"
OUT = HERE / "results/transfer-diagnostics"
WEBOTS = Path("/Users/muadhsambul/embodied/work/Webots.app/Contents/MacOS/webots")
POLICY = ROOT / "assets/navigation-rooms-experimental.bin"
SELECTED = {
    "first_partition_contact": "f15-dev-0004-sd401bbed-w5e651fdf",
    "second_partition_floor_contact": "f15-dev-0001-s6958c653-w5e651fdf-mirror-y",
    "common_hard_success": "f15-dev-0014-scd18eac9-w5e651fdf",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def custom_data(source: str, diagnostic_prefix: str, nav_scale: float) -> str:
    import re
    match = re.search(r'customData\s+"([^"]*)"', source)
    if not match:
        raise ValueError("selected Webots route has no Robot.customData")
    values = dict(part.split("=", 1) for part in match.group(1).split(";") if "=" in part)
    values["diagnostic_prefix"] = diagnostic_prefix
    values["diagnostic_nav_scale"] = f"{nav_scale:.6f}"
    return source[:match.start()] + 'customData "' + ";".join(
        f"{key}={value}" for key, value in values.items()) + '"' + source[match.end():]


def verify_selection(rows: dict[str, dict]) -> None:
    for role, failure_id in SELECTED.items():
        if failure_id not in rows:
            raise ValueError(f"missing selected case {failure_id}")
        row = rows[failure_id]
        if row["difficulty_band"] != "hard":
            raise ValueError(f"selected case is not hard: {failure_id}")
        if role == "common_hard_success":
            if row["focused_metal_success"] != "1" or row["webots_success"] != "1":
                raise ValueError(f"control is not a common success: {failure_id}")
        elif row["focused_metal_success"] != "1" or row["webots_collision"] != "1":
            raise ValueError(f"contact selection is not Metal-pass/Webots-contact: {failure_id}")


def run_case(role: str, failure_id: str, row: dict[str, str], webots: Path, nav_scale: float = 1.0) -> dict:
    case_label = role if nav_scale == 1.0 else f"{role}-nav-scale-{nav_scale:g}"
    case_dir = OUT / case_label
    case_dir.mkdir(parents=True, exist_ok=True)
    source_world = Path(row["route_world"])
    if not source_world.is_file():
        raise FileNotFoundError(source_world)
    run_world = HERE / "worlds" / f".transfer-diagnostic-{case_label}.wbt"
    prefix = f"transfer-diagnostic-{case_label}"
    run_world.write_text(custom_data(source_world.read_text(), prefix, nav_scale))
    shutil.copy2(source_world, case_dir / "baseline-route.wbt")
    shutil.copy2(run_world, case_dir / "diagnostic-route.wbt")
    shutil.copy2(Path(row["episode_json"]), case_dir / "matrix-episode.json")
    shutil.copy2(Path(row["raw_trace_csv"]), case_dir / "matrix-trace.csv")
    matrix_case = Path(row["episode_json"]).parent
    for input_name in ("scene-metadata.json", "bank-record.json", "focused-selected-dev.csv"):
        source_input = matrix_case / input_name
        if source_input.is_file():
            shutil.copy2(source_input, case_dir / input_name)
    command = [str(webots), "--port=23456", "--minimize", "--batch", "--mode=fast", "--no-rendering",
               "--stdout", "--stderr", str(run_world)]
    try:
        process_list = subprocess.run(["ps", "-axo", "command="], capture_output=True, text=True, check=False).stdout
        owned_webots = [line for line in process_list.splitlines()
                        if str(webots) in line and str(HERE / "worlds") in line]
        if owned_webots:
            raise RuntimeError("an RL Webots process is already active; preserving shared controller outputs")
        for stale in (RESULTS / "last-run.json", RESULTS / "last-run-exit.marker", RESULTS / "last-run-trace.csv",
                      RESULTS / f"{prefix}-dense.csv", RESULTS / f"{prefix}-nav.csv", RESULTS / f"{prefix}-contacts.csv"):
            stale.unlink(missing_ok=True)
        process = subprocess.run(command, cwd=HERE, capture_output=True, text=True, timeout=90, check=False)
        (case_dir / "webots.log").write_text(process.stdout + process.stderr)
        if process.returncode:
            raise RuntimeError(f"Webots exit={process.returncode}; see {case_dir/'webots.log'}")
        shared = {
            "episode.json": RESULTS / "last-run.json",
            "exit.marker": RESULTS / "last-run-exit.marker",
            "dense-100hz.csv": RESULTS / f"{prefix}-dense.csv",
            "nav-20hz.csv": RESULTS / f"{prefix}-nav.csv",
            "contacts.csv": RESULTS / f"{prefix}-contacts.csv",
        }
        for name, path in shared.items():
            if not path.is_file():
                raise RuntimeError(f"missing diagnostic output {path}")
            shutil.copy2(path, case_dir / name)
        episode = json.loads((case_dir / "episode.json").read_text())
        if not episode.get("collision") and role != "common_hard_success":
            raise RuntimeError(f"diagnostic rerun did not reproduce contact for {failure_id}")
        if role == "common_hard_success" and not episode.get("success"):
            raise RuntimeError(f"diagnostic control did not reproduce success for {failure_id}")
        manifest = {
            "schema": "webots-transfer-diagnostic-v1", "role": role, "case_label": case_label,
            "failure_id": failure_id, "diagnostic_nav_scale": nav_scale,
            "command": command, "protocol": {"webots": "R2025a", "ode_step_ms": 1,
                "raptor_hz": 100, "navigation_hz": 20, "motor_sampling": "average",
                "physics_profile": "hover", "max_steps": 2000, "collision_radius_m": 0.18,
                "rendering": False, "actor_truth_inputs": False},
            "source_world_sha256": sha256(case_dir / "baseline-route.wbt"),
            "diagnostic_world_sha256": sha256(case_dir / "diagnostic-route.wbt"),
            "policy_sha256": sha256(POLICY), "controller_source_sha256": sha256(HERE / "controllers/raptor_webots/raptor_webots.cpp"),
            "episode": episode,
        }
        (case_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return {"role": role, "case_label": case_label, "failure_id": failure_id, "nav_scale": nav_scale,
                "success": episode.get("success"),
                "collision": episode.get("collision"), "steps": episode.get("steps"),
                "time_s": episode.get("time_s"), "dense_samples": sum(1 for _ in (case_dir / "dense-100hz.csv").open()) - 1,
                "nav_samples": sum(1 for _ in (case_dir / "nav-20hz.csv").open()) - 1,
                "contact_samples": sum(1 for _ in (case_dir / "contacts.csv").open()) - 1,
                "case_dir": str(case_dir)}
    finally:
        run_world.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--webots", type=Path, default=WEBOTS)
    parser.add_argument("--roles", default=",".join(SELECTED))
    parser.add_argument("--nav-scale", type=float, default=1.0,
                        help="diagnostic post-policy velocity/yaw scale in (0,1]; 1 preserves the saved matrix")
    args = parser.parse_args()
    if not args.webots.is_file():
        parser.error(f"Webots executable not found: {args.webots}")
    if not 0.0 < args.nav_scale <= 1.0:
        parser.error("--nav-scale must be in (0,1]")
    OUT.mkdir(parents=True, exist_ok=True)
    with SOURCE.open(newline="") as source:
        rows = {row["failure_id"]: row for row in csv.DictReader(source)}
    verify_selection(rows)
    selected_roles = args.roles.split(",")
    if any(role not in SELECTED for role in selected_roles):
        parser.error(f"roles must be from {','.join(SELECTED)}")
    outputs = []
    for role in selected_roles:
        failure_id = SELECTED[role]
        print(f"run role={role} failure_id={failure_id}", flush=True)
        outputs.append(run_case(role, failure_id, rows[failure_id], args.webots, args.nav_scale))
    summary_path = OUT / "summary.json"
    prior = json.loads(summary_path.read_text()) if summary_path.is_file() else {"cases": []}
    merged = {case["case_label"]: case for case in prior.get("cases", [])}
    merged.update({case["case_label"]: case for case in outputs})
    summary = {"schema": "webots-transfer-diagnostics-summary-v1", "cases": list(merged.values()),
               "matrix_csv_sha256": sha256(SOURCE), "policy_sha256": sha256(POLICY),
               "interpretation": "Selected DEV case replays only; no final split. nav_scale=1 preserves control; lower values are explicit external-action diagnostic variants."}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
