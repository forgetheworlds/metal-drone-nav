#!/usr/bin/env python3
"""Build reproducible figures from recorded navigation experiments.

Run from any directory with:
    python3 /path/to/RL/evidence.py

The script reads only recorded CSV/TSV tables and the checked-in benchmark
record. It does not run training, evaluation, or simulation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "artifacts"
INPUTS = ROOT / "evidence/inputs"
TRAINING_TSV = INPUTS / "training.tsv"
STATIC_EVAL = INPUTS / "evaluation.csv"
CONTINUATION = INPUTS / "continuation-evaluation.csv"
OLD_THREATS = INPUTS / "threat-evaluation.csv"
JOINT_THREATS = INPUTS / "threat-joint-controlled.csv"
CHALLENGE_BANK = INPUTS / "challenge-bank-v1.jsonl"
SPEED_SWEEP = INPUTS / "static-speed.csv"
INFERENCE_ABLATIONS = INPUTS / "threat-joint-ablations.csv"
CORNER_LOG = INPUTS / "corner-control.log"
CORNER_TRAINING = INPUTS / "corner-control-training.tsv"
MIRRORED_BANK = INPUTS / "challenge-bank-mirrored-v1.jsonl"
WITNESS_1MPS = INPUTS / "bank-witness-1mps.csv"
WITNESS_15MPS = INPUTS / "bank-witness-1.5mps.csv"
WITNESS_LOG_1MPS = INPUTS / "bank-witness-1mps.log"
WITNESS_LOG_15MPS = INPUTS / "bank-witness-1.5mps.log"
ARRIVAL_DIR = INPUTS / "arrival-training"
ARRIVAL_HISTORY = ARRIVAL_DIR / "history.csv"
ARRIVAL_EVALUATIONS = ARRIVAL_DIR / "evaluations.csv"
ARRIVAL_MANIFEST = ARRIVAL_DIR / "manifest.json"
ARRIVAL_RETENTION = ARRIVAL_DIR / "retention.csv"
WEBOTS_STABLE_DIR = INPUTS / "webots-stable-arrival"
WEBOTS_STABLE_EPISODES = WEBOTS_STABLE_DIR / "episodes.csv"
WEBOTS_STABLE_RUNS = WEBOTS_STABLE_DIR / "runs.json"
WEBOTS_STABLE_MANIFEST = WEBOTS_STABLE_DIR / "manifest.json"
BENCHMARKS = ROOT / "docs" / "BENCHMARKS.md"
CHALLENGE_FILES = {
    "Original PPO · mode 17": (INPUTS / "bank-original.csv", "guided-table-memory.bin.best", "17"),
    "Clean/stress PPO · mode 17": (INPUTS / "bank-static.csv", "guided-clean-stress.bin.best", "17"),
    "Threat-joint PPO · mode 17": (INPUTS / "bank-dynamic.csv", "guided-threat-joint.bin.best", "17"),
    "Geometry prior · mode 13": (INPUTS / "bank-geometry.csv", "guided-table-memory.bin.best", "13"),
    "Goal script · mode 2": (INPUTS / "bank-goal.csv", "guided-table-memory.bin.best", "2"),
}

COLORS = {
    "blue": "#176B87",
    "orange": "#D97732",
    "green": "#47856D",
    "red": "#B44B45",
    "purple": "#7763A8",
    "gray": "#68737D",
    "navy": "#253746",
}
EVAL_RE = re.compile(r"^eval family=\d+ mode=\d+ episodes=\d+ .*? min_clearance=\S+ wall_s=\S+$")
METRIC_RE = re.compile(r"([a-z_]+)=([^\s]+)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_training(path: Path) -> list[dict[str, Any]]:
    """Decode the headerless row schema written by main.mm: 8 TSV columns."""
    names = ["checkpoint", "rollout", "wall_s", "gpu_s", "success", "collision", "timeout", "goal_time_s"]
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as stream:
        for line_number, fields in enumerate(csv.reader(stream, delimiter="\t"), 1):
            if len(fields) != len(names):
                raise ValueError(f"{path}:{line_number}: expected 8 columns, got {len(fields)}")
            row: dict[str, Any] = dict(zip(names, fields))
            for name in names[1:]:
                row[name] = float(row[name])
            row["rollout"] = int(row["rollout"])
            row["checkpoint"] = str(row["checkpoint"])
            rows.append(row)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No data rows in {path}")
    return rows


def load_and_validate_arrival_evidence() -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, list[dict[str, str]]], dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    """Load the recorded open-domain training run and matched fresh-seed rows."""
    manifest = json.loads(ARRIVAL_MANIFEST.read_text())
    if manifest.get("schema") != "arrival-training-evidence-v1":
        raise ValueError("Unknown arrival-training evidence schema")
    if (int(manifest.get("train_seed", -1)) != 42 or
            int(manifest.get("selection_seed", -1)) != 800001 or
            int(manifest.get("fresh_evaluation_seed", -1)) != 820001 or
            int(manifest.get("selected_rollout", -1)) != 900 or
            int(manifest.get("transitions", -1)) != 4_096_000):
        raise ValueError("Arrival-training manifest does not match the recorded experiment")

    for filename, expected_hash in manifest.get("input_hashes", {}).items():
        path = ARRIVAL_DIR / filename
        if not path.is_file() or sha256(path) != expected_hash:
            raise ValueError(f"Arrival evidence input hash mismatch: {path}")
    for relative, expected_hash in manifest.get("source_hashes", {}).items():
        path = ROOT / relative
        if path.is_file() and (relative.endswith(".bin.best") or relative.startswith("assets/navigation-")) and sha256(path) != expected_hash:
            raise ValueError(f"Arrival source/checkpoint hash mismatch: {path}")

    with ARRIVAL_HISTORY.open(newline="") as stream:
        history = list(csv.DictReader(stream))
    required_history = {"rollout", "total_transitions", "wall_s", "success", "collision", "timeout", "mean_arrival_s"}
    if not history or not required_history.issubset(history[0]):
        raise ValueError("Arrival history has an unexpected schema")
    for row in history:
        for key in required_history:
            row[key] = float(row[key])
        row["rollout"] = int(row["rollout"])
        row["total_transitions"] = int(row["total_transitions"])
        if row["total_transitions"] != row["rollout"] * 4096:
            raise ValueError("Arrival history transition count does not match 128×32 rollout collection")
        if abs(row["success"] + row["collision"] + row["timeout"] - 1.0) > 1e-5:
            raise ValueError("Arrival history terminal rates do not sum to one")
    expected_rollouts = list(range(0, 1001, 50))
    if [row["rollout"] for row in history] != expected_rollouts:
        raise ValueError("Arrival history must contain the initial point and 50-rollout checkpoints through 1000")
    if any(b["wall_s"] <= a["wall_s"] for a, b in zip(history, history[1:])):
        raise ValueError("Arrival history wall time is not increasing")
    selected_rollout = int(manifest["selected_rollout"])
    selected = next((row for row in history if row["rollout"] == selected_rollout), None)
    if selected is None or selected["success"] != 1.0 or abs(selected["mean_arrival_s"] - 4.32735) > 1e-4:
        raise ValueError("Arrival selected rollout does not match the recorded selection result")

    initial_dev = read_csv(ARRIVAL_DIR / "initial-dev.csv")
    selected_dev = read_csv(ARRIVAL_DIR / "selected-dev.csv")
    if len(initial_dev) != 128 or len(selected_dev) != 128:
        raise ValueError("Arrival selection evaluation must use 128 episodes before and after training")
    for rows, label in ((initial_dev, "initial"), (selected_dev, "selected")):
        if any((r["seed"] != "800001" or r["stage"] != "0" or r["family"] != "0" or
                r["mode"] != "17" or r["domain_amplitude"] != "1") for r in rows):
            raise ValueError(f"Arrival {label} selection rows do not match the open-domain validation profile")
        if any(sum(int(r[k]) for k in ("success", "collision", "timeout")) != 1 for r in rows):
            raise ValueError(f"Arrival {label} selection rows contain an invalid terminal outcome")
    initial_rate = sum(int(r["success"]) for r in initial_dev) / len(initial_dev)
    selected_rate = sum(int(r["success"]) for r in selected_dev) / len(selected_dev)
    selected_mean = sum(float(r["time_s"]) for r in selected_dev if r["success"] == "1") / max(1, sum(int(r["success"]) for r in selected_dev))
    if abs(initial_rate - history[0]["success"]) > 1e-5 or abs(selected_rate - selected["success"]) > 1e-5 or abs(selected_mean - selected["mean_arrival_s"]) > 1e-4:
        raise ValueError("Arrival selection CSV does not match its history checkpoints")

    summaries = read_csv(ARRIVAL_EVALUATIONS)
    expected_cases = {(0, 0, 0), (0, 0, 1), (1, 0, 1), (2, 1, 1), (2, 2, 1), (2, 4, 1), (2, 5, 1)}
    if len(summaries) != 14:
        raise ValueError("Arrival fresh-seed evaluation must contain 14 paired domain rows")
    episodes: dict[str, list[dict[str, str]]] = {}
    for summary in summaries:
        if summary["policy"] not in ("original", "arrival"):
            raise ValueError("Unknown policy in arrival evaluation table")
        key = (int(summary["stage"]), int(summary["family"]), int(summary["domain_amplitude"]))
        if key not in expected_cases or summary["seed"] != "820001" or int(summary["episodes"]) != 128:
            raise ValueError("Arrival fresh-seed row has an unexpected stage, family, amplitude, seed, or count")
        if int(summary["successes"]) + int(summary["collisions"]) + int(summary["timeouts"]) != 128:
            raise ValueError("Arrival fresh-seed summary outcomes do not sum to 128")
        csv_path = ARRIVAL_DIR / Path(summary["csv_path"]).name
        rows = read_csv(csv_path)
        if len(rows) != 128 or len({row["env"] for row in rows}) != 128:
            raise ValueError(f"{csv_path.name} must contain one row for each of 128 environments")
        if any(row["seed"] != "820001" or int(row["stage"]) != key[0] or int(row["family"]) != key[1] or int(row["domain_amplitude"]) != key[2] or row["mode"] != "17" for row in rows):
            raise ValueError(f"{csv_path.name} does not match its summary condition")
        for row in rows:
            if sum(int(row[k]) for k in ("success", "collision", "timeout")) != 1:
                raise ValueError(f"{csv_path.name} contains a non-terminal or duplicate outcome")
        counts = (sum(int(row["success"]) for row in rows), sum(int(row["collision"]) for row in rows), sum(int(row["timeout"]) for row in rows))
        wanted_counts = tuple(int(summary[k]) for k in ("successes", "collisions", "timeouts"))
        if counts != wanted_counts:
            raise ValueError(f"{csv_path.name} outcomes differ from the summary table")
        episodes[f"{key[0]}-{key[1]}-{key[2]}-{summary['policy']}"] = rows

    pair_fields = ("env", "stage", "family", "domain_amplitude", "start_x", "start_y", "start_z", "start_yaw",
                   "initial_vx", "initial_vy", "initial_vz", "goal_x", "goal_y", "goal_z", "initial_distance_m",
                   "direct_clearance_m", "witness_length_m", "mass_kg")
    seen_cases: set[tuple[int, int, int]] = set()
    for key in expected_cases:
        original = episodes.get(f"{key[0]}-{key[1]}-{key[2]}-original")
        arrival = episodes.get(f"{key[0]}-{key[1]}-{key[2]}-arrival")
        if original is None or arrival is None:
            raise ValueError(f"Arrival fresh-seed pair is missing for {key}")
        for left, right in zip(original, arrival):
            if any(left[field] != right[field] for field in pair_fields):
                raise ValueError(f"Arrival fresh-seed pair uses different task or plant data for {key} env={left['env']}")
        seen_cases.add(key)
    if seen_cases != expected_cases:
        raise ValueError("Arrival fresh-seed matrix is incomplete")

    rich_path = ARRIVAL_DIR / "rich-dev.csv"
    rich_dev = read_csv(rich_path)
    challenge_rows = [json.loads(line) for line in MIRRORED_BANK.read_text().splitlines()]
    challenge_by_id = {row["failure_id"]: row for row in challenge_rows}
    dev_ids = {key for key, row in challenge_by_id.items() if row["split"] == "dev"}
    checkpoint_hash = manifest["source_hashes"]["assets/checkpoints/arrival-open-domain-experimental.bin.best"]
    bank_hash = sha256(MIRRORED_BANK)
    if len(rich_dev) != 90 or {row["failure_id"] for row in rich_dev} != dev_ids:
        raise ValueError("Arrival rich-bank result must cover the 90 mirrored dev levels")
    for row in rich_dev:
        level = challenge_by_id[row["failure_id"]]
        if (row["split"] != "dev" or int(row["episodes"]) != 1 or row["mode"] != "17" or
                row["speed_mps"] != "1.5" or row["max_steps"] != "400" or
                row["checkpoint_sha256"] != checkpoint_hash or row["bank_sha256"] != bank_hash or
                row["scene_sha256"] != level["scene_sha256"] or int(row["family"]) != int(level["family"])):
            raise ValueError(f"Arrival rich-bank provenance mismatch for {row['failure_id']}")
        if sum(int(row[k]) for k in ("success", "collision", "timeout")) != 1:
            raise ValueError(f"Arrival rich-bank row has a nonterminal result: {row['failure_id']}")
    family_outcomes = {}
    for family in (14, 15, 16):
        selected_rows = [row for row in rich_dev if int(row["family"]) == family]
        family_outcomes[family] = tuple(sum(int(row[k]) for row in selected_rows) for k in ("success", "collision", "timeout"))
        if len(selected_rows) != 30:
            raise ValueError(f"Arrival rich-bank family {family} does not contain 30 dev levels")
    if family_outcomes != {14: (0, 30, 0), 15: (0, 30, 0), 16: (25, 5, 0)}:
        raise ValueError(f"Arrival rich-bank development outcomes changed: {family_outcomes}")

    retention = read_csv(ARRIVAL_RETENTION)
    expected_retention = {4, 5, 7, 8, 10, 11, 12}
    if len(retention) != 14:
        raise ValueError("Arrival legacy-retention comparison must contain seven matched family pairs")
    retention_keys = set()
    for row in retention:
        key = (row["policy"], int(row["family"]))
        retention_keys.add(key)
        if (row["policy"] not in ("original", "arrival") or int(row["family"]) not in expected_retention or
                row["seed"] != "800001" or int(row["episodes"]) != 128 or row["budget_s"] != "10" or
                row["speed_cap_mps"] != "1.5" or row["success_rule"] != "first_goal_region_entry"):
            raise ValueError("Arrival legacy-retention row does not match the saved first-entry protocol")
        counts = [round(float(row[name]) * int(row["episodes"])) for name in ("success", "collision", "timeout")]
        if sum(counts) != 128:
            raise ValueError("Arrival legacy-retention outcome rates do not sum to 128 episodes")
    if retention_keys != {(policy, family) for policy in ("original", "arrival") for family in expected_retention}:
        raise ValueError("Arrival legacy-retention comparison is missing a policy/family pair")
    return history, summaries, episodes, manifest, rich_dev, retention


def load_and_validate_webots_stable_arrival() -> tuple[list[dict[str, str]], dict[str, Any]]:
    manifest = json.loads(WEBOTS_STABLE_MANIFEST.read_text())
    if manifest.get("schema") != "webots-stable-arrival-v1" or int(manifest.get("records", -1)) != 36:
        raise ValueError("Unknown or incomplete Webots stable-arrival evidence manifest")
    for name, expected_hash in manifest.get("input_hashes", {}).items():
        path = WEBOTS_STABLE_DIR / name
        if not path.is_file() or sha256(path) != expected_hash:
            raise ValueError(f"Webots stable-arrival input hash mismatch: {path}")

    rows = read_csv(WEBOTS_STABLE_EPISODES)
    run_records = json.loads(WEBOTS_STABLE_RUNS.read_text())
    if len(rows) != 36 or len(run_records) != 36:
        raise ValueError("Webots stable-arrival package must contain 36 episode records and run manifests")
    if abs(float(manifest["radius_m"]) - 0.35) > 1e-9 or abs(float(manifest["speed_max_mps"]) - 0.5) > 1e-9 or abs(float(manifest["dwell_s"]) - 0.2) > 1e-9 or abs(float(manifest["budget_s"]) - 20) > 1e-9:
        raise ValueError("Webots stable-arrival contract differs from the declared goal-hold protocol")
    if manifest.get("sensors") != "idealGPS/IMU/Gyro;cleanRangeFinder" or abs(float(manifest.get("collision_radius_m", 0.18)) - 0.18) > 1e-9:
        raise ValueError("Webots stable-arrival sensor or body-scoring contract changed")

    policies = {"navigation.bin": "original", "navigation-arrival-experimental.bin": "arrival"}
    actor_hashes = {"navigation.bin": manifest["source_hashes"]["assets/navigation.bin"],
                    "navigation-arrival-experimental.bin": manifest["source_hashes"]["assets/navigation-arrival-experimental.bin"]}
    for actor, expected_hash in actor_hashes.items():
        path = ROOT / "assets" / actor
        if not path.is_file() or sha256(path) != expected_hash:
            raise ValueError(f"Webots stable-arrival actor hash mismatch: {path}")
    families = {"doorway", "table_overhang", "mixed_clutter"}
    expected_seeds = set(range(41001, 41007))
    episodes_by_key = {}
    for row in rows:
        key = (row["policy"], row["world"], int(row["seed"]))
        if key in episodes_by_key:
            raise ValueError(f"Duplicate Webots stable-arrival row: {key}")
        episodes_by_key[key] = row
        if row["policy"] not in policies or row["family"] not in families or row["goal_objective"] != "hold":
            raise ValueError("Unexpected Webots policy, scene family, or objective")
        if int(row["seed"]) not in expected_seeds or int(row["steps"]) > 2000 or float(row["time_s"]) > 20.0 + 1e-6:
            raise ValueError("Webots episode exceeds the fixed 20-second/2000-step profile")
        if row["physics_step_ms"] != "1" or row["motor_sampling"] != "average" or row["physics_profile"] != "hover":
            raise ValueError("Webots episode physics or motor-sampling profile changed")
        if row["navigation_loaded"] != "True":
            raise ValueError("Webots episode did not load its navigation actor")
        if sum(row[field] == "True" for field in ("success", "collision", "timeout")) != 1:
            raise ValueError("Webots episode does not have one terminal outcome")
        if row["success"] == "True" and (float(row["final_error_m"]) > 0.35 + 1e-6 or
                                           float(row["final_world_speed_mps"]) > 0.5 + 1e-6 or
                                           float(row["goal_dwell_s"]) < 0.2 - 1e-6):
            raise ValueError("Webots success does not satisfy the stable-arrival rule")

    run_by_key = {}
    for record in run_records:
        episode = record["episode"]
        key = (episode["policy"], episode["world"], int(episode["seed"]))
        if key in run_by_key or key not in episodes_by_key:
            raise ValueError(f"Duplicate or unmatched Webots run manifest: {key}")
        run_by_key[key] = record
        episode_row = episodes_by_key[key]
        if (bool(episode["success"]) != (episode_row["success"] == "True") or
                bool(episode["collision"]) != (episode_row["collision"] == "True") or
                bool(episode["timeout"]) != (episode_row["timeout"] == "True")):
            raise ValueError(f"Webots run record outcome differs from episodes.csv: {key}")
        run_config = record["run_manifest"]
        if run_config.get("files_sha256", {}).get("navigation_actor") != actor_hashes[episode["policy"]]:
            raise ValueError(f"Webots run manifest actor hash mismatch: {key}")
        if (run_config.get("goal_objective") != "hold" or int(run_config.get("max_steps", 0)) != 2000 or
                int(run_config.get("physics_basic_time_step_ms", 0)) != 1 or
                int(run_config.get("raptor_control_period_ms", 0)) != 10 or
                int(run_config.get("navigation_period_ms", 0)) != 50 or
                abs(float(run_config.get("goal_radius_m", 0)) - 0.35) > 1e-9 or
                abs(float(run_config.get("goal_dwell_threshold_s", 0)) - 0.2) > 1e-9 or
                abs(float(run_config.get("goal_dwell_max_speed_mps", 0)) - 0.5) > 1e-9):
            raise ValueError(f"Webots hold/scorer configuration mismatch: {key}")
        if run_config.get("range_noise_stddev_m") != 0.0 or "ideal GPS" not in run_config.get("ego_sensors", ""):
            raise ValueError(f"Webots run does not use ideal ego sensing and clean range: {key}")
        if "sampled at 100 Hz" not in run_config.get("goal_contract", ""):
            raise ValueError(f"Webots hold score is not recorded at 100 Hz: {key}")
    if set(run_by_key) != set(episodes_by_key):
        raise ValueError("Webots episodes and run manifests do not match")

    for family in families:
        for policy in policies:
            selected = [row for row in rows if row["family"] == family and row["policy"] == policy]
            if len(selected) != 6 or {int(row["seed"]) for row in selected} != expected_seeds:
                raise ValueError(f"Expected the same six Webots seeds for {policy}/{family}")
    for row in rows:
        key = (row["policy"], row["world"], int(row["seed"]))
        paired_policy = "navigation.bin" if row["policy"] == "navigation-arrival-experimental.bin" else "navigation-arrival-experimental.bin"
        paired = episodes_by_key.get((paired_policy, row["world"], int(row["seed"])))
        if paired is None or paired["family"] != row["family"]:
            raise ValueError(f"Webots policy comparison is not paired by saved world/seed: {key}")
        if (paired["generator_parameters"] != row["generator_parameters"] or
                paired["obstacle_count"] != row["obstacle_count"] or
                paired["route_witness_length_m"] != row["route_witness_length_m"]):
            raise ValueError(f"Webots policy comparison uses different saved geometry: {key}")

    expected_totals = {"original": (0, 4, 14), "arrival": (17, 1, 0)}
    for label, policy_file in policies.items():
        selected = [row for row in rows if row["policy"] == label]
        actual = tuple(sum(row[field] == "True" for row in selected) for field in ("success", "collision", "timeout"))
        expected = expected_totals[policy_file]
        if actual != expected:
            raise ValueError(f"Webots stable-arrival totals changed for {policy_file}: {actual}")
        recorded = manifest["results"][policy_file]
        if actual != tuple(int(recorded[key]) for key in ("success", "collision", "timeout")):
            raise ValueError(f"Webots manifest outcome totals disagree for {policy_file}")
    return rows, manifest


def load_and_validate_challenge_evidence() -> tuple[dict[str, list[dict[str, str]]], dict[str, dict[str, Any]]]:
    records = [json.loads(line) for line in CHALLENGE_BANK.read_text().splitlines()]
    if len(records) != 270:
        raise ValueError(f"Expected the frozen 270-level bank, got {len(records)}")
    bank_by_id: dict[str, dict[str, Any]] = {}
    counts: dict[tuple[str, int], int] = {}
    for record in records:
        failure_id = record["failure_id"]
        if failure_id in bank_by_id:
            raise ValueError(f"Duplicate bank failure_id: {failure_id}")
        if record["split"] not in ("train", "dev", "final"):
            raise ValueError(f"Invalid bank split in {failure_id}")
        bank_by_id[failure_id] = record
        key = (record["split"], int(record["family"]))
        counts[key] = counts.get(key, 0) + 1
    for split in ("train", "dev", "final"):
        for family in (14, 15, 16):
            if counts.get((split, family)) != 30:
                raise ValueError(f"Expected 30 family-{family} levels in {split}; got {counts.get((split, family), 0)}")
    bank_hash = sha256(CHALLENGE_BANK)
    dev_ids = {key for key, record in bank_by_id.items() if record["split"] == "dev"}
    result_sets: dict[str, list[dict[str, str]]] = {}
    for label, (path, checkpoint_name, mode) in CHALLENGE_FILES.items():
        rows = read_csv(path)
        ids = {row["failure_id"] for row in rows}
        if len(rows) != 90 or ids != dev_ids:
            raise ValueError(f"{path.name} must contain exactly the 90 frozen dev levels")
        if len(ids) != len(rows):
            raise ValueError(f"{path.name} contains duplicate failure IDs")
        expected_checkpoint = sha256(ROOT / "assets" / "checkpoints" / checkpoint_name)
        for row in rows:
            record = bank_by_id[row["failure_id"]]
            if (row["split"] != "dev" or row["mode"] != mode or row["episodes"] != "1" or
                    row["speed_mps"] != "1.5" or row["max_steps"] != "400"):
                raise ValueError(f"{path.name} has an unexpected split, mode, episode count, speed, or budget")
            if row["bank_sha256"] != bank_hash or row["scene_sha256"] != record["scene_sha256"]:
                raise ValueError(f"{path.name} provenance does not match the challenge bank for {row['failure_id']}")
            if row["checkpoint_sha256"] != expected_checkpoint:
                raise ValueError(f"{path.name} checkpoint hash does not match {checkpoint_name}")
            if int(row["family"]) != int(record["family"]) or row["world_source_sha256"] != record["world_source_sha256"]:
                raise ValueError(f"{path.name} scene metadata mismatch for {row['failure_id']}")
            if sum(int(row[k]) for k in ("success", "collision", "timeout")) != 1:
                raise ValueError(f"{path.name} has a non-terminal episode result for {row['failure_id']}")
        result_sets[label] = rows
    return result_sets, bank_by_id


def load_and_validate_speed_sweep() -> list[dict[str, str]]:
    rows = read_csv(SPEED_SWEEP)
    expected = {(group, family, cap) for group, families in
                (("speed_clean", ("5", "7", "8")), ("speed_stressed", ("5", "8")))
                for family in families for cap in ("0.75", "1.0", "1.5", "2.0", "3.0")}
    observed = {(r["group"], r["family"], r["speed_mps"]) for r in rows}
    if len(rows) != 25 or observed != expected:
        raise ValueError("Speed sweep must contain the fixed 25 clean/combined-stress cases")
    for row in rows:
        if (row["mode"] != "17" or row["seed"] != "800001" or row["episodes"] != "128" or
                row["distance_m"] != "4.0"):
            raise ValueError("Speed sweep cases do not share mode, seed, episodes, and goal distance")
        if row["group"] == "speed_clean":
            disturbance = ("sensor_delay_frames", "wind_param", "depth_noise_m", "dropout_probability", "command_delay_steps")
            if any(float(row[field]) != 0 for field in disturbance):
                raise ValueError("Clean speed case unexpectedly contains a disturbance")
        else:
            wanted = {"sensor_delay_frames": 2, "wind_param": .5, "depth_noise_m": .05,
                      "dropout_probability": .1, "command_delay_steps": 1}
            if any(abs(float(row[field]) - value) > 1e-8 for field, value in wanted.items()):
                raise ValueError("Stressed speed case differs from the recorded combined-stress profile")
    return rows


def load_and_validate_inference_ablations() -> list[dict[str, str]]:
    rows = read_csv(INFERENCE_ABLATIONS)
    expected = {(group, family, mode) for group in ("ablation_clean", "ablation_stressed")
                for family in ("5", "7", "8", "10", "11") for mode in ("17", "18", "19")}
    observed = {(r["group"], r["family"], r["mode"]) for r in rows}
    if len(rows) != 30 or observed != expected:
        raise ValueError("Inference ablation must contain 30 matched mode/family/stress cases")
    for row in rows:
        if row["seed"] != "800001" or row["episodes"] != "128" or row["speed_mps"] != "1.5":
            raise ValueError("Inference ablations do not share the fixed seed, episode count, and speed")
        wanted = {"sensor_delay_frames": 0, "wind_param": 0, "depth_noise_m": 0,
                  "dropout_probability": 0, "command_delay_steps": 0}
        if row["group"] == "ablation_stressed":
            wanted = {"sensor_delay_frames": 2, "wind_param": .5, "depth_noise_m": .05,
                      "dropout_probability": .1, "command_delay_steps": 1}
        if any(abs(float(row[field]) - value) > 1e-8 for field, value in wanted.items()):
            raise ValueError("Inference ablation disturbance profile does not match its group")
    return rows


def load_and_validate_corner_control() -> tuple[list[dict[str, Any]], dict[str, float], dict[str, float]]:
    training = read_training(CORNER_TRAINING)
    if len(training) != 100 or any(Path(row["checkpoint"]).name != "guided-corner-control.bin" for row in training):
        raise ValueError("Corner control history must contain the 100 recorded rollout checkpoints")
    if [row["rollout"] for row in training] != list(range(10, 1001, 10)):
        raise ValueError("Corner control history has missing or reordered rollout rows")
    if any(row["success"] != 0 or row["collision"] != 1 for row in training):
        raise ValueError("Corner control training-selection validation is not uniformly 0%/100% success/collision")
    log_text = CORNER_LOG.read_text()
    eval_rows = [dict(METRIC_RE.findall(line)) for line in log_text.splitlines()
                 if EVAL_RE.match(line)]
    eval_rows = [row for row in eval_rows if row.get("family") == "14" and row.get("mode") == "17"]
    if len(eval_rows) < 2 or any(row.get("episodes") != "128" or row.get("success") != "0" or row.get("collision") != "1" for row in eval_rows):
        raise ValueError("Corner-control log lacks consistent before/after 128-episode validation")
    train_config = next((line for line in log_text.splitlines() if line.startswith("PPO training ")), "")
    if not all(token in train_config for token in ("family=14", "envs=128", "horizon=32", "speed=1.5", "distance=8")):
        raise ValueError("Corner-control log training configuration does not match the recorded targeted run")
    train_metrics = [dict(METRIC_RE.findall(line)) for line in log_text.splitlines() if line.startswith("train episodes=")]
    if not train_metrics or int(train_metrics[-1].get("episodes", "0")) != 82729:
        raise ValueError("Corner-control log is missing the final in-training episode summary")
    if float(train_metrics[-1]["success"]) != 0 or float(train_metrics[-1]["collision"]) < .999:
        raise ValueError("Corner-control in-training outcome does not match the reported failure")
    return training, eval_rows[0], eval_rows[-1]


def load_and_validate_mirrored_witnesses() -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, dict[str, Any]]]:
    records = [json.loads(line) for line in MIRRORED_BANK.read_text().splitlines()]
    if len(records) != 270:
        raise ValueError(f"Expected 270 mirrored-bank levels, got {len(records)}")
    bank = {r["failure_id"]: r for r in records}
    if len(bank) != len(records):
        raise ValueError("Mirrored bank contains duplicate failure IDs")
    dev_ids = {fid for fid, record in bank.items() if record["split"] == "dev"}
    if len(dev_ids) != 90 or sum(record.get("coordinate_transform") == "mirror_y" for record in records) != 135:
        raise ValueError("Mirrored bank does not contain the documented 90-level dev split and alternate-side scenes")
    bank_hash = sha256(MIRRORED_BANK)
    outputs = []
    for path, speed, budget in ((WITNESS_1MPS, 1.0, 60), (WITNESS_15MPS, 1.5, 20)):
        rows = read_csv(path)
        if len(rows) != 90 or {r["failure_id"] for r in rows} != dev_ids:
            raise ValueError(f"{path.name} does not cover the exact 90 mirrored dev levels")
        for row in rows:
            record = bank[row["failure_id"]]
            if (row["split"] != "dev" or row["controller"] != "frozen_RAPTOR" or
                    row["privileged_route"] != "true" or float(row["speed_cap_mps"]) != speed or
                    float(row["budget_s"]) != budget or row["bank_sha256"] != bank_hash):
                raise ValueError(f"{path.name} provenance/config mismatch for {row['failure_id']}")
            if int(row["family"]) != int(record["family"]):
                raise ValueError(f"{path.name} family mismatch for {row['failure_id']}")
            if sum(int(row[k]) for k in ("success", "collision", "timeout")) != 1:
                raise ValueError(f"{path.name} has nonterminal witness outcome for {row['failure_id']}")
        outputs.append(rows)
    one, one_five = outputs
    if any(int(row["success"]) != 1 for row in one):
        raise ValueError("1.0 m/s, 60-second witness run is not a 90/90 pass")
    if sum(int(row["success"]) for row in one_five) != 69 or sum(int(row["timeout"]) for row in one_five) != 21 or any(int(row["collision"]) for row in one_five):
        raise ValueError("1.5 m/s, 20-second witness outcomes do not match the recorded 69 pass / 21 timeout result")
    return one, one_five, bank


def wilson(success: float, episodes: int) -> tuple[float, float]:
    """95% Wilson interval for the recorded binomial episode rate."""
    if episodes <= 0:
        return 0.0, 1.0
    z = 1.959963984540054
    center = (success + z * z / (2 * episodes)) / (1 + z * z / episodes)
    half = z * math.sqrt(success * (1 - success) / episodes + z * z / (4 * episodes * episodes)) / (1 + z * z / episodes)
    return max(0.0, center - half), min(1.0, center + half)


def fixed_case(rows: list[dict[str, str]], *, mode: str, family: str,
               group: str = "baseline", seed: str = "800001") -> dict[str, str]:
    matches = [r for r in rows if r["mode"] == mode and r["family"] == family
               and r["group"] == group and r["seed"] == seed]
    if len(matches) != 1:
        raise ValueError(f"Expected one {group}/mode{mode}/family{family}/seed{seed} row; found {len(matches)}")
    return matches[0]


def check_threat_matrices(old: list[dict[str, str]], new: list[dict[str, str]]) -> None:
    def key(row: dict[str, str]) -> tuple[str, ...]:
        return tuple(row[k] for k in ("case_id", "mode", "family", "seed", "threat_kind", "threat_speed_mps", "nominal_ttc_s", "initial_vx_mps", "episodes"))

    old_keys = {key(r) for r in old}
    new_keys = {key(r) for r in new}
    if len(old) != 24 or len(new) != 24 or old_keys != new_keys:
        raise ValueError("Threat matrices are not the same complete 24-case set")
    for label, rows in (("original", old), ("joint-trained", new)):
        if any(int(r["episodes"]) != 128 for r in rows):
            raise ValueError(f"{label} threat matrix has a case other than 128 episodes")


def add_footer(fig: plt.Figure, text: str) -> None:
    fig.text(0.025, 0.018, text, ha="left", va="bottom", fontsize=8.2, color=COLORS["gray"])


def save_figure(fig: plt.Figure, out: Path, stem: str) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    png, svg = out / f"{stem}.png", out / f"{stem}.svg"
    fig.savefig(png, dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [png.name, svg.name]


def training_figure(rows: list[dict[str, Any]], out: Path) -> list[str]:
    series = [
        ("Raw 661", "raw-broad.bin", COLORS["blue"]),
        ("2×2 min-pooled 181", "pooled-broad.bin", COLORS["orange"]),
    ]
    selected: dict[str, list[dict[str, Any]]] = {}
    for label, filename, _ in series:
        selected[label] = sorted((r for r in rows if Path(r["checkpoint"]).name == filename), key=lambda r: r["rollout"])
        if not selected[label]:
            raise ValueError(f"Training history has no rows for {filename}")
    if len({r["rollout"] for r in selected["Raw 661"]}) < 2 or len({r["rollout"] for r in selected["2×2 min-pooled 181"]}) < 2:
        raise ValueError("Broad architecture training curves need at least two recorded points each")

    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    for label, _, color in series:
        data = selected[label]
        best = max(data, key=lambda r: r["success"])
        legend = f"{label} · best {best['success']:.1%} at rollout {best['rollout']}"
        ax.plot([r["wall_s"] for r in data], [r["success"] for r in data], label=legend,
                color=color, linewidth=2.2, marker="o", markersize=2.8)
        ax.scatter([best["wall_s"]], [best["success"]], color=color, edgecolor="white", linewidth=1.1, zorder=5)
    ax.set_title("Feature-compression experiment: broad-scene validation", loc="left", weight="bold", fontsize=14)
    ax.set_xlabel("Elapsed wall time at validation record (s)")
    ax.set_ylabel("Validation success fraction")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(axis="y", alpha=0.24)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.text(0.01, 0.98, "Same broad 0–6 scene mix and seed; validation uses 128 fixed-seed episodes.",
            transform=ax.transAxes, va="top", fontsize=9, color=COLORS["gray"])
    add_footer(fig, "Each point is a recorded training-selection evaluation. The best points are selected from the same history; they are not fresh-seed test scores.")
    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.21)
    return save_figure(fig, out, "training-broad-validation")


def arrival_training_figure(history: list[dict[str, Any]], out: Path) -> list[str]:
    wall = np.array([row["wall_s"] for row in history], dtype=float)
    transitions = np.array([row["total_transitions"] for row in history], dtype=float) / 1e6
    rollouts = np.array([row["rollout"] for row in history], dtype=int)
    success = np.array([row["success"] for row in history], dtype=float)
    collision = np.array([row["collision"] for row in history], dtype=float)
    timeout = np.array([row["timeout"] for row in history], dtype=float)
    arrival_time = np.array([row["mean_arrival_s"] for row in history], dtype=float)
    selected_index = int(np.where(rollouts == 900)[0][0])

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.0), sharex=True)
    axes[0].plot(wall, success, color=COLORS["blue"], marker="o", markersize=3.5, linewidth=2)
    axes[0].scatter([wall[selected_index]], [success[selected_index]], s=68, color=COLORS["green"],
                    edgecolor="white", linewidth=1.2, zorder=5, label="Selected checkpoint · rollout 900")
    axes[0].set_ylabel("Success")
    axes[0].set_ylim(-0.04, 1.06)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    axes[0].legend(frameon=False, loc="lower right", fontsize=8.5)
    axes[0].set_title("Open-domain PPO: stable arrival during training", loc="left", weight="bold", fontsize=14)

    axes[1].plot(wall, collision, color=COLORS["red"], marker="o", markersize=3, linewidth=1.7, label="Collision")
    axes[1].plot(wall, timeout, color=COLORS["orange"], marker="o", markersize=3, linewidth=1.7, label="Timeout")
    axes[1].set_ylabel("Episode fraction")
    axes[1].set_ylim(-0.03, 0.38)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    axes[1].legend(frameon=False, loc="upper right", ncol=2, fontsize=8.5)

    axes[2].plot(wall, arrival_time, color=COLORS["purple"], marker="o", markersize=3.5, linewidth=2)
    axes[2].scatter([wall[selected_index]], [arrival_time[selected_index]], s=68, color=COLORS["green"],
                    edgecolor="white", linewidth=1.2, zorder=5)
    axes[2].set_ylabel("Mean arrival time\n(successes only, s)")
    axes[2].set_xlabel("Elapsed wall time (s)")
    axes[2].set_ylim(0, max(arrival_time) * 1.16)
    axes[2].annotate("4.33 s at selected rollout 900", (wall[selected_index], arrival_time[selected_index]),
                     xytext=(-115, 13), textcoords="offset points", fontsize=8.5,
                     arrowprops={"arrowstyle": "-", "color": COLORS["gray"], "lw": 0.8})

    for ax in axes:
        ax.axvline(wall[selected_index], color=COLORS["green"], linestyle="--", linewidth=0.9, alpha=0.75)
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    def wall_to_transitions(value: np.ndarray) -> np.ndarray:
        return np.interp(value, wall, transitions)
    def transitions_to_wall(value: np.ndarray) -> np.ndarray:
        return np.interp(value, transitions, wall)
    top_axis = axes[0].secondary_xaxis("top", functions=(wall_to_transitions, transitions_to_wall))
    top_axis.set_xlabel("Collected transitions (millions)")
    top_axis.set_xticks(np.arange(0.0, 4.01, 0.5))
    add_footer(fig, "4.096M transitions over 34.75 s. Points are periodic fixed-seed validation records, not every rollout. Mean arrival time is conditional on success.")
    fig.subplots_adjust(left=0.14, right=0.98, top=0.91, bottom=0.11, hspace=0.22)
    return save_figure(fig, out, "arrival-open-training")


def arrival_domain_figure(summaries: list[dict[str, str]], out: Path) -> list[str]:
    rows = {(int(row["stage"]), int(row["family"]), int(row["domain_amplitude"]), row["policy"]): row
            for row in summaries}
    cases = [
        ((0, 0, 0), "Open room · nominal plant"),
        ((0, 0, 1), "Open room · randomized plant"),
        ((1, 0, 1), "Near-goal hold · randomized plant"),
        ((2, 1, 1), "Static boxes"),
        ((2, 2, 1), "Vertical poles"),
        ((2, 4, 1), "Doorway"),
        ((2, 5, 1), "Table / counter"),
    ]
    fig, ax = plt.subplots(figsize=(10.3, 6.6))
    ybase = np.arange(len(cases))[::-1]
    offsets = {"original": -0.14, "arrival": 0.14}
    styles = {"original": (COLORS["gray"], "o", "Original guided policy"),
              "arrival": (COLORS["blue"], "D", "Open-domain candidate")}
    for key, name in styles.items():
        color, marker, _ = name
        xs, ys, left_err, right_err = [], [], [], []
        annotations = []
        for index, (case, _) in enumerate(cases):
            row = rows[(*case, key)]
            episodes = int(row["episodes"])
            successes = int(row["successes"])
            rate = successes / episodes
            low, high = wilson(rate, episodes)
            xs.append(rate * 100)
            ys.append(ybase[index] + offsets[key])
            left_err.append((rate - low) * 100)
            right_err.append((high - rate) * 100)
            annotations.append(f"C {int(row['collisions'])} · T {int(row['timeouts'])}")
        ax.errorbar(xs, ys, xerr=np.array([left_err, right_err]), fmt=marker, color=color,
                    markersize=6.5, capsize=2.5, linewidth=1.4, label=styles[key][2], zorder=3)
        for x, y, note in zip(xs, ys, annotations):
            ax.text(103.0, y, note, va="center", fontsize=8.2, color=color)
    ax.set_yticks(ybase, [label for _, label in cases])
    ax.set_xlim(0, 117)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlabel("Stable-arrival success (% of 128 episodes; 95% Wilson interval)")
    ax.text(103.0, ybase[0] + 0.43, "Collision · timeout", va="center", fontsize=8.2,
            color=COLORS["gray"], weight="bold")
    ax.legend(frameon=False, loc="lower left", ncol=2, fontsize=8.5)
    ax.grid(axis="x", alpha=0.22)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_ylim(-0.58, len(cases) - 0.25)
    fig.suptitle("Fresh-seed outcomes: gains and remaining clutter failures", x=0.14, y=0.97,
                 ha="left", weight="bold", fontsize=14)
    fig.text(0.14, 0.92, "Seed 820001 · paired task rows and recorded mass · 20 s limit",
             ha="left", fontsize=9, color=COLORS["gray"])
    add_footer(fig, "One fresh evaluation seed, 128 episodes per condition. Amplitude 1 is a declared parameter-randomization range, not measured hardware uncertainty. Clutter includes direct paths and simple geometric witnesses; this is not a full route-planning benchmark.")
    fig.subplots_adjust(left=0.29, right=0.98, top=0.85, bottom=0.15)
    return save_figure(fig, out, "arrival-fresh-seed-domain")


def arrival_rich_bank_figure(rows: list[dict[str, str]], out: Path) -> list[str]:
    families = [(14, "Bent hallway corners"), (15, "Connected rooms"), (16, "Vertical choices")]
    outcomes = ("success", "collision", "timeout")
    colors = {"success": COLORS["green"], "collision": COLORS["red"], "timeout": COLORS["gray"]}
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ys = np.arange(len(families))[::-1]
    for y, (family, label) in zip(ys, families):
        selected = [row for row in rows if int(row["family"]) == family]
        total = len(selected)
        left = 0.0
        for outcome in outcomes:
            count = sum(int(row[outcome]) for row in selected)
            width = count / total * 100
            ax.barh(y, width, left=left, height=0.48, color=colors[outcome],
                    label=outcome.capitalize() if y == ys[0] else None)
            if count:
                ax.text(left + width / 2, y, str(count), ha="center", va="center",
                        color="white", fontsize=9, weight="bold")
            left += width
        ax.text(102, y, f"{sum(int(row['success']) for row in selected)}/{total}",
                va="center", fontsize=10, weight="bold", color=COLORS["navy"])
    ax.set_yticks(ys, [label for _, label in families])
    ax.set_xlim(0, 116)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlabel("Outcome fraction (% of 30 development levels)")
    ax.grid(axis="x", alpha=0.22)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle("Open-domain candidate fails the richer route bank", x=0.15, y=0.97,
                 ha="left", weight="bold", fontsize=14)
    fig.text(0.15, 0.92, "Mirrored dev split · mode 17 · 1.5 m/s requested cap · 20 s limit · green success, red collision",
             ha="left", fontsize=9, color=COLORS["gray"])
    add_footer(fig, "Final split untouched. This is a separate bank and not a paired comparison against the earlier checkpoint. It rejects the open-domain candidate as a general replacement.")
    fig.subplots_adjust(left=0.25, right=0.98, top=0.84, bottom=0.18)
    return save_figure(fig, out, "arrival-rich-bank-limit")


def arrival_legacy_retention_figure(rows: list[dict[str, str]], out: Path) -> list[str]:
    labels = [(4, "Doorway"), (5, "Table / counter"), (7, "Broad mixed scenes 0–6"),
              (8, "Held two-door scene"), (10, "Moving-sphere approach"),
              (11, "Moving-sphere crossing"), (12, "Rehearsal mix")]
    indexed = {(int(row["family"]), row["policy"]): row for row in rows}
    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    ybase = np.arange(len(labels))[::-1]
    styles = {"original": (COLORS["gray"], "o", "Original guided policy"),
              "arrival": (COLORS["blue"], "D", "Open-domain candidate")}
    for policy, (color, marker, legend) in styles.items():
        xs, ys, low_errors, high_errors = [], [], [], []
        notes = []
        for index, (family, _) in enumerate(labels):
            row = indexed[(family, policy)]
            episodes = int(row["episodes"])
            successes = round(float(row["success"]) * episodes)
            rate = successes / episodes
            low, high = wilson(rate, episodes)
            xs.append(rate * 100)
            ys.append(ybase[index] + (-0.14 if policy == "original" else 0.14))
            low_errors.append((rate - low) * 100)
            high_errors.append((high - rate) * 100)
            notes.append(int(round(float(row["collision"]) * episodes)))
        ax.errorbar(xs, ys, xerr=np.array([low_errors, high_errors]), fmt=marker,
                    color=color, markersize=6.3, capsize=2.5, linewidth=1.4, label=legend, zorder=3)
        if policy == "arrival":
            for index, (y, collisions) in enumerate(zip(ys, notes)):
                family = labels[index][0]
                previous = int(round(float(indexed[(family, "original")]["collision"]) * 128))
                ax.text(103, y, f"{previous} → {collisions}", va="center", fontsize=8.1, color=color)
    for index, (family, _) in enumerate(labels):
        before = round(float(indexed[(family, "original")]["success"]) * 128)
        after = round(float(indexed[(family, "arrival")]["success"]) * 128)
        y = ybase[index]
        ax.plot([before / 128 * 100, after / 128 * 100], [y - 0.14, y + 0.14],
                color=COLORS["gray"], linewidth=0.8, alpha=0.55, zorder=1)
        ax.text(137, y, f"{before} → {after}", ha="center", va="center", fontsize=8.1, color=COLORS["navy"])
    ax.set_yticks(ybase, [label for _, label in labels])
    ax.set_xlim(0, 155)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlabel("First-goal-entry success (% of 128 episodes; 95% Wilson interval)")
    ax.text(103, ybase[0] + 0.48, "C old → new", va="center", fontsize=8.1, color=COLORS["gray"], weight="bold")
    ax.text(137, ybase[0] + 0.48, "S old → new", ha="center", va="center", fontsize=8.1, color=COLORS["gray"], weight="bold")
    ax.legend(frameon=False, loc="lower left", ncol=2, fontsize=8.5)
    ax.grid(axis="x", alpha=0.22)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle("The new task improves arrival but loses some legacy performance", x=0.14, y=0.97,
                 ha="left", weight="bold", fontsize=14)
    fig.text(0.14, 0.92, "Seed 800001 · 128 episodes · same 1.5 m/s requested cap and 10 s budget",
             ha="left", fontsize=9, color=COLORS["gray"])
    add_footer(fig, "Both policies use the legacy first-goal-region-entry score here. It does not require the new stable hold. The rich-bank failure is separate evidence against using this candidate as a general replacement.")
    fig.subplots_adjust(left=0.30, right=0.98, top=0.86, bottom=0.16)
    return save_figure(fig, out, "arrival-legacy-retention-regression")


def webots_stable_arrival_figure(rows: list[dict[str, str]], out: Path) -> list[str]:
    families = [("doorway", "Doorways"), ("table_overhang", "Table overhangs"),
                ("mixed_clutter", "Mixed clutter")]
    outcomes = ("success", "collision", "timeout")
    outcome_color = {"success": COLORS["green"], "collision": COLORS["red"], "timeout": COLORS["gray"]}
    policy_specs = [("navigation.bin", "Original policy", -0.19, ""),
                    ("navigation-arrival-experimental.bin", "Arrival candidate", 0.19, "///")]
    fig, ax = plt.subplots(figsize=(8.9, 5.2))
    centers = np.arange(len(families))
    width = 0.32
    for policy, label, offset, hatch in policy_specs:
        for index, (family, _) in enumerate(families):
            selected = [row for row in rows if row["policy"] == policy and row["family"] == family]
            counts = {outcome: sum(row[outcome] == "True" for row in selected) for outcome in outcomes}
            x = centers[index] + offset
            bottom = 0.0
            for outcome in outcomes:
                count = counts[outcome]
                height = count / 6
                ax.bar(x, height, width, bottom=bottom, color=outcome_color[outcome],
                       edgecolor=COLORS["navy"] if hatch else "white", linewidth=0.6,
                       hatch=hatch, zorder=3)
                if count:
                    ax.text(x, bottom + height / 2, str(count), ha="center", va="center",
                            fontsize=8.5, color="white", weight="bold")
                bottom += height
            ax.text(x, 1.025, f"{counts['success']}/6", ha="center", va="bottom",
                    fontsize=8.5, color=COLORS["navy"], weight="bold")
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="white", edgecolor=COLORS["gray"], label="Original policy"),
        Patch(facecolor="white", edgecolor=COLORS["navy"], hatch="///", label="Arrival candidate"),
        *[Patch(facecolor=outcome_color[outcome], edgecolor="white", label=outcome.capitalize()) for outcome in outcomes],
    ]
    ax.legend(handles=handles, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.19), fontsize=8.5)
    ax.set_xticks(centers, [label for _, label in families])
    ax.set_ylim(0, 1.15)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.set_ylabel("Episode outcome (6 matched seeds per family)")
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Webots stable-arrival check", x=0.11, y=0.97, ha="left", weight="bold", fontsize=14)
    fig.text(0.11, 0.92, "R2025a · same 18 static scenes/seeds · 20 s · radius 0.35 m · speed ≤0.5 m/s · dwell 0.2 s",
             ha="left", fontsize=9, color=COLORS["gray"])
    add_footer(fig, "100 Hz hold scoring, 1 ms physics, ideal GPS/IMU/Gyro, clean RangeFinder, 0.18 m collision sphere. The arrival candidate passes 17/18 here; this small static-scene transfer check is not hardware evidence or broad generalization.")
    fig.subplots_adjust(left=0.12, right=0.98, top=0.77, bottom=0.18)
    return save_figure(fig, out, "webots-stable-arrival")


def mode_comparison_figure(rows: list[dict[str, str]], out: Path) -> list[str]:
    family_order = [("7", "Mixed"), ("8", "Two doors"), ("5", "Table/counter"), ("3", "Moving spheres")]
    modes = [("17", "Guided learned", COLORS["blue"]), ("13", "Geometry prior", COLORS["green"]), ("2", "Goal script", COLORS["orange"])]
    fig, ax = plt.subplots(figsize=(10.2, 6.0))
    group_width = 0.78
    bar_width = group_width / len(modes)
    centers = np.arange(len(family_order))
    for mi, (mode, name, color) in enumerate(modes):
        xs, ys, low, high = [], [], [], []
        for fi, (family, _) in enumerate(family_order):
            row = fixed_case(rows, mode=mode, family=family)
            if int(row["episodes"]) != 128:
                raise ValueError("Static mode matrix must use 128 completed episodes")
            rate = float(row["success"])
            lo, hi = wilson(rate, int(row["episodes"]))
            xs.append(centers[fi] - group_width / 2 + (mi + 0.5) * bar_width)
            ys.append(rate)
            low.append(rate - lo)
            high.append(hi - rate)
        ax.bar(xs, ys, width=bar_width * 0.92, color=color, label=name, zorder=3)
        ax.errorbar(xs, ys, yerr=np.array([low, high]), fmt="none", ecolor=COLORS["navy"], capsize=2.5, linewidth=1, zorder=4)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.025, f"{y:.0%}", ha="center", va="bottom", fontsize=7.7, rotation=0)
    ax.set_xticks(centers, [label for _, label in family_order])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Episode success fraction")
    ax.set_title("One evaluation matrix, three navigation modes", loc="left", weight="bold", fontsize=14)
    ax.text(0.01, 0.98, "Same seed 800001; 128 episodes per bar. Whiskers are 95% Wilson intervals.",
            transform=ax.transAxes, va="top", fontsize=9, color=COLORS["gray"])
    ax.grid(axis="y", alpha=0.2, zorder=0)
    fig.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.91))
    add_footer(fig, "Family 8 was held out during training. These fixed-scene tests do not establish broad real-world generalization.")
    fig.subplots_adjust(left=0.11, right=0.98, top=0.79, bottom=0.22)
    return save_figure(fig, out, "static-scene-policy-comparison")


def transfer_figure(original: list[dict[str, str]], continuation: list[dict[str, str]], out: Path) -> list[str]:
    candidates = [
        ("Selected table-memory", "original", COLORS["navy"]),
        ("Broad stress", "broad_stress", COLORS["blue"]),
        ("Door stress", "door_stress", COLORS["orange"]),
        ("Rehearsal", "rehearsal", COLORS["green"]),
    ]
    profiles = [
        ("Clean\ntwo doors", "baseline", "8"),
        ("Clean\ntable/counter", "baseline", "5"),
        ("Combined stress\ntwo doors", "combined", "8"),
        ("Combined stress\ntable/counter", "combined", "5"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    x = np.arange(len(profiles))
    width = 0.18
    for ci, (label, candidate, color) in enumerate(candidates):
        ys, lows, highs = [], [], []
        for group, family in ((group, family) for _, group, family in profiles):
            if candidate == "original":
                row = fixed_case(original, mode="17", family=family, group=group)
            else:
                matches = [r for r in continuation if r["candidate"] == candidate and r["mode"] == "17" and r["family"] == family and r["group"] == group and r["seed"] == "800001"]
                if len(matches) != 1:
                    raise ValueError(f"Expected one {candidate}/{group}/family{family} case; found {len(matches)}")
                row = matches[0]
            rate = float(row["success"])
            lo, hi = wilson(rate, int(row["episodes"]))
            ys.append(rate)
            lows.append(rate - lo)
            highs.append(hi - rate)
        xs = x - (len(candidates) - 1) * width / 2 + ci * width
        ax.bar(xs, ys, width=width * 0.94, label=label, color=color, zorder=3)
        ax.errorbar(xs, ys, yerr=np.array([lows, highs]), fmt="none", ecolor=COLORS["navy"], capsize=2, linewidth=0.9, zorder=4)
    ax.set_xticks(x, [label for label, _, _ in profiles])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Episode success fraction")
    ax.set_title("Robustness training moved the tradeoff", loc="left", weight="bold", fontsize=14)
    ax.text(0.01, 0.98, "Matched evaluation settings and seed 800001; 128 episodes per case.",
            transform=ax.transAxes, va="top", fontsize=9, color=COLORS["gray"])
    ax.grid(axis="y", alpha=0.2, zorder=0)
    fig.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.91))
    add_footer(fig, "Candidates were selected with different training profile sets. Per-profile results expose regressions: the door-stress candidate loses table skill; no bar is a universal score.")
    fig.subplots_adjust(left=0.11, right=0.98, top=0.79, bottom=0.22)
    return save_figure(fig, out, "clean-and-stress-transfer")


def threat_figure(old: list[dict[str, str]], new: list[dict[str, str]], out: Path) -> list[str]:
    check_threat_matrices(old, new)
    sets = [("Original selected policy", old), ("Threat-joint candidate", new)]
    categories = []
    for kind in ("approach", "crossing"):
        for speed in ("0.5", "2.0"):
            for ttc in ("0.5", "1.0"):
                categories.append((kind, speed, ttc))
    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    x = np.arange(len(categories))
    colors = [COLORS["blue"], COLORS["red"]]
    threat_success: list[list[float]] = []
    for label, data, color in ((sets[0][0], sets[0][1], colors[0]), (sets[1][0], sets[1][1], colors[1])):
        y = []
        for kind, speed, ttc in categories:
            matches = [r for r in data if r["mode"] == "17" and r["threat_kind"] == kind
                       and float(r["threat_speed_mps"]) == float(speed)
                       and float(r["nominal_ttc_s"]) == float(ttc)]
            if len(matches) != 1:
                raise ValueError(f"Expected one guided threat row for {kind}/{speed}/{ttc}")
            y.append(float(matches[0]["success"]))
        ax.plot(x, y, marker="o", markersize=5, linewidth=2.1, label=label, color=color)
        threat_success.append(y)
    for xi, before, after in zip(x, threat_success[0], threat_success[1]):
        if abs(before - after) >= 0.05:
            offset = (14, -20) if xi == 0 else ((-34, -20) if xi == len(x) - 1 else (0, 10))
            ax.annotate(f"{before:.0%} → {after:.0%}", (xi, max(before, after)),
                        xytext=offset, textcoords="offset points", ha="center", fontsize=8.2,
                        color=COLORS["navy"], weight="bold")
    labels = [f"{kind}\n{speed} m/s · TTC {ttc}s" for kind, speed, ttc in categories]
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylim(-0.08, 1.16)
    ax.set_ylabel("Episode success fraction")
    ax.set_title("Fast moving threats remain a hard boundary", loc="left", weight="bold", fontsize=14)
    ax.text(0.01, 0.98, "Guided mode 17; 128 episodes per case; same 24-case matrix and seed.",
            transform=ax.transAxes, va="top", fontsize=9, color=COLORS["gray"])
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, loc="upper right")
    add_footer(fig, "The later candidate improves 2 m/s approach at nominal TTC 0.5s (0%→75%) and crossing TTC 1s (0%→100%). Both policies remain at 0% for 2 m/s crossing at TTC 0.5s. TTC is a scene parameter, not measured path collision time.")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.86, bottom=0.25)
    return save_figure(fig, out, "controlled-threat-before-after")


def challenge_dev_figure(result_sets: dict[str, list[dict[str, str]]], out: Path) -> list[str]:
    from matplotlib.lines import Line2D

    family_info = [
        (14, "Bent hallway corner"),
        (15, "Connected rooms + doors"),
        (16, "Over / under choice"),
    ]
    policies = list(CHALLENGE_FILES.keys())
    short_labels = ["Original\nPPO 17", "Clean/stress\nPPO 17", "Threat-joint\nPPO 17", "Geometry\nprior 13", "Goal\nscript 2"]
    outcomes = [
        ("success", "Success", COLORS["green"]),
        ("collision", "Collision", COLORS["red"]),
        ("timeout", "Timeout", COLORS["orange"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.2), sharey=True)
    x = np.arange(len(policies))
    width = 0.63
    for ax, (family, family_label) in zip(axes, family_info):
        for index, policy in enumerate(policies):
            rows = [row for row in result_sets[policy] if int(row["family"]) == family]
            if len(rows) != 30:
                raise ValueError(f"Expected 30 dev rows for family {family} in {policy}")
            bottom = 0.0
            for field, _, color in outcomes:
                count = sum(int(row[field]) for row in rows)
                fraction = count / len(rows)
                if fraction:
                    ax.bar(index, fraction, width, bottom=bottom, color=color, edgecolor="white", linewidth=0.7)
                    if fraction >= 0.13:
                        ax.text(index, bottom + fraction / 2, str(count), ha="center", va="center",
                                fontsize=8, color="white", weight="bold")
                bottom += fraction
            rate = sum(int(row["success"]) for row in rows) / len(rows)
            lo, hi = wilson(rate, len(rows))
            ax.errorbar(index, rate, yerr=[[rate - lo], [hi - rate]], fmt="_", color=COLORS["navy"],
                        capsize=3, linewidth=1.2, markersize=12, zorder=5)
            counts = [sum(int(row[field]) for row in rows) for field, _, _ in outcomes]
            ax.text(index, 1.045, f"{counts[0]}/{counts[1]}/{counts[2]}", ha="center", va="bottom",
                    fontsize=7.2, color=COLORS["navy"])
        ax.set_title(f"Family {family}: {family_label}", loc="left", weight="bold", fontsize=10.5)
        ax.set_xticks(x, short_labels, fontsize=8)
        ax.set_ylim(0, 1.22)
        ax.grid(axis="y", alpha=0.2, zorder=0)
    axes[0].set_ylabel("Fraction of 30 held development levels")
    fig.suptitle("Frozen challenge bank exposes a generalization gap", x=0.04, y=0.985,
                 ha="left", weight="bold", fontsize=15)
    fig.text(0.04, 0.94, "Same 90 dev level IDs · 30 per family · speed 1.5 m/s · at most 400 steps (20 s) · each row is one episode",
             fontsize=9, color=COLORS["gray"])
    handles = [plt.Rectangle((0, 0), 1, 1, color=color, label=label) for _, label, color in outcomes]
    handles.append(Line2D([], [], color=COLORS["navy"], marker="_", linestyle="None", markersize=12,
                          label="95% Wilson interval on success"))
    fig.legend(handles=handles, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.89), fontsize=8.5)
    fig.text(0.025, 0.018, "Counts above bars are success/collision/timeout. Final split (90 levels) has no evaluation rows. The geometry prior is a non-learned baseline; geometric witness routes only show that a path exists.",
             fontsize=8.2, color=COLORS["gray"])
    fig.subplots_adjust(left=0.07, right=0.99, top=0.79, bottom=0.20, wspace=0.10)
    return save_figure(fig, out, "challenge-bank-held-dev-outcomes")


def challenge_worlds_figure(bank_by_id: dict[str, dict[str, Any]], out: Path) -> list[str]:
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    representatives = []
    for family in (14, 15, 16):
        options = [record for record in bank_by_id.values()
                   if record["split"] == "dev" and int(record["family"]) == family]
        if len(options) != 30:
            raise ValueError(f"Expected 30 dev bank levels for family {family}")
        representatives.append(min(options, key=lambda record: (abs(float(record["difficulty"]) - 0.5),
                                                                  -float(record["witness_min_clearance_m"]))))

    def cuboid_faces(center: list[float], half: list[float]) -> list[list[tuple[float, float, float]]]:
        x0, x1 = center[0] - half[0], center[0] + half[0]
        y0, y1 = center[1] - half[1], center[1] + half[1]
        z0, z1 = center[2] - half[2], center[2] + half[2]
        v = [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),
             (x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)]
        return [[v[i] for i in face] for face in (
            (0,1,2,3),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7))]

    fig = plt.figure(figsize=(17.0, 7.4))
    legend_handles = [
        Line2D([0],[0], color=COLORS["green"], linewidth=2.4, marker="o", markersize=4, label="Geometric witness route"),
        Line2D([0],[0], color=COLORS["red"], linewidth=1.5, linestyle="--", label="Direct start-to-goal line"),
        Line2D([0],[0], marker="*", color="w", markerfacecolor=COLORS["orange"], markersize=12, label="Goal"),
        plt.Rectangle((0,0),1,1,facecolor="#718096",edgecolor="#344054",alpha=0.36,label="Saved AABB obstacle"),
    ]
    for index, record in enumerate(representatives, 1):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        for obstacle in record["obstacles"]:
            if int(obstacle["kind"]) != 0:
                raise ValueError("Challenge bank representative renderer expects its saved AABB schema")
            faces = cuboid_faces(obstacle["center"], obstacle["half_extent"])
            ax.add_collection3d(Poly3DCollection(faces, facecolors="#718096", edgecolors="#344054",
                                                 linewidths=0.28, alpha=0.36))
        route = np.asarray(record["witness_route"], dtype=float)
        goal = np.asarray(record["goal"], dtype=float)
        ax.plot(route[:,0], route[:,1], route[:,2], color=COLORS["green"], linewidth=2.4,
                marker="o", markersize=2.7, zorder=5)
        ax.plot([0, goal[0]], [0, goal[1]], [1.5, goal[2]], color=COLORS["red"],
                linewidth=1.5, linestyle="--", alpha=0.85)
        ax.scatter([0], [0], [1.5], color=COLORS["blue"], s=35, depthshade=False)
        ax.scatter([goal[0]], [goal[1]], [goal[2]], marker="*", color=COLORS["orange"],
                   edgecolor="white", s=140, depthshade=False)
        # Draw the fixed room wireframe; this is saved bank geometry, not a learned trajectory.
        bounds = ((-2, 14), (-5, 5), (0, 5))
        corners = [(x,y,z) for x in bounds[0] for y in bounds[1] for z in bounds[2]]
        for i, first in enumerate(corners):
            for second in corners[i+1:]:
                differing = [axis for axis in range(3) if first[axis] != second[axis]]
                if len(differing) == 1:
                    ax.plot([first[0],second[0]],[first[1],second[1]],[first[2],second[2]],
                            color="#AAB4BE", linewidth=0.45, alpha=0.5)
        ax.set(xlim=bounds[0], ylim=bounds[1], zlim=bounds[2], xlabel="World X (m)",
               ylabel="World Y (m)", zlabel="World Z (m)")
        ax.set_box_aspect((16,10,5))
        ax.view_init(elev=27, azim=-59)
        ax.set_title(f"Family {record['family']} · {record['family_name']}\n"
                     f"{record['failure_id']} · {record['difficulty_band']} ({record['difficulty']:.2f})\n"
                     f"{len(record['obstacles'])} obstacles · witness clearance {record['witness_min_clearance_m']:.2f} m",
                     loc="left", fontsize=8.7, weight="bold", pad=8)
    fig.suptitle("Challenge bank representative worlds", x=0.035, y=0.99, ha="left", weight="bold", fontsize=15)
    fig.text(0.035, 0.935, "One medium-difficulty dev scene per family · exact saved AABBs and goal · room frame XYZ, Z up · geometry from challenge-bank-v1.jsonl",
             fontsize=9, color=COLORS["gray"])
    fig.legend(handles=legend_handles, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.90), fontsize=8.5)
    fig.text(0.025, 0.018, "Green route is the bank’s geometric clearance witness for a 0.18 m vehicle; it proves only that a collision-free path exists. It is not a learned flight or policy output.",
             fontsize=8.2, color=COLORS["gray"])
    fig.subplots_adjust(left=0.02, right=0.99, top=0.79, bottom=0.12, wspace=0.03)
    return save_figure(fig, out, "challenge-bank-witness-worlds")


def corner_control_figure(training: list[dict[str, Any]], baseline: dict[str, float],
                          final: dict[str, float], out: Path) -> list[str]:
    rollouts = [0] + [row["rollout"] for row in training]
    success = [float(baseline["success"])] + [row["success"] for row in training]
    collision = [float(baseline["collision"])] + [row["collision"] for row in training]
    if float(final["success"]) != success[-1] or float(final["collision"]) != collision[-1]:
        raise ValueError("Final corner validation does not match the last training-selection row")
    fig, ax = plt.subplots(figsize=(9.4, 5.5))
    ax.plot(rollouts, success, color=COLORS["green"], linewidth=2.1, marker="o", markersize=2.8,
            label="Validation success")
    ax.plot(rollouts, collision, color=COLORS["red"], linewidth=2.1, marker="o", markersize=2.8,
            label="Validation collision")
    ax.scatter([0, 1000], [0, 0], color=COLORS["green"], edgecolor="white", zorder=5)
    ax.annotate("before warm start\n0 / 128 successes", (0, 0), xytext=(22, 28), textcoords="offset points",
                fontsize=8.5, color=COLORS["green"])
    ax.annotate("after 1,000 PPO rollouts\n0 / 128 successes", (1000, 0), xytext=(-150, 30), textcoords="offset points",
                fontsize=8.5, color=COLORS["green"])
    ax.set_xlim(-25, 1025)
    ax.set_ylim(-0.12, 1.12)
    ax.set_xlabel("PPO training rollouts")
    ax.set_ylabel("Family-14 validation outcome fraction")
    ax.set_title("Targeted corner training did not learn a detour", loc="left", weight="bold", fontsize=14)
    ax.text(0.01, 0.98, "Family 14 · 1.5 m/s intent · 8 m goal · same 128-episode selection seed before and after",
            transform=ax.transAxes, va="top", fontsize=8.8, color=COLORS["gray"])
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, loc="center right")
    add_footer(fig, "This single 1,000-rollout warm-start run used 4.096M transitions and logged 82,729 training episodes at 99.994% collision. It shows this data/schedule was insufficient; it does not prove family 14 is unlearnable.")
    fig.subplots_adjust(left=0.12, right=0.98, top=0.86, bottom=0.22)
    return save_figure(fig, out, "corner-control-training-failure")


def inference_ablation_figure(rows: list[dict[str, str]], out: Path) -> list[str]:
    from matplotlib.patches import Patch

    families = [("5", "Table/counter"), ("8", "Held two doors")]
    modes = [("17", "Full history·17", COLORS["blue"]),
             ("18", "Newest-only guidance·18", COLORS["orange"]),
             ("19", "Fixed nonzero intent·19", COLORS["purple"])]
    groups = [("ablation_clean", "Clean"), ("ablation_stressed", "Combined stress")]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.8), sharey=True)
    centers = np.arange(len(groups))
    width = 0.23
    for ax, (family, family_name) in zip(axes, families):
        for mi, (mode, label, color) in enumerate(modes):
            xs, values, lows, highs = [], [], [], []
            for gi, (group, _) in enumerate(groups):
                case = [r for r in rows if r["family"] == family and r["mode"] == mode and r["group"] == group]
                if len(case) != 1:
                    raise ValueError(f"Missing/duplicate inference ablation {family}/{mode}/{group}")
                rate = float(case[0]["success"])
                lo, hi = wilson(rate, int(case[0]["episodes"]))
                xs.append(centers[gi] - width + mi * width)
                values.append(rate)
                lows.append(rate - lo)
                highs.append(hi - rate)
            ax.bar(xs, values, width=width * .94, color=color, label=label, zorder=3)
            ax.errorbar(xs, values, yerr=np.array([lows, highs]), fmt="none", ecolor=COLORS["navy"],
                        capsize=2.5, linewidth=1, zorder=4)
            for x0, value in zip(xs, values):
                ax.text(x0, value + .025, f"{value:.0%}", ha="center", va="bottom", fontsize=7.6)
        ax.set_title(family_name, loc="left", weight="bold")
        ax.set_xticks(centers, [name for _, name in groups])
        ax.set_ylim(0, 1.13)
        ax.grid(axis="y", alpha=.2, zorder=0)
    axes[0].set_ylabel("Success in 128 episodes")
    fig.suptitle("Inference perturbations show history helps some tasks, not all", x=.04, y=.985,
                 ha="left", fontsize=14, weight="bold")
    fig.text(.04, .94, "Same threat-joint checkpoint and seed 800001 · no retraining · combined stress: 100 ms sensor + 50 ms command + noise/dropout/wind",
             fontsize=8.5, color=COLORS["gray"])
    fig.legend(handles=[Patch(facecolor=color, label=label) for _, label, color in modes],
               frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5,.89), fontsize=8.4)
    fig.text(.025,.018,"Mode 18 duplicates the prior-depth actor channel with current depth and limits pose/depth guidance to the newest frame; geometric guidance stays enabled. Mode 19 rescales each nonzero command to the requested cap. This is an inference ablation, not a trained-architecture comparison.",
             fontsize=7.8,color=COLORS["gray"])
    fig.subplots_adjust(left=.1,right=.98,top=.78,bottom=.22,wspace=.14)
    return save_figure(fig,out,"inference-history-and-speed-ablations")


def speed_sweep_figure(rows: list[dict[str, str]], out: Path) -> list[str]:
    caps = np.asarray([.75, 1.0, 1.5, 2.0, 3.0])
    families = [("5", "Table/counter", COLORS["blue"]),
                ("7", "Mixed scenes", COLORS["green"]),
                ("8", "Held two doors", COLORS["orange"])]
    fig, (success_ax, speed_ax) = plt.subplots(1,2,figsize=(13.8,6.0),gridspec_kw={"width_ratios":[1.55,1.0]})
    for family, name, color in families:
        clean = sorted((r for r in rows if r["family"]==family and r["group"]=="speed_clean"),key=lambda r:float(r["speed_mps"]))
        if len(clean)!=5: raise ValueError(f"Incomplete clean speed sweep for family {family}")
        y=[float(r["success"]) for r in clean]
        yerr=[]
        for r,rate in zip(clean,y):
            lo,hi=wilson(rate,int(r["episodes"]));yerr.append((rate-lo,hi-rate))
        success_ax.errorbar(caps,y,yerr=np.asarray(yerr).T,marker="o",linewidth=2,color=color,capsize=2.5,label=f"{name} · clean")
    for family, name, color in (families[0],families[2]):
        stress=sorted((r for r in rows if r["family"]==family and r["group"]=="speed_stressed"),key=lambda r:float(r["speed_mps"]))
        if len(stress)!=5: raise ValueError(f"Incomplete stressed speed sweep for family {family}")
        y=[float(r["success"]) for r in stress]
        yerr=[]
        for r,rate in zip(stress,y):
            lo,hi=wilson(rate,int(r["episodes"]));yerr.append((rate-lo,hi-rate))
        success_ax.errorbar(caps,y,yerr=np.asarray(yerr).T,marker="s",linestyle="--",linewidth=1.8,color=color,capsize=2.5,label=f"{name} · combined stress")
    success_ax.set(xlabel="Requested 3D velocity-intent cap (m/s)",ylabel="Episode success fraction",ylim=(-.03,1.05),title="Success falls at higher caps")
    success_ax.grid(alpha=.2);success_ax.legend(frameon=False,fontsize=7.6,ncol=2,loc="lower left")

    table_cases=sorted((r for r in rows if r["family"]=="5"),key=lambda r:(r["group"],float(r["speed_mps"])))
    for group,label,color,linestyle in (("speed_clean","Mean path speed · clean",COLORS["blue"],"-"),
                                        ("speed_stressed","Mean path speed · stress",COLORS["orange"],"-"),
                                        ("speed_clean","Peak observed speed · clean",COLORS["blue"],"--"),
                                        ("speed_stressed","Peak observed speed · stress",COLORS["orange"],"--")):
        selected=sorted((r for r in table_cases if r["group"]==group),key=lambda r:float(r["speed_mps"]))
        metric="mean_speed_mps" if "Mean" in label else "peak_speed_mps"
        speed_ax.plot([float(r["speed_mps"]) for r in selected],[float(r[metric]) for r in selected],
                      marker="o",linestyle=linestyle,color=color,linewidth=1.8,label=label)
    speed_ax.set(xlabel="Requested 3D velocity-intent cap (m/s)",ylabel="Measured path speed (m/s)",title="Command cap is not a speed bound")
    speed_ax.grid(alpha=.2);speed_ax.legend(frameon=False,fontsize=7.4,loc="upper left")
    fig.suptitle("Fixed-policy speed-cap sweep",x=.04,y=.985,ha="left",fontsize=14,weight="bold")
    fig.text(.04,.94,"Same guided checkpoint · mode 17 · seed 800001 · 128 episodes/case · 10 s episode limit · no retraining",
             fontsize=8.7,color=COLORS["gray"])
    fig.text(.025,.018,"Caps are requested commands, not measured vehicle speed limits. Mean path speed includes failures; peak is the maximum observed over all episodes and may be an outlier. At low caps, many cases time out at 10 s; check the source CSV for success/collision/timeout counts. Goal time is conditioned on success.",
             fontsize=7.8,color=COLORS["gray"])
    fig.subplots_adjust(left=.07,right=.98,top=.80,bottom=.22,wspace=.22)
    return save_figure(fig,out,"static-policy-speed-cap-sweep")


def parse_scaling_table(text: str) -> tuple[list[int], list[float], list[float]]:
    table = re.search(r"\| N \| GPU wall3rollouts s \| CPU wall3rollouts s \| CPU/GPU \|\n\|[-| :]+\|\n(?P<rows>(?:\|.*\n)+)", text)
    if not table:
        raise ValueError("Could not find the recorded CPU/GPU scaling table in docs/BENCHMARKS.md")
    n_values, gpu, cpu = [], [], []
    for line in table.group("rows").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        n_values.append(int(cells[0].replace(",", "")))
        gpu.append(float(cells[1]))
        cpu.append(float(cells[2]))
    return n_values, gpu, cpu


def scaling_figure(out: Path) -> list[str]:
    n, gpu, cpu = parse_scaling_table(BENCHMARKS.read_text())
    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    ax.plot(n, gpu, marker="o", linewidth=2.2, color=COLORS["blue"], label="M3 Metal GPU wall")
    ax.plot(n, cpu, marker="o", linewidth=2.2, color=COLORS["orange"], label="Accelerate CPU wall")
    for line_index, (xs, ys, color) in enumerate(((n, gpu, COLORS["blue"]), (n, cpu, COLORS["orange"]))):
        for x, y in zip(xs, ys):
            offset = (-12, 12) if x == 128 and line_index == 0 else ((18, -18) if x == 128 else ((0, 8) if line_index == 0 else (0, -14)))
            ax.annotate(f"{y:.3f}s", (x, y), xytext=offset, textcoords="offset points", ha="center", fontsize=8, color=color)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(n, [f"{v:,}" for v in n])
    ax.set_xlabel("Environment count (log scale)")
    ax.set_ylabel("Wall time for 3 complete rollouts (s, log scale)")
    ax.set_title("Full-workload scaling: GPU and optimized CPU", loc="left", weight="bold", fontsize=14)
    ax.text(0.01, 0.98, "32-step horizon · 2 PPO epochs · 256 minibatch · 320 rays/history · actual RAPTOR/physics",
            transform=ax.transAxes, va="top", fontsize=8.7, color=COLORS["gray"])
    ax.grid(which="both", alpha=0.2)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.88))
    add_footer(fig, "GPU wall includes host encoding and wait. CPU uses Accelerate SGEMM and GCD. Setup and evaluation are excluded. These are complete training workloads, not ray-kernel timings.")
    fig.subplots_adjust(left=0.14, right=0.98, top=0.88, bottom=0.21)
    return save_figure(fig, out, "cpu-gpu-training-scaling")


def checkpoint_inputs() -> list[Path]:
    candidates = [
        ROOT / "assets/checkpoints/guided-table-memory.bin.best",
        ROOT / "assets/checkpoints/guided-clean-stress.bin.best",
        ROOT / "assets/checkpoints/guided-stress.bin.best",
        ROOT / "assets/checkpoints/guided-door-stress.bin.best",
        ROOT / "assets/checkpoints/guided-rehearsal.bin.best",
        ROOT / "assets/checkpoints/guided-threat-joint.bin.best",
    ]
    return [p for p in candidates if p.is_file()]


def load_trace_pair() -> list[dict[str, Any]]:
    pair = []
    for name in ("early-crossing", "later-crossing"):
        metadata_path = INPUTS / f"{name}.json"
        trace_path = INPUTS / f"{name}.csv"
        metadata = json.loads(metadata_path.read_text())
        with trace_path.open(newline="") as stream:
            trace = list(csv.DictReader(stream))
        if metadata["family"] != 11 or metadata["episode_family"] != 11 or metadata["mode"] != 17:
            raise ValueError(f"{name} is not a guided family-11 trace")
        if not trace or any(int(row["sensor_ready"]) != 1 for row in trace):
            raise ValueError(f"{name} trace has no rows or includes unready sensor frames")
        times = [float(row["t_s"]) for row in trace]
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError(f"{name} trace time must be strictly increasing")
        interval = float(metadata["sensor_interval_s"])
        if any(abs((b - a) - interval) > 1e-5 for a, b in zip(times, times[1:])):
            raise ValueError(f"{name} pose rows do not match the declared sensor interval")
        if not {"position", "quaternion_wxyz", "linear_velocity", "angular_velocity_body"}.issubset(metadata["terminal_state"]):
            raise ValueError(f"{name} lacks exact terminal pose and velocity metadata")
        if not times[-1] <= float(metadata["duration_s"]) <= times[-1] + interval + 1e-5:
            raise ValueError(f"{name} duration does not align with its final 20 Hz pose and terminal state")
        metadata["trace"] = [{key: (float(value) if key != "sensor_ready" else int(value)) for key, value in row.items()} for row in trace]
        pair.append(metadata)
    a, b = pair
    stable_fields = ("seed", "environment_index", "world_seed", "family", "episode_family", "mode", "dt_s", "native_dt_s", "sensor_interval_s")
    if any(a[key] != b[key] for key in stable_fields) or a["obstacles"] != b["obstacles"]:
        raise ValueError("Paired scene traces do not share the exact recorded world and timing")
    return pair


def scene_figure(pair: list[dict[str, Any]], out: Path) -> list[str]:
    fig, (xy, altitude) = plt.subplots(1, 2, figsize=(11.2, 5.4), gridspec_kw={"width_ratios": [1.0, 1.15]})
    configs = [
        ("early-crossing", "Original selected policy", COLORS["red"]),
        ("later-crossing", "Threat-joint candidate", COLORS["blue"]),
    ]
    first = pair[0]
    sphere = first["obstacles"][0]
    center0 = np.asarray(sphere["center_t0"], dtype=float)
    velocity = np.asarray(sphere["velocity"], dtype=float)
    duration = max(float(p["duration_s"]) for p in pair)
    obstacle_t = np.linspace(0, duration, 90)
    obstacle_xyz = center0[None, :] + obstacle_t[:, None] * velocity[None, :]
    goal = first["goal"]
    xy.plot(obstacle_xyz[:, 0], obstacle_xyz[:, 1], linestyle="--", color=COLORS["purple"], alpha=0.7, label="Sphere centre path")
    for metadata, (expected, label, color) in zip(pair, configs):
        if Path(metadata["checkpoint"]).name not in ("guided-table-memory.bin.best", "guided-threat-joint.bin.best"):
            raise ValueError(f"Unexpected trace checkpoint for {expected}: {metadata['checkpoint']}")
        rows = metadata["trace"]
        px = [r["x"] for r in rows]
        py = [r["y"] for r in rows]
        pz = [r["z"] for r in rows]
        pt = [r["t_s"] for r in rows]
        outcome = "success" if metadata["success"] else "collision" if metadata["collision"] else "timeout"
        xy.plot(px, py, color=color, linewidth=2.2, label=f"{label} · {outcome}")
        altitude.plot(pt, pz, color=color, linewidth=2.2, label=f"{label} · {outcome}")
        xy.scatter([px[0]], [py[0]], marker="o", color=color, edgecolor="white", zorder=4)
        terminal = metadata["terminal_state"]["position"]
        xy.scatter([terminal[0]], [terminal[1]], marker="X", color=color, edgecolor="white", zorder=4)
        altitude.scatter([pt[0]], [pz[0]], marker="o", color=color, edgecolor="white", zorder=4)
        altitude.scatter([float(metadata["duration_s"])], [terminal[2]], marker="s", color=color, edgecolor="white", zorder=4)
    xy.scatter([goal[0]], [goal[1]], marker="*", s=180, color=COLORS["green"], edgecolor="white", zorder=5, label="Goal")
    altitude.axhline(float(goal[2]), linestyle=":", color=COLORS["green"], linewidth=1.5, label="Goal altitude")
    sphere_x = float(center0[0])
    sphere_y = float(center0[1])
    sphere_z = float(center0[2])
    radius = float(sphere["size"][0])
    xy.add_patch(plt.Circle((sphere_x, sphere_y), radius, color=COLORS["purple"], alpha=0.22, label="Moving sphere radius"))
    altitude.axhspan(sphere_z - radius, sphere_z + radius, color=COLORS["purple"], alpha=0.12, label="Sphere vertical extent")
    for axis, xlabel, ylabel in ((xy, "World X (m)", "World Y (m)"), (altitude, "Simulator time (s)", "World Z (m)")):
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=8, loc="best")
    xy.set_xlim(-0.2, 4.8)
    xy.set_ylim(-1.4, 1.4)
    xy.set_aspect("equal", adjustable="box")
    altitude.set_xlim(0, duration)
    altitude.set_ylim(0.8, 2.5)
    xy.set_title("Top view", loc="left", weight="bold")
    altitude.set_title("Altitude over time", loc="left", weight="bold")
    fig.suptitle("Same crossing obstacle, same world seed", x=0.04, y=0.98, ha="left", weight="bold", fontsize=14)
    fig.text(0.04, 0.915, "Recorded 20 Hz simulator traces · family 11 · seed 800001 · obstacle motion shown from logged world geometry", fontsize=8.8, color=COLORS["gray"])
    fig.text(0.025, 0.018, "Circle is the logged 0.35 m sphere radius. Curves use 20 Hz poses; X marks exact terminal state from JSON metadata. Ground-truth scene geometry is display-only.", fontsize=8.2, color=COLORS["gray"])
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.18, wspace=0.22)
    return save_figure(fig, out, "paired-crossing-scene")


def scene_animation(pair: list[dict[str, Any]], out: Path) -> str:
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(8.4, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    configs = [(pair[0], COLORS["red"], "Original policy"), (pair[1], COLORS["blue"], "Threat-joint candidate")]
    sphere = pair[0]["obstacles"][0]
    center0 = np.asarray(sphere["center_t0"], dtype=float)
    velocity = np.asarray(sphere["velocity"], dtype=float)
    radius = float(sphere["size"][0])
    max_time = max(float(p["duration_s"]) for p in pair)
    frame_count = int(round(max_time / float(pair[0]["sensor_interval_s"]))) + 1
    frame_times = np.linspace(0, max_time, frame_count)
    u = np.linspace(0, 2 * np.pi, 22)
    v = np.linspace(0, np.pi, 14)
    unit_x = np.outer(np.cos(u), np.sin(v))
    unit_y = np.outer(np.sin(u), np.sin(v))
    unit_z = np.outer(np.ones_like(u), np.cos(v))
    obstacle_center = center0
    ax.plot_surface(obstacle_center[0] + radius * unit_x, obstacle_center[1] + radius * unit_y,
                    obstacle_center[2] + radius * unit_z, color=COLORS["purple"], alpha=0.36, linewidth=0)
    for metadata, color, label in configs:
        rows = metadata["trace"]
        ax.plot([r["x"] for r in rows], [r["y"] for r in rows], [r["z"] for r in rows],
                color=color, linewidth=1.5, alpha=0.46, label=f"{label} trail")
    goal = np.asarray(pair[0]["goal"], dtype=float)
    ax.scatter([goal[0]], [goal[1]], [goal[2]], marker="*", s=150, color=COLORS["green"], label="Goal")
    ax.plot([0, goal[0]], [0, 0], [0.18, 0.18], color=COLORS["gray"], alpha=0.22, linestyle=":")
    ax.set(xlim=(-0.2, 4.8), ylim=(-1.6, 1.6), zlim=(0.6, 2.5),
           xlabel="World X (m)", ylabel="World Y (m)", zlabel="World Z (m)")
    ax.set_box_aspect((5.0, 3.2, 2.0))
    ax.view_init(elev=25, azim=-62)
    ax.set_title("Recorded family-11 crossing episode · 20 Hz pose samples", loc="left", weight="bold", pad=14)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.text(0.5, 0.018, "Logged obstacle geometry is shown for visualization only. End states come from exact JSON terminal metadata; playback is 8 fps.",
             ha="center", fontsize=8, color=COLORS["gray"])
    drones = []
    headings = []
    for _, color, _ in configs:
        point, = ax.plot([], [], [], marker="o", markersize=6, color=color, linestyle="None")
        heading = None
        drones.append(point)
        headings.append(heading)
    dynamic_sphere = None
    label = fig.text(0.5, 0.02, "", ha="center", fontsize=10, color=COLORS["navy"])

    def sample(metadata: dict[str, Any], t: float) -> dict[str, float]:
        rows = metadata["trace"]
        index = min(int(math.floor(t / float(metadata["sensor_interval_s"]) + 1e-7)), len(rows) - 1)
        if t >= float(metadata["duration_s"]):
            terminal = metadata["terminal_state"]
            return {"x": terminal["position"][0], "y": terminal["position"][1], "z": terminal["position"][2],
                    "qw": terminal["quaternion_wxyz"][0], "qx": terminal["quaternion_wxyz"][1],
                    "qy": terminal["quaternion_wxyz"][2], "qz": terminal["quaternion_wxyz"][3]}
        return rows[index]

    def update(frame: int):
        nonlocal dynamic_sphere
        t = float(frame_times[frame])
        for idx, (metadata, color, _) in enumerate(configs):
            row = sample(metadata, t)
            xyz = (row["x"], row["y"], row["z"])
            drones[idx].set_data([xyz[0]], [xyz[1]])
            drones[idx].set_3d_properties([xyz[2]])
            if headings[idx] is not None:
                headings[idx].remove()
            w, x, y, z = row["qw"], row["qx"], row["qy"], row["qz"]
            forward = (1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y))
            headings[idx] = ax.quiver(*xyz, *(0.28 * np.asarray(forward)), color=color, linewidth=1.6, arrow_length_ratio=0.28)
        center = center0 + velocity * t
        if dynamic_sphere is not None:
            dynamic_sphere.remove()
        dynamic_sphere = ax.plot_surface(center[0] + radius * unit_x,
                                         center[1] + radius * unit_y,
                                         center[2] + radius * unit_z,
                                         color=COLORS["purple"], alpha=0.48, linewidth=0)
        old = sample(pair[0], t)
        new = sample(pair[1], t)
        old_state = f"collision @ {float(pair[0]['duration_s']):.2f}s" if t >= float(pair[0]["duration_s"]) else "moving"
        new_state = f"success @ {float(pair[1]['duration_s']):.2f}s" if t >= float(pair[1]["duration_s"]) else "moving"
        label.set_text(f"t={t:.2f}s   original: {old_state}   threat-joint: {new_state}")
        return [*drones, label]

    animation = FuncAnimation(fig, update, frames=frame_count, interval=120, blit=False)
    path = out / "paired-crossing-scene.gif"
    animation.save(path, writer=PillowWriter(fps=8))
    plt.close(fig)
    return path.name


def verify_candidate_hashes(continuation: list[dict[str, str]]) -> dict[str, str]:
    by_candidate: dict[str, str] = {}
    for row in continuation:
        name = row.get("candidate", "")
        recorded = row.get("checkpoint_sha256", "")
        if not name or not recorded:
            continue
        previous = by_candidate.setdefault(name, recorded)
        if previous != recorded:
            raise ValueError(f"Candidate {name} contains multiple checkpoint hashes")
    files = {p.name: sha256(p) for p in checkpoint_inputs()}
    expected = {
        "controlled_threats": "guided-table-memory.bin.best",
        "broad_stress": "guided-stress.bin.best",
        "door_stress": "guided-door-stress.bin.best",
        "rehearsal": "guided-rehearsal.bin.best",
    }
    for candidate, filename in expected.items():
        if candidate in by_candidate and filename in files and by_candidate[candidate] != files[filename]:
            raise ValueError(f"Checkpoint hash mismatch for {candidate}: CSV does not match {filename}")
    return by_candidate


def room_transfer_figure(out: Path) -> list[str]:
    rows = read_csv(INPUTS / "room-webots-transfer/cases.csv")
    if len(rows) != 30 or len({row["failure_id"] for row in rows}) != 30:
        raise ValueError("Room transfer figure requires the complete matched 30-level matrix")
    groups = ("easy", "medium", "hard", "all")
    counts = [sum(group == "all" or row["difficulty_band"] == group for row in rows) for group in groups]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for offset, column, label, color in ((-.18, "focused_metal_success", "Metal", COLORS["blue"]),
                                       (.18, "webots_success", "Webots", COLORS["orange"])):
        successes = [sum(int(row[column]) for row in rows
                         if group == "all" or row["difficulty_band"] == group) for group in groups]
        positions = np.arange(len(groups)) + offset
        ax.bar(positions, [success / count for success, count in zip(successes, counts)],
               width=.34, label=label, color=color)
        for x, success, count in zip(positions, successes, counts):
            ax.text(x, success / count + .025, f"{success}/{count}", ha="center", fontsize=10)
    ax.set_xticks(np.arange(len(groups)), [group.title() for group in groups])
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Goal-entry success fraction")
    ax.set_title("Hard rooms expose the simulator transfer gap", loc="left", weight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.2)
    add_footer(fig, "Same 30 development rooms and policy. 20 s budget; 1.5 m/s requested cap. Webots: 12 contacts, no timeouts. Clean depth and ideal ego sensors; final split untouched.")
    fig.subplots_adjust(bottom=.23)
    return save_figure(fig, out, "room-webots-transfer")


def imitation_failure_figure(out: Path) -> list[str]:
    rows = read_csv(INPUTS / "navigation-imitation/progression.csv")
    proof = json.loads((INPUTS / "navigation-imitation/proof.json").read_text())
    if any(int(row["episodes"]) != 90 for row in rows):
        raise ValueError("imitation progression requires the complete 90-case DEV suite")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    for stage, label, color in (("root-run", "Expert flights", "#24658f"),
                                ("aggregate-run", "Mixture-state labels", "#bc7040")):
        selected = [row for row in rows if row["stage"] == stage and row["batch_half_squared_command_error"]]
        axes[0].plot([int(row["total_actor_updates"]) for row in selected],
                     [float(row["batch_half_squared_command_error"]) for row in selected],
                     marker="o", markersize=3, label=label, color=color)
    axes[0].set(title="Recorded minibatch loss", xlabel="Actor updates",
                ylabel="Mean half-squared command error")
    axes[0].legend(frameon=False, fontsize=8)
    for key, label, color in (("corner_success", "Corners", "#af4044"),
                              ("room_success", "Rooms", "#24658f"),
                              ("vertical_success", "Vertical routes", "#498556")):
        axes[1].plot([int(row["total_actor_updates"]) for row in rows],
                     [int(row[key]) for row in rows], marker="o", markersize=3,
                     label=label, color=color)
    axes[1].set(title="Actual held-out DEV completion", xlabel="Actor updates",
                ylabel="Successful cases per family (30)", ylim=(-1, 31))
    axes[1].legend(frameon=False, fontsize=8)
    initial = proof["fit_diagnostics"]["initial_rows"]
    index = np.arange(len(initial))
    axes[2].scatter(index, [row["target"][1] for row in initial],
                    label="Privileged teacher", marker="x", color="#24658f")
    axes[2].scatter(index, [row["predicted"][1] for row in initial],
                    label="After 1,000 updates", s=16, color="#bc7040")
    axes[2].set(title="Critical initial turn: 24 TRAIN flights", xlabel="Successful teacher case",
                ylabel="Normalized body Y command", ylim=(-1.1, 1.1))
    axes[2].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, .8))
    for ax in axes:
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Lower imitation loss did not produce corner navigation", fontsize=14, weight="bold")
    fig.text(.02, .015, "One source-only seed. Data changes after update 1,000. All corner DEV scores remain 0/30. Failed candidates were not promoted.", fontsize=8)
    fig.tight_layout(rect=(0, .055, 1, .92))
    return save_figure(fig, out, "imitation-learning-failure")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory (default: artifacts)")
    parser.add_argument("--imitation-only", action="store_true", help="render the recorded imitation experiment without regenerating other figures")
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    if args.imitation_only:
        print("generated:", ", ".join(imitation_failure_figure(out)))
        return 0

    training = read_training(TRAINING_TSV)
    static_eval = read_csv(STATIC_EVAL)
    continuation = read_csv(CONTINUATION)
    old_threats = read_csv(OLD_THREATS)
    joint_threats = read_csv(JOINT_THREATS)
    traces = load_trace_pair()
    candidate_hashes = verify_candidate_hashes(continuation)
    challenge_results, challenge_levels = load_and_validate_challenge_evidence()
    speed_rows = load_and_validate_speed_sweep()
    ablation_rows = load_and_validate_inference_ablations()
    corner_training, corner_baseline, corner_final = load_and_validate_corner_control()
    witness_1mps, witness_15mps, mirrored_levels = load_and_validate_mirrored_witnesses()
    arrival_history, arrival_summaries, arrival_episodes, arrival_manifest, arrival_rich_dev, arrival_retention = load_and_validate_arrival_evidence()
    webots_stable_rows, webots_stable_manifest = load_and_validate_webots_stable_arrival()

    outputs: list[str] = []
    outputs += room_transfer_figure(out)
    outputs += training_figure(training, out)
    outputs += mode_comparison_figure(static_eval, out)
    outputs += transfer_figure(static_eval, continuation, out)
    outputs += threat_figure(old_threats, joint_threats, out)
    outputs += scaling_figure(out)
    outputs += scene_figure(traces, out)
    outputs.append(scene_animation(traces, out))
    outputs += challenge_dev_figure(challenge_results, out)
    outputs += challenge_worlds_figure(challenge_levels, out)
    outputs += corner_control_figure(corner_training, corner_baseline, corner_final, out)
    outputs += inference_ablation_figure(ablation_rows, out)
    outputs += speed_sweep_figure(speed_rows, out)
    outputs += arrival_training_figure(arrival_history, out)
    outputs += arrival_domain_figure(arrival_summaries, out)
    outputs += arrival_rich_bank_figure(arrival_rich_dev, out)
    outputs += arrival_legacy_retention_figure(arrival_retention, out)
    outputs += webots_stable_arrival_figure(webots_stable_rows, out)

    source_paths = [
        ROOT / "evidence.py", BENCHMARKS,
        ROOT / "docs" / "CODE_DIRECTION.md", ROOT / "docs" / "NEXT_PHASE.md", ROOT / "docs" / "RESEARCH_LOG.md",
        *sorted(INPUTS.rglob("*")),
        *checkpoint_inputs(),
        ROOT / "assets/checkpoints/arrival-open-domain-experimental.bin.best",
        ROOT / "assets/navigation-arrival-experimental.bin",
    ]
    inputs = [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p), "bytes": p.stat().st_size}
              for p in source_paths if p.is_file()]
    manifests = {
        "generated_by": "evidence.py",
        "command": "python3 evidence.py --out artifacts",
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds"),
        "runtime": {"python": sys.version.split()[0], "matplotlib": matplotlib.__version__, "numpy": np.__version__},
        "outputs": outputs,
        "inputs": inputs,
        "recorded_candidate_hashes": candidate_hashes,
        "challenge_bank_provenance": {
            "input": str(CHALLENGE_BANK.relative_to(ROOT)),
            "sha256": sha256(CHALLENGE_BANK),
            "level_count": len(challenge_levels),
            "dev_level_count": sum(record["split"] == "dev" for record in challenge_levels.values()),
            "final_level_count": sum(record["split"] == "final" for record in challenge_levels.values()),
            "final_split_evaluated": False,
            "results": {label: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path),
                                 "checkpoint": str((ROOT / "assets" / "checkpoints" / checkpoint).relative_to(ROOT)),
                                 "checkpoint_sha256": sha256(ROOT / "assets" / "checkpoints" / checkpoint), "mode": mode}
                        for label, (path, checkpoint, mode) in CHALLENGE_FILES.items()},
        },
        "targeted_corner_training": {
            "log": str(CORNER_LOG.relative_to(ROOT)),
            "filtered_history": str(CORNER_TRAINING.relative_to(ROOT)),
            "history_rows": len(corner_training),
            "rollouts": 1000,
            "transitions": 128 * 32 * 1000,
            "training_episodes": 82729,
            "training_collision_fraction": 0.99994,
            "selection_validation": {"before": {"episodes": int(corner_baseline["episodes"]), "success": float(corner_baseline["success"]), "collision": float(corner_baseline["collision"])},
                                      "after": {"episodes": int(corner_final["episodes"]), "success": float(corner_final["success"]), "collision": float(corner_final["collision"])}},
            "configuration": "family 14, mode 17, 1.5 m/s cap, 8 m goal, 20 s episode limit",
        },
        "speed_sweep_provenance": {
            "input": str(SPEED_SWEEP.relative_to(ROOT)),
            "sha256": sha256(SPEED_SWEEP),
            "checkpoint": "assets/checkpoints/guided-clean-stress.bin.best",
            "checkpoint_sha256": sha256(ROOT / "assets/checkpoints/guided-clean-stress.bin.best"),
            "command": "python3 evaluation.py --speed-sweep --checkpoint assets/checkpoints/guided-clean-stress.bin.best --output results/static-speed.csv",
            "episode_limit_s": 10,
            "episodes_per_case": 128,
            "retrained_at_caps": False,
        },
        "inference_ablation_provenance": {
            "input": str(INFERENCE_ABLATIONS.relative_to(ROOT)),
            "sha256": sha256(INFERENCE_ABLATIONS),
            "checkpoint": "assets/checkpoints/guided-threat-joint.bin.best",
            "checkpoint_sha256": sha256(ROOT / "assets/checkpoints/guided-threat-joint.bin.best"),
            "command": "python3 evaluation.py --ablations --checkpoint assets/checkpoints/guided-threat-joint.bin.best --output results/threat-joint-ablations.csv",
            "retrained_by_mode": False,
        },
        "privileged_witness_route_runs": {
            "bank": str(MIRRORED_BANK.relative_to(ROOT)),
            "bank_sha256": sha256(MIRRORED_BANK),
            "bank_level_count": len(mirrored_levels),
            "dev_levels": sum(record["split"] == "dev" for record in mirrored_levels.values()),
            "final_split_evaluated": False,
            "generator_command": "python3 challenge_bank.py --seed 20261001 --distance 8 --per-split 30 --families 14,15,16 --mirror-y --out evidence/inputs/challenge-bank-mirrored-v1.jsonl",
            "semantics": "Saved witness waypoints were followed by a scripted host controller through frozen RAPTOR and simulator physics. The waypoint path is privileged input, not a learned-navigation policy result.",
            "at_1mps_cap_60s": {"csv": str(WITNESS_1MPS.relative_to(ROOT)), "sha256": sha256(WITNESS_1MPS),
                                 "successes": sum(int(r["success"]) for r in witness_1mps), "episodes": len(witness_1mps),
                                 "outcomes_by_family": {family: {"success": sum(int(r["success"]) for r in witness_1mps if r["family"] == family), "collision": sum(int(r["collision"]) for r in witness_1mps if r["family"] == family), "timeout": sum(int(r["timeout"]) for r in witness_1mps if r["family"] == family), "mean_elapsed_s": sum(float(r["elapsed_s"]) for r in witness_1mps if r["family"] == family) / 30} for family in ("14", "15", "16")}},
            "at_1_5mps_cap_20s": {"csv": str(WITNESS_15MPS.relative_to(ROOT)), "sha256": sha256(WITNESS_15MPS),
                                   "successes": sum(int(r["success"]) for r in witness_15mps), "timeouts": sum(int(r["timeout"]) for r in witness_15mps),
                                   "collisions": sum(int(r["collision"]) for r in witness_15mps), "episodes": len(witness_15mps),
                                   "outcomes_by_family": {family: {"success": sum(int(r["success"]) for r in witness_15mps if r["family"] == family), "collision": sum(int(r["collision"]) for r in witness_15mps if r["family"] == family), "timeout": sum(int(r["timeout"]) for r in witness_15mps if r["family"] == family)} for family in ("14", "15", "16")}},
            "logs": [str(WITNESS_LOG_1MPS.relative_to(ROOT)), str(WITNESS_LOG_15MPS.relative_to(ROOT))],
        },
        "arrival_open_domain_candidate": {
            "input_directory": str(ARRIVAL_DIR.relative_to(ROOT)),
            "input_manifest_sha256": sha256(ARRIVAL_MANIFEST),
            "training_command": arrival_manifest["training_command"],
            "train_seed": arrival_manifest["train_seed"],
            "selection_seed": arrival_manifest["selection_seed"],
            "selected_rollout": arrival_manifest["selected_rollout"],
            "total_transitions": arrival_manifest["transitions"],
            "elapsed_wall_s": arrival_manifest["wall_s"],
            "checkpoint": "assets/checkpoints/arrival-open-domain-experimental.bin.best",
            "checkpoint_sha256": sha256(ROOT / "assets/checkpoints/arrival-open-domain-experimental.bin.best"),
            "actor_export": "assets/navigation-arrival-experimental.bin",
            "actor_export_sha256": sha256(ROOT / "assets/navigation-arrival-experimental.bin"),
            "selection_validation": {
                "episodes": 128,
                "seed": 800001,
                "initial_success": arrival_history[0]["success"],
                "selected_success": next(row["success"] for row in arrival_history if row["rollout"] == arrival_manifest["selected_rollout"]),
                "selected_mean_arrival_s_successes_only": next(row["mean_arrival_s"] for row in arrival_history if row["rollout"] == arrival_manifest["selected_rollout"]),
            },
            "fresh_seed_evaluation": {
                "seed": 820001,
                "episodes_per_condition": 128,
                "paired_task_and_plant_rows": True,
                "conditions": [{"policy": row["policy"], "stage": int(row["stage"]), "family": int(row["family"]),
                                "domain_amplitude": int(row["domain_amplitude"]), "successes": int(row["successes"]),
                                "collisions": int(row["collisions"]), "timeouts": int(row["timeouts"])} for row in arrival_summaries],
            },
            "rich_bank_development": {
                "bank": str(MIRRORED_BANK.relative_to(ROOT)),
                "bank_sha256": sha256(MIRRORED_BANK),
                "rows": len(arrival_rich_dev),
                "final_split_evaluated": False,
                "outcomes_by_family": {str(family): {"success": sum(int(row["success"]) for row in arrival_rich_dev if int(row["family"]) == family),
                                                              "collision": sum(int(row["collision"]) for row in arrival_rich_dev if int(row["family"]) == family),
                                                              "timeout": sum(int(row["timeout"]) for row in arrival_rich_dev if int(row["family"]) == family)}
                                        for family in (14, 15, 16)},
            },
            "legacy_first_entry_retention": {
                "input": str(ARRIVAL_RETENTION.relative_to(ROOT)),
                "sha256": sha256(ARRIVAL_RETENTION),
                "seed": 800001,
                "episodes_per_family_policy": 128,
                "budget_s": 10,
                "requested_speed_cap_mps": 1.5,
                "success_rule": "first_goal_region_entry for both policies; these rows are not stable-arrival scores",
                "families": [{"family": family, "original_successes": round(float(next(row["success"] for row in arrival_retention if int(row["family"]) == family and row["policy"] == "original")) * 128),
                              "arrival_successes": round(float(next(row["success"] for row in arrival_retention if int(row["family"]) == family and row["policy"] == "arrival")) * 128)}
                             for family in (4, 5, 7, 8, 10, 11, 12)],
            },
            "source_hashes_at_run": arrival_manifest["source_hashes"],
            "limitations": arrival_manifest["limitations"] + [
                "This single training seed improves the evaluated open and near-goal tasks and selected simple static scenes; it is not a full generalization result.",
                "The rich mirrored-bank result uses a different scene set and does not form a paired checkpoint comparison.",
                "The same-seed selection curve is model-selection evidence, not an independent final test.",
                "Legacy first-entry retention rows use a different success rule than the new stable-arrival task; use the matched legacy matrix only for retention comparisons.",
            ],
        },
        "webots_stable_arrival_transfer": {
            "input_directory": str(WEBOTS_STABLE_DIR.relative_to(ROOT)),
            "manifest_sha256": sha256(WEBOTS_STABLE_MANIFEST),
            "episodes_csv_sha256": sha256(WEBOTS_STABLE_EPISODES),
            "runs_json_sha256": sha256(WEBOTS_STABLE_RUNS),
            "webots_release": "R2025a",
            "scene_count": 18,
            "episodes": len(webots_stable_rows),
            "seeds_per_family_policy": 6,
            "families": ["doorway", "table_overhang", "mixed_clutter"],
            "goal_contract": {"radius_m": webots_stable_manifest["radius_m"],
                               "max_actual_speed_mps": webots_stable_manifest["speed_max_mps"],
                               "continuous_dwell_s": webots_stable_manifest["dwell_s"],
                               "budget_s": webots_stable_manifest["budget_s"],
                               "scorer_hz": 100},
            "simulation": {"physics_step_ms": 1, "controller_period_ms": 10,
                           "navigation_period_ms": 50, "motor_sampling": "interval-average",
                           "physics_profile": "hover", "collision_sphere_radius_m": 0.18},
            "sensors": {"ego": "ideal GPS/IMU/Gyro", "range": "clean RangeFinder", "range_noise_stddev_m": 0},
            "results_by_policy": {
                label: {"success": sum(row["success"] == "True" for row in webots_stable_rows if row["policy"] == policy),
                        "collision": sum(row["collision"] == "True" for row in webots_stable_rows if row["policy"] == policy),
                        "timeout": sum(row["timeout"] == "True" for row in webots_stable_rows if row["policy"] == policy),
                        "by_family": {family: {"success": sum(row["success"] == "True" for row in webots_stable_rows if row["policy"] == policy and row["family"] == family),
                                                "collision": sum(row["collision"] == "True" for row in webots_stable_rows if row["policy"] == policy and row["family"] == family),
                                                "timeout": sum(row["timeout"] == "True" for row in webots_stable_rows if row["policy"] == policy and row["family"] == family)}
                                        for family in ("doorway", "table_overhang", "mixed_clutter")}}
                for policy, label in (("navigation.bin", "original"), ("navigation-arrival-experimental.bin", "arrival_candidate"))},
            "reproduction": webots_stable_manifest["reproduction"],
            "source_hashes": webots_stable_manifest["source_hashes"],
            "limitations": webots_stable_manifest["limitations"],
        },
        "figure_sources": {
            "room-webots-transfer": {"inputs": ["evidence/inputs/room-webots-transfer/cases.csv", "evidence/inputs/room-webots-transfer/proof.json"], "filter": "all 30 paired family-15 development IDs; group by declared difficulty; same selected policy and first-entry rule; final split untouched"},
            "training-broad-validation": {"input": "evidence/inputs/training.tsv", "filter": "checkpoint basename raw-broad.bin or pooled-broad.bin; recorded validation rows; elapsed wall time from training invocation"},
            "static-scene-policy-comparison": {"input": "evidence/inputs/evaluation.csv", "filter": "group=baseline, seed=800001, families 7/8/5/3, modes 17/13/2"},
            "clean-and-stress-transfer": {"inputs": ["evidence/inputs/evaluation.csv", "evidence/inputs/continuation-evaluation.csv"], "filter": "mode=17, seed=800001; clean baseline and combined stress for families 8 and 5; candidates from continuation CSV"},
            "controlled-threat-before-after": {"inputs": ["evidence/inputs/threat-evaluation.csv", "evidence/inputs/threat-joint-controlled.csv"], "filter": "guided mode 17; exact matched 24-case family-0 threat matrix"},             "cpu-gpu-training-scaling": {"input": "docs/BENCHMARKS.md", "filter": "recorded environment ladder for three complete training rollouts"},
            "paired-crossing-scene": {"inputs": ["evidence/inputs/early-crossing.json", "evidence/inputs/early-crossing.csv", "evidence/inputs/later-crossing.json", "evidence/inputs/later-crossing.csv"], "filter": "same family 11, seed 800001, environment 0, world seed, obstacle list, and clean timing"},
            "paired-crossing-scene.gif": {"inputs": ["evidence/inputs/early-crossing.json", "evidence/inputs/early-crossing.csv", "evidence/inputs/later-crossing.json", "evidence/inputs/later-crossing.csv"], "filter": "recorded 20 Hz poses and logged obstacle motion; playback at 8 fps; ground truth is display-only"},
            "challenge-bank-held-dev-outcomes": {"inputs": ["evidence/inputs/challenge-bank-v1.jsonl", *[str(path.relative_to(ROOT)) for path, _, _ in CHALLENGE_FILES.values()]], "filter": "the same 90 frozen dev IDs; 30 per family; 1.5 m/s; max 400 steps; final split has no result rows"},
            "challenge-bank-witness-worlds": {"input": "evidence/inputs/challenge-bank-v1.jsonl", "filter": "one dev level per family nearest medium difficulty 0.5, tie-broken by larger witness clearance"},
            "corner-control-training-failure": {"inputs": [str(CORNER_LOG.relative_to(ROOT)), str(CORNER_TRAINING.relative_to(ROOT))], "filter": "family-14 warm-start run; before/after seed-700001 validation and ten-rollout training history records"},
            "inference-history-and-speed-ablations": {"input": str(INFERENCE_ABLATIONS.relative_to(ROOT)), "filter": "same checkpoint, modes 17/18/19; table and held-door clean/combined-stress cases"},
            "static-policy-speed-cap-sweep": {"input": str(SPEED_SWEEP.relative_to(ROOT)), "filter": "same checkpoint weights and seed; five requested caps; three clean and two combined-stress static families"},
            "arrival-open-training": {"input": str(ARRIVAL_HISTORY.relative_to(ROOT)), "filter": "all 21 logged evaluations from rollout 0 through 1000; seed 800001; selected at rollout 900"},
            "arrival-fresh-seed-domain": {"inputs": [str(ARRIVAL_EVALUATIONS.relative_to(ROOT)), str((ARRIVAL_DIR / "initial-dev.csv").relative_to(ROOT)), str((ARRIVAL_DIR / "selected-dev.csv").relative_to(ROOT))], "filter": "fresh seed 820001; 128 paired episode rows per condition; identical start/goal/scene/mass between policies"},
            "arrival-rich-bank-limit": {"inputs": [str((ARRIVAL_DIR / "rich-dev.csv").relative_to(ROOT)), str(MIRRORED_BANK.relative_to(ROOT))], "filter": "candidate-only, 90 mirrored challenge-bank dev levels; 30 each families 14/15/16; final split untouched"},
            "arrival-legacy-retention-regression": {"input": str(ARRIVAL_RETENTION.relative_to(ROOT)), "filter": "same seed 800001; 128 episodes, 10 s budget, requested cap 1.5 m/s; both checkpoints scored with legacy first-goal-region-entry rule"},
            "webots-stable-arrival": {"inputs": [str(WEBOTS_STABLE_EPISODES.relative_to(ROOT)), str(WEBOTS_STABLE_RUNS.relative_to(ROOT)), str(WEBOTS_STABLE_MANIFEST.relative_to(ROOT))], "filter": "36 episodes; 3 static scene families×6 saved seeds×2 policies; 20 s stable-arrival contract; matched by scene and seed"},
        },
        "training_tsv_schema": ["checkpoint", "rollout", "elapsed_wall_s", "gpu_s", "validation_success", "collision", "timeout", "goal_time_s"],
        "metric_definitions": {
            "validation_success": "The score returned by evaluate_training_policy at the recorded rollout. For ordinary families it is one fixed-seed evaluation; for training families 12/13 it is minimum success across their documented clean/stress profiles.",
            "episode_success": "Success fraction from the CSV row; each plotted row is 128 completed episodes unless stated otherwise.",
            "wilson_intervals": "Two-sided 95% Wilson score intervals for a binomial episode rate; they show finite-sample uncertainty, not seed-to-seed variance.",
            "wall_time": "Training TSV elapsed wall seconds are measured since the training invocation at each logged validation. Scaling values are the recorded complete three-rollout workload wall times.",
            "threat_ttc": "Nominal initial scene time-to-collision parameter. It is not actual closest-approach or policy-path collision time.",
            "speed_sweep": "speed_mps is the requested 3D velocity-intent norm cap under contract 1, not measured speed. mean_speed_mps is path divided by elapsed time across successes and failures. peak_speed_mps is the maximum vehicle speed observed in the case and can reflect an outlier.",
            "corner_control": "Training TSV rows are periodic fixed-seed selection evaluations. The initial and final evaluations use the same family-14 configuration; this is one targeted warm-start experiment.",
            "stable_arrival": "Success requires position within 0.35 m and actual speed at or below 0.5 m/s for a continuous 0.2 s hold. Mean arrival time is averaged over successful episodes only.",
            "arrival_training_validation": "The open-domain training curve uses a fixed 128-episode selection set (seed 800001). Checkpoint selection uses that set; these points are not an independent test.",
            "arrival_fresh_seed": "Seed 820001 uses paired start, goal, scene and recorded mass rows for original and candidate policies, with 128 episodes per condition. It is one fresh evaluation seed.",
            "arrival_domain_amplitude": "Amplitude 1 applies the declared parameter-randomization ranges. These ranges are assumptions, not measured hardware uncertainty.",
            "legacy_first_entry_retention": "Legacy family retention uses first goal-region entry as success. It cannot be compared numerically with the candidate's stable-arrival evaluation, which also requires speed and hold time.",
            "webots_stable_arrival": "Webots success requires a 0.35 m radius, actual speed at or below 0.5 m/s, and continuous 0.2 s hold scored at 100 Hz. The compared runs share each saved world and seed.",
        },
        "limitations": [
            "All evidence is from simulation. No real camera, transport, or flight validation is represented.",
            "The baseline static matrix uses seed 800001 and 128 episodes per row; Wilson intervals do not capture seed variance.",
            "Clean/stress candidates were selected using different profile sets. The figure compares per-profile outcomes and does not claim a matched training trial or a universal aggregate score.",
            "The training TSV records sparse validation checkpoints, not every rollout. Its wall time includes periodic validation work.",             "CPU/GPU scaling is the full PPO workload recorded in docs/BENCHMARKS.md. Do not compare it with depth-only microbenchmarks.",
            "The fixed-weight cap sweep is an inference stress test, not a trained speed curriculum or safe-speed certification. Its 10-second limit causes timeouts at some low caps, and measured vehicle speed can exceed the requested cap.",
            "The family-14 PPO control is one warm-start/schedule result. It shows that this added family and schedule did not teach detours; it does not show that corners are unlearnable.",
            "Separate PPO runs on the mirrored bank are tracked outside this figure package. Their preliminary development scores are not included here; no final split has been evaluated.",
            "The 3D view and GIF render recorded simulator traces and logged scene geometry. They are not camera footage. Pose paths are sampled at 20 Hz; terminal state metadata records the exact final 100 Hz state.",
            "Challenge-bank outcome bars use 30 levels per family from the frozen development split. They are not final holdout results, and the five policy/inference variants were not trained as a single matched experiment.",
            "Challenge-bank witness routes are geometric feasibility witnesses only; they are not learned trajectories or evidence that the tested policy can fly them.",
            "The open-domain candidate is a single-seed foundation experiment. Its gains on tested open/near-goal and simple static scenes do not establish full navigation generalization; it fails the mirrored corner and connected-room dev strata.",
            "The arrival rich-bank evaluation uses one candidate checkpoint on 90 mirrored development levels only. It is separate from the earlier bank and provides no matched baseline; the final split remains untouched.",
            "The Webots stable-arrival result covers 18 static scenes and ideal ego/range sensors. It is a small simulator transfer check, not broad generalization or hardware evidence.",
        ],
        "visualization_trace_contract": {
            "sources": ["evidence/inputs/early-crossing.json", "evidence/inputs/early-crossing.csv", "evidence/inputs/later-crossing.json", "evidence/inputs/later-crossing.csv"],
            "metadata": ["family/mode/seed/environment/world_seed", "dt_s/native_dt_s/sensor_interval_s", "world frame and units", "room bounds", "goal xyz", "drone radius", "obstacle type/center/size/velocity", "outcome/path/clearance", "exact terminal state at 100 Hz"],
            "time_series": ["20 Hz t_s/position xyz/quaternion wxyz/world velocity/body omega/reference/executed intent/yaw", "instantaneous clearance", "sensor readiness/capture time", "80 row-major pooled depth ranges in metres"],
            "separation": "Ground-truth obstacle fields support display only and are not policy inputs. The GIF uses logged poses and obstacle motion at 8 fps; it does not represent camera footage.",
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifests, indent=2) + "\n")
    print(f"generated={len(outputs)} figures={len(outputs)//2} output={out}")
    print(f"training_rows={len(training)} static_rows={len(static_eval)} continuation_rows={len(continuation)} threat_cases={len(old_threats)}")
    print(f"challenge_dev_levels={len(challenge_levels) and sum(r['split']=='dev' for r in challenge_levels.values())} speed_cases={len(speed_rows)} ablation_cases={len(ablation_rows)} corner_validations={len(corner_training)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
