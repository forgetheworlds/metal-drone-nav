#!/usr/bin/env python3
"""Recompute the October 4 local-navigation experiments from archived flight rows."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent
SPLITS = ("dev-a", "dev-b", "dev-c", "clutter", "open")


def read_records(path):
    with tarfile.open(path, "r:gz") as archive:
        records = {m.name: archive.extractfile(m).read()
                   for m in archive.getmembers() if m.isfile()}
    hashes = json.loads(records.pop("SHA256.json"))
    if set(records) != set(hashes):
        raise ValueError("Archive member list differs from its manifest")
    for name, data in records.items():
        if hashlib.sha256(data).hexdigest() != hashes[name]:
            raise ValueError(f"Archive hash mismatch: {name}")
    return records


def csv_rows(records, name):
    return list(csv.DictReader(io.StringIO(records[name].decode())))


def flight_counts(rows, expected):
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} flights, got {len(rows)}")
    for row in rows:
        outcomes = [int(row[k]) for k in ("success", "collision", "timeout")]
        if any(value not in (0, 1) for value in outcomes) or sum(outcomes) != 1:
            raise ValueError("Flight must have exactly one terminal outcome")
    return {"n": len(rows), **{k: sum(int(r[k]) for r in rows)
            for k in ("success", "collision", "timeout")}}


def goal_bearing(row):
    dx = float(row["goal_x"]) - float(row["start_x"])
    dy = float(row["goal_y"]) - float(row["start_y"])
    return math.remainder(math.atan2(dy, dx) - float(row["start_yaw"]), 2 * math.pi)


def review_experiments(records):
    review = {"schema": 1, "scope": "exposed development; no FINAL or adoption",
              "archive_members_verified": len(records), "observe": {},
              "shaping": {}, "source_audit": {}}
    for arm in ("armT", "armC"):
        history = csv_rows(records, f"observe/{arm}.bin.history.csv")
        updates = [int(r["rollout"]) for r in history]
        if updates != sorted(set(updates)) or updates[-1] != 10000:
            raise ValueError(f"Incomplete or duplicate training history: {arm}")
        if any(int(r["transitions"]) != int(r["rollout"]) * 128 * 32 for r in history):
            raise ValueError(f"Training transition budget mismatch: {arm}")
    for split in SPLITS:
        arms = {}
        for arm in ("fast", "bc", "armT", "armC"):
            rows = csv_rows(records, f"observe/{arm}-{split}-mode17.csv")
            stats = flight_counts(rows, 128)
            successes = [r for r in rows if int(r["success"])]
            stats["successful_arrival_mean_s"] = sum(float(r["time_s"]) for r in successes) / len(successes)
            detours = [r for r in rows if int(r["route_class"]) == 1]
            outside = [r for r in detours if abs(goal_bearing(r)) > math.pi / 4]
            stats["detour_success_n"] = [sum(int(r["success"]) for r in detours), len(detours)]
            stats["horizontal_out_of_view_detour_success_n"] = [sum(int(r["success"]) for r in outside), len(outside)]
            arms[arm] = stats
        treatment = csv_rows(records, f"observe/armT-{split}-mode17.csv")
        control = csv_rows(records, f"observe/armC-{split}-mode17.csv")
        control_by_env = {r["env"]: r for r in control}
        if len(control_by_env) != 128 or len({r["env"] for r in treatment}) != 128:
            raise ValueError("Duplicate task identity")
        wins, losses = 0, 0
        for t in treatment:
            c = control_by_env[t["env"]]
            for key in ("scene_seed", "start_x", "start_y", "start_z", "start_yaw", "goal_x", "goal_y", "goal_z"):
                if t[key] != c[key]:
                    raise ValueError(f"Paired task mismatch: {key}")
            wins += int(t["success"]) and not int(c["success"])
            losses += int(c["success"]) and not int(t["success"])
        arms["paired_final_T_vs_C"] = {"T_only_success": wins, "C_only_success": losses, "net": wins - losses}
        review["observe"][split] = arms
        review["shaping"][split] = {
            f"{arm}-s{seed}": flight_counts(csv_rows(records, f"shaping/{arm}-s{seed}-{split}.csv"), 128)
            for seed in (1, 2) for arm in ("control", "shaped")}
    for arm in ("policy", "blind", "hover"):
        rows = csv_rows(records, f"audit/source-{arm}-episodes.csv")
        if len({(r["env"], r["slot"]) for r in rows}) != 1024:
            raise ValueError("Full TRAIN slot coverage missing")
        review["source_audit"][arm] = flight_counts(rows, 1024)
    review["limitations"] = [
        "Observe comparison has one training seed; final 10000 is primary, best is secondary.",
        "BC dev-c was viewed before PPO; no design change reported. These are development results.",
        "Shaped seed 1 ran 3700 updates; best selected at 1950 lies within the 2000 selection budget.",
        "Shaping tables compare selected best policies; no final-2000 shaped seed-1 weights retained.",
        "Horizontal angle labels do not establish full-frustum visibility or a visible opening.",
        "Successful-arrival means omit failed flights; retain outcome counts beside them.",
        "Archived flight rows permit result reproduction, not a new simulator-transfer claim."]
    return review


def plot_experiments(records, review, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for arm, label in (("armT", "BC → PPO"), ("armC", "PPO control")):
        rows = csv_rows(records, f"observe/{arm}.bin.history.csv")
        axes[0].plot([int(r["rollout"]) for r in rows], [100 * float(r["dev_success"]) for r in rows], label=label, alpha=.8)
    axes[0].set(xlabel="PPO rollouts", ylabel="DEV-a success (%)", ylim=(0, 100), title="Selection curve; one training seed")
    axes[0].legend()
    for arm, label in (("bc", "BC before PPO"), ("armT", "BC → PPO final"), ("armC", "PPO control final")):
        axes[1].plot(SPLITS, [review["observe"][s][arm]["success"] for s in SPLITS], marker="o", label=label)
    axes[1].set(ylabel="Successes / 128", title="Frozen final outcomes; failures retained")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / "local-learning-review.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for seed in (1, 2):
        deltas = [review["shaping"][s][f"shaped-s{seed}"]["success"] - review["shaping"][s][f"control-s{seed}"]["success"] for s in SPLITS]
        ax.plot(SPLITS, deltas, marker="o", label=f"Training seed {seed}")
    ax.axhline(0, color="black", linewidth=.8)
    ax.set(ylabel="Shaping − control successes / 128", title="Geodesic shaping: selected policy comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "local-shaping-review.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=ROOT / "evidence/inputs/local-training-review/records.tar.gz")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--figures", type=Path)
    args = parser.parse_args()
    records = read_records(args.records)
    review = review_experiments(records)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n")
    if args.figures:
        plot_experiments(records, review, args.figures)
    print(f"Verified {len(records)} inputs; final T−C deltas: " + ", ".join(f"{s} {review['observe'][s]['paired_final_T_vs_C']['net']:+d}" for s in SPLITS))


if __name__ == "__main__":
    main()
