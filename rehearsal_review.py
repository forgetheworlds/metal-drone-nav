#!/usr/bin/env python3
"""Review the fixed rehearsal schedule using frozen, paired flight records."""
import argparse
import csv
import hashlib
import json
import statistics
import tarfile
from pathlib import Path


SPLITS = ["dev-h", "dev-g1", "dev-g2", "dev-a", "dev-b", "dev-c", "open", "clutter"]
SEEDS = [20261015, 20261016]


def summarize(rows):
    successes = [r for r in rows if int(r["success"])]
    blocked = [r for r in rows if int(r["route_class"]) != 0]
    return {
        "episodes": len(rows),
        "successes": len(successes),
        "contacts": sum(int(r["collision"]) for r in rows),
        "timeouts": sum(int(r["timeout"]) for r in rows),
        "blocked_episodes": len(blocked),
        "blocked_successes": sum(int(r["success"]) for r in blocked),
        "successful_arrival_mean_s": statistics.mean(float(r["time_s"]) for r in successes),
        "episode_mean_s": statistics.mean(float(r["time_s"]) for r in rows),
    }


def review(archive):
    with tarfile.open(archive, "r:gz") as tar:
        hashes = json.load(tar.extractfile("SHA256.json"))
        contents = {}
        for name, expected in hashes.items():
            data = tar.extractfile(name).read()
            assert hashlib.sha256(data).hexdigest() == expected, name
            contents[name] = data
    result = {"input_files": len(hashes), "splits": {}}
    for selected in [False, True]:
        stage = "selected_best" if selected else "final10000"
        result["splits"][stage] = {}
        for split in SPLITS:
            arms = {}
            by_weight = {}
            for weight in [0, 1]:
                rows = []
                for seed in SEEDS:
                    label = f"armG3-weight{weight}-s{seed}" + ("-best" if selected else "")
                    name = f"evals/{label}-{split}-mode17.csv"
                    batch = list(csv.DictReader(contents[name].decode().splitlines()))
                    assert len(batch) == 128, name
                    assert len({int(r["env"]) for r in batch}) == 128, name
                    for row in batch:
                        assert int(row["success"]) + int(row["collision"]) + int(row["timeout"]) == 1
                        if int(row["success"]):
                            assert float(row["stable_hold_s"]) >= .199
                            assert float(row["final_distance_m"]) <= .35001
                            assert float(row["final_speed_mps"]) <= .50001
                        row["training_seed"] = seed
                    rows.extend(batch)
                arms[str(weight)] = summarize(rows)
                by_weight[weight] = {(r["training_seed"], r["env"]): r for r in rows}
            wins = losses = 0
            for key, control in by_weight[0].items():
                treatment = by_weight[1][key]
                for field in ["scene_seed", "start_x", "start_y", "start_z", "goal_x", "goal_y", "goal_z"]:
                    assert control[field] == treatment[field], (split, key, field)
                wins += int(treatment["success"]) > int(control["success"])
                losses += int(treatment["success"]) < int(control["success"])
            result["splits"][stage][split] = {"arms": arms, "paired_wins": wins, "paired_losses": losses}
    finals = result["splits"]["final10000"]
    h = finals["dev-h"]["arms"]
    c = finals["dev-c"]["arms"]
    result["matched_gates"] = {
        "fresh_blocked_no_loss": h["1"]["blocked_successes"] >= h["0"]["blocked_successes"],
        "old_dev_c_gain_at_least6": c["1"]["successes"] - c["0"]["successes"] >= 6,
    }
    floors = {"dev-a": 190, "dev-b": 195, "dev-c": 193, "open": 253, "clutter": 186}
    result["absolute_retention_gates"] = {
        split: finals[split]["arms"]["1"]["successes"] >= floor for split, floor in floors.items()
    }
    result["verdict"] = "NOT ADOPTED"
    assert not all(result["matched_gates"].values())
    result["scope"] = "Two matched source training seeds; exposed development retention and fresh nonselector dev-h. No native or blind FINAL claim. Fixed episode weighting, not adaptive failure replay."
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("evidence/inputs/rehearsal/records.tar.gz"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    result = review(args.archive)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    if args.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for weight, label, offset, color in [(0, "50% source episodes", -.18, "#4878a8"), (1, "67% source episodes", .18, "#c77748")]:
            values = [result["splits"]["final10000"][split]["arms"][str(weight)]["successes"] for split in SPLITS]
            ax.bar([i + offset for i in range(len(SPLITS))], values, width=.36, label=label, color=color)
        ax.set_xticks(range(len(SPLITS)), SPLITS)
        ax.set_ylabel("Successful flights / 256 (two training seeds)")
        ax.set_ylim(0, 280)
        ax.set_title("Fixed rehearsal weighting did not preserve navigation capability")
        ax.legend(loc="lower right")
        fig.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=160)
        plt.close(fig)
    print(json.dumps({"verdict": result["verdict"], "matched_gates": result["matched_gates"], "input_files": result["input_files"]}))


if __name__ == "__main__":
    main()
