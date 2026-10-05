#!/usr/bin/env python3
"""Recompute completed training-strategy comparisons from archived flight rows."""
import argparse
import json
from pathlib import Path

from local_training_review import csv_rows, flight_counts, read_records

ROOT = Path(__file__).resolve().parent


def review_strategies(records):
    review = {"scope": "source development evidence; no adoption or FINAL", "mixture": {}, "penalty": {}, "dynamic": {}}
    for split in ("dev-g1", "dev-g2", "dev-a", "dev-b", "dev-c", "open", "clutter"):
        arms = {}
        for arm in ("C", "T"):
            rows = []
            for seed in (1, 2):
                rows += csv_rows(records, f"mixture/evals/armG2-{arm}-s{seed}-{split}-mode17.csv")
            stats = flight_counts(rows, 256)
            stats["blocked_n"] = sum(int(r["route_class"]) == 1 for r in rows)
            stats["blocked_success"] = sum(int(r["success"]) for r in rows if int(r["route_class"]) == 1)
            arms[arm] = stats
        review["mixture"][split] = arms
    for split in ("dev-r1", "dev-r2", "dev-a", "dev-b", "dev-c", "open", "clutter"):
        arms = {}
        for penalty in (10, 50):
            rows = []
            for seed in (20261014, 20261015):
                rows += csv_rows(records, f"penalty/evals/pen{penalty}-s{seed}-{split}-m17.csv")
            stats = flight_counts(rows, 256)
            successes = [r for r in rows if int(r["success"])]
            stats["successful_arrival_mean_s"] = sum(float(r["time_s"]) for r in successes) / len(successes)
            arms[str(penalty)] = stats
        review["penalty"][split] = arms
    for arm in ("warmstart", "full-s1", "ablate-s1", "full-s2", "ablate-s2"):
        review["dynamic"][arm] = flight_counts(csv_rows(records, f"dynamic/evals/{arm}-devdyn.csv"), 128)
    review["limitations"] = [
        "Dynamic results use selected checkpoints on the development selector, not a blind test.",
        "Original dynamic ablation removed direct previous depth while cached geometry history remained.",
        "Mixture improves new blocked tasks but fails the old DEV-c retention gate.",
        "Penalty50 reduces contacts but increases timeouts and successful arrival time.",
        "Successful-arrival means exclude failures; outcome denominators remain visible.",
        "Archive supports result analysis, not complete training or hardware reproduction."]
    return review


def plot_strategies(review, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].bar(["Current bank", "New mixture"], [sum(review["mixture"][s][a]["blocked_success"] for s in ("dev-g1", "dev-g2")) for a in ("C", "T")])
    axes[0].set(ylabel="Successes / 128 blocked flights", ylim=(0, 128), title="New skills; retention gate failed")
    for x, p in enumerate(("10", "50")):
        r = review["penalty"]["dev-r2"][p]
        axes[1].bar(x, r["collision"], color="#a64040", label="Contact" if x == 0 else None)
        axes[1].bar(x, r["timeout"], bottom=r["collision"], color="#c59a36", label="Timeout" if x == 0 else None)
    axes[1].set_xticks((0, 1), ("Penalty 10", "Penalty 50"))
    axes[1].set(ylabel="Failures / 256 flights", ylim=(0, 256), title="Fewer contacts; more stalls")
    axes[1].legend()
    keys = ("warmstart", "full-s1", "full-s2")
    axes[2].bar(("Starting", "Seed 1", "Seed 2"), [review["dynamic"][k]["success"] for k in keys])
    axes[2].set(ylabel="Successes / 128 flights", ylim=(0, 128), title="Dynamic learning; selector evidence")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/training-strategy-review.json")
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    records = read_records(ROOT / "evidence/inputs/training-strategies/records.tar.gz")
    review = review_strategies(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(review, indent=2) + "\n")
    if args.figure:
        plot_strategies(review, args.figure)
    print("Verified original strategy records; no follow-up training data included")


if __name__ == "__main__":
    main()
