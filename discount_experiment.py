#!/usr/bin/env python3
"""Reproducible plots for the omp-credit-assignment gamma time-scale experiment.

Reads ONLY the recorded result CSVs owned by evidence/inputs/discount-experiment:
  arm-gamma0990/credit-stages.csv   (DEV eval every 100 rollouts)
  arm-gamma0995/credit-stages.csv
  arm-gamma0990/credit-runs.csv     (per-training-rollout metrics, 800 rows)
  arm-gamma0995/credit-runs.csv

Writes PNGs into artifacts/.
No data is synthesised; every point is a recorded measurement.

Run:
  python3 discount_experiment.py
"""
import csv
import os
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
CREDIT = os.path.join(ROOT, "evidence", "inputs", "discount-experiment")
OUT = os.path.join(ROOT, "artifacts")
ARMS = [("gamma0990", 0.99, "#1f77b4"), ("gamma0995", 0.995, "#d62728")]


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(rows, key):
    return [float(r[key]) for r in rows]


def plot_dev_curves():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    metrics = [
        ("dev_corner", "DEV corners", 30),
        ("dev_rooms", "DEV rooms", 30),
        ("dev_vertical", "DEV vertical", 30),
        ("dev_total", "DEV total", 90),
    ]
    stages = {}
    for name, gamma, color in ARMS:
        stages[name] = read_csv(os.path.join(CREDIT, name, "credit-stages.csv"))
    for ax, (key, label, denom) in zip(axes.ravel(), metrics):
        for name, gamma, color in ARMS:
            rows = stages[name]
            ax.plot(fnum(rows, "rollout"), fnum(rows, key), "o-", color=color,
                    label="gamma=%.3f" % gamma)
        ax.set_title("%s  (denominator %d)" % (label, denom))
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.5, denom + 0.5)
    axes[0][0].set_ylabel("successes")
    axes[1][0].set_ylabel("successes")
    axes[1][0].set_xlabel("training rollout")
    axes[1][1].set_xlabel("training rollout")
    axes[0][0].legend(loc="best", fontsize=8)
    fig.suptitle("All-90 original-start DEV evaluation vs training rollout\n"
                 "both arms share warm start; neither arm ever scores a DEV corner",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = os.path.join(OUT, "credit-arms-dev-outcomes.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def plot_train_loss():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, gamma, color in ARMS:
        rows = read_csv(os.path.join(CREDIT, name, "credit-runs.csv"))
        x = fnum(rows, "transitions")
        axes[0].plot(x, fnum(rows, "value_loss"), color=color, lw=0.9,
                     label="gamma=%.3f" % gamma)
        axes[1].plot(x, fnum(rows, "policy_loss"), color=color, lw=0.9,
                     label="gamma=%.3f" % gamma)
    axes[0].set_title("value loss (sampled minibatch)")
    axes[1].set_title("policy loss (sampled minibatch)")
    for ax in axes:
        ax.set_xlabel("cumulative TRAIN transitions")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Recorded per-rollout training losses; loss magnitude is not a "
                 "navigation result", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = os.path.join(OUT, "credit-arms-train-loss.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def plot_cumulative_family():
    # Cumulative TRAIN episodes classified by the transition's own family.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    fams = [("corner", "corners", 30), ("rooms", "rooms", 30), ("vertical", "vertical", 30)]
    for ax, (fam, label, _) in zip(axes, fams):
        for name, gamma, color in ARMS:
            rows = read_csv(os.path.join(CREDIT, name, "credit-runs.csv"))
            succ = []
            for r in rows:
                parts = dict(p.split("=") for p in r["cumulative_family_outcomes"].split(";"))
                fields = parts[fam].split("/")  # ep/succ/collision/timeout
                succ.append(float(fields[1]))
            ax.plot(fnum(rows, "transitions"), succ, color=color, lw=0.9,
                    label="gamma=%.3f" % gamma)
        ax.set_title("cumulative TRAIN %s successes" % label)
        ax.set_xlabel("cumulative TRAIN transitions")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("cumulative successes")
    fig.suptitle("TRAIN success attributed to each transition's own family: "
                 "zero corner successes in both arms", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = os.path.join(OUT, "credit-arms-cumulative-train.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def main():
    global CREDIT, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", default=CREDIT)
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()
    CREDIT, OUT = args.inputs, args.out
    os.makedirs(OUT, exist_ok=True)
    for fn in (plot_dev_curves, plot_train_loss, plot_cumulative_family):
        print("wrote", fn())


if __name__ == "__main__":
    main()
