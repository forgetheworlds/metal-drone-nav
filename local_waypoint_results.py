#!/usr/bin/env python3
"""Rebuild the local waypoint figures from published experiment records."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "evidence/inputs/local-waypoint"
OUTPUT = ROOT / "artifacts"
ACTORS = ("arrival", "candidate", "fast")
LABELS = ("Arrival baseline", "Reliability candidate", "Faster candidate")
COLORS = ("#64748b", "#2563eb", "#f59e0b")


def records(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def save(figure, name, caption):
    figure.text(.5, .015, caption, ha="center", fontsize=8)
    figure.tight_layout(rect=(0, .05, 1, 1))
    figure.savefig(OUTPUT / name, dpi=160)
    plt.close(figure)


def main():
    table = records(DATA / "evaluation-records.csv")
    splits = ("open", "deva", "devb", "clutter")
    groups = {(actor, split): [row for row in table if row["record_file"] == f"{actor}-{split}-mode17.csv"]
              for actor in ACTORS for split in splits}
    assert all(len(rows) == 128 for rows in groups.values())
    figure, axis = plt.subplots(figsize=(8, 4.6))
    for index, (actor, label, color) in enumerate(zip(ACTORS, LABELS, COLORS)):
        values = [sum(int(row["success"]) for row in groups[actor, split]) for split in splits]
        bars = axis.bar([x + (index-1)*.24 for x in range(4)], [100*n/128 for n in values],
                        width=.23, label=label, color=color)
        axis.bar_label(bars, labels=[str(n) for n in values], fontsize=8, padding=3)
    axis.set_xticks(range(4), ("Open", "DEV-a", "DEV-b", "Clutter"))
    axis.set_ylim(0, 115); axis.set_ylabel("Success (%)"); axis.legend(fontsize=8)
    axis.set_title("Local waypoint control: 128 tasks per bank")
    save(figure, "local-waypoint-eval-success.png", "Labels are successes /128. DEV-a selected checkpoints; single learning seed.")

    history = records(DATA / "training/runs-history.csv")
    figure, axis = plt.subplots(figsize=(8, 4.6))
    for run, label, color in (("candidate-local", "Local bank, time cost0.2", COLORS[1]),
                               ("candidate-fast", "Local bank, time cost1.0", COLORS[2])):
        rows = [row for row in history if row["run"] == run]
        axis.plot([int(row["rollout"]) for row in rows], [100*int(row["successes_of_128"])/128 for row in rows],
                  label=label, color=color)
    axis.set_ylim(0, 100); axis.set_xlabel("PPO rollouts"); axis.set_ylabel("DEV-a success (%)")
    axis.set_title("Observed learning progression on the selection bank"); axis.legend()
    save(figure, "local-waypoint-training-curves.png", "2000 rollouts per arm; repeated DEV-a feedback, not blind final evaluation.")

    figure, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for index, (actor, label, color) in enumerate(zip(ACTORS, LABELS, COLORS)):
        rows = groups[actor, "deva"]; success = [row for row in rows if int(row["success"])]
        axes[0].bar(index, sum(int(row["collision"]) for row in rows), color=color)
        axes[1].bar(index, sum(float(row["time_s"]) for row in success)/len(success), color=color)
    for axis in axes:
        axis.set_xticks(range(3), ("Baseline", "Reliable", "Faster"))
    axes[0].set_ylabel("Contacts /128"); axes[1].set_ylabel("Arrival seconds, successful flights only")
    figure.suptitle("Safety and arrival trade-off on DEV-a")
    save(figure, "local-waypoint-behaviour.png", "Different actors succeed on different tasks; conditional times do not isolate speed causally.")

    traces = records(DATA / "trajectory-records.csv")
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    for axis, filename, title in zip(axes, ("arrival-dev-a-mode17.csv", "fast-dev-a-mode17.csv"),
                                    ("Arrival baseline", "Faster candidate")):
        for env in range(4):
            rows = [row for row in traces if row["record_file"] == filename and int(row["env"]) == env]
            if not rows:
                continue
            axis.plot([float(row["x"]) for row in rows], [float(row["y"]) for row in rows], label=f"Task{env}")
            axis.scatter(float(rows[0]["goal_x"]), float(rows[0]["goal_y"]), marker="x")
        axis.set_title(title); axis.set_xlabel("World X (m)"); axis.set_ylabel("World Y (m)")
        axis.set_aspect("equal", adjustable="datalim"); axis.legend(fontsize=7)
    save(figure, "local-waypoint-trajectories.png", "First four recorded local tasks; XY projection omits vertical motion and geometry. These are plots, not videos.")


if __name__ == "__main__":
    main()
