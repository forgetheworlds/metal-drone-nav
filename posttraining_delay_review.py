#!/usr/bin/env python3
"""Review or rerun the frozen-policy DEV-b delay experiment."""
import argparse
import json
from pathlib import Path
import subprocess

from local_training_review import csv_rows, flight_counts, read_records

ROOT = Path(__file__).resolve().parent
ACTORS = ("fast", "bc", "armT", "armC")
DELAYS = {"nominal": (0, 0), "sensor100ms": (2, 0),
          "command100ms": (0, 2), "both100ms": (2, 2)}


def review_delays(records):
    results = []
    identities = None
    for actor in ACTORS:
        for condition, (sensor, command) in DELAYS.items():
            rows = csv_rows(records, f"flights/{actor}-{condition}.csv")
            counts = flight_counts(rows, 128)
            task_ids = {(r["env"], r["scene_seed"], r["start_x"], r["start_y"],
                         r["start_z"], r["start_yaw"], r["goal_x"], r["goal_y"],
                         r["goal_z"]) for r in rows}
            if len(task_ids) != 128:
                raise ValueError("Duplicate task identity")
            if identities is None:
                identities = task_ids
            if task_ids != identities:
                raise ValueError("Delay experiment changed task identities")
            if any(int(r["sensor_delay"]) != sensor or int(r["command_delay"]) != command for r in rows):
                raise ValueError("Recorded delay does not match the experiment")
            if any(float(r["stable_hold_s"]) < .1999 or float(r["final_speed_mps"]) > .5001
                   or float(r["final_distance_m"]) > .3501
                   for r in rows if int(r["success"])):
                raise ValueError("Success without the declared stable hold")
            results.append({"actor": actor, "condition": condition, **counts})
    return {"scope": "one exposed DEV-b bank; frozen source policies; no training or FINAL",
            "delay_tick_s": .05, "rows": results,
            "limitations": ["Cold-start sensor padding and zero command padding are included.",
                            "No noise, wind, dynamics variation, or native simulator result is implied.",
                            "An improvement with delay can reflect smoothing or startup timing."]}


def rerun(records, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for actor in ACTORS:
        checkpoint = directory / f"{actor}.bin"
        checkpoint.write_bytes(records[f"weights/{actor}.bin"])
        for condition, (sensor, command) in DELAYS.items():
            output = directory / f"{actor}-{condition}.csv"
            args = [str(ROOT / "build/metal_nav_waypoint"), "local-eval", str(checkpoint),
                    str(output), "--spec", "dev-b", "--mode", "17", "--seed", "700001",
                    "--sensor-delay", str(sensor), "--command-delay", str(command)]
            with output.with_suffix(".log").open("w") as log:
                subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)
            records[f"flights/{actor}-{condition}.csv"] = output.read_bytes()


def plot_delays(review, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    names = {"fast": "Preserved fast", "bc": "Imitation before PPO",
             "armT": "Imitation → PPO final", "armC": "PPO control final"}
    for actor in ACTORS:
        rows = [r for r in review["rows"] if r["actor"] == actor]
        ax.plot(range(4), [r["success"] for r in rows], marker="o", label=names[actor])
    ax.set_xticks(range(4), ["Nominal", "Sensor +100 ms", "Command +100 ms", "Both +100 ms"])
    ax.set(ylabel="Stable arrivals / 128", title="Frozen policies on the same DEV-b tasks", ylim=(0, 128))
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Rerun source evaluation; hold the shared GPU lock externally")
    parser.add_argument("--out", type=Path, default=ROOT / "results/posttraining-delay-replay")
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    records = read_records(ROOT / "evidence/inputs/posttraining-delay/records.tar.gz")
    if args.run:
        rerun(records, args.out)
    review = review_delays(records)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "review.json").write_text(json.dumps(review, indent=2) + "\n")
    if args.figure:
        plot_delays(review, args.figure)
    for actor in ACTORS:
        print(actor, [r["success"] for r in review["rows"] if r["actor"] == actor])


if __name__ == "__main__":
    main()
