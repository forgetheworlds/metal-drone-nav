#!/usr/bin/env python3
"""Verify and summarize the frozen BC/fast Webots development comparison."""
import argparse
import hashlib
import json
from pathlib import Path

from local_training_review import read_records

ROOT = Path(__file__).resolve().parent


def review_transfer(records):
    manifest = json.loads(records["selection-manifest.json"])
    flights = json.loads(records["flights.json"])
    planned = {(r["slug"], r["actor"], r["index"]) for r in manifest["planned_flights"]}
    if len(flights) != 256 or {(r["slug"], r["actor"], r["index"]) for r in flights} != planned:
        raise ValueError("Incomplete or changed flight cohort")
    for actor, asset in manifest["assets"].items():
        if hashlib.sha256(records[f"nav/{actor}.bin"]).hexdigest() != asset["nav_sha256"]:
            raise ValueError("Frozen NAV hash mismatch")
    overlaps = []
    for row in flights:
        prefix = f"flights/{row['slug']}/"
        run = json.loads(records[prefix + "run.json"])
        episode = json.loads(records[prefix + "episode.json"])
        if not run["valid"] or run["exit_code"] != 0:
            raise ValueError(f"Invalid flight: {row['slug']}")
        if not episode["navigation_loaded"] or not episode["raptor_loaded"] or episode["grading"] != "navigation_task_step":
            raise ValueError("Missing policy/controller or wrong grader")
        if episode["sensor_profile"] != "legacy" or episode["policy_version"] != 1:
            raise ValueError("Changed policy/sensor contract")
        if not any(episode[k] for k in ("success", "collision", "timeout")) or (episode["success"] and episode["collision"]):
            raise ValueError("Invalid terminal result")
        if sum(int(episode[k]) for k in ("success", "collision", "timeout")) > 1:
            overlaps.append(row["slug"])
        if episode["success"]:
            if episode["goal_dwell_s"] < .1999 or episode["final_world_speed_mps"] > .5001 or episode["final_error_m"] > .3501:
                raise ValueError("Success without the declared stable arrival")
        world = records["worlds/" + Path(row["world"]).name]
        if hashlib.sha256(world).hexdigest() != row["world_sha256"]:
            raise ValueError("Recorded world changed")
    actors = {}
    for actor in ("bc", "fast"):
        rows = [r for r in flights if r["actor"] == actor]
        if len(rows) != 128:
            raise ValueError("Missing actor flights")
        # The shared grader can flag contact and deadline on the same tick.
        # Match the existing source/report precedence and preserve raw flags.
        successes = [r for r in rows if r["success"] and not r["collision"]]
        collisions = [r for r in rows if r["collision"]]
        timeouts = [r for r in rows if r["timeout"] and not r["success"] and not r["collision"]]
        if len(successes) + len(collisions) + len(timeouts) != 128:
            raise ValueError("Terminal denominator does not match")
        actors[actor] = {"n": 128, "success": len(successes), "collision": len(collisions), "timeout": len(timeouts),
                         "mean_successful_arrival_s": sum(float(r["time_s"]) for r in successes) / len(successes),
                         "by_route": {str(rc): {"n": sum(r["route_class"] == rc for r in rows),
                                                "success": sum(bool(r["success"]) for r in rows if r["route_class"] == rc)} for rc in (0, 1)}}
    by_task = {(r["index"], r["actor"]): r for r in flights}
    pairs = {}
    for label, indices in (("primary_balanced", manifest["primary_indices"]), ("full_bank", range(128))):
        wins = sum(bool(by_task[i, "bc"]["success"]) and not by_task[i, "fast"]["success"] for i in indices)
        losses = sum(bool(by_task[i, "fast"]["success"]) and not by_task[i, "bc"]["success"] for i in indices)
        pairs[label] = {"n_tasks": len(indices), "bc_only_success": wins, "fast_only_success": losses, "net": wins - losses}
    return {"scope": "exposed nominal DEV-b; frozen navigation, no target training; not FINAL or broad robustness",
            "actors": actors, "pairs": pairs, "raw_terminal_flag_overlap": overlaps,
            "archive_inputs_verified": len(records)}


def plot_transfer(review, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    names = ["Preserved fast", "Imitation before PPO"]
    values = [review["actors"][a] for a in ("fast", "bc")]
    axes[0].bar(names, [v["success"] for v in values], color=["#777777", "#2378a8"])
    axes[0].set(ylabel="Stable arrivals / 128", ylim=(0, 128), title="Frozen native Webots comparison")
    for i, value in enumerate(values):
        axes[0].text(i, value["success"] + 2, str(value["success"]), ha="center")
    axes[1].bar(names, [v["mean_successful_arrival_s"] for v in values], color=["#777777", "#2378a8"])
    axes[1].set(ylabel="Mean successful arrival (s)", title="Reliability comes with slower arrivals")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/native-bc-review.json")
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    records = read_records(ROOT / "evidence/inputs/native-bc-fast/records.tar.gz")
    review = review_transfer(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(review, indent=2) + "\n")
    if args.figure:
        plot_transfer(review, args.figure)
    print("Verified 256 native runs; BC/fast stable arrivals:", review["actors"]["bc"]["success"], review["actors"]["fast"]["success"])


if __name__ == "__main__":
    main()
