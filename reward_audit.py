#!/usr/bin/env python3
"""Analyze recorded source flights; this script does not train or simulate."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
CONTROLS = ("actor", "route_teacher", "scan_then_teacher", "hover")


def read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as stream:
        return list(csv.DictReader(stream))


def analyze(episodes: list[dict[str, str]], transitions: list[dict[str, str]]) -> dict:
    keys = {(row["failure_id"], row["control"]) for row in episodes}
    if len(episodes) != 120 or len(keys) != 120:
        raise ValueError("expected four controls on the same 30 source TRAIN corners")
    trajectories: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in transitions:
        trajectories.setdefault((row["failure_id"], row["control"]), []).append(row)
    if keys != trajectories.keys():
        raise ValueError("episode and transition identities differ")
    reconstruction_error = 0.0
    for row in episodes:
        key = (row["failure_id"], row["control"])
        data = trajectories[key]
        if [int(step["tick"]) for step in data] != list(range(1, len(data) + 1)):
            raise ValueError(f"non-contiguous transition ticks: {key}")
        replay = sum(.99 ** tick * float(step["reward"]) for tick, step in enumerate(data))
        if abs(replay - float(row["return_gamma99"])) > 1e-7:
            raise ValueError(f"recorded return differs from transition replay: {key}")
        if any(int(step["terminated"]) or int(step["truncated"]) for step in data[:-1]):
            raise ValueError(f"transition after an episode boundary: {key}")
        final = data[-1]
        if int(final["terminated"]) != int(row["success"]) + int(row["collision"]):
            raise ValueError(f"termination metadata differs: {key}")
        if int(final["truncated"]) != int(row["timeout"]):
            raise ValueError(f"timeout metadata differs: {key}")
        for step in data:
            parts = sum(float(step[name]) for name in
                        ("progress_reward", "time_cost", "risk_cost", "terminal_reward"))
            reconstruction_error = max(reconstruction_error, abs(parts - float(step["reward"])))
    if reconstruction_error > 1e-7:
        raise ValueError("reward decomposition does not match native rewards")
    hover = {row["failure_id"]: row for row in episodes if row["control"] == "hover"}
    aggregates = {}
    for control in CONTROLS:
        subset = [row for row in episodes if row["control"] == control]
        success = [row for row in subset if int(row["success"])]
        def inverted(row, bootstrap=False):
            comparison = hover[row["failure_id"]]
            value = float(comparison["return_gamma99"])
            if bootstrap:
                value += float(comparison["bootstrap_return"])
            return float(row["return_gamma99"]) < value
        aggregates[control] = {
            "episodes": len(subset),
            "successes": sum(int(row["success"]) for row in subset),
            "contacts": sum(int(row["collision"]) for row in subset),
            "timeouts": sum(int(row["timeout"]) for row in subset),
            "mean_return": statistics.mean(float(row["return_gamma99"]) for row in subset),
            "mean_success_return": statistics.mean(float(row["return_gamma99"]) for row in success) if success else None,
            "successful_below_hover": [row["failure_id"] for row in success if inverted(row)],
            "successful_below_hover_with_frozen_critic_bootstrap": [row["failure_id"] for row in success if inverted(row, True)],
        }
    counterfactuals = []
    for gamma in (.99, .995, .999, 1.0):
        returns = {key: sum(gamma ** tick * float(step["reward"])
                           for tick, step in enumerate(data))
                   for key, data in trajectories.items()}
        for control in ("route_teacher", "scan_then_teacher"):
            success = [row for row in episodes if row["control"] == control and int(row["success"])]
            counterfactuals.append({
                "gamma": gamma, "control": control, "successful_episodes": len(success),
                "successes_below_hover": sum(returns[(row["failure_id"], control)] <
                                            returns[(row["failure_id"], "hover")] for row in success),
                "mean_success_return": statistics.mean(returns[(row["failure_id"], control)] for row in success),
                "counterfactual_only_no_new_flights": True,
            })
    return {"control_aggregates": aggregates, "reward_gamma_counterfactuals": counterfactuals,
            "transition_count": len(transitions), "reconstruction_max_error": reconstruction_error,
            "discount_half_life_s": math.log(.5) / math.log(.99) / 20,
            "gae_half_life_s": math.log(.5) / math.log(.99 * .95) / 20,
            "rollout_s": 32 / 20}


def plot(episodes: list[dict[str, str]], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    hover = {row["failure_id"]: float(row["return_gamma99"]) for row in episodes if row["control"] == "hover"}
    for control, label, color in (("route_teacher", "Route teacher", "#176B87"),
                                  ("scan_then_teacher", "2 s scan + teacher", "#D97732")):
        success = [row for row in episodes if row["control"] == control and int(row["success"])]
        difference = sorted(float(row["return_gamma99"]) - hover[row["failure_id"]] for row in success)
        axes[0].plot(range(1, len(difference) + 1), difference, "o-", ms=4, label=f"{label}: {len(success)}/30", color=color)
    axes[0].axhline(0, color="black", lw=1)
    axes[0].set(xlabel="Successful flight, sorted independently", ylabel="Discounted return minus matched hover",
                title="Some real successes score below hovering")
    axes[0].legend(fontsize=9)
    times = [tick / 20 for tick in range(401)]
    for gamma, label, color in ((.99, "Actual reward discount γ = .99", "#176B87"),
                                (.99 * .95, "GAE weight formula γλ = .9405", "#D97732")):
        axes[1].semilogy(times, [gamma ** (20 * time) for time in times], label=label, color=color)
    axes[1].axvline(1.6, color="gray", ls="--", label="32-tick rollout boundary")
    axes[1].set(xlabel="Delay (seconds)", ylabel="Multiplicative weight", ylim=(1e-12, 1.1),
                title="Long routes rely on critic bootstrapping")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2)
    figure.suptitle("Source TRAIN audit · real RAPTOR/Metal flights · no policy improvement claimed", fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=ROOT / "evidence/inputs/reward-audit")
    parser.add_argument("--figure", type=Path, default=ROOT / "artifacts/reward-credit-audit.png")
    args = parser.parse_args()
    episodes = read_csv(args.inputs / "episodes.csv")
    transitions = read_csv(args.inputs / "transitions.csv.gz")
    metrics = analyze(episodes, transitions)
    # Print the derived evidence rather than overwriting its provenance manifest.
    print(json.dumps(metrics, indent=2))
    plot(episodes, args.figure)


if __name__ == "__main__":
    main()
