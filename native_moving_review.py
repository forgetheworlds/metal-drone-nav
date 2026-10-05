#!/usr/bin/env python3
"""Review the frozen 128-flight native moving-obstacle benchmark."""
import argparse
import hashlib
import json
import statistics
import tarfile
from pathlib import Path


def review(path):
    with tarfile.open(path, "r:gz") as archive:
        hashes = json.load(archive.extractfile("SHA256.json"))
        data = {}
        for name, expected in hashes.items():
            value = archive.extractfile(name).read()
            assert hashlib.sha256(value).hexdigest() == expected, name
            data[name] = value
    rows = json.loads(data["progress.json"])
    assert len(rows) == 128
    assert len({(r["panel"], r["draw"], r["actor"]) for r in rows}) == 128
    for row in rows:
        slug = f"rootmoving-{row['panel']}-draw{row['draw']}-{row['actor']}"
        manifest = json.loads(data[f"flights/{slug}/manifest.json"])
        run = json.loads(data[f"flights/{slug}/run.json"])
        episode = json.loads(data[f"flights/{slug}/episode.json"])
        assert run["valid"] and run["exit_code"] == 0, slug
        assert episode["navigation_loaded"] and episode["raptor_loaded"], slug
        assert episode["grading"] == "navigation_task_step", slug
        assert manifest["world_sha256"] == hashes[f"flights/{slug}/world.wbt"], slug
        assert manifest["nav_sha256"] == hashes[f"nav/nav-{row['actor']}.bin"], slug
        assert manifest["mover_driver_sha256"] == hashes["source/local_moving_obstacle.py"], slug
        for name in ["success", "collision", "timeout", "time_s"]:
            assert row[name] == episode[name], (slug, name)
        assert sum(bool(episode[x]) for x in ["success", "collision", "timeout"]) == 1
        assert row["mover_samples"] >= 3 and row["mover_max_error_m"] <= .01
        if episode["success"]:
            assert episode["goal_dwell_s"] >= .199
            assert episode["final_error_m"] <= .35001
            assert episode["final_world_speed_mps"] <= .50001
    result = {"valid_flights": 128, "hashed_inputs": len(hashes), "panels": {}}
    for panel in ["diagnostic", "challenge"]:
        arms = {}
        keyed = {}
        for actor in ["fast", "full"]:
            group = [r for r in rows if r["panel"] == panel and r["actor"] == actor]
            assert len(group) == 32
            successes = [r for r in group if r["success"]]
            arms[actor] = {
                "episodes": len(group), "successes": len(successes),
                "contacts": sum(r["collision"] for r in group),
                "timeouts": sum(r["timeout"] for r in group),
                "successful_arrival_mean_s": statistics.mean(r["time_s"] for r in successes),
            }
            keyed[actor] = {r["draw"]: r for r in group}
        assert keyed["fast"].keys() == keyed["full"].keys()
        both = [k for k in keyed["fast"] if keyed["fast"][k]["success"] and keyed["full"][k]["success"]]
        wins = sum(keyed["full"][k]["success"] and not keyed["fast"][k]["success"] for k in keyed["fast"])
        losses = sum(keyed["fast"][k]["success"] and not keyed["full"][k]["success"] for k in keyed["fast"])
        result["panels"][panel] = {
            "arms": arms, "paired_full_only_wins": wins, "paired_fast_only_wins": losses,
            "both_success_n": len(both),
            "paired_full_minus_fast_arrival_s": statistics.mean(keyed["full"][k]["time_s"] - keyed["fast"][k]["time_s"] for k in both),
        }
    result["scope"] = "Frozen source weights and navigation; independent native ODE/sensors/motors/collisions; no target-side training. Diagnostic nonselector is easy; challenge was exposed source selector. Development evidence."
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("evidence/inputs/native-moving/records.tar.gz"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    result = review(args.archive)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    if args.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.5))
        labels = ["Diagnostic\nfast", "Diagnostic\nlearned", "Challenge\nfast", "Challenge\nlearned"]
        groups = [result["panels"][p]["arms"][a] for p in ["diagnostic", "challenge"] for a in ["fast", "full"]]
        ax.bar(range(4), [g["successes"] for g in groups], label="Goal reached", color="#4878a8")
        ax.bar(range(4), [g["contacts"] for g in groups], bottom=[g["successes"] for g in groups], label="Contact", color="#c77748")
        ax.set_xticks(range(4), labels)
        ax.set_ylim(0, 35)
        ax.set_ylabel("Flights / 32")
        ax.set_title("Frozen source policy in independent moving Webots scenes")
        ax.legend(loc="upper left")
        fig.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=160)
        plt.close(fig)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
