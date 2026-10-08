"""Grade the complete critic-geometry pilot without selecting checkpoints."""

import argparse
import hashlib
import json
from pathlib import Path
import struct

from navigation_physical_retention_results import flights, totals

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/root-critic-geometry"
IDENTITY_FIELDS = [
    "env", "scene_seed", "route_class", "family", "start_x", "start_y",
    "start_z", "start_yaw", "goal_x", "goal_y", "goal_z",
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_flights(control, geometry):
    for env in range(128):
        for field in IDENTITY_FIELDS:
            if control[env][field] != geometry[env][field]:
                raise ValueError(f"Flight identity differs: env {env}, {field}")
    common = [env for env in range(128)
              if int(control[env]["success"]) and int(geometry[env]["success"])]
    deltas = [float(geometry[e]["time_s"]) - float(control[e]["time_s"])
              for e in common]
    return {
        "control": totals(control), "geometry": totals(geometry),
        "wins": [e for e in range(128) if not int(control[e]["success"])
                 and int(geometry[e]["success"])],
        "losses": [e for e in range(128) if int(control[e]["success"])
                   and not int(geometry[e]["success"])],
        "common_successes": len(common), "arrival_deltas_s": deltas,
    }


def review(folder):
    request = json.loads((folder / "pilot-request.json").read_text())
    freeze = json.loads((folder / "freeze.json").read_text())
    jobs = json.loads((folder / "jobs.json").read_text())
    receipts = json.loads((folder / "eval-jobs.json").read_text())
    if len(jobs) != 4 or any(j.get("exit") != 0 for j in jobs):
        raise ValueError("Four completed producers are required")
    if len(receipts) != 92 or any(r.get("exit") != 0 for r in receipts):
        raise ValueError("All 92 actual evaluation receipts are required")
    for path, digest in freeze.items():
        if sha(path) != digest:
            raise ValueError(f"Frozen input changed: {path}")
    budget = {}
    for job in jobs:
        checkpoint = Path(job["checkpoint"])
        if sha(checkpoint) != job["checkpoint_sha256"]:
            raise ValueError("Final checkpoint hash differs")
        header = checkpoint.read_bytes()[:136]
        if struct.unpack_from("<4I", header, 12) != (967688, 14593, 32, 512):
            raise ValueError("Checkpoint architecture differs")
        if struct.unpack_from("<I", header, 36)[0] != 256 or struct.unpack_from("<Q", header, 40)[0] != 32768:
            raise ValueError("Final rollout/optimizer budget differs")
        counts = Path(str(checkpoint) + ".entry-transitions.bin").read_bytes()
        transitions = sum(struct.unpack("<" + str(len(counts) // 8) + "Q", counts))
        if transitions != 4194304:
            raise ValueError("Actual source exposure differs")
        budget[f"{job['group']}-s{job['seed']}"] = {
            "transitions": transitions, "optimizer_steps": 32768,
            "training_wall_s": job["wall_s"],
        }
    identity = lambda r: (r["label"], r["seed"], r["panel"], r["profile"])
    expected = {identity(r) for r in request["evaluation"]}
    if len(expected) != 92 or {identity(r) for r in receipts} != expected:
        raise ValueError("Evaluation cases do not match the frozen request")
    data = {}
    for receipt in receipts:
        if sha(receipt["csv"]) != receipt["csv_sha256"] or sha(receipt["bank"]) != receipt["bank_sha256"]:
            raise ValueError("Flight or bank hash differs")
        data[identity(receipt)] = flights(receipt["csv"])
    # Reuse the already scored parent and unmasked candidate. Check their exact
    # files and paired task identities; do not launch their flights again.
    references = json.loads((ROOT / "results/omp-functional-retention/root-eval-receipts.json").read_text())
    reference_roles = {"composed_parent": "composed_parent", "treatment": "prior_unmasked"}
    reference_count = 0
    for receipt in references:
        if receipt["label"] not in reference_roles:
            continue
        if receipt.get("exit") != 0 or sha(receipt["csv"]) != receipt["csv_sha256"]:
            raise ValueError("Cached reference flight receipt differs")
        if sha(receipt["bank"]) != receipt["bank_sha256"]:
            raise ValueError("Cached reference bank differs")
        key = (reference_roles[receipt["label"]], receipt["seed"], receipt["panel"], receipt["profile"])
        data[key] = flights(receipt["csv"])
        reference_count += 1
    if reference_count != 92:
        raise ValueError("All 92 cached parent/prior-model cases are required")
    panels, per_seed, gates = {}, {}, []

    def gate(name, passed, actual, required):
        gates.append({"gate": name, "pass": bool(passed), "actual": actual,
                      "required": required})

    for panel, profile in sorted({(r["panel"], r["profile"]) for r in receipts}):
        key = f"{panel}-{profile}"
        pooled = {role: {metric: 0 for metric in ["success", "collision", "timeout"]}
                  for role in ["control", "geometry", "composed_parent", "prior_unmasked"]}
        seed_results, arrival = {}, []
        for seed in [1, 2]:
            summary = compare_flights(data["control", seed, panel, profile],
                                      data["geometry", seed, panel, profile])
            reference_deltas = {}
            for role in ["composed_parent", "prior_unmasked"]:
                comparison = compare_flights(data[role, seed, panel, profile],
                                             data["geometry", seed, panel, profile])
                summary[role] = totals(data[role, seed, panel, profile])
                reference_deltas[role] = {
                    "wins": comparison["wins"], "losses": comparison["losses"],
                    "common_successes": comparison["common_successes"],
                    "common_arrival_delta_s": sum(comparison["arrival_deltas_s"]) / comparison["common_successes"]
                    if comparison["common_successes"] else None,
                }
            summary["reference_comparisons"] = reference_deltas
            seed_results[str(seed)] = summary
            for role in pooled:
                for metric in pooled[role]:
                    pooled[role][metric] += summary[role][metric]
            arrival.extend(summary["arrival_deltas_s"])
            if panel == "dev-c" and profile == "nominal":
                delta = summary["geometry"]["success"] - summary["control"]["success"]
                gate(key + f" seed{seed} success", delta >= 0, delta, ">=0")
        c, t = pooled["control"], pooled["geometry"]
        delta_s = sum(arrival) / len(arrival) if arrival else None
        panels[key] = {"tasks": 256, **pooled, "common_successes": len(arrival),
                       "common_arrival_delta_s": delta_s}
        per_seed[key] = seed_results
        gate(key + " success retention", t["success"] >= c["success"] - 3,
             t["success"] - c["success"], ">=-3")
        for metric in ["collision", "timeout"]:
            gate(key + " " + metric, t[metric] <= c[metric] + 3,
                 t[metric] - c[metric], "<=3")
        if profile == "nominal" and panel == "dev-c":
            gate(key + " primary success", t["success"] >= c["success"] + 4,
                 t["success"] - c["success"], ">=4")
            gate(key + " primary contacts", t["collision"] <= c["collision"] - 3,
                 t["collision"] - c["collision"], "<=-3")
        if profile == "nominal" and panel in ["open", "fresh-open", "long-open", "fresh-long-open"]:
            gate(key + " absolute floor", t["success"] >= 253, t["success"], ">=253")
        if panel == "composite" and profile == "nominal":
            gate(key + " course floor", t["success"] >= 241, t["success"], ">=241")
        if panel == "fresh-course-reflected" and profile == "combined":
            gate(key + " stress floor", t["success"] >= 178, t["success"], ">=178")
        if profile == "nominal" and panel in ["long-open", "composite", "open"]:
            limit = .2 if panel == "open" else .5
            gate(key + " common arrival speed", delta_s is not None and delta_s <= limit,
                 delta_s, f"<={limit}")
    return {"scope": "Exposed source development/stress, final256 only; no independent transfer or blind FINAL.",
            "status": "PASSES_PILOT_GATES_REQUIRES_FURTHER_VALIDATION" if all(g["pass"] for g in gates) else "NOT_ADOPTED",
            "budget": budget, "actual_flights": len(receipts) * 128,
            "reused_reference_flights": reference_count * 128,
            "evaluation_wall_s": sum(r["wall_s"] for r in receipts),
            "panels": panels, "per_seed": per_seed, "gates": gates,
            "failed_gates": [g for g in gates if not g["pass"]]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=OUT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = review(args.folder)
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text)
