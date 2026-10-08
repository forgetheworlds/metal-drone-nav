"""Export one exact WPBANK task with bounded sphere motion to an isolated project."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "webots"))
from obstacle_motion import position
import local_waypoint_transfer as transfer
import metal_scene


def export(bank, index, policy, project, slug, policy_version=1):
    if policy_version not in [1, 4]:
        raise ValueError("bounded guided export requires policy version1 or4")
    record = transfer.read_bank(bank)[index]
    shapes, schedules = [], []
    for i, obstacle in enumerate(record["obstacles"][:record["world_count"]]):
        kind = obstacle["kind"]
        if kind not in [0, 1, 2, 3]:
            raise ValueError("unsupported obstacle kind")
        visual = copy.deepcopy(obstacle)
        if kind == 3:
            schedules.append({"def": f"ChallengeObstacle{i:02d}", "kind": kind,
                              "center": obstacle["center"], "size": obstacle["half_extent"],
                              "velocity": obstacle["velocity"]})
            visual["kind"] = 1
            visual["center"] = list(position(kind, obstacle["center"], obstacle["half_extent"], obstacle["velocity"], 0))
            visual["half_extent"] = [obstacle["half_extent"][0]] * 3
        elif any(obstacle["velocity"]):
            raise ValueError("this exporter requires bounded moving spheres, not legacy linear movers")
        visual["velocity"] = [0, 0, 0]
        shapes.append(visual)
    project.mkdir(parents=True, exist_ok=True)
    (project / "worlds").mkdir(exist_ok=True)
    (project / "controllers").mkdir(exist_ok=True)
    for name in ["assets", "protos"]:
        target = project / name
        source = ROOT / "assets" if name == "assets" else ROOT / "webots/protos"
        if not target.exists():target.symlink_to(source, target_is_directory=True)
    nav = project / "controllers/local_waypoint_transfer"
    if not nav.exists():nav.symlink_to(ROOT / "webots/controllers/local_waypoint_transfer", target_is_directory=True)
    mover = project / "controllers/bounded_obstacle"
    mover.mkdir(exist_ok=True)
    shutil.copy2(ROOT / "webots/controllers/bounded_obstacle/bounded_obstacle.py", mover / "bounded_obstacle.py")
    shutil.copy2(ROOT / "obstacle_motion.py", mover / "obstacle_motion.py")
    record["obstacles_used"] = shapes
    transfer.WORLD_DIR = project / "worlds"
    world = transfer.write_world(record, "bounded", policy, None, slug=slug)
    text = world.read_text()
    # The controller resolves absolute NAV paths; isolate receipt paths by project.
    text = re.sub(r"policy=[^;\"]+", "policy=" + str(policy.resolve()), text, count=1)
    text = text.replace("policy_version=1", f"policy_version={policy_version}", 1)
    if schedules:
        custom = json.dumps(schedules, separators=(",", ":")).replace('"', '\\"')
        text += '\nDEF BoundedMoverDriver Robot { supervisor TRUE controller "bounded_obstacle" customData "' + custom + '" }\n'
    world.write_text(text)
    manifest = {"bank": str(bank.resolve()), "bank_sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
                "index": index, "record_sha256": record["record_sha256"], "record": record,
                "schedules": schedules, "world_sha256": hashlib.sha256(world.read_bytes()).hexdigest(),
                "policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
                "policy_version": policy_version,
                "motion_module_sha256": hashlib.sha256((mover / "obstacle_motion.py").read_bytes()).hexdigest()}
    world.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    return world


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bank", type=Path);parser.add_argument("index", type=int)
    parser.add_argument("policy", type=Path);parser.add_argument("project", type=Path)
    parser.add_argument("--slug", default="bounded-motion-smoke")
    args = parser.parse_args()
    print(export(args.bank, args.index, args.policy, args.project.resolve(), args.slug))
