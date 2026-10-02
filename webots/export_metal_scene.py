#!/usr/bin/env python3
"""Export one selected Metal challenge record as an independent Webots world."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "worlds" / "a_to_b_bounded.wbt"
EXPECTED_BOUNDS = {"x": [-2.0, 14.0], "y": [-5.0, 5.0], "z": [0.0, 5.0]}


def _number(value: float) -> str:
    return f"{float(value):.9g}"


def _solid(index: int, obstacle: dict) -> str:
    kind = int(obstacle["kind"])
    center = [_number(value) for value in obstacle["center"]]
    half = [_number(value) for value in obstacle["half_extent"]]
    if kind == 0:
        geometry = f"Box {{ size {2*float(half[0]):.9g} {2*float(half[1]):.9g} {2*float(half[2]):.9g} }}"
        bounding = geometry
    elif kind == 1:
        radius = float(half[0])
        geometry = f"Sphere {{ radius {_number(radius)} }}"
        bounding = geometry
    elif kind == 2:
        radius, height = float(half[0]), 2.0 * float(half[2])
        geometry = f"Cylinder {{ radius {_number(radius)} height {_number(height)} }}"
        bounding = geometry
    else:
        raise ValueError(f"unsupported Metal primitive kind: {kind}")
    name = f"ChallengeObstacle{index:02d}"
    return (
        f'DEF {name} Solid {{ name "metal_{index:02d}_kind{kind}" '
        f'translation {" ".join(center)} children [ Shape {{ '
        f'appearance PBRAppearance {{ baseColor 0.63 0.34 0.18 roughness 1 }} '
        f'geometry {geometry} }} ] boundingObject {bounding} locked TRUE }}'
    )


def export_scene(scene_path: Path, output_path: Path, policy: str, max_steps: int = 2000) -> dict:
    scene = json.loads(scene_path.read_text())
    if scene.get("schema") != "webots-metal-scene-v1":
        raise ValueError("expected a webots-metal-scene-v1 scoring record")
    if scene.get("room_bounds_m") != EXPECTED_BOUNDS:
        raise ValueError("selected scene room bounds do not match the bounded Webots template")
    obstacles = scene.get("obstacles", [])
    if not obstacles or len(obstacles) > 16:
        raise ValueError("Metal scene must contain 1 to 16 supported obstacles")
    start = scene["start_xyz_m"]
    goal = scene["goal_xyz_m"]
    if len(start) != 3 or len(goal) != 3:
        raise ValueError("scene start and goal must be 3D points")
    distance = math.sqrt(sum((float(goal[i]) - float(start[i])) ** 2 for i in range(3)))
    text = TEMPLATE.read_text()
    solids = "\n".join(_solid(i, obstacle) for i, obstacle in enumerate(obstacles))
    if text.count("# WEBOTS_OBSTACLES_INSERTION_POINT") != 1:
        raise ValueError("bounded Webots template needs one obstacle insertion point")
    text = text.replace("# WEBOTS_OBSTACLES_INSERTION_POINT", solids, 1)
    title = str(scene["scene_name"]).replace('"', "")
    text = re.sub(r'title "[^"]*"', f'title "{title}"', text, count=1)
    text = re.sub(
        r'RaptorCrazyflie \{ translation [^}]*?customData "[^"]*" \}',
        'RaptorCrazyflie { translation ' + " ".join(_number(v) for v in start) +
        ' name "RaptorCrazyflie" controller "raptor_webots" supervisor TRUE customData "'
        f'phase=navigation;seed={int(scene.get("seed", 1))};policy={policy};speed=1.5;distance={distance:.9g};'
        'goal=' + ",".join(_number(v) for v in goal) + f';max_steps={int(max_steps)};motor_sampling=average;goal_objective=entry' + '" }',
        text,
        count=1,
    )
    if "# WEBOTS_OBSTACLES_INSERTION_POINT" in text:
        raise ValueError("failed to insert Metal obstacles")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text)
    return {
        "scene_name": scene["scene_name"],
        "scene_metadata": str(scene_path.resolve()),
        "world_path": str(output_path.resolve()),
        "start_xyz_m": [float(v) for v in start],
        "goal_xyz_m": [float(v) for v in goal],
        "room_bounds_m": scene["room_bounds_m"],
        "obstacle_count": len(obstacles),
        "actor_input": "deployed navigation policy; no scene metadata or witness route is passed",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path, help="selected webots-metal-scene-v1 record")
    parser.add_argument("output", type=Path)
    parser.add_argument("--policy", default="../assets/navigation-rooms-experimental.bin")
    parser.add_argument("--max-steps", type=int, default=2000)
    args = parser.parse_args()
    print(json.dumps(export_scene(args.scene, args.output, args.policy, args.max_steps), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
