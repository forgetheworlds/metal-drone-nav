"""Render declared challenge geometry; these images do not depict flown paths."""
import argparse
import json
from pathlib import Path
import struct

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from navigation_distance_tasks import read_bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bank", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    _, entries = read_bank(args.bank)
    labels = json.loads(args.manifest.read_text())["records"]
    first = {}
    for i, label in enumerate(labels):
        first.setdefault(label["kind"], i)
    fig = plt.figure(figsize=(15, 9))
    for panel, (kind, index) in enumerate(first.items(), 1):
        ax = fig.add_subplot(2, 3, panel, projection="3d")
        entry = entries[index]
        count = struct.unpack_from("<I", entry, 640)[0]
        for obstacle in range(count):
            shape, *values = struct.unpack_from("<I9f", entry, obstacle * 40)
            center, size, velocity = values[:3], values[3:6], values[6:]
            moving = any(velocity)
            center = [center[j] + 5 * velocity[j] for j in range(3)]
            color = "#ef4444" if moving else "#64748b"
            if shape == 0:
                ax.bar3d(*[center[j] - size[j] for j in range(3)], *[2 * s for s in size],
                         color=color, alpha=.13 if obstacle < 2 else .5, shade=True)
            elif shape == 1:
                u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 16), np.linspace(0, np.pi, 12))
                ax.plot_surface(center[0] + size[0] * np.cos(u) * np.sin(v),
                                center[1] + size[0] * np.sin(u) * np.sin(v),
                                center[2] + size[0] * np.cos(v), color=color, alpha=.8)
            else:
                angle, height = np.meshgrid(np.linspace(0, 2 * np.pi, 20),
                                           [center[2] - size[2], center[2] + size[2]])
                ax.plot_surface(center[0] + size[0] * np.cos(angle),
                                center[1] + size[0] * np.sin(angle), height, color=color, alpha=.65)
        start, goal = struct.unpack_from("<3f", entry, 676), struct.unpack_from("<3f", entry, 700)
        ax.scatter(*start, c="#2563eb", s=35, label="Start")
        ax.scatter(*goal, c="#16a34a", s=45, marker="*", label="Goal")
        ax.set(xlim=(-1, 12), ylim=(-2, 2), zlim=(0, 5), title=kind.replace("_", " "),
               xlabel="x (m)", ylabel="y (m)", zlabel="z (m)")
        ax.set_box_aspect((13, 4, 5));ax.view_init(elev=25, azim=-60)
        ax.set_yticks([-1.5, 0, 1.5]);ax.set_zticks([0, 2.5, 5])
    fig.suptitle("Source challenge geometry — declared primitives at t = 5 s", fontsize=16)
    fig.text(.02, .02, "Blue: initial position. Green: goal. Red: moving obstacle. Geometry render; no flown path is shown.", fontsize=11)
    fig.tight_layout(rect=(0, .05, 1, .94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140)


if __name__ == "__main__":
    main()
