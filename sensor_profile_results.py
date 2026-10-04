#!/usr/bin/env python3
"""Plot frozen-policy Metal camera sensitivity from the published DEV records."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "evidence/inputs/sensor-profile"

def read_outcomes(name):
    with (DATA / name).open() as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 90 and all(row["split"] == "dev" for row in rows)
    groups = []
    for family in (14, 15, 16):
        cases = [row for row in rows if int(row["family"]) == family]
        assert len(cases) == 30
        assert all(sum(int(row[key]) for key in ("success", "collision", "timeout")) == 1 for row in cases)
        groups.append(sum(int(row["success"]) for row in cases))
    return rows, groups

def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    legacy_rows, legacy = read_outcomes("legacy-dev.csv")
    native_rows, native = read_outcomes("native-dev.csv")
    assert [row["failure_id"] for row in legacy_rows] == [row["failure_id"] for row in native_rows]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for offset, values, color, label in ((-.18, legacy, "#2563eb", "Legacy source camera"),
                                         (.18, native, "#f59e0b", "Calibrated camera in Metal")):
        bars = axis.bar([index + offset for index in range(3)], [100 * n / 30 for n in values],
                        width=.34, color=color, label=label)
        axis.bar_label(bars, labels=[f"{n}/30" for n in values], padding=4)
    axis.set_xticks(range(3), ["Bent hallways", "Connected rooms", "Vertical choices"])
    axis.set_ylim(0, 112)
    axis.set_yticks(range(0, 101, 20))
    axis.set_ylabel("Successful episodes (%)")
    axis.set_title("Camera sensitivity: frozen weights, 90 development tasks")
    axis.legend(loc="upper left")
    axis.spines[["top", "right"]].set_visible(False)
    figure.text(.5, .02, "Both runs use Metal physics. This is not an independent transfer benchmark.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .04, 1, 1))
    output = ROOT / "artifacts/sensor-profile-dev.png"
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(f"Legacy {sum(legacy)}/90; calibrated {sum(native)}/90; plot {output}")

if __name__ == "__main__":
    main()
