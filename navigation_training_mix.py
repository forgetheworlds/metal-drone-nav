"""Reserve training transitions by environment group, not by episode duration."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from navigation_distance_tasks import read_bank, write_bank


def build(static_path, long_path, challenge_path, output):
    static_period, static_records = read_bank(static_path)
    long_period, old_mix = read_bank(long_path)
    challenge_period, challenges = read_bank(challenge_path)
    assert static_period == 8 and long_period == 12 and challenge_period == 6
    assert len(static_records) == 1024 and len(old_mix) == 1536 and len(challenges) == 768
    long_records = [row for env in range(128) for row in old_mix[env * 12 + 8:env * 12 + 12]]
    period = math.lcm(16, 16, 24)
    output.mkdir(parents=True, exist_ok=True)
    banks, groups = {}, []
    for arm in ["control", "combined"]:
        records = []
        class_index = {"static": 0, "long": 0, "challenge": 0}
        # Interleave groups within every minibatch. Neither group nor index is an actor input.
        for env in range(128):
            lane = env % 4
            kind = "static" if lane < 2 else "long" if lane == 2 else "challenge"
            if kind == "static":
                index = class_index[kind];class_index[kind] += 1
                choices = static_records[index * 16:(index + 1) * 16]
            elif kind == "long":
                index = class_index[kind];class_index[kind] += 1
                choices = long_records[index * 16:(index + 1) * 16]
            else:
                index = class_index[kind];class_index[kind] += 1
                choices = (challenges[index * 24:(index + 1) * 24] if arm == "combined"
                           else long_records[index * 16:(index + 1) * 16])
                if arm == "control":kind = "long"
            assert choices and period % len(choices) == 0
            records.extend(choices * (period // len(choices)))
            if arm == "combined":groups.append(kind)
        assert len(records) == 128 * period
        banks[arm] = records
        write_bank(output / (arm + ".bin"), period, records)
    for env in range(128):
        if env % 4 < 3:
            assert banks["control"][env * period:(env + 1) * period] == banks["combined"][env * period:(env + 1) * period]
    assert set(static_records).issubset(set(banks["combined"]))
    assert set(long_records).issubset(set(banks["combined"]))
    assert set(challenges).issubset(set(banks["combined"]))
    manifest = {"period": period, "records_per_arm": 128 * period,
                "control_transition_share": {"static": .5, "long": .5},
                "combined_transition_share": {"static": .5, "long": .25, "challenge": .25},
                "combined_group_per_env": groups,
                "unique_bank_entries": {arm: len(set(rows)) for arm, rows in banks.items()},
                "static_long_lanes_byte_identical": True,
                "all_parent_and_challenge_entries_retained": True,
                "source_files": {str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                 for path in [static_path, long_path, challenge_path]}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("static", type=Path)
    parser.add_argument("long_mix", type=Path)
    parser.add_argument("challenge", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.static, args.long_mix, args.challenge, args.output), indent=2))
