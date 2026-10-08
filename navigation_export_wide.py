"""Export a frozen 184/H/4 PPO actor without its critic or optimizer state."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def export(checkpoint, destination):
    data = checkpoint.read_bytes()
    if len(data) < 136 or data[:8] != b"PPOFIX1\0":
        raise ValueError("Expected a PPOFIX1 checkpoint")
    version, actor_count, critic_count = struct.unpack_from("<3I", data, 8)
    width, remainder = divmod(actor_count - 8, 189)
    if version not in [6, 8, 9, 10] or remainder or width not in [64, 256, 768, 2560, 5120]:
        raise ValueError("Unsupported guided 184/H/4 actor or checkpoint version")
    mode = struct.unpack_from("<I", data, 56)[0]
    substeps, sensor_period = struct.unpack_from("<2I", data, 64)
    speed = struct.unpack_from("<f", data, 92)[0]
    velocity_contract, geometry_memory = struct.unpack_from("<2I", data, 124)
    if mode not in [17, 22] or substeps != 5 or sensor_period != 1:
        raise ValueError("Unsupported action map or control cadence")
    if velocity_contract != 1 or geometry_memory != 1 or not math.isfinite(speed) or not 0 < speed <= 10:
        raise ValueError("Unsupported body-velocity or geometry-memory contract")
    if len(data) < 136 + 4 * (actor_count + critic_count):
        raise ValueError("Truncated actor/critic parameter payload")
    weights = data[136:136 + actor_count * 4]
    if any(not math.isfinite(x[0]) for x in struct.iter_unpack("<f", weights)):
        raise ValueError("Non-finite actor parameter")
    profile = 2 if version == 10 else 1
    sidecar = Path(str(checkpoint) + ".train.json")
    if sidecar.exists():
        contract = json.loads(sidecar.read_text())
        if contract.get("actor_obs_dim") != 184 or contract.get("actor_hidden_dim") != width:
            raise ValueError("Source architecture sidecar differs")
        if contract.get("sensor_profile") != ("native" if profile == 2 else "legacy"):
            raise ValueError("Source camera sidecar differs")
    metadata = struct.pack("<6If2I3f2I", 4, 184, width, 4, 17, actor_count,
                           speed, 16, 20, 12.0, .05, .01, 8, 0x00554C46)
    camera = struct.pack("<I6fI", profile, 1.0, .8 if profile == 2 else .75,
                         .08 if profile == 2 else 0, .03, 12.0, 0, 0)
    source_hash = 14695981039346656037
    for byte in data:
        source_hash = ((source_hash ^ byte) * 1099511628211) & ((1 << 64) - 1)
    payload = b"NAVWID4\0" + metadata + camera + weights + b"NAVSRC4\0" + struct.pack("<Q", source_hash)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    record = {
        "schema": "wide-guided-nav-v4", "source": str(checkpoint),
        "source_sha256": hashlib.sha256(data).hexdigest(),
        "source_fnv1a64": source_hash, "source_version": version,
        "actor_inputs": 184, "actor_hidden": width, "actor_parameters": actor_count,
        "source_critic_parameters": critic_count, "serialized_critic_parameters": 0,
        "sensor_profile": profile, "max_speed_mps": speed,
        "weights_sha256": hashlib.sha256(weights).hexdigest(),
        "nav_sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
        "scope": "Frozen deterministic mode17 actor export; not independent flight proof.",
    }
    Path(str(destination) + ".json").write_text(json.dumps(record, indent=2) + "\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.checkpoint, args.output), indent=2))
