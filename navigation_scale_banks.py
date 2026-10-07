"""Tile a frozen TRAIN bank to scale parallelism without changing its task distribution."""
import argparse
from pathlib import Path
import json
import hashlib
from navigation_distance_tasks import read_bank, write_bank


def tile(source, environments, output, phase_balance=False):
    period, records = read_bank(source)
    original = len(records) // period
    if environments % original or environments < original:
        raise ValueError('Environment count must be an integer multiple of the original count')
    scaled = []
    for index in range(environments):
        source_env = index % original
        block = records[source_env*period:(source_env+1)*period]
        # Equal total samples at large N means fewer episodes per environment.
        # Spread initial phases so the hardware comparison is not first-slot only.
        offset = (source_env*17 + (index//original)*13) % period if phase_balance else 0
        scaled.extend(block[offset:] + block[:offset])
    write_bank(output, period, scaled)
    if read_bank(output) != (period, scaled):
        raise ValueError('Scaled bank wire parity failed')
    manifest = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'scaled_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                'original_environments': original, 'environments': environments,
                'period': period, 'unique_records': len(set(records)),
                'tiling_changes_diversity': False, 'phase_balanced': phase_balance,
                'initial_phase_rule': '(source_env*17 + tile*13) % period' if phase_balance else 'legacy',
                'scope': 'Parallelism axis only; more unique worlds is a separate experiment.'}
    output.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--envs', type=int, required=True)
    parser.add_argument('--phase-balance', action='store_true')
    args = parser.parse_args();print(json.dumps(tile(args.source,args.envs,args.output,args.phase_balance)))
