"""Audit complete matched delay learning, paired flights and frozen acceptance gates."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import struct

PANELS = ('long-open', 'long-hallway', 'dev-a', 'dev-b', 'dev-c', 'open', 'clutter', 'composite')
PROFILES = ('nominal', 'sensor-delay', 'command-delay', 'both-delay', 'combined')


def read_flights(path):
    with path.open() as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 128 or len({row['env'] for row in rows}) != 128:
        raise ValueError(f'Incomplete or repeated flight rows: {path}')
    for row in rows:
        if sum(int(row[key]) for key in ('success', 'collision', 'timeout')) != 1:
            raise ValueError(f'Nonexclusive outcome: {path}')
        if not all(math.isfinite(float(row[key])) for key in ('time_s', 'path_m', 'final_distance_m', 'final_speed_mps')):
            raise ValueError(f'Nonfinite flight: {path}')
        if int(row['success']) and not (float(row['stable_hold_s']) >= .199 and
                                       float(row['final_distance_m']) <= .35001 and
                                       float(row['final_speed_mps']) <= .50001):
            raise ValueError(f'Incomplete stable arrival: {path}')
    return {row['env']: row for row in rows}


def summarize(rows):
    return {key: sum(int(row[column]) for row in rows)
            for key, column in [('success', 'success'), ('contacts', 'collision'), ('timeouts', 'timeout')]}


def review(folder):
    jobs = json.loads((folder / 'provenance/jobs.json').read_text())
    if len(jobs) != 4 or any(row.get('exit') != 0 for row in jobs):
        raise ValueError('Four completed producer receipts required')
    receipt_path = folder / 'provenance/eval-receipts.json'
    if not receipt_path.exists():
        raise ValueError('All 48 completed evaluator receipts required')
    receipts = json.loads(receipt_path.read_text())
    if len(receipts) != 48 or any(row.get('exit') != 0 for row in receipts):
        raise ValueError('All 48 completed evaluator receipts required')
    exposure = {}
    for job in jobs:
        checkpoint = folder / 'runs' / (job['arm'] + '.bin')
        data = checkpoint.read_bytes()
        if struct.unpack_from('<I', data, 36)[0] != 10000 or struct.unpack_from('<Q', data, 40)[0] != 320000:
            raise ValueError(f'Incomplete saved header: {checkpoint}')
        with Path(str(checkpoint) + '.delay-exposure.csv').open() as file:
            rows = list(csv.DictReader(file))
        # Uncheckpointed observations may survive an interrupted segment.
        # Last occurrence belongs to the resumed continuation; retain duplicates
        # in the raw journal rather than adding abandoned work to the budget.
        by_rollout = {int(row['rollout']): row for row in rows}
        if set(by_rollout) != set(range(1, 10001)):
            raise ValueError(f'Incomplete exposure journal: {checkpoint}')
        stale = sum(int(row['positive_age_rows']) for row in by_rollout.values())
        if ('-C-' in job['arm'] and stale != 0) or ('-T-' in job['arm'] and stale == 0):
            raise ValueError(f'Delay exposure arm mismatch: {checkpoint}')
        exposure[job['arm']] = {'unique_rollouts': len(by_rollout), 'raw_rows': len(rows),
                                'positive_age_rows': stale, 'assigned_rows_per_stratum': 20480000}
    results = {}
    for panel in PANELS:
        for profile in PROFILES if panel == 'composite' else ('nominal',):
            paired, summaries, individual = {}, {}, {}
            for arm in ('C', 'T'):
                batches = []
                individual[arm] = []
                paired[arm] = {}
                for seed in (1, 2):
                    path = folder / 'evals' / f'delay-{arm}-s{seed}-{panel}-{profile}.csv'
                    batch = read_flights(path)
                    batches.extend(batch.values())
                    individual[arm].append(summarize(batch.values()))
                    paired[arm].update({(seed, env): row for env, row in batch.items()})
                summaries[arm] = summarize(batches)
            wins = losses = 0
            delays = []
            for key, control in paired['C'].items():
                treatment = paired['T'][key]
                identity = ('scene_seed', 'start_x', 'start_y', 'start_z', 'start_yaw',
                            'goal_x', 'goal_y', 'goal_z', 'sensor_delay', 'command_delay')
                if any(control[field] != treatment[field] for field in identity):
                    raise ValueError(f'Paired task/timing differs: {panel}/{profile}/{key}')
                previous, current = int(control['success']), int(treatment['success'])
                wins += current > previous
                losses += previous > current
                if current and previous:
                    delays.append(float(treatment['time_s']) - float(control['time_s']))
            for seed in (1, 2):
                a = folder / 'evals' / f'delay-C-s{seed}-{panel}-{profile}.csv.physics.bin'
                b = folder / 'evals' / f'delay-T-s{seed}-{panel}-{profile}.csv.physics.bin'
                if a.read_bytes() != b.read_bytes():
                    raise ValueError(f'Paired plant differs: {panel}/{profile}/seed{seed}')
            results[f'{panel}/{profile}'] = {**summaries, 'individual_seeds': individual,
                                           'wins': wins, 'losses': losses,
                                           'common_success_delay_s': math.fsum(delays)/len(delays) if delays else None}
    gates = {}
    primary = results['composite/both-delay']
    gates['delay_success_gain'] = primary['T']['success'] >= primary['C']['success'] + 10
    gates['delay_contact_reduction'] = primary['T']['contacts'] <= primary['C']['contacts'] - 10
    gates['delay_each_seed_success_gain'] = all(a['success'] > b['success'] for a, b in
                                                zip(primary['individual_seeds']['T'], primary['individual_seeds']['C']))
    for key, result in results.items():
        if key.endswith('/nominal'):
            gates[key + ':success_retention'] = result['T']['success'] >= result['C']['success'] - 3
            for metric in ('contacts', 'timeouts'):
                gates[key + ':' + metric] = result['T'][metric] <= result['C'][metric] + 3
        gates[key + ':speed'] = result['common_success_delay_s'] is not None and result['common_success_delay_s'] <= .5
    combined = results['composite/combined']
    gates['combined_success_retention'] = combined['T']['success'] >= combined['C']['success']
    for metric in ('contacts', 'timeouts'):
        gates['combined_' + metric] = combined['T'][metric] <= combined['C'][metric] + 3
    for panel, floor in [('open', 253), ('long-open', 253), ('long-hallway', 253), ('dev-b', 223), ('clutter', 219)]:
        gates[panel + ':absolute_floor'] = results[panel + '/nominal']['T']['success'] >= floor
    return {'results': results, 'exposure': exposure, 'gates': gates,
            'failed_gates': [name for name, passed in gates.items() if not passed],
            'decision': 'passes declared source gates' if all(gates.values()) else 'not adopted',
            'scope': 'Exposed source development; passing these gates does not prove full goal or independent transfer.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    print(json.dumps(review(args.folder), indent=2))
