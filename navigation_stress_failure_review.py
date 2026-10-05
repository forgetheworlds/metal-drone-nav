"""Recompute paired stress failure evidence from immutable flight/trace records."""
import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import tarfile


def review(archive):
    with tarfile.open(archive) as package:
        hashes = json.load(package.extractfile('SHA256.json'))
        content = {}
        for name, digest in hashes.items():
            data = package.extractfile(name).read()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f'Input hash differs: {name}')
            content[name] = data

    def rows(name):
        return list(csv.DictReader(io.StringIO(content[name].decode())))

    def flights(name):
        records = rows(name)
        if len(records) != 128 or len({row['env'] for row in records}) != 128:
            raise ValueError(f'Incomplete flight panel: {name}')
        for row in records:
            if sum(int(row[key]) for key in ('success', 'collision', 'timeout')) != 1:
                raise ValueError(f'Invalid terminal grade: {name}')
            if not all(math.isfinite(float(row[key])) for key in ('time_s', 'path_m', 'final_distance_m')):
                raise ValueError(f'Non-finite flight: {name}')
        return {row['env']: row for row in records}

    result = {'input_files': len(hashes), 'lost_success_contacts': {}, 'age_correction': {}}
    for profile in ('both-delay', 'combined'):
        cases = []
        corrected_success = original_success = wins = losses = 0
        for seed in (1, 2):
            nominal = flights(f'traces/T-s{seed}-nominal.csv')
            stressed = flights(f'traces/T-s{seed}-{profile}.csv')
            lost = {env for env in stressed if stressed[env]['collision'] == '1'
                    and nominal[env]['success'] == '1'}
            trajectories = {env: [] for env in sorted(lost, key=int)}
            for row in rows(f'traces/T-s{seed}-{profile}-trace.csv'):
                if row['env'] in trajectories:
                    trajectories[row['env']].append(row)
            for env, trajectory in trajectories.items():
                end = float(trajectory[-1]['time_s'])
                final_window = [row for row in trajectory if float(row['time_s']) >= end - .4]
                def speed(row, prefix):
                    return math.sqrt(sum(float(row[prefix + axis]) ** 2 for axis in 'xyz'))
                cases.append({
                    'seed': seed, 'env': int(env), 'family': stressed[env]['family_name'],
                    'min_depth_m': min(float(row['min_depth_m']) for row in final_window),
                    'mean_command_gap_mps': statistics.mean(float(row['command_gap_mps']) for row in final_window),
                    'latest_requested_slower': speed(trajectory[-1], 'requested_v') <
                                              speed(trajectory[-1], 'applied_v') - .1,
                })
            corrected = flights(f'age/T-s{seed}-composite-{profile}.csv')
            for env, actual in corrected.items():
                reference = stressed[env]
                if any(actual[key] != reference[key] for key in ('scene_seed', 'goal_x', 'goal_y', 'goal_z')):
                    raise ValueError('Paired geometry differs')
                current, previous = int(actual['success']), int(reference['success'])
                corrected_success += current
                original_success += previous
                wins += current > previous
                losses += previous > current
        result['lost_success_contacts'][profile] = {
            'count': len(cases),
            'geometry_under_1m_last_0_4s': sum(row['min_depth_m'] < 1 for row in cases),
            'latest_requested_slower': sum(row['latest_requested_slower'] for row in cases),
            'mean_command_gap_mps': statistics.mean(row['mean_command_gap_mps'] for row in cases),
            'cases': cases,
        }
        result['age_correction'][profile] = {
            'original_success': original_success, 'corrected_success': corrected_success,
            'paired_wins': wins, 'paired_losses': losses,
        }
    for seed in (1, 2):
        for panel in ('composite', 'open', 'long-hall'):
            current = flights(f'age/T-s{seed}-{panel}-nominal.csv')
            reference = flights(f'baseline/T-s{seed}-{panel}-nominal.csv')
            if current != reference:
                raise ValueError('Nominal flight parity failed')
    result['nominal_parity_panels'] = 6
    result['decision'] = 'Not adopted: delay outcomes worsen despite combined-stress gains.'
    result['scope'] = 'Source development replay. Nearby measured geometry is not contact-object attribution or proof of sufficient warning.'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive')
    args = parser.parse_args()
    print(json.dumps(review(args.archive), indent=2))
