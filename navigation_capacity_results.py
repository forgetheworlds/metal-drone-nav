"""Review raw capacity flights and plot the fixed-recipe comparison."""
import argparse
import json
from pathlib import Path
import struct
import hashlib
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from navigation_scaling_results import read_grades, pool_grades

WIDTHS = [64, 256, 768, 2560, 5120]
NOMINAL = ['long-open', 'long-hallway', 'dev-a', 'dev-b', 'dev-c', 'open', 'clutter']
STRESS = ['nominal', 'sensor-delay', 'command-delay', 'both-delay', 'combined']
FRESH = ['fresh-static', 'fresh-clutter', 'fresh-open', 'fresh-long-open', 'fresh-long-hallway', 'fresh-course', 'fresh-course-reflected']


def checkpoint_header(path):
    if not path.exists():
        path = Path(str(path) + '.header')
    with path.open('rb') as file:
        data = file.read(136)
    if len(data) != 136 or data[:7] != b'PPOFIX1':
        raise ValueError('Invalid saved checkpoint header')
    return data


def review(folder):
    jobs = json.loads((folder / 'jobs.json').read_text())
    if {job['arm'] for job in jobs} != {f'w{width}-s{seed}' for width in WIDTHS for seed in [1, 2]}:
        raise ValueError('Incomplete producer matrix')
    for job in jobs:
        header = checkpoint_header(folder / 'runs' / (job['arm'] + '.bin'))
        if job['exit'] != 0 or struct.unpack_from('<4I', header, 12) != (189 * job['width'] + 8, 4225, 32, 512):
            raise ValueError('Producer architecture mismatch')
        if struct.unpack_from('<I', header, 36)[0] != 2560 or struct.unpack_from('<Q', header, 40)[0] != 327680:
            raise ValueError('Producer did not finish matched budget')
    receipts = json.loads((folder / 'eval-receipts.json').read_text())
    batches = {}
    for receipt in receipts:
        key = (receipt['arm'], receipt['prefix'], receipt['panel'], receipt['profile'])
        if key in batches or receipt['exit'] != 0:
            raise ValueError('Duplicate or failed evaluation')
        rows = read_grades(folder / 'evals' / Path(receipt['csv']).name)
        if {int(row['env']) for row in rows} != set(range(128)):
            raise ValueError('Missing task IDs')
        batches[key] = rows
    expected = set()
    curves, fresh = {}, {}
    for width in WIDTHS:
        curves[str(width)], fresh[str(width)] = {}, {}
        for prefix in range(256, 2561, 256):
            panels = [(panel, 'nominal') for panel in NOMINAL] + [('composite', profile) for profile in STRESS]
            if prefix in [512, 1280, 2560]:
                panels += [(panel, profile) for panel in FRESH for profile in (['nominal', 'both-delay', 'combined'] if panel.startswith('fresh-course') else ['nominal'])]
            for panel, profile in panels:
                keys = [(f'w{width}-s{seed}', prefix, panel, profile) for seed in [1, 2]]
                expected.update(keys)
                if any(key not in batches for key in keys):
                    raise ValueError('Missing evaluation cell')
                pooled = pool_grades([batches[key] for key in keys])
                destination = fresh if panel.startswith('fresh-') else curves
                destination[str(width)].setdefault(str(prefix * 512 * 32), {})[panel + '/' + profile] = pooled
    if set(batches) != expected or len(expected) != 1530:
        raise ValueError('Evaluation matrix differs from declared cells')
    return {'scope': 'Fixed-recipe actor capacity; two seeds. Source development, not independent FINAL. Final endpoints primary.',
            'training_transitions': 419430400, 'evaluations': len(batches), 'flights': len(batches) * 128,
            'curves': curves, 'fresh': fresh, 'jobs': jobs}


def plot(report, output):
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    panels = [('open/nominal', 'Short open'), ('dev-c/nominal', 'Static C'), ('clutter/nominal', 'Clutter'),
              ('long-hallway/nominal', 'Long hallway'), ('composite/nominal', 'Course'), ('composite/combined', 'Combined stress')]
    labels = ['12k', '50k', '150k', '500k', '1M']
    for axis, (panel, title) in zip(axes.flat, panels):
        for width, label in zip(WIDTHS, labels):
            curve = report['curves'][str(width)]
            x = [int(samples) / 1e6 for samples in curve]
            y = [curve[samples][panel]['success'] / 256 * 100 for samples in curve]
            axis.plot(x, y, label=label, linewidth=1.7)
        axis.set(title=title, xlabel='Training transitions (millions)', ylabel='Stable success (%)', ylim=(0, 100))
        axis.grid(alpha=.2)
        axis.legend(frameon=False, fontsize=8, ncol=2)
    figure.suptitle('Actor capacity under the same PPO recipe: two-seed navigation outcomes')
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def update_drift(folder):
    observations = np.fromfile(folder / 'gates/real-observations.f32', dtype='<f4').reshape(256, 184).astype('float64')

    def load_actor(path):
        data = path.read_bytes()
        count = struct.unpack_from('<I', data, 12)[0]
        return np.frombuffer(data, dtype='<f4', count=count, offset=136).astype('float64')

    def means(actor, width):
        bias = width * 184
        output = bias + width
        hidden = np.tanh(observations @ actor[:bias].reshape(width, 184).T + actor[bias:output])
        prior = np.column_stack((observations[:, -3:], np.zeros(256)))
        return hidden @ actor[output:output + 4 * width].reshape(4, width).T + actor[-8:-4] + prior

    records = []
    for width in WIDTHS:
        warm = folder / 'preflight' / f'w{width}.bin'
        smoke = folder / 'gates' / f'smoke-w{width}.bin'
        initial, trained = load_actor(warm), load_actor(smoke)
        initial[-4:] = -1  # The matched training recipe resets action std.
        before, after = means(initial, width), means(trained, width)
        old_std, new_std = np.exp(initial[-4:]), np.exp(trained[-4:])
        kl = np.sum(np.log(new_std / old_std) + (old_std ** 2 + (before - after) ** 2) / (2 * new_std ** 2) - .5, axis=1)
        records.append({'width': width, 'optimizer_steps': struct.unpack_from('<Q', smoke.read_bytes(), 40)[0],
                        'probe_rows': 256, 'raw_mean_rms_change': float(np.sqrt(np.mean((after - before) ** 2))),
                        'old_to_new_gaussian_kl_mean': float(np.mean(kl)), 'kl_max': float(np.max(kl)),
                        'warm_sha256': hashlib.sha256(warm.read_bytes()).hexdigest(),
                        'smoke_sha256': hashlib.sha256(smoke.read_bytes()).hexdigest()})
    return {'scope': 'Post-hoc fixed256 real source-observation probe after one source rollout/128Adam updates, one seed. KL is latent Gaussian policy shift, not closed-loop improvement or proof of failure cause. Same warm means, reset logstd-1.',
            'records': records}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--out', type=Path, default=Path('artifacts/plots/actor-capacity.png'))
    parser.add_argument('--drift', action='store_true')
    args = parser.parse_args()
    result = review(args.folder)
    (args.folder / 'root-capacity-review.json').write_text(json.dumps(result, indent=2) + '\n')
    plot(result, args.out)
    if args.drift:
        (args.folder / 'root-first-update-drift.json').write_text(json.dumps(update_drift(args.folder), indent=2) + '\n')
    print('Verified ten saved budgets, 1530 panels and 195840 raw flight grades')
