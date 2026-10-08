"""Independently check the frozen window audit without starting a new learner."""
import argparse
import hashlib
import tarfile
import tempfile
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
FOLDER = ROOT / 'results/omp-erosion-contract-repair'
FIELDS = ['tick', 'env', 'reward', 'value', 'next_value', 'advantage_raw',
          'advantage_normalized', 'terminated', 'truncated', 'clearance_m']


def episodes(data):
    """Completed bank tasks, paired by environment and deterministic slot ordinal."""
    result = {}
    steps, envs, _ = data.shape
    for env in range(envs):
        start = 0
        ordinal = 0
        for tick in range(steps):
            row = data[tick, env]
            if not (row[7] or row[8]):
                continue
            outcome = 'contact' if row[7] and row[2] < 0 else (
                'success' if row[7] else 'timeout')
            result[env, ordinal] = (outcome, float(data[start:tick + 1, env, 9].min()))
            ordinal += 1
            start = tick + 1
    return result


def check_run(receipt):
    path = Path(receipt['csv'])
    reduced = FOLDER / 'arrays' / (receipt['label'] + '.npz')
    if reduced.exists():
        record = json.loads((FOLDER / 'array-manifest.json').read_text())[receipt['label']]
        assert hashlib.sha256(reduced.read_bytes()).hexdigest() == record['sha256']
        with np.load(reduced, allow_pickle=False) as saved:
            assert list(saved['fields']) == FIELDS
            data = saved['data']
        digest = record['source_csv_sha256']
    else:
        if not path.is_absolute():
            path = ROOT / path
        with path.open() as file:
            columns = file.readline().strip().split(',')
        data = np.loadtxt(path, delimiter=',', skiprows=1,
                          usecols=[columns.index(name) for name in FIELDS], dtype=np.float64)
    steps = int(data[:, 0].max()) + 1
    envs = int(data[:, 1].max()) + 1
    assert len(data) == steps * envs
    np.testing.assert_array_equal(data[:, 0], np.repeat(np.arange(steps), envs))
    np.testing.assert_array_equal(data[:, 1], np.tile(np.arange(envs), steps))
    data = data.reshape(steps, envs, len(FIELDS))
    worst_raw = worst_normalized = 0.0
    for start in range(0, steps, 32):
        chunk = data[start:start + 32]
        carry = np.zeros(envs)
        replay = np.empty((32, envs))
        for tick in range(31, -1, -1):
            row = chunk[tick]
            term, trunc = row[:, 7] != 0, row[:, 8] != 0
            delta = row[:, 2] + .99 * np.where(term, 0.0, row[:, 4]) - row[:, 3]
            carry = delta + .99 * .95 * np.where(term | trunc, 0.0, carry)
            replay[tick] = carry
        raw = chunk[:, :, 5]
        # Double precision replay of decimal CSV, rather than the agent's float
        # implementation. Small float accumulation and text-rounding error is expected.
        worst_raw = max(worst_raw, float(np.max(np.abs(replay - raw))))
        norm = (raw - raw.mean()) / np.sqrt(np.mean((raw - raw.mean()) ** 2) + 1e-8)
        worst_normalized = max(worst_normalized, float(np.max(np.abs(norm - chunk[:, :, 6]))))
    assert worst_raw < 1e-4, worst_raw
    assert worst_normalized < 1e-3, worst_normalized
    tasks = episodes(data)
    if not reduced.exists():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return tasks, {'rows': len(data) * envs, 'windows': steps // 32,
                   'completed_tasks': len(tasks), 'csv_sha256': digest,
                   'max_double_replay_gae_error': worst_raw,
                   'max_double_replay_normalization_error': worst_normalized}


def review():
    receipts = json.loads((FOLDER / 'window-receipts-gate.json').read_text())
    original = json.loads((FOLDER / 'dump-manifest.json').read_text())['gate_dumps']
    runs, tasks = {}, {}
    for receipt in receipts:
        assert receipt['exit'] == 0
        label = receipt['label']
        tasks[label], runs[label] = check_run(receipt)
        assert runs[label]['csv_sha256'] == original[label + '.csv']['sha256']
        print(label, runs[label], flush=True)
    pairs = {}
    for width in ['wide', 'narrow']:
        for seed in [1, 2]:
            early, full = tasks[f'{width}-early-s{seed}'], tasks[f'{width}-full-s{seed}']
            shared = early.keys() & full.keys()
            solved = [key for key in shared if early[key][0] == 'success']
            bands = {}
            for low, high in [(0, .05), (.05, .1), (.1, .2), (.2, .4), (.4, float('inf'))]:
                keys = [key for key in solved if low <= early[key][1] < high]
                bands[f'{low}-{high}'] = {'early_success': len(keys),
                    'lost': sum(full[key][0] != 'success' for key in keys)}
            pairs[f'{width}-s{seed}'] = {'shared_tasks': len(shared), 'bands': bands,
                'early_success': len(solved),
                'full_success': sum(full[key][0] == 'success' for key in shared)}
    review = {'runs': runs, 'pairs': pairs, 'decision': 'No new learner authorized by this diagnostic alone',
        'scope': [
            'Frozen sampled actors; no optimizer updates between consecutive windows.',
            'Bank payloads align by environment/episode; action-noise draws do not remain paired after different episode lengths.',
            'clearance_m is pre-action clearance of the 0.18 m collision sphere, not minimum clearance over all 100 Hz physics substeps.',
            'action_0..3 are sampled Gaussian latents, not BODY velocity commands after guidance, tanh, delay and RAPTOR.',
            'Margin association is replicated; it does not identify the cause of drift or justify a specific anchor strength.',
            'Ratios of mean advantages and selected completed-episode Monte Carlo errors do not prove route credit or rule out value errors.',
            'Parameter anchor targets the original warmstart, not the useful early trained policy; prior 0 versus .01 study already exists.'
        ]}
    (FOLDER / 'root-review.json').write_text(json.dumps(review, indent=2) + '\n')
    print('ROOT_WINDOW_REVIEW_PASS', len(runs), 'runs', flush=True)
    return review


def main():
    global FOLDER, ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path, help='Audit folder or retained records.tar.gz')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--plot', type=Path)
    args = parser.parse_args()
    if args.inputs.is_dir():
        FOLDER = args.inputs.resolve()
        result = review()
    else:
        with tempfile.TemporaryDirectory() as directory:
            FOLDER = Path(directory).resolve()
            with tarfile.open(args.inputs, 'r:gz') as archive:
                for member in archive.getmembers():
                    destination = (FOLDER / member.name).resolve()
                    if not destination.is_relative_to(FOLDER) or not (member.isfile() or member.isdir()):
                        raise ValueError('Unsafe archive member: ' + member.name)
                archive.extractall(FOLDER)
            result = review()
            if args.output:
                args.output.write_bytes((FOLDER / 'root-review.json').read_bytes())
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        labels, close, roomy = [], [], []
        for name, pair in result['pairs'].items():
            bands = pair['bands']
            low = [bands['0-0.05'], bands['0.05-0.1']]
            close.append(100 * sum(x['lost'] for x in low) / sum(x['early_success'] for x in low))
            high = bands['0.4-inf']
            roomy.append(100 * high['lost'] / high['early_success'])
            labels.append(name)
        positions = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(positions - .18, close, .36, label='Early sampled clearance below 0.10 m')
        ax.bar(positions + .18, roomy, .36, label='Early sampled clearance at least 0.40 m')
        ax.set_xticks(positions, labels)
        ax.set_ylabel('Previously successful bank tasks lost (%)')
        ax.set_title('Continued PPO loses more of the close-clearance solutions')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 60)
        fig.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=160)
        plt.close(fig)


if __name__ == '__main__':
    main()
