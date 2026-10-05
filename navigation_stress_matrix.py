"""Frozen-policy diagnostics; no training or checkpoint selection.

Run `preflight` before `matrix`. All Metal commands use the shared root lock.
A failed receipt stops execution. Raw sampled plants remain beside flight CSVs.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import struct
import subprocess

ROOT = Path(__file__).resolve().parent
PROFILES = ('nominal', 'depth-noise', 'dropout', 'sensor-delay',
            'command-delay', 'both-delay', 'dynamics', 'combined')
PREFLIGHT_PROFILES = ('nominal', 'depth-noise', 'dynamics', 'combined')
PANELS = {
    'open': ROOT / 'results/omp-local-dynamics/banks/open-static.bin',
    'long-hall': ROOT / 'results/root-distance-learning/dev/hallway.bin',
    'composite': ROOT / 'results/root-motion-fidelity/dev/challenges.bin',
}
BINARY = ROOT / 'build/navigation_stress_eval'
BC = ROOT / 'results/omp-local-capability/runs/bc.bin'
PPO = ROOT / 'results/root-consolidated-ppo'


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def check_freeze(folder):
    for name, digest in json.loads((folder / 'freeze.json').read_text()).items():
        if file_hash(Path(name)) != digest:
            raise RuntimeError(f'Frozen input changed: {name}')


def save_receipts(path, receipts):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(receipts, indent=2) + '\n')
    temporary.replace(path)


def evaluate(checkpoint, bank, output, profile):
    if output.exists():
        raise RuntimeError(f'Preserve existing result before rerun: {output}')
    checkpoint_hash = file_hash(checkpoint)
    argv = ['python3', str(ROOT / 'run_locked.py'), '--', str(BINARY),
            str(checkpoint), str(bank), str(output), profile]
    with output.with_suffix('.log').open('w') as log:
        code = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT).returncode
    if code:
        raise RuntimeError(f'Stress flight failed ({code}): {output}')
    if file_hash(checkpoint) != checkpoint_hash:
        raise RuntimeError(f'Actor changed during evaluation: {checkpoint}')
    rows = read_rows(output)
    if len(rows) != 128:
        raise RuntimeError(f'Expected 128 completed tasks: {output}')
    for row in rows:
        if sum(int(row[key]) for key in ('success', 'collision', 'timeout')) != 1:
            raise RuntimeError(f'Incomplete or ambiguous task: {row}')
    return {
        'profile': profile, 'argv': argv, 'exit': code, 'tasks': len(rows),
        'checkpoint_sha256': checkpoint_hash, 'bank_sha256': file_hash(bank),
        'csv_sha256': file_hash(output),
        'physics_sha256': file_hash(output.with_name(output.name + '.physics.bin')),
        'summary': {key: sum(int(row[column]) for row in rows)
                    for key, column in [('S', 'success'), ('C', 'collision'), ('T', 'timeout')]},
    }


def preflight(folder):
    check_freeze(folder)
    receipts = []
    for profile in PREFLIGHT_PROFILES:
        output = folder / f'preflight-{profile}.csv'
        receipt = evaluate(BC, PANELS['composite'], output, profile)
        if profile == 'nominal':
            rows = read_rows(output)
            baseline = read_rows(ROOT / 'results/root-motion-fidelity/bc-dev.csv')
            if len(baseline) != len(rows):
                raise RuntimeError('Nominal parity row count differs')
            for actual, reference in zip(rows, baseline):
                if {k: v for k, v in actual.items() if k != 'split'} != {
                        k: v for k, v in reference.items() if k != 'split'}:
                    raise RuntimeError('Nominal scored-flight parity failed')
        receipts.append(receipt)
        save_receipts(folder / 'preflight-receipts.json', receipts)
        print(profile, receipt['summary'], flush=True)
    print('STRESS_PREFLIGHT_COMPLETE', flush=True)


def matrix(folder):
    check_freeze(folder)
    checks = json.loads((folder / 'preflight-receipts.json').read_text())
    if len(checks) != len(PREFLIGHT_PROFILES) or {
            row['profile'] for row in checks} != set(PREFLIGHT_PROFILES):
        raise RuntimeError('All required stress preflights must pass first')
    for receipt in checks:
        if receipt['exit'] or receipt['tasks'] != 128:
            raise RuntimeError('Invalid preflight receipt')
        output = folder / f"preflight-{receipt['profile']}.csv"
        if file_hash(output) != receipt['csv_sha256']:
            raise RuntimeError('Preflight flight table changed')
    training = json.loads((PPO / 'provenance/jobs.json').read_text())
    evaluation = json.loads((PPO / 'provenance/eval-receipts.json').read_text())
    if len(training) != 4 or any(row.get('exit') != 0 or row['rollouts'] != 10000
                                or row['optimizer_step'] != 320000 for row in training):
        raise RuntimeError('Four matched PPO runs must finish first')
    if len(evaluation) != 64 or any(row.get('exit') != 0 for row in evaluation):
        raise RuntimeError('Matched PPO evaluation must finish first')
    actors = {'BC': BC}
    for seed in (1, 2):
        for arm in ('C', 'T'):
            actors[f'{arm}-s{seed}'] = PPO / 'runs' / f'consolidated-{arm}-s{seed}.bin'
    results = folder / 'evals'
    results.mkdir(exist_ok=True)
    receipts = []
    for label, checkpoint in actors.items():
        if label != 'BC' and struct.unpack_from('<I', checkpoint.read_bytes(), 36)[0] != 10000:
            raise RuntimeError(f'Incomplete checkpoint: {checkpoint}')
        for panel, bank in PANELS.items():
            for profile in PROFILES:
                output = results / f'{label}-{panel}-{profile}.csv'
                receipt = evaluate(checkpoint, bank, output, profile)
                receipt.update(actor=label, split=panel)
                receipts.append(receipt)
                save_receipts(folder / 'receipts.json', receipts)
                print(label, panel, profile, receipt['summary'], flush=True)
    print('STRESS120EVAL_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('preflight', 'matrix'))
    parser.add_argument('--out', type=Path, default=ROOT / 'results/root-stress-matrix')
    args = parser.parse_args()
    {'preflight': preflight, 'matrix': matrix}[args.stage](args.out.resolve())
