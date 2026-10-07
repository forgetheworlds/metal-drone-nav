"""Resume only missing frozen native flights with a verified background launcher."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import shutil
import signal
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'webots'))
import local_waypoint_transfer as transfer
import bounded_motion_scene as scenes


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_launcher_change(helper, original_hash):
    current = helper.read_text()
    background = '''        environment = os.environ.copy()
        if sys.platform == "darwin" and not record_movie:
            # Batch/minimize still lets Qt promote each new macOS instance.
            # Keep its Cocoa/OpenGL backend but disable foreground promotion.
            environment["QT_MAC_DISABLE_FOREGROUND_APPLICATION_TRANSFORM"] = "1"
        process = subprocess.Popen(command, cwd=str(WEBOTS_DIR), stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True,
                                   close_fds=True, env=environment)'''
    original = '''        process = subprocess.Popen(command, cwd=str(WEBOTS_DIR), stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)'''
    if current.count(background) != 1 or hashlib.sha256(current.replace(background, original).encode()).hexdigest() != original_hash:
        raise ValueError('Helper changes exceed the verified foreground-suppression patch')


def resume(folder):
    freeze = json.loads((folder / 'freeze.json').read_text())
    helper = ROOT / 'webots/local_waypoint_transfer.py'
    for name, expected in freeze.items():
        path = Path(name)
        if digest(path) != expected:
            if path != helper:
                raise ValueError(f'Frozen input changed: {path}')
            check_launcher_change(helper, expected)
    background = json.loads((folder / 'background-check.json').read_text())
    if not background['valid_native_flight'] or background['webots_became_foreground']:
        raise ValueError('A valid non-foreground launch check is required')
    actors = json.loads((folder / 'actors.json').read_text())
    selection = json.loads((folder / 'selection.json').read_text())
    records = json.loads((folder / 'receipts.json').read_text())
    expected = {(case['reflection'], case['index'], label) for case in selection for label in actors}
    completed = set()
    for row in records:
        key = (row['reflection'], row['index'], row['actor'])
        if key in completed or key not in expected or not row['run']['valid']:
            raise ValueError('Invalid or duplicated completed flight ledger')
        flight = folder / 'flights' / row['slug']
        if json.loads((flight / 'episode.json').read_text()) != row['run']['receipt']:
            raise ValueError(f'Completed receipt differs: {flight}')
        world = folder / 'project/worlds' / ('local-waypoint-transfer-' + row['slug'] + '.wbt')
        if digest(world) != row['world_sha256']:
            raise ValueError(f'Completed world changed: {world}')
        completed.add(key)
    for actor in actors.values():
        if digest(Path(actor['nav'])) != actor['nav_sha256'] or digest(Path(actor['checkpoint'])) != actor['checkpoint_sha256']:
            raise ValueError('Frozen actor changed')
    supplemental = {
        'started': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
        'original_helper_sha256': freeze[str(helper)], 'background_helper_sha256': digest(helper),
        'completed_before_resume': len(completed), 'remaining': len(expected - completed),
        'selection_sha256': digest(folder / 'selection.json'),
        'resume_code_sha256': digest(Path(__file__)), 'record_movie': False,
    }
    (folder / 'resume-provenance.json').write_text(json.dumps(supplemental, indent=2) + '\n')
    accepted = dict(freeze)
    accepted[str(helper)] = digest(helper)
    project = folder / 'project'
    transfer.WEBOTS_DIR = project
    transfer.WORLD_DIR = project / 'worlds'
    transfer.OUT_DIR = folder
    transfer.FLIGHT_DIR = folder / 'flights'
    def cancel(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    print('NATIVE_RESUME_PID', os.getpid(), 'remaining', len(expected - completed), flush=True)
    for case in selection:
        for label, actor in actors.items():
            key = (case['reflection'], case['index'], label)
            if key in completed:
                continue
            for name, expected_hash in accepted.items():
                if digest(Path(name)) != expected_hash:
                    raise ValueError(f'Input changed during resume: {name}')
            slug = f"delay-fresh-{case['reflection']}-{case['index']}-{label}"
            flight = folder / 'flights' / slug
            if flight.exists():
                retained = folder / 'interrupted-attempts' / (slug + '-' + str(len(records)))
                retained.parent.mkdir(exist_ok=True)
                if retained.exists():
                    raise ValueError('Interrupted attempt destination already exists')
                shutil.move(str(flight), str(retained))
            world = scenes.export(Path(case['bank']), case['index'], Path(actor['nav']), project, slug)
            result = transfer.run_flight(world, flight, False)
            row = {**case, 'actor': label, 'slug': slug, 'nav_sha256': actor['nav_sha256'],
                   'world_sha256': digest(world), 'run': result}
            if not result['valid']:
                (folder / 'invalid-resume.json').write_text(json.dumps(row, indent=2) + '\n')
                raise RuntimeError('Invalid execution preserved; repair cause before continuing')
            records.append(row)
            temporary = folder / 'receipts.tmp'
            temporary.write_text(json.dumps(records, indent=2) + '\n')
            temporary.replace(folder / 'receipts.json')
            completed.add(key)
            print(slug, 'VALID', len(completed), '/', len(expected), flush=True)
    print('FRESH_NATIVE192_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=ROOT / 'results/root-delay-native')
    args = parser.parse_args()
    resume(args.folder.resolve())
