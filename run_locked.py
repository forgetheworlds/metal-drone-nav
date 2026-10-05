#!/usr/bin/env python3
"""Run a command while holding an exclusive flock on results/metal-training.lock.

The lock is acquired before the child starts and released only after the child
exits, so every TRAIN/EVAL subprocess of this mission is mutually exclusive
with other agents' training or evaluation runs.

usage: run_locked.py -- CMD [ARGS...]
"""
import fcntl
import os
import sys
import time


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0] != "--":
        sys.stderr.write("usage: run_locked.py -- CMD [ARGS...]\n")
        return 2
    args = args[1:]
    project_root = os.path.dirname(os.path.abspath(__file__))
    lock_path = os.path.join(project_root, "results", "metal-training.lock")
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o666)
    started = time.time()
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.stderr.write("waiting for metal-training.lock\n")
        fcntl.flock(fd, fcntl.LOCK_EX)
    waited = time.time() - started
    if waited > 0.5:
        print(f"metal-training.lock acquired after {waited:.1f}s", flush=True)
    else:
        print("metal-training.lock acquired", flush=True)
    # Keep the lock in the command itself. Replacing this process also makes
    # the launcher's PID/exit status refer to the actual command: cancelling
    # a short-lived wrapper must not leave Webots or training alive behind it.
    # A command that starts separate process groups must still own their cleanup.
    os.set_inheritable(fd, True)
    try:
        os.execvp(args[0], args)
    finally:
        # Reached only when exec fails. A successful command releases the lock
        # when it exits; this Python wrapper no longer exists at that point.
        os.close(fd)


if __name__ == "__main__":
    sys.exit(main())
