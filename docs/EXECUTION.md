# Heavy-job execution

Training, evaluation and native Webots flights share one exclusive lock:
`results/metal-training.lock` in the primary repository. Worktree jobs must use
the primary repository launcher, not a separate lock in their worktree.

```sh
python3 /Users/muadhsambul/RL/run_locked.py -- COMMAND ARGUMENTS
```

The launcher waits for the lock, then replaces itself with the command while
retaining the open lock descriptor. Its PID and exit status now belong to that
command. Cancelling the launcher therefore cancels the actual command rather
than leaving it running behind a terminated wrapper. The lock releases when
the command exits. The old saved-command path under
`results/mimo-learning-next/run_locked.py` forwards to this public launcher.

A command that creates separate process groups must still own their cleanup.
For a Webots batch, launch each simulator in a new process group, keep the real
process handle, bound its runtime, terminate that group on failure, wait for
the actual exit and save a per-flight receipt. Include load errors, missing
robot or policy, invalid sensor/motor initialization and absent episode result
as execution failures. A mover driver continuing after a failed vehicle
controller is not a completed navigation flight.

Use persistent parent-owned execution for full training and benchmark batches.
An external agent ending while awaiting a background callback has repeatedly
disposed its jobs. A CLI being alive is not evidence that training started.
Inspect the actual child, checkpoint header, exit status and complete outcomes.

The October 5 cancellation check used a private temporary lock. It verified
that the actual command PID equals the launcher PID, a second command waits
during execution, cancellation ends the first command, and the waiting command
then finishes. It did not acquire or alter the shared production lock.

Normal RL Webots runs use port 23456, `--minimize --batch`. The shared Webots
installation, other-project files/settings and other-project processes remain
outside this repository's execution ownership. Visible recording is used only
for actual scene movies and is disclosed with their evidence.
