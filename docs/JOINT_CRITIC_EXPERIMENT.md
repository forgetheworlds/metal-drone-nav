# One value function for static and moving tasks

The moving-obstacle policy transfers well to the native challenge, but it also
lost some static-task performance. I am testing the controller-state critic
in the original mixed trainer rather than adding another navigation specialist.

That trainer cycles four dynamic tasks and eight static tasks per environment.
Its 512 dynamic records and 1,024 static records remain unchanged. The static
bank is byte-identical to the canonical source bank. Episode proportions are
one-third dynamic and two-thirds static; transition proportions depend on
actual flight duration and are recorded by the trainer.

Both arms keep the original 184-input actor, FULL depth history, body action
mapping, legacy camera, physics and reward. They use the original time cost
0.2 per second, contact penalty 10 and arrival bonus 10. Both critics have
64 inputs and 64 hidden units; only controller-state information differs.
The control's extra inputs are zero. Both use the same fast warmstart and the
same zero-padded migration of its 32-input value weights.

The new [wrapper](../navigation_joint_critic.mm) uses the tested root critic
core. The old dynamic module and snapshot remain preserved. A one-rollout
smoke passed reset position, reference, goal, yaw and episode-clock checks,
then saved one rollout and 32 optimizer steps with 12,104 actor parameters
and 4,225 critic parameters.

Four fresh 5,000-rollout runs compare control/treatment across the original
two seeds. Final checkpoints are primary and source dev-dyn selects checkpoints.
The evaluation includes fresh dev-dyn2 and all static retention banks. Gates
require retaining dynamic success, improving static blocked routes, reaching
at least 254/256 open goals, and preserving speed. They were fixed before the
full runs. No simulator-side policy training or asset replacement is implied.

Results are pending. This comparison asks whether the value-state correction
helps one actor retain both skills; it does not change the task mixture, make
the actor larger, or replace PPO.
