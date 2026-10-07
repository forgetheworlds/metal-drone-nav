For the next phase, focus entirely on making the learned navigation policy genuinely strong through training scale, curriculum, and controlled experimentation.

You already know the repo, architecture, simulator, policy, training code, benchmarks, and current limitations. Do not spend time re-auditing or explaining the repository to me.

The research question is:

    How far can we push navigation performance by dramatically improving
    the amount, diversity, and ordering of experience before we need a
    fundamentally different policy architecture?

Treat this as an open-ended research run. Use the available compute aggressively.

Start from the strongest working baseline and iterate experimentally.

Priorities:

1. SCALE EXPERIENCE
   Push parallel simulation and total experience much harder.
   Benchmark useful environment counts and find the regime that maximizes
   learning progress per wall-clock hour, not merely raw FPS.

   Scale from the current regime toward thousands of parallel environments
   where practical.

   Track:
   - samples/sec
   - training throughput
   - PPO update cost
   - CPU/GPU utilization
   - memory
   - wall-clock time to reach performance thresholds

2. BUILD A MEANINGFUL CURRICULUM
   Progress from:
   - easy open navigation
   - sparse obstacles
   - denser static clutter
   - narrow passages / difficult geometry
   - longer local navigation
   - higher speeds
   - moving obstacles
   - difficult interaction / short-TTC cases
   - noise, latency, disturbances, and mixed difficult worlds

   Do not make the curriculum arbitrary.

   Use measured competence to decide when difficulty should increase.

3. INVESTIGATE ADAPTIVE TASK SAMPLING
   Explore training near the policy's learning frontier.

   Spend less training on:
   - trivially mastered cases
   - currently impossible cases

   Spend more training on:
   - cases with partial success
   - recent failures the policy can plausibly learn from
   - difficulty bands where learning progress is highest

   Consider simple learning-progress sampling, failure replay, PLR-like
   ideas, or curriculum mutation.

   Start with the simplest useful implementation and measure whether it
   actually helps.

4. KEEP A DIRECT-MIXED BASELINE
   Curriculum must beat something.

   Run a comparable policy trained directly on the mixed task distribution
   with roughly matched sample/compute budget.

   If curriculum does not improve final held-out performance or time to
   competence, simplify or remove it.

5. SCALE NETWORK CAPACITY AS A SEPARATE AXIS
   Do not simultaneously change architecture and curriculum.

   Start with the current small policy and run a capacity sweep.

   Roughly explore:
       current
       ~50k params
       ~150k
       ~500k
       ~1M

   Exact sizes should follow the current architecture.

   Determine whether:
   - the current policy is under-capacity,
   - more experience fixes most failures,
   - larger policies only help on harder distributions,
   - or performance saturates regardless of size.

6. KEEP PPO AS THE REFERENCE
   Do not jump to GRPO or another algorithm merely because it is interesting.

   First establish what well-scaled PPO can do.

   Once that baseline is strong, you may run controlled algorithm comparisons
   if there is a concrete reason to expect improvement.

7. AUDIT THE REWARD WHILE SCALING
   Watch for reward exploitation and shaping dependence.

   Determine what each reward component is actually teaching.

   If useful, use stronger shaping early and reduce dependence on it later,
   but validate this experimentally rather than assuming it is necessary.

8. BUILD A REAL HELD-OUT EVALUATION SUITE
   Separate training and evaluation distributions.

   Evaluate independently on things such as:
   - normal static navigation
   - dense clutter
   - narrow passages
   - longer navigation
   - dynamic obstacles
   - high-speed avoidance
   - unseen layouts
   - noise / latency / disturbances
   - retention of easy skills after harder training

   Track at least:
   - success
   - collisions
   - time to goal
   - path efficiency
   - speed
   - failure type

   Use multiple seeds.

9. PRESERVE CAUSALITY IN EXPERIMENTS
   Change one major thing at a time.

   Avoid:
       bigger network
       + different reward
       + new curriculum
       + new PPO settings
       + new environment distribution

   all in one experiment.

   I want to know WHY something got better.

10. KEEP RUNNING
    This is not a one-experiment task.

    Use a loop like:

        observe current failure mode
        → form hypothesis
        → run smallest decisive experiment
        → analyze
        → keep/reject hypothesis
        → scale the winner
        → expose next bottleneck

    Continue autonomously instead of stopping after every minor result.

The most important metric is not raw simulator FPS or reward.

It is:

    general navigation capability reached per unit wall-clock training time,
    followed by final held-out capability after sufficient scale.

Generate plots, tables, videos, and failure classifications as useful.
Preserve results from unsuccessful experiments so we do not repeat them.

Do not spend this phase cleaning architecture or rewriting working components
unless they materially block the experiments.

At meaningful checkpoints, summarize:

- what hypothesis was tested
- experiment configuration
- result
- whether the hypothesis survived
- what currently appears to be the dominant bottleneck
- what you are testing next

The broad direction is:

    EXPERIENCE SCALE
          +
    CURRICULUM / TASK DISTRIBUTION
          +
    POLICY CAPACITY SWEEP
          ↓
    determine the actual scaling law / bottleneck of this navigation policy.

Keep pushing until the results make it clear what the next fundamental
limitation is.
