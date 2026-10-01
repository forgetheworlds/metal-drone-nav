# State — 2026-10-01

Goal: a verified Metal-native navigation trainer with RAPTOR motor control. The full outcome is active, not yet achieved.

Machine: Apple M3, 10 GPU cores, 16 GB unified memory. Apple clang 17 and macOS SDK are installed. Full Xcode and offline `metal` compiler are absent. Runtime Metal compilation is the first test; installation is authorized if needed.

Repository started empty. Only Luna agents are used. No skills are used, per the current user request.

Ownership: root owns build/runtime, worlds, integration, tests and consolidated evidence. Luna raptor owns raptor.hpp/raptor.metal and the official weight export. Luna physics owns physics.hpp/physics.metal. Luna ppo owns ppo.hpp/ppo.metal.

Current work: obtain pinned upstream references; prove runtime compilation; agree packed interfaces; implement CPU/upstream/Metal parity before closed-loop training.

Completion gates: official RAPTOR parity; upstream physics parity; independent ray/collision tests; GPU-resident rollouts; tested PPO gradients and updates; held-out goal/avoidance improvement; matched end-to-end optimization.

Verified: runtime Metal compilation; shared CPU/MSL box/sphere/cylinder rays; independent collision clearance; deterministic moving geometry; 40,960 ray and 128 clearance parity cases. Depth benchmark ladder reaches 32,768 environments. See BENCHMARKS.md.

Verified control/learning operators: official RAPTOR 16-step outputs on 128 GPU environments (max error 5.96e-7); 512 official L2F physics fixture records (max error 9.54e-7); independent PX4 observation-transform check; 32 complete CPU/GPU control trajectories over 160 native ticks (state error 9.09e-6). PPO forward, GAE terminal/truncation, selected finite-difference gradients, backprop/reduction and Adam parity pass. A fixed-target optimizer smoke test moves the mean toward its target; it is not yet evidence of navigation learning.

GPU loop: 16x20 pinhole range/history, deployable actor state, privileged critic, Gaussian navigation commands, finite yaw/velocity target adapter, actual RAPTOR GRU at 100 Hz, L2F RK4, independent collision/reward/reset. Host encodes long command batches and waits after the batch, not after each step. Buffers are allocated at setup.

Training now runs entirely on GPU, including rollout, GAE, normalization, two PPO epochs, backward/reduction, gradient clipping and Adam. Checkpoint v3 stores actor/critic/optimizer plus physics, worlds, recurrent state, RNG, depth/command delay rings and config for exact continuation. `./build/metal_nav train 50 0 results/open.bin` continues the open-world run. Training uses128 environments,32-step rollouts,256-sample minibatches,661 actor inputs,64 hidden units, four actions; requested speed≤1m/s and3m goals.

Fixed reference bug: reanchoring target position to measured position while separately commanding target velocity caused altitude loss in upstream dynamics. Persistent target position now integrates desired velocity on every10ms tick. The actor also sees target-position tracking error; SimRun is184 bytes. Source transform and clips still match selected PX4. Old adapter experiment is isolated in results/open-old-adapter.bin. New controller reaches all128 open held-out goals at1m/s. Upstream20s constant2m/s probe stays at1.448m altitude, versus floor failure with the old adapter. See RESEARCH_LOG.md.

Current learned result after30 rollouts: deterministic held-out open success21.1%, zero collisions,78.9%timeouts. This is improvement from the initial policy but does not meet the goal. Continue learning, then static clutter versus goal-scripted baseline (~39% success with new adapter). PPO update currently costs~1.67s per4096-transition rollout; optimize measured gradient bottleneck. PPO Luna worker owns ppo.hpp/ppo.metal optimization and provides a host patch in/tmp; root owns main.mm. M3 research Luna agent owns M3_RESEARCH.md.

Overnight continuation: app heartbeat `overnight-metal-navigation-build` every 15 minutes, same chat. Continue the active goal from this state; use only Luna agents; no skills. Root commits use Forge the World, GitHub account forgetheworlds. Xcode install is allowed if needed. ARM VPS `oracle-vps` is available but has not been needed.
