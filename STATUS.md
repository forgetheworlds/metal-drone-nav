# State — 2026-10-01

Goal: a verified Metal-native navigation trainer with RAPTOR motor control. The full outcome is active, not yet achieved.

Machine: Apple M3, 10 GPU cores, 16 GB unified memory. Apple clang 17 and macOS SDK are installed. Full Xcode and offline `metal` compiler are absent. Runtime Metal compilation is the first test; installation is authorized if needed.

Repository started empty. Only Luna agents are used. No skills are used, per the current user request.

Ownership: root owns build/runtime, worlds, integration, tests and consolidated evidence. Luna raptor owns raptor.hpp/raptor.metal and the official weight export. Luna physics owns physics.hpp/physics.metal. Luna ppo owns ppo.hpp/ppo.metal.

Current work: obtain pinned upstream references; prove runtime compilation; agree packed interfaces; implement CPU/upstream/Metal parity before closed-loop training.

Completion gates: official RAPTOR parity; upstream physics parity; independent ray/collision tests; GPU-resident rollouts; tested PPO gradients and updates; held-out goal/avoidance improvement; matched end-to-end optimization.

No performance or learning claims have passed yet.
