# Benchmarks

Machine: Apple M3, 10 GPU cores, 16 GB unified memory; Apple clang 17. Release `-O3`. Date: 2026-10-01.

Initial gate: runtime-compiled raw Metal kernel writes 4096 known float values. Every value agrees exactly. First cold GPU execution: 6.375 microseconds (not a sustained workload benchmark). Command: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j 4 && ./build/metal_nav`.

No navigation-throughput, optimization-speedup or learning-quality claim yet.

## Primitive ray range, first measured ladder

Commit base `7fcf5d2` plus world implementation. FP32 safe/precise, eight AABBs per world plus six room bounds, 16x20 rays, same scalar geometry code on CPU and GPU. Command: `./build/metal_nav bench-depth`. Times are minimum of four GPU runs; CPU is a single scalar run. This is a sensor component benchmark, not training throughput or a comparison to an optimized CPU trainer.

| Environments | One thread/ray ms | One thread/world ms | Scalar CPU ms |
|---:|---:|---:|---:|
| 1 | 0.0123 | 2.001 | 0.0134 |
| 32 | 0.0216 | 2.014 | 0.396 |
| 128 | 0.0694 | 1.947 | 1.603 |
| 512 | 0.261 | 1.965 | 6.107 |
| 2,048 | 0.906 | 2.030 | 24.640 |
| 8,192 | 2.841 | 9.325 | 94.950 |
| 32,768 | 11.318 | 43.425 | 406.212 |

Analytic tests cover misses, inside exits, tangents, parallel slabs, cylinder sides/caps, moving geometry, range clamp and collision independent of depth. All 40,960 CPU/GPU rays and 128 clearances match exactly for the test scene set. Full workload impact remains to be measured.

## Correctness gates

Command: `./build/metal_nav test`, Release, FP32 safe/precise.

| Gate | Workload | Max absolute error |
|---|---|---:|
| RAPTOR official oracle | 128 envs ×16 recurrent steps | 5.96e-7 |
| RAPTOR recurrent state | CPU vs Metal | 1.19e-7 |
| L2F official physics | 64 states ×8 reference steps | 9.54e-7 |
| PX4 observation transform | independent expanded formula | 5.96e-8 |
| Complete control loop | 32 envs ×160 native ticks | state 9.09e-6; hidden 2.50e-6 |
| PPO operators | actor/critic, boundary GAE, gradients, Adam | within 3e-5 test bounds |

GPU fixed-target optimizer smoke: action-mean error 0.587→0.062; this does not establish navigation learning.

Scripted goal direction at requested horizontal speed≤2 m/s, 4m goal distance, 128 held-out first episodes: open success38.3%, boxes28.1%, poles4.7%, dynamic spheres34.4%. These low baseline results are recorded rather than hidden. They are not a trained policy result. GPU execution for each 200-step episode batch is roughly 0.05s; wall about0.05–0.08s on these short early-terminating scenes. Count actual live steps before quoting transitions/s.
