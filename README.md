# Metal navigation research engine

Outcome and requirements: [GOAL.md](GOAL.md). Current evidence and next steps: [STATUS.md](STATUS.md).

Build on Apple Silicon with macOS 15 or newer and Command Line Tools:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
./build/metal_nav
./build/metal_nav test
./build/metal_nav sim
./build/metal_nav bench-depth
./build/metal_nav bench-loop
./build/metal_nav train 50 0 results/open.bin
```

Metal kernels compile at runtime through `MTLDevice`. Full Xcode is not required for the verified path. FP32, safe arithmetic, and precise floating-point functions are used for parity.

No Python, PyTorch, MLX or tensor runtime is used in the training hot loop.

`test` checks analytic geometry, official RAPTOR outputs, official L2F physics fixtures, the PX4 target adapter, PPO operators and an integrated 160-control-step CPU/GPU trajectory. `sim` measures the simple goal-direction controller on four scene families with one completed episode per held-out seed.

`train ITERATIONS FAMILY CHECKPOINT` trains and resumes exact GPU state. Families:0 open,1 boxes,2 poles,3 moving spheres. Checkpoints are saved atomically every10 rollouts. The current trained policy does not yet meet the held-out navigation goal.

`train ITERATIONS FAMILY CHECKPOINT WARMSTART` starts a new curriculum stage from actor/critic parameters and resets optimizer/exploration. Latest checkpoint resumes all state; `CHECKPOINT.best` keeps the best validation policy. One `results/training.tsv` records evaluation history. `eval CHECKPOINT MODE FAMILY SEED SPEED DISTANCE` evaluates first episodes on a fresh seed; modes4 learned mean,1 random,2 goal-direction script.

References are pinned outside the build. To regenerate the cold assets:

```sh
git clone https://github.com/rl-tools/raptor /tmp/raptor-reference
git -C /tmp/raptor-reference checkout 2c789dfcf16cc96fe697704492b3bf79dd2cc5a0
git -C /tmp/raptor-reference submodule update --init rl-tools data
python3 export_raptor.py /tmp/raptor-reference/data/raptor-policy-checkpoint.tar.gz
clang++ -std=c++17 -O2 -I/tmp/raptor-reference/rl-tools/include reference.cpp -o build/reference
./build/reference assets/physics.bin
```

RAPTOR and RLtools code/weights are used under their MIT notices in `THIRD_PARTY_LICENSES.txt`.
