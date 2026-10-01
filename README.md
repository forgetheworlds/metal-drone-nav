# Metal navigation research engine

Outcome and requirements: [GOAL.md](GOAL.md). Current evidence and next steps: [STATUS.md](STATUS.md).

Build on Apple Silicon with macOS 15 or newer and Command Line Tools:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
./build/metal_nav
```

Metal kernels compile at runtime through `MTLDevice`. Full Xcode is not required for the verified path. FP32, safe arithmetic, and precise floating-point functions are used for parity.

No Python, PyTorch, MLX or tensor runtime is used in the training hot loop.
