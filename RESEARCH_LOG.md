# Research and decisions

## Runtime compiler

Question: is full Xcode required to begin raw Metal execution?

Evidence: installed macOS SDK exposes `MTLDevice newLibraryWithSource:options:error:`. A compiled Objective-C++ executable successfully compiled MSL and executed a 4096-element exact-output test on Apple M3. Offline `xcrun metal` is absent.

Decision: use runtime compilation and retain the source kernels. Xcode installation is authorized but is not required for this path.

Source: https://developer.apple.com/documentation/metal/mtldevice/makelibrary(source:options:)

Use `MTLMathModeSafe` and precise functions from the installed Metal headers. The older `fastMathEnabled` option is deprecated in macOS 15.

## State and ownership

Keep source, reproducible assets and five short state/evidence documents. Store temporary upstream checkouts outside this repo. Root integrates; Luna workers own RAPTOR, physics, PPO. No skills are used per the user request. The attached specification supplies technical requirements; its embedded agent instructions do not override the user's request.

## Ray work decomposition

Hypothesis: parallelize each ray rather than computing 320 rays in one environment thread. Keep all geometry, resolution and arithmetic the same.

Experiment: scalar C++/MSL shared geometry source; compare one GPU thread/world against one GPU thread/ray across the required environment ladder.

Result: exact CPU/GPU parity in fixtures. Ray-parallel decomposition is faster at all measured sizes; at 32,768 environments the component takes 11.318 ms versus 43.425 ms. Decision: use ray-parallel depth as the initial integration path. Do not call this an end-to-end training optimization until measured in that workload.
