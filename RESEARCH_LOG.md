# Research and decisions

## Runtime compiler

Question: is full Xcode required to begin raw Metal execution?

Evidence: installed macOS SDK exposes `MTLDevice newLibraryWithSource:options:error:`. A compiled Objective-C++ executable successfully compiled MSL and executed a 4096-element exact-output test on Apple M3. Offline `xcrun metal` is absent.

Decision: use runtime compilation and retain the source kernels. Xcode installation is authorized but is not required for this path.

Source: https://developer.apple.com/documentation/metal/mtldevice/makelibrary(source:options:)

Use `MTLMathModeSafe` and precise functions from the installed Metal headers. The older `fastMathEnabled` option is deprecated in macOS 15.

## State and ownership

Keep source, reproducible assets and five short state/evidence documents. Store temporary upstream checkouts outside this repo. Root integrates; Luna workers own RAPTOR, physics, PPO. No skills are used per the user request. The attached specification supplies technical requirements; its embedded agent instructions do not override the user's request.
