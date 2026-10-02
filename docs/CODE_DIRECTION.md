# Code Direction

Write code that is easy for another engineer to understand, modify, benchmark, and verify without sacrificing measured performance.

The goal is:

> simple structure, obvious names, explicit data flow, minimal abstraction, and fast execution where performance matters.

## Readability

Prefer names from the actual problem domain.

Good:

```text
NavigationPolicy
OrderBook
AudioBuffer
FrameEncoder
PhysicsState
JobQueue
ObservationBuilder
```

Avoid vague names unless they genuinely describe the concept:

```text
Manager
Processor
Handler
Helper
Utils
System
DataManager
```

Classes and structs should represent clear nouns.

Functions should describe clear actions:

```text
buildObservation()
encodeFrame()
updateReference()
processPacket()
computeReward()
saveCheckpoint()
```

Avoid names such as:

```text
process()
handle()
runLogic()
doStuff()
updateData()
```

when a more precise name exists.

## Make the architecture visible

A developer should be able to understand the main data flow without reading implementation details.

Prefer entry points that read approximately like:

```cpp
input = sensor.read();
state = estimator.update(input);
action = policy.infer(state);
command = controller.compute(action);
output.send(command);
```

The high-level code should explain **what happens**.

Lower-level files should explain **how it happens**.

Complexity should be revealed gradually.

## Keep responsibilities clear

Each important component should have one main reason to exist.

Prefer:

```text
Parser
Scheduler
Renderer
Policy
Controller
Database
Evaluator
Trainer
```

over one giant class that owns unrelated behavior.

Do not split code into tiny abstractions merely to satisfy a style rule.

Create a new class/module when it represents a useful conceptual boundary.

## Prefer simple data structures

Represent important concepts explicitly.

Prefer:

```cpp
struct Pose {
    Vec3 position;
    Quaternion orientation;
};
```

over repeatedly passing unrelated primitive values.

Keep structures lightweight where they are performance-sensitive.

Do not turn simple records into elaborate object hierarchies.

## Prefer composition

Avoid deep inheritance trees unless polymorphism is genuinely required.

Prefer:

```text
Application
├── Renderer
├── Physics
├── NetworkClient
└── Scheduler
```

over layers such as:

```text
BaseSystem
→ AbstractSystem
→ RuntimeSystem
→ SpecializedRuntimeSystem
→ FinalRuntimeSystem
```

Use interfaces where there is a real interchangeable boundary.

Do not create interfaces preemptively.

## Avoid unnecessary abstraction

Do not build:

- generic frameworks for one concrete use case
- wrapper layers around already-clear APIs
- factories when direct construction is sufficient
- dependency injection machinery where normal constructors work
- generic utility libraries for a few functions
- elaborate configuration systems before they are needed

Start concrete.

Generalize when multiple real cases prove that a shared abstraction is useful.

## Performance-sensitive code is allowed to look different

Readability does not mean forcing every hot path into high-level object-oriented abstractions.

Separate:

```text
orchestration code
from
performance-critical kernels
```

The orchestration layer should remain easy to understand.

The hot path may use:

- contiguous arrays
- SoA instead of AoS
- explicit memory layouts
- SIMD
- GPU kernels
- preallocated buffers
- fixed-size structures
- batching
- custom allocators
- manual loops
- reduced indirection
- cache-aware layouts
- specialized implementations

when measurements justify them.

Document **why** unusual low-level code exists.

## Prefer zero-cost abstractions

Use abstractions that compile away or have negligible runtime cost when possible.

Examples:

```text
small structs
templates/generics
inline functions
constexpr/static configuration
references/spans/views
RAII
typed enums
```

Do not introduce runtime polymorphism, allocation, copying, synchronization, or indirection simply to make an API look cleaner.

## Keep the hot path obvious

For performance-critical systems, an engineer should be able to trace:

```text
input
→ transformation
→ computation
→ output
```

without jumping through many abstraction layers.

Avoid hidden work in hot loops such as:

- unexpected allocation
- filesystem access
- logging
- string formatting
- copies
- virtual dispatch
- locks
- dynamic container growth

unless justified.

## Allocation

Prefer predictable ownership.

In repeated high-frequency operations:

```text
allocate once
reuse
```

rather than:

```text
allocate
use
destroy
allocate
use
destroy
...
```

Do not prematurely introduce memory pools everywhere.

Use them where profiling shows allocation matters.

## Data movement matters

Avoid unnecessary copies, especially for:

- images
- tensors
- audio/video frames
- large vectors
- simulation state
- network buffers

Prefer references, views, spans, moves, shared buffers, or direct ownership transfer where appropriate.

Do not sacrifice clear ownership merely to avoid a tiny copy that does not matter.

## Concurrency

Do not add threads because something "could be parallel."

Parallelize when there is a measurable benefit.

Make ownership and synchronization obvious.

Prefer designs where workers operate on independent data over designs requiring frequent shared-state locking.

## Error handling

Failures should be explicit.

Do not silently ignore errors.

Use the language's normal error-handling mechanism consistently.

Make invalid states difficult to represent when doing so remains simple.

## Comments

Comments should explain:

```text
why this exists
why this unusual approach is necessary
what invariant must remain true
what performance constraint caused the design
```

Do not comment obvious syntax.

Bad:

```cpp
// increment i
i++;
```

Useful:

```cpp
// Buffer is reused here because this executes at 100 kHz and allocation
// accounted for ~18% of runtime in profiling.
```

## Files

Organize files around recognizable concepts.

A directory should help someone predict where code lives.

Avoid:

```text
misc/
helpers/
utils/
common/
stuff/
```

becoming dumping grounds.

Small projects should remain small.

Do not create a large directory hierarchy before the project needs one.

## Main entry points

Keep entry points boring.

They should primarily:

```text
load configuration
construct major components
connect them
start the program
```

Business logic should not accumulate inside `main()`.

## Tests

Test public behavior and important invariants.

Performance-critical components should have correctness tests before aggressive optimization.

When replacing a clear reference implementation with an optimized version, preserve the reference where useful and verify parity.

A strong pattern is:

```text
simple reference implementation
        ↓
optimized implementation
        ↓
parity tests
        ↓
benchmark
```

## Performance decisions require evidence

Do not make code harder to understand because something *might* be faster.

First establish a baseline.

Measure:

```text
wall-clock time
latency
throughput
memory
allocations
CPU/GPU utilization
cache behavior
```

as relevant.

Then optimize the measured bottleneck.

After optimization, verify:

```text
correctness
+
actual performance improvement
```

Keep the simpler implementation when the optimized version provides no meaningful benefit.

## Preserve causal clarity

Every major subsystem should have a clear answer to:

```text
Why does this exist?
Who calls it?
What goes in?
What comes out?
What state does it own?
What assumptions does it make?
```

If those questions are difficult to answer, reconsider the abstraction.

## Avoid clever code

Prefer obvious code over compressed code.

Do not optimize for:

```text
fewest lines
maximum abstraction
language tricks
clever metaprogramming
```

Optimize for:

```text
correctness
clarity
maintainability
measured performance
```

## When performance conflicts with elegance

Performance wins when the difference materially affects the project outcome.

But isolate the complexity.

Prefer:

```text
clean public interface
        ↓
specialized ugly-but-fast implementation
```

rather than spreading optimization-specific complexity throughout the entire codebase.

Explain the tradeoff and preserve benchmarks proving why it exists.

## Before adding a new abstraction

Ask:

1. What real concept does this represent?
2. Does it simplify the code that uses it?
3. Is there more than one real use case?
4. Does it hide important behavior?
5. Does it add allocation, copying, indirection, synchronization, or runtime dispatch?
6. Can the same result be achieved more simply?

If the abstraction has no clear answer, do not add it.

## Before optimizing

Ask:

1. Is this actually on the hot path?
2. Do we have a measurement?
3. What resource is limiting us?
4. What is the simplest experiment?
5. Did the optimized version materially improve the relevant metric?
6. What readability or maintainability cost did we introduce?

## Standard

Someone unfamiliar with the project should be able to:

1. inspect the repository structure;
2. locate the entry point;
3. identify the major domain concepts;
4. follow the main data flow;
5. find the implementation of a specific behavior;
6. understand where performance-critical code begins;
7. modify one subsystem without understanding the entire repository.

When adding or changing code, leave the surrounding system at least as understandable as you found it.

Do not sacrifice performance for cosmetic cleanliness.

Do not sacrifice clarity for hypothetical performance.

Prefer **simple, explicit, measured engineering**.