# Reproducibility contract

OpenMVG-rs treats deterministic execution as a user-visible contract. Given
the same prepared input, configuration, full-width root seed, RNG protocol,
target, numerical backend, and pinned toolchain, a supported run must produce
the same scientific output at every supported thread count. The
retained determinism tests check replay contracts; corpus-level evidence comes from release
qualification on a fixed dataset corpus outside this repository.

Scientific output includes model choices, inlier sets, registered views,
reconstruction values, and the digest of those values. Wall-clock measurements,
resource usage, timestamps, and progress events are telemetry: they may differ
and never influence a numerical decision or scientific digest.

## Random streams

Applications own one `RunSeed([u8; 32])`. Library algorithms use the
declared `AlgorithmRng` protocol and derive child streams from versioned domain
names and stable problem identifiers. Streams must not depend on worker index,
task completion order, container iteration order, or incidental call timing.
Any invocation index is part of the stable domain identity. Parallel work is
reduced into canonical ID order.

A numeric seed label is only an input to a particular RNG protocol. Equal
labels under two different protocol versions are independent streams, not
paired samples. Protocol comparisons therefore use fixed multi-seed
distributions and predeclared non-inferiority limits rather than matching one
historically favorable draw.

## Receipts and protocol changes

Persisted receipts record the root seed, seed-derivation algorithm, RNG and
transformation version, target, backend, toolchain, input fingerprints, and
scientific-output digest. A missing or mismatched field fails acceptance.

Changing seed derivation, the generator, or a random transformation creates a
new protocol version. Existing baselines remain immutable. The new protocol is
accepted only after exact replay within and across supported thread counts on
the same target, backend, and toolchain, a comparison on the release qualification corpus,
and an explicitly named new baseline.
Thresholds, retry policy, solver policy, and sample counts are not tuned after
observing the new stream.

This is not a claim of bit-identical floating-point output across targets,
standard-library math implementations, compiler toolchains, or numerical
backends. Cross-target claims require their own declared tolerance and gate.
