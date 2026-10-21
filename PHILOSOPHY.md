# Project philosophy

## Mission

Enable **reproducible research** in Multiple View Geometry (MVG) and Structure from Motion (SfM) applications.

## Core principles

### 1. Educational first, research ready

This library prioritizes **clarity and understanding** over raw performance. Code should be readable by students and researchers learning computer vision algorithms. When a choice exists between a clever optimization and a clear implementation, choose clarity.

### 2. Accuracy over speed

Correctness and numerical stability come first. Optimizations are welcome only when they don't compromise accuracy or readability.

### 3. Pure Rust

Target a pure Rust implementation with no C/C++ dependencies. This gives:

- simpler cross-platform builds;
- memory safety;
- a dependency tree that Cargo manages on its own.

The Python bindings (`openmvg-rs`) call into the Rust core; the algorithms stay in Rust.

### 4. Reproducibility

- Algorithms should produce deterministic results given the same inputs
- Reference implementations should match published papers
- Include citations to source papers in documentation

## Target audience

1. Students learning computer vision and 3D reconstruction
2. Researchers prototyping and validating new algorithms
3. Practitioners needing reliable, well-documented MVG tools
