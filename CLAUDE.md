# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAMA (Rapid Algorithm for Multicut Problem) is a GPU-accelerated solver for the multicut/correlation clustering problem. It uses CUDA kernels and Thrust/CCCL primitives for GPU computation. Published at CVPR 2022.

## Build Commands

```bash
# Build (from repo root)
mkdir build && cd build
cmake .. && make -j 4

# Run tests (from build directory)
ctest

# Install Python package
pip install .
# With PyTorch tensor support:
WITH_TORCH=ON pip install .

# Run CLI tool
./build/rama_text_input -f graph.txt
./build/rama_text_input --help
```

## Build Requirements

- CMake 3.22.1+, CUDA 11.2+, GCC 10+
- Dependencies auto-downloaded via CPM.cmake (pybind11, CCCL v2.8.3, Eigen, CLI11)
- Optional: OpenMP, PyTorch (via `WITH_TORCH=ON`)

## Architecture

### Algorithm Variants

The solver supports four modes selected via string parameter ("P", "PD", "PD+", "D"):
- **P** (Purely Primal): Fastest, lower quality — primal heuristics only
- **PD** (default): Best time/quality tradeoff — primal + dual relaxation
- **PD+**: Best quality — more dual optimization iterations
- **D**: Dual only — computes lower bound, no primal solution

### Core Pipeline (in `src/rama_cuda.cu`)

1. **Dual Solver** (`dual_solver.cu`) — computes LP relaxation lower bound via conflicted cycles
2. **Conflicted Cycles Detection** (`conflicted_cycles_cuda.cu`) — GPU kernels finding violated triangle/quadrangle/pentagon constraints
3. **Edge Contraction** (`edge_contractions_woc.cu`) — coarsens graph using either maximum matching or MST depending on contraction rate
4. **Maximum Matching** (`maximum_matching_vertex_based.cu`) — greedy GPU matching
5. **Message Passing** (`multicut_message_passing.cu`) — dual optimization on edge/triangle constraints

### Key Data Structures

**dCOO** (`include/dCOO.h`, `src/dCOO.cu`): Legacy GPU-native sparse matrix in COO format. Uses matrix terminology (rows/cols/row_ids/col_ids/data). Still used throughout the pipeline but being replaced by Graph.

**Graph** (`include/graph.h`): Templatized replacement for dCOO. Uses graph-theoretic naming (num_nodes/tails/heads/costs). Templatized on `VectorType` (`thrust::device_vector` or `thrust::host_vector`) for CPU/GPU support. Always stores symmetric edges (both directions). Key differences from dCOO:
- Single `num_nodes_` instead of separate `rows_`/`cols_`
- `num_edges()` = undirected count, `num_directed_edges()` = stored count
- Input validation: `is_single_orientation()`, `has_duplicate_edges()`
- `is_symmetric` constructor flag for passing already-symmetric edges
- Type aliases: `DeviceGraph`, `HostGraph`

### Configuration

All solver parameters are centralized in `include/multicut_solver_options.h` using CLI11 for parsing.

### Interfaces

- **C++ API**: `rama_cuda()` in `include/rama_cuda.h` — takes edge lists (i, j, costs) + options, returns (node_labels, lower_bound, duration, timeline)
- **Python API**: `rama_py` module in `src/rama_py.cu` — provides `rama_cuda()`, `rama_cuda_gpu_pointers()`, and `rama_torch()` (when compiled with PyTorch)
- **CLI**: `src/rama_text_input.cu` — reads text files in `MULTICUT\ni j cost` format

### External Libraries (in `external/`)

- **ECL-CC**: GPU connected components (single-kernel algorithm)
- **cudaMST**: GPU minimum spanning tree (Boruvka and Kruskal variants)

## Ongoing Migration: Thrust Abstraction for CPU/GPU

The codebase is being ported from raw CUDA kernels to Thrust library algorithms to enable both CPU and GPU execution backends. Thrust provides backend-agnostic parallel primitives (via `thrust::device_vector` / `thrust::host_vector` and execution policies like `thrust::device` / `thrust::host`).

### Migration Status

| Component | Status | Notes |
|-----------|--------|-------|
| Graph (dCOO replacement) | **Done** | `include/graph.h` — templatized, symmetric, graph naming |
| Edge contraction | Not started | `contract_cuda()` not yet ported to Graph |
| Dual solver | Not started | Uses dCOO |
| Conflicted cycles | Not started | Uses dCOO |
| Maximum matching | **Done** | `include/maximum_matching.h` — templatized, uses Graph |
| Message passing | Not started | Uses dCOO |

When modifying or adding code:
- Prefer Thrust algorithms over raw CUDA kernels where possible
- Use Thrust execution policies to allow backend selection at compile/run time
- CPU vs GPU is selected by using `thrust::host_vector` (CPU) or `thrust::device_vector` (GPU) — algorithms operate on whichever vector type is provided
- All functionality lives in templatized headers in `include/` — template instantiation happens in `.cpp` files for CPU or `.cu` files for GPU
- The CCCL v2.8.3 dependency already provides Thrust
- Use `Graph` instead of `dCOO` for new code

## Code Conventions

- Headers in `include/`, CUDA implementations in `src/`, tests in `test/`
- GPU-first design: data lives on device via `thrust::device_vector`, minimal CPU-GPU transfers
- All `.cu` files — even "C++" logic uses CUDA compilation for Thrust/CCCL interop
- Move semantics used for efficient GPU data transfer
- Use lambdas (with `__host__ __device__` annotations via macros) instead of functor structs for Thrust algorithm callbacks

### Testing Convention

Tests for templatized components use a three-file pattern with separate CPU and GPU instantiations:

1. **`test/<component>_test.h`** — templatized test functions (`template<template<typename> class VectorType>`)
2. **`test/<component>_gpu_test.cu`** — instantiates tests with `thrust::device_vector`
3. **`test/<component>_cpu_test.cpp`** — instantiates tests with `thrust::host_vector` (compiled as CUDA via `set_source_files_properties`)

Example (see `test/graph_test.h`, `test/graph_gpu_test.cu`, `test/graph_cpu_test.cpp`):
```cpp
// test/foo_test.h — templatized tests
template<template<typename> class VectorType>
void test_something() { /* ... */ }

template<template<typename> class VectorType>
void run_all_foo_tests() { test_something<VectorType>(); }

// test/foo_gpu_test.cu
#include "foo_test.h"
int main() { run_all_foo_tests<thrust::device_vector>(); }

// test/foo_cpu_test.cpp
#include "foo_test.h"
int main() { run_all_foo_tests<thrust::host_vector>(); }
```

Use the `test()` helper from `include/test.h` for assertions.

When running tests, always run the CPU test first (it's faster and doesn't require GPU availability), then the GPU test.

## Git

- Do not add any Claude acknowledgment or co-author lines in commit messages.
