# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAMA (Rapid Algorithm for Multicut Problem) is a GPU-accelerated solver for the multicut/correlation clustering problem. It uses CUDA kernels and Thrust/CCCL primitives for GPU computation. Published at CVPR 2022.

## The Multicut Problem

The **multicut** (correlation clustering) problem partitions a graph into clusters by deciding which edges to "cut". Each edge has a real-valued cost:

- **Positive edges** (cost > 0): Endpoints are similar and prefer to be in the **same** cluster. Cutting a positive edge incurs a penalty equal to its cost.
- **Negative (repulsive) edges** (cost < 0): Endpoints are dissimilar and prefer to be in **different** clusters. Keeping a repulsive edge uncut incurs a penalty equal to |cost|.

The objective is to find a partition that minimizes the total cost of cut positive edges plus uncut negative edges. The problem is NP-hard.

### LP Relaxation and Lower Bounds

The solver computes a lower bound via an LP (linear programming) relaxation. Each edge gets a relaxed label in [0,1] (0 = uncut, 1 = cut). The LP enforces **cycle inequalities**: for any cycle, no single edge label can exceed the sum of all other edge labels in that cycle. The tightest constraints come from short cycles (triangles).

**Triangle inequality**: For a triangle on nodes (i, j, k), the constraint `x_ij <= x_ik + x_jk` must hold for all three edge permutations. This prevents inconsistent labelings (e.g., i and j in different clusters while both are with k).

### Conflicted Cycles and Triangulation

A **conflicted cycle** is one containing at least one repulsive edge — these are the cycles where LP constraints may be violated and tightening is beneficial. The solver detects conflicted cycles of length 3 (triangles), 4 (quadrangles), and 5 (pentagons).

**Triangulation**: All higher-order conflicted cycles are decomposed into triangles for uniform treatment:
- Quadrangles → 2 triangles (via a shared diagonal)
- Pentagons → 3 triangles

Triangle detection (`find_triangles.h`): For each repulsive edge (u, v), find all common neighbours w in the positive graph. Each such w forms a conflicted triangle (u, v, w) — exactly one repulsive edge (u,v) and two positive edges (u,w) and (v,w).

### Message Passing (Dual Optimization)

Message passing iteratively tightens the LP lower bound by reparametrizing costs between edges and triangles:

1. **Edge → triangle**: Each edge's cost is distributed equally among all triangles containing it.
2. **Triangle → edge**: Min-marginal messages are computed per triangle and sent back to edges, using `min_marginal(x,y,z) = min(x+y, x+z, x+y+z, 0) - min(0, y+z)`.

This preserves the objective value while making the relaxation tighter. The resulting lower bound = sum of negative edge costs + sum of triangle lower bounds.

### Primal Solver (Edge Contraction)

The primal phase iteratively coarsens the graph:
1. Select edges to merge via maximum matching or MST (on reparametrized costs)
2. Contract selected edges, merging their endpoints into single nodes
3. Repeat until no more contractions improve the solution

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

Use `generate_random_graph()` from `test/random_graph.h` to construct random graphs in tests. It returns a `RandomGraph` struct with `num_nodes`, `tails`, `heads`, and `costs` vectors.

When running tests, always run the CPU test first (it's faster and doesn't require GPU availability), then the GPU test.

## Visualizing Graphs

Use `graph-easy` (installed at `/usr/bin/graph-easy`) to draw ASCII graphs when explaining graph structures or examples. Edge labels carry the cost.

Example — a conflicted triangle (one repulsive edge, two positive edges):
```bash
echo '[0] -- +3 --> [1] -- -1 --> [2] -- +2 --> [0]' | graph-easy --as ascii
```
```
      +2
  +-----------------------+
  v                       |
+---+  +3   +---+  -1   +---+
| 0 | ----> | 1 | ----> | 2 |
+---+       +---+       +---+
```

## Git

- Do not add any Claude acknowledgment or co-author lines in commit messages.
