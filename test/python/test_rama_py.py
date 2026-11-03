"""Tests for the rama_py bindings. CPU tests always run; GPU tests run only if the
module was built with CUDA and a GPU is present.

Run directly (python test_rama_py.py) or with pytest. rama_py must be importable,
e.g. via PYTHONPATH=<build>/src.
"""
import itertools
import random
import sys

import rama_py


def random_graph(num_nodes, num_edges, seed):
    rng = random.Random(seed)
    edges = set()
    # spanning path so that every node has an incident edge
    for v in range(num_nodes - 1):
        edges.add((v, v + 1))
    while len(edges) < num_edges:
        u, v = rng.sample(range(num_nodes), 2)
        edges.add((min(u, v), max(u, v)))
    edges = sorted(edges)
    i = [u for u, _ in edges]
    j = [v for _, v in edges]
    costs = [rng.uniform(-1.0, 1.0) for _ in edges]
    return i, j, costs


def objective(labels, i, j, costs):
    return sum(c for u, v, c in zip(i, j, costs) if labels[u] != labels[v])


def set_partitions(n):
    """All partitions of range(n), as label lists in restricted growth form."""
    def rec(prefix, max_label):
        if len(prefix) == n:
            yield list(prefix)
            return
        for label in range(max_label + 2):
            yield from rec(prefix + [label], max(max_label, label))
    yield from rec([0], 0)


def brute_force_optimum(num_nodes, i, j, costs):
    return min(objective(p, i, j, costs) for p in set_partitions(num_nodes))


def check_solution(solve, num_nodes, i, j, costs, opts):
    labels, lb, duration_ms, timeline = solve(i, j, costs, opts)
    assert len(labels) == num_nodes
    assert all(l >= 0 for l in labels)
    assert duration_ms >= 0
    obj = objective(labels, i, j, costs)
    assert lb <= obj + 1e-3, (lb, obj)
    return labels, lb, obj


# --- CPU ---

def test_cpu_solution_is_consistent():
    for solver_type in ["P", "PD", "PD+"]:
        for seed in range(3):
            i, j, costs = random_graph(50, 200, seed)
            check_solution(rama_py.rama_cpu, 50, i, j, costs, rama_py.multicut_solver_options(solver_type))


def test_cpu_bounds_bracket_optimum():
    for seed in range(5):
        i, j, costs = random_graph(7, 15, seed)
        _, lb, obj = check_solution(rama_py.rama_cpu, 7, i, j, costs, rama_py.multicut_solver_options("PD"))
        opt = brute_force_optimum(7, i, j, costs)
        assert lb <= opt + 1e-3 and opt <= obj + 1e-3, (lb, opt, obj)


def test_cpu_lower_bound_only():
    i, j, costs = random_graph(30, 100, 0)
    _, lb, _, _ = rama_py.rama_cpu(i, j, costs, rama_py.multicut_solver_options("D"))
    assert lb <= sum(c for c in costs if c > 0) and lb >= sum(c for c in costs if c < 0) - 1e-3


def test_cpu_sanitize_graph():
    # Nodes 2, 3 and 4 have no incident edges, edge (0, 1) appears twice in opposite directions.
    i = [0, 1, 5, 1, 6]
    j = [1, 0, 6, 5, 0]
    costs = [1.0, 0.5, 2.0, -3.0, 1.0]
    opts = rama_py.multicut_solver_options("PD")
    opts.sanitize_graph = True
    labels, lb, _, _ = rama_py.rama_cpu(i, j, costs, opts)
    assert len(labels) == 7
    assert labels[2] == labels[3] == labels[4] == -1
    assert all(labels[v] >= 0 for v in [0, 1, 5, 6])
    assert lb <= objective(labels, i, j, costs) + 1e-3


def test_generic_rama_dispatch():
    i, j, costs = random_graph(40, 150, 1)
    opts = rama_py.multicut_solver_options("PD")
    cpu = rama_py.rama_cpu(i, j, costs, opts)
    generic = rama_py.rama(i, j, costs, opts, device="cpu")
    assert cpu[0] == generic[0] and cpu[1] == generic[1]
    labels, _, _, _ = rama_py.rama(i, j, costs)  # default options and device="auto"
    assert len(labels) == 40


def test_invalid_input():
    for args, kwargs in [(([0], [1, 2], [1.0]), {}),
                         (([], [], []), {}),
                         (([0], [1], [1.0]), {"device": "tpu"})]:
        try:
            rama_py.rama(*args, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {args} {kwargs}")


def test_cuda_unavailable_without_cuda_build():
    if rama_py.with_cuda:
        return
    try:
        rama_py.rama_cuda([0], [1], [1.0])
    except RuntimeError:
        return
    raise AssertionError("rama_cuda should raise when built without CUDA")


# --- GPU ---

def gpu_tests_enabled():
    if not rama_py.with_cuda:
        return False
    try:
        rama_py.rama_cuda([0], [1], [1.0])
        return True
    except RuntimeError:
        return False


def test_gpu_solution_is_consistent():
    if not gpu_tests_enabled():
        return
    for seed in range(3):
        i, j, costs = random_graph(50, 200, seed)
        opts = rama_py.multicut_solver_options("PD")
        check_solution(rama_py.rama_cuda, 50, i, j, costs, opts)
        check_solution(lambda *a: rama_py.rama(*a, device="cuda"), 50, i, j, costs, opts)


def test_gpu_and_cpu_agree_on_bounds():
    if not gpu_tests_enabled():
        return
    for seed in range(5):
        i, j, costs = random_graph(7, 15, seed)
        opt = brute_force_optimum(7, i, j, costs)
        for solve in [rama_py.rama_cpu, rama_py.rama_cuda]:
            _, lb, obj = check_solution(solve, 7, i, j, costs, rama_py.multicut_solver_options("PD"))
            assert lb <= opt + 1e-3 and opt <= obj + 1e-3, (solve.__name__, lb, opt, obj)


if __name__ == "__main__":
    tests = [(name, f) for name, f in globals().items() if name.startswith("test_") and callable(f)]
    print(f"rama_py with_cuda={rama_py.with_cuda}, gpu tests enabled={gpu_tests_enabled()}")
    for name, f in tests:  # definition order: CPU tests first, then GPU
        f()
        print(f"{name}: OK")
    sys.exit(0)
