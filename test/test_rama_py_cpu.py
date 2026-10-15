# Tests for the CPU Python interface (rama_py_cpu). Needs no GPU.
# Run with the directory containing rama_py_cpu*.so on PYTHONPATH.
import itertools
import os
import random
import tempfile

import rama_py_cpu

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None


def objective(labels, i, j, costs):
    return sum(c for a, b, c in zip(i, j, costs) if labels[a] != labels[b])


def set_partitions(n):
    # All labelings of n nodes in restricted growth form (one per partition).
    def rec(k, labels, num_labels):
        if k == n:
            yield list(labels)
            return
        for l in range(num_labels + 1):
            labels.append(l)
            yield from rec(k + 1, labels, max(num_labels, l + 1))
            labels.pop()
    yield from rec(0, [], 0)


def brute_force_optimum(n, i, j, costs):
    return min(objective(p, i, j, costs) for p in set_partitions(n))


def options(mode):
    opts = rama_py_cpu.multicut_solver_options(mode)
    opts.verbose = False
    return opts


# Triangle: cutting the repulsive edge (1,2) forces a second cut; the cheapest
# is (0,1), so node 1 is separated from {0, 2} with objective 1.1 - 2 = -0.9.
TRI_I, TRI_J, TRI_COSTS = [0, 1, 2], [1, 2, 0], [1.1, -2.0, 3.0]


def check_triangle_labels(labels):
    assert len(labels) == 3, labels
    assert labels[0] == labels[2] and labels[0] != labels[1], labels


def test_triangle():
    for mode in ["P", "PD", "PD+"]:
        labels, lb, time_ms, timeline = rama_py_cpu.rama_cpu(TRI_I, TRI_J, TRI_COSTS, options(mode))
        check_triangle_labels(labels)
        assert abs(objective(labels, TRI_I, TRI_J, TRI_COSTS) - (-0.9)) < 1e-5
        assert lb <= -0.9 + 1e-5, lb
        assert time_ms >= 0


def test_only_lower_bound():
    opts = options("D")
    labels, lb, _, _ = rama_py_cpu.rama_cpu(TRI_I, TRI_J, TRI_COSTS, opts)
    assert opts.only_compute_lb
    assert len(labels) == 0, labels
    # The triangle inequalities make the LP relaxation tight on a single triangle.
    assert abs(lb - (-0.9)) < 1e-4, lb


def test_options():
    opts = rama_py_cpu.multicut_solver_options()
    opts.sanitize_graph = True
    opts.max_time_sec = 10
    assert opts.sanitize_graph and opts.max_time_sec == 10
    assert isinstance(repr(opts), str) and len(repr(opts)) > 0


def test_random_graphs_against_brute_force():
    rng = random.Random(0)
    for trial in range(30):
        n = rng.randint(3, 8)
        edges = [(a, b) for a in range(n) for b in range(a + 1, n) if rng.random() < 0.6]
        # Path through all nodes so that every node index occurs in some edge.
        edges = sorted(set(edges) | {(k, k + 1) for k in range(n - 1)})
        i = [a for a, _ in edges]
        j = [b for _, b in edges]
        costs = [rng.uniform(-3, 3) for _ in edges]
        opt = brute_force_optimum(n, i, j, costs)
        for mode in ["P", "PD"]:
            labels, lb, _, _ = rama_py_cpu.rama_cpu(i, j, costs, options(mode))
            assert len(labels) == n
            obj = objective(labels, i, j, costs)
            assert obj >= opt - 1e-4, (trial, mode, obj, opt)
            assert lb <= opt + 1e-4, (trial, mode, lb, opt)


def test_sanitize_graph():
    # Node 3 has no edges, and edge (0,1) is given twice in both orientations.
    i = [0, 1, 2, 4, 1]
    j = [1, 2, 0, 2, 0]
    costs = [1.1, -2.0, 3.0, 1.0, 1.1]
    opts = options("PD")
    opts.sanitize_graph = True
    labels, _, _, _ = rama_py_cpu.rama_cpu(i, j, costs, opts)
    assert len(labels) == 5, labels
    assert labels[3] == -1, labels
    assert all(l >= 0 for k, l in enumerate(labels) if k != 3), labels


def test_pointers():
    if np is None:
        print("numpy not available, skipping test_pointers")
        return
    expected, _, _, _ = rama_py_cpu.rama_cpu(TRI_I, TRI_J, TRI_COSTS, options("P"))
    i = np.array(TRI_I, dtype=np.int32)
    j = np.array(TRI_J, dtype=np.int32)
    costs = np.array(TRI_COSTS, dtype=np.float32)
    num_nodes = int(max(i.max(), j.max())) + 1
    node_labels = np.full(num_nodes, -7, dtype=np.int32)
    opts = options("P")
    opts.dump_timeline = True
    timeline = rama_py_cpu.rama_cpu_pointers(i.ctypes.data, j.ctypes.data, costs.ctypes.data,
                                             node_labels.ctypes.data, num_nodes, len(i), opts)
    assert list(node_labels) == list(expected), (node_labels, expected)
    assert len(timeline) > 0
    assert all(len(t) == num_nodes for t in timeline)


def test_torch():
    if torch is None or not hasattr(rama_py_cpu, "rama_torch"):
        print("torch interface not available, skipping test_torch")
        return
    expected, expected_lb, _, _ = rama_py_cpu.rama_cpu(TRI_I, TRI_J, TRI_COSTS, options("PD"))
    i = torch.tensor(TRI_I, dtype=torch.int32)
    j = torch.tensor(TRI_J, dtype=torch.int32)
    costs = torch.tensor(TRI_COSTS, dtype=torch.float32)
    node_mapping, lb = rama_py_cpu.rama_torch(i, j, costs, options("PD"))
    assert node_mapping.device.type == "cpu" and node_mapping.dtype == torch.int32
    assert torch.equal(node_mapping, torch.tensor(expected, dtype=torch.int32))
    assert lb.dtype == torch.float64 and abs(lb.item() - expected_lb) < 1e-6, (lb, expected_lb)

    if torch.cuda.is_available():
        try:
            rama_py_cpu.rama_torch(i.cuda(), j.cuda(), costs.cuda(), options("PD"))
        except RuntimeError as e:
            assert "CPU tensor" in str(e), e
        else:
            raise AssertionError("CUDA tensors must be rejected by the CPU solver")


def test_read_multicut_file():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "triangle.txt")
        with open(path, "w") as f:
            f.write("MULTICUT\n")
            for a, b, c in zip(TRI_I, TRI_J, TRI_COSTS):
                f.write(f"{a} {b} {c}\n")
        i, j, costs, li, lj, lc = rama_py_cpu.read_multicut_file(path)
    assert (i, j) == (TRI_I, TRI_J)
    assert all(abs(a - b) < 1e-6 for a, b in zip(costs, TRI_COSTS))
    assert li == [] and lj == [] and lc == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name} passed")
