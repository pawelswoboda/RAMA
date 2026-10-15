# Checks that the CPU (rama_py_cpu) and GPU (rama_py) Python interfaces give
# identical results. Needs a GPU. Optionally pass multicut .txt instances as
# command line arguments to compare on them as well.
import random
import sys

import rama_py
import rama_py_cpu

try:
    import torch
except ImportError:
    torch = None


def options(module, mode):
    opts = module.multicut_solver_options(mode)
    opts.verbose = False
    return opts


def random_graph(rng, n, avg_degree=4.0):
    # Sparse graphs: dense ones produce so many conflicted cycles that the
    # sequential CPU backend takes tens of seconds in dual mode.
    p = min(1.0, avg_degree / n)
    edges = {(a, b) for a in range(n) for b in range(a + 1, n) if rng.random() < p}
    edges = sorted(edges | {(k, k + 1) for k in range(n - 1)})
    i = [a for a, _ in edges]
    j = [b for _, b in edges]
    costs = [rng.uniform(-3, 3) for _ in edges]
    return i, j, costs


def compare(i, j, costs, mode, sanitize=False, name=""):
    opts_cpu = options(rama_py_cpu, mode)
    opts_gpu = options(rama_py, mode)
    opts_cpu.sanitize_graph = opts_gpu.sanitize_graph = sanitize
    labels_cpu, lb_cpu, _, _ = rama_py_cpu.rama_cpu(i, j, costs, opts_cpu)
    labels_gpu, lb_gpu, _, _ = rama_py.rama_cuda(i, j, costs, opts_gpu)
    assert labels_cpu == labels_gpu, (name, mode)
    assert abs(lb_cpu - lb_gpu) <= 1e-4 * max(1.0, abs(lb_gpu)), (name, mode, lb_cpu, lb_gpu)


def test_options_across_modules():
    # Both modules register multicut_solver_options as module-local; pybind11
    # still converts between them since they wrap the same C++ type.
    tri = ([0, 1, 2], [1, 2, 0], [1.1, -2.0, 3.0])
    expected, _, _, _ = rama_py_cpu.rama_cpu(*tri, options(rama_py_cpu, "PD"))
    labels, _, _, _ = rama_py_cpu.rama_cpu(*tri, options(rama_py, "PD"))
    assert labels == expected
    labels, _, _, _ = rama_py.rama_cuda(*tri, options(rama_py_cpu, "PD"))
    assert labels == expected


def test_random_graphs():
    rng = random.Random(1)
    for trial in range(20):
        i, j, costs = random_graph(rng, rng.randint(5, 500))
        for mode in ["P", "PD", "D"]:
            compare(i, j, costs, mode, name=f"random {trial}")


def test_sanitize_graph():
    i = [0, 1, 2, 4, 1, 7]
    j = [1, 2, 0, 2, 0, 4]
    costs = [1.1, -2.0, 3.0, 1.0, 1.1, -0.5]
    compare(i, j, costs, "PD", sanitize=True, name="sanitize")


def test_torch():
    if torch is None or not hasattr(rama_py, "rama_torch") or not hasattr(rama_py_cpu, "rama_torch"):
        print("torch interfaces not available, skipping test_torch")
        return
    rng = random.Random(2)
    i, j, costs = random_graph(rng, 300)
    ti = torch.tensor(i, dtype=torch.int32)
    tj = torch.tensor(j, dtype=torch.int32)
    tc = torch.tensor(costs, dtype=torch.float32)
    labels_cpu, lb_cpu = rama_py_cpu.rama_torch(ti, tj, tc, options(rama_py_cpu, "PD"))
    labels_gpu, lb_gpu = rama_py.rama_torch(ti.cuda(), tj.cuda(), tc.cuda(), options(rama_py, "PD"))
    assert labels_gpu.is_cuda and not labels_cpu.is_cuda
    assert torch.equal(labels_cpu, labels_gpu.cpu())
    assert lb_cpu.dtype == lb_gpu.dtype == torch.float64
    assert abs(lb_cpu.item() - lb_gpu.item()) <= 1e-4 * max(1.0, abs(lb_gpu.item()))


def test_pointers():
    if torch is None:
        print("torch not available, skipping test_pointers")
        return
    rng = random.Random(3)
    i, j, costs = random_graph(rng, 300)
    num_nodes = max(max(i), max(j)) + 1
    opts_cpu = options(rama_py_cpu, "PD")
    opts_gpu = options(rama_py, "PD")
    tensors_cpu = [torch.tensor(i, dtype=torch.int32), torch.tensor(j, dtype=torch.int32),
                   torch.tensor(costs, dtype=torch.float32)]
    tensors_gpu = [t.cuda() for t in tensors_cpu]
    labels_cpu = torch.empty(num_nodes, dtype=torch.int32)
    labels_gpu = torch.empty(num_nodes, dtype=torch.int32, device="cuda")
    rama_py_cpu.rama_cpu_pointers(*[t.data_ptr() for t in tensors_cpu], labels_cpu.data_ptr(),
                                  num_nodes, len(i), opts_cpu)
    rama_py.rama_cuda_gpu_pointers(*[t.data_ptr() for t in tensors_gpu], labels_gpu.data_ptr(),
                                   num_nodes, len(i), labels_gpu.device.index, opts_gpu)
    assert torch.equal(labels_cpu, labels_gpu.cpu())


def test_instances(paths):
    for path in paths:
        i, j, costs, li, _, _ = rama_py_cpu.read_multicut_file(path)
        assert not li, "lifted instances are not supported by this comparison"
        for mode in ["P", "PD"]:
            compare(i, j, costs, mode, name=path)
        print(f"  {path}: CPU and GPU agree")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and name != "test_instances" and callable(fn):
            fn()
            print(f"{name} passed")
    test_instances(sys.argv[1:])
