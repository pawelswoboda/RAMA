import torch
import rama_py

opts = rama_py.multicut_solver_options("P")
opts.verbose=False
res = rama_py.rama_cuda([0, 1, 2], [1, 2, 0], [1.1, -2, 3], opts)
expected = res[0]

i = torch.Tensor([0, 1, 2]).to('cuda').to(torch.int32)
j = torch.Tensor([1, 2, 0]).to('cuda').to(torch.int32)
costs = torch.Tensor([1.1, -2, 3]).to('cuda').to(torch.float32)

node_labels = torch.ones((3), device = i.device).to(torch.int32)
rama_py.rama_cuda_gpu_pointers(i.data_ptr(), j.data_ptr(), costs.data_ptr(), node_labels.data_ptr(), 3, 3, i.device.index, opts)
assert(torch.all(node_labels.cpu() == torch.Tensor(expected)))
