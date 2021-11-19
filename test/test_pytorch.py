import torch
import rama_py

opts = rama_py.multicut_solver_options("P")
opts.verbose=False
res = rama_py.rama_cuda([0, 1, 2], [1, 2, 0], [1.1, -2, 3], opts)
expected = res[0]

i = torch.Tensor([0, 1, 2]).to('cuda').to(torch.int32)
j = torch.Tensor([1, 2, 0]).to('cuda').to(torch.int32)
costs = torch.Tensor([1.1, -2, 3]).to('cuda').to(torch.float32)
opts = rama_py.multicut_solver_options("PD")
opts.verbose=False
node_mapping, lb = rama_py.rama_torch(i, j, costs, opts)
assert(torch.all(node_mapping.cpu() == torch.Tensor(expected)))