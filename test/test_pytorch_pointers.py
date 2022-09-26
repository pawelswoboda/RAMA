import torch
import rama_py

opts = rama_py.multicut_solver_options("P")
opts.verbose=False
res = rama_py.rama_cuda([0, 1, 2], [1, 2, 0], [1.1, -2, 3], opts)
expected = res[0]

i = torch.Tensor([0, 1, 2]).to('cuda').contiguous().to(torch.int32) # Can only be of dtype int32!
j = torch.Tensor([1, 2, 0]).to('cuda').contiguous().to(torch.int32) # Can only be of dtype int32!
costs = torch.Tensor([1.1, -2, 3]).to('cuda').contiguous().to(torch.float32) # Can only be of dtype float32!

num_nodes = max(i.max(), j.max()) + 1
node_labels = torch.ones(num_nodes, device = i.device).to(torch.int32)
num_edges = i.numel()
opts.dump_timeline = True # Set to true to get intermediate results.
timeline = rama_py.rama_cuda_gpu_pointers(i.data_ptr(), j.data_ptr(), costs.data_ptr(), node_labels.data_ptr(), num_nodes, num_edges, i.device.index, opts)
assert(torch.all(node_labels.cpu() == torch.Tensor(expected)))
