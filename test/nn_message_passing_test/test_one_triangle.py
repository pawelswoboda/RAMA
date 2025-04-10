import rama_py

opts = rama_py.multicut_solver_options("PD")
opts.verbose = False
i = [0, 0, 1]
j = [1, 2, 2]
costs = [1.5, -2.3, 0.8]
_, lb, _, _ = rama_py.rama_cuda(i, j, costs, opts) #node_mapping, lower_bound, duration, timeline
print(lb)
assert(-1.51 <= lb <= -1.49)




