import rama_py

i = [0, 1, 2]
j = [1, 2, 3]
costs = [1.5, -2.3, 0.8]

opts = rama_py.multicut_solver_options("PD")

node_mapping, lower_bound, duration, timeline = rama_py.rama_cuda(i, j, costs, opts)

print("Node Mapping:", node_mapping)
print("Lower Bound:", lower_bound)
print("Compute Duration:", duration, "ms")