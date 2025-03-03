import rama_py

A = rama_py.dCOO()

solver = rama_py.MulticutMessagePassing(A, [1, 2], [2, 3], [3, 1], True)

t1, t2, t3 = solver.get_triangles()
print("Dreiecke:", t1, t2, t3)

multipliers = solver.get_lagrange_multipliers()
print("Lagrange Multiplikatoren:", multipliers)

i, j, costs = solver.get_edges()
print("Kanten:", i, j)
print("Kosten:", costs)


