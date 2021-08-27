#! /usr/bin/env python2
import sys

edges = []
with open(sys.argv[1]) as fh:
    fh.readline()
    for line in fh.readlines():
        u, v, w = line.strip().split()
        u = int(u)
        v = int(v)
        w = float(w)
        edges.append((u, v, w))

weight = 0.
vertices = set()
with open(sys.argv[2]) as fh:
    fh.readline()
    for line in fh.readlines():
        i = int(line.strip())
        weight += edges[i][2]
        vertices.add(edges[i][0])
        vertices.add(edges[i][1])

print weight
print len(vertices)

