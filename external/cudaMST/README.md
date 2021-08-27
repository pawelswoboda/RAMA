CUDA MST
========

This repo implements following two papers using CUDA so as to accelerate MST computation:

1. *Chapter 7 Fast Minimum Spanning Tree Computation* of *GPU Computing Gems*, a data-parallel MST based on Boruvka's algorithm.
2. *Fast and Memory-Efficient Minimum Spanning Tree on the GPU*, a data-parallel Kruskal's MST algorithm, which lies between serial Kruskal's and parallel Boruvka's in terms of parallelism.

This is originally part of the [benchmark suite](http://www.cs.cmu.edu/~pbbs/benchmarks.html).
[Thrust](http://thrust.github.io/) is heavily used.

-----

The baselines are:
* serial Kruskal's algorithm with union find with weight and path-compression
* CPU multi-core parallel Kruskal's algorithm

The compare flows are **gpuMST** and **gpuMSTdpk** which are my implementations.

-----

For data-parallel Boruvka's, result shows that on a GTX750ti & E1231v2 machine, 
on-par performance is achieved on random graphs, whereas better performances can be observed with sparse (grid) graph and power-law graphs.

And this approach is quite memory costly -- I cannot go beyond 2 million vertices with 10 million edges for random graph with my 2GB GPU memory.

##### gpuMST result

    1 : randLocalGraph_WE_5_2000000 :  -r 1 -o /tmp/ofile765486_563367 : '0.759'
    1 : rMatGraph_WE_5_2000000 :  -r 1 -o /tmp/ofile138860_218538 : '0.655'
    1 : 2Dgrid_WE_2000000 :  -r 1 -o /tmp/ofile8903_852545 : '0.159'
    gpuMST : 0 : weighted time, min=0.524 median=0.524 mean=0.524


##### serialMST result

    1 : randLocalGraph_WE_5_2000000 :  -r 1 -o /tmp/ofile399034_715347 : '0.65'
    1 : rMatGraph_WE_5_2000000 :  -r 1 -o /tmp/ofile477678_439826 : '0.697'
    1 : 2Dgrid_WE_2000000 :  -r 1 -o /tmp/ofile983504_141272 : '0.395'
    serialMST : 0 : weighted time, min=0.58 median=0.58 mean=0.58

-----

For data-parallel Kruskal's, result is much better -- it is more memory efficient and faster.
With my 2GB GPU memory, I can push to 5 million vertices with 25 million edges (and maybe higher below 10 million vertices with 50 million edges, never tried...

Result shows that the performance is also better -- the speedup is more than 3x.
In particular, as usual, for sparse 2d grid graph, the speedup is the best, while now the response to random graph is a little bit better than power-law, just as the pure serial one does.

In general, I think the performance gain mentioned in the paper is very similar to what I have observed, compared to either data-parallel Boruvka's or serial Kruskal's.

##### gpuMSTdpk result

    1 : randLocalGraph_WE_5_5000000 :  -r 1 -o /tmp/ofile609269_741536 : '0.604'
    1 : rMatGraph_WE_5_5000000 :  -r 1 -o /tmp/ofile631862_452509 : '0.886'
    1 : 2Dgrid_WE_5000000 :  -r 1 -o /tmp/ofile972092_45516 : '0.363'
    gpuMSTdpk : 0 : weighted time, min=0.617 median=0.617 mean=0.617

##### serialMST result
    1 : randLocalGraph_WE_5_5000000 :  -r 1 -o /tmp/ofile514914_278972 : '1.99'
    1 : rMatGraph_WE_5_5000000 :  -r 1 -o /tmp/ofile295582_919992 : '2.54'
    1 : 2Dgrid_WE_5000000 :  -r 1 -o /tmp/ofile576909_787203 : '1.19'
    serialMST : 0 : weighted time, min=1.906 median=1.906 mean=1.906

-----

