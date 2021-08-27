#include <iostream>
#include <algorithm>
#include <cstdint>

#include "graph.cuh"
#include "gettime.h"
#include "gpuMST.h"
// #include "MST.h"
// #include "parallel.h"
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sequence.h>

const int BlockSize = 256;

namespace MST_boruvka
{
__global__
void init_Edges(wghEdge<int> *input, int size, int *u, int *v, float *w, int *id) {
  const int pos = threadIdx.x + blockIdx.x * blockDim.x;
  if (pos < size) {
    wghEdge<int> e = input[pos];
    u[pos] = e.u;
    v[pos] = e.v;
    w[pos] = e.weight;
    id[pos] = pos;

    u[pos+size] = e.v;
    v[pos+size] = e.u;
    w[pos+size] = e.weight;
    id[pos+size] = pos;
  }
}


struct Edges {
  thrust::device_vector<int> u;
  thrust::device_vector<int> v;
  thrust::device_vector<int> id;
  thrust::device_vector<float> w;

  int n_edges = 0;
  int n_vertices = 0;

  Edges() { }

  Edges(const thrust::device_vector<int>& _u, const thrust::device_vector<int>& _v, const thrust::device_vector<float>& _costs) :
  u(2 * _u.size()), v(2 * _v.size()), id(2 * _u.size()), w(2 * _costs.size()), n_edges(2 * _costs.size())
  {
    assert(_u.size() == _v.size());
    assert(_u.size() == _costs.size());

    thrust::copy(_u.begin(), _u.end(), u.begin());
    thrust::copy(_u.begin(), _u.end(), v.begin() + _u.size());

    thrust::copy(_v.begin(), _v.end(), v.begin());
    thrust::copy(_v.begin(), _v.end(), u.begin() + _v.size());

    thrust::copy(_costs.begin(), _costs.end(), w.begin());
    thrust::copy(_costs.begin(), _costs.end(), w.begin() + _costs.size());

    n_edges = w.size();
    id = thrust::device_vector<int>(n_edges);
    thrust::sequence(id.begin(), id.begin() + _costs.size());
    thrust::sequence(id.begin() + _costs.size(), id.end());

    n_vertices = *thrust::max_element(u.begin(), u.end()) + 1;
    assert(n_vertices == *thrust::max_element(v.begin(), v.end()) + 1); // should be undirected graph.
  }

  Edges(const wghEdgeArray<int>& G) :
    u(G.m*2), v(G.m*2), id(G.m*2), w(G.m*2), n_edges(G.m*2), n_vertices(G.n) { 
    thrust::device_vector<wghEdge<int>> E(G.E, G.E + G.m);

    init_Edges<<<(G.m + BlockSize - 1) / BlockSize, BlockSize>>>
      (thrust::raw_pointer_cast(E.data()), G.m, 
       thrust::raw_pointer_cast(u.data()),
       thrust::raw_pointer_cast(v.data()),
       thrust::raw_pointer_cast(w.data()),
       thrust::raw_pointer_cast(id.data()));
  }

  Edges(int m, int n) : u(m), v(m), id(m), w(m), n_edges(m), n_vertices(n) { }
};


template<typename T>
void print_vector(const T& vec, std::string text) {
  std::cout << text << std::endl;
  for (size_t i = 0; i < vec.size() && i < 100; ++i) {
    std::cout << " " << vec[i];
  }
  std::cout << std::endl;
}

//--------------------------------------------------------------------------------
// kernels for mst
//--------------------------------------------------------------------------------
__global__
void remove_circles(int *input, size_t size, int* output, int *aux)
{
  const uint32_t pos = threadIdx.x + blockIdx.x * blockDim.x;
  if (pos < size) {
    int successor   = input[pos];
    int s_successor = input[successor];

    successor = ((successor > pos) && (s_successor == pos)) ? pos : successor;
    //if ((successor > pos) && (s_successor == pos)) {
    //  successor = pos;
    //}
    aux[pos] = (successor != pos);
    output[pos] = successor;
  }
}

__global__
void merge_vertices(int *successors, size_t size)
{
  const uint32_t pos = threadIdx.x + blockIdx.x * blockDim.x;

  if (pos < size) {
    bool goon = true;
    int i = 0;

    while (goon && (i++ < 50)) {
      int successor = successors[pos];
      int ssuccessor= successors[successor];
      __syncthreads();

      if (ssuccessor != successor) {
        successors[pos] = ssuccessor;
      }
      goon = __any(ssuccessor != successor);
      __syncthreads();
    }
  }
}

__global__
void mark_segments(int *input, int *output, size_t size)
{
  const uint32_t pos = threadIdx.x + blockIdx.x * blockDim.x;

  if (pos < size) {
    output[pos] = ((pos == size-1) || (input[pos] != input[pos+1]));
  }
}

__global__
void mark_edges_to_keep(
    const int *u, const int *v,
    int *new_vertices, int *output, size_t size)
{
  const uint32_t pos = threadIdx.x + blockIdx.x * blockDim.x;

  if (pos < size) {
    // true means the edge will be kept
    output[pos] = (new_vertices[u[pos]] != new_vertices[v[pos]]);
  }
}

__global__
void update_edges_with_new_vertices(
    int *u, int *v, int *new_vertices, size_t size)
{
  const uint32_t pos = threadIdx.x + blockIdx.x * blockDim.x;

  if (pos < size) {
    u[pos] = new_vertices[u[pos]];
    v[pos] = new_vertices[v[pos]];
  }
}

//--------------------------------------------------------------------------------
// functors
//--------------------------------------------------------------------------------
__host__ __device__ bool operator< (const int2& a, const int2& b) {
    return (a.x == b.x) ? (a.y < b.y) : (a.x < b.x);
};

struct binop_tuple_minimum {
  typedef thrust::tuple<float, int, int> T; // (w, v, id)
  __host__ __device__ 
  T operator() (const T& a, const T& b) const {
    return (thrust::get<0>(a) == thrust::get<0>(b)) ? 
      ((thrust::get<1>(a) < thrust::get<1>(b)) ? a : b) :
      ((thrust::get<0>(a) < thrust::get<0>(b)) ? a : b);
  }
};

// --------------------------------------------------------------------------------
// GPU MST
// --------------------------------------------------------------------------------
void recursive_mst_loop(
    Edges& edges,
    thrust::device_vector<int>&    mst_edges,
    int &n_mst)
{
  size_t n_edges = edges.n_edges;
  size_t n_vertices = edges.n_vertices;

  thrust::device_vector<int> succ(n_vertices);
  thrust::device_vector<int> succ_id(n_vertices);
  thrust::device_vector<int> succ_indices(n_vertices);
  thrust::device_vector<int> succ_temp(n_vertices);

  thrust::device_vector<int>  indices(n_edges);
  thrust::device_vector<int>  flags(n_edges);
  Edges edges_temp(edges.n_edges, edges.n_vertices);

  while (1) {
    if (n_edges == 1) {
      mst_edges[n_mst++] = edges.id[0];
      return;
    }

    thrust::sequence(indices.begin(), indices.begin() + n_edges);
    thrust::sort_by_key(edges.u.begin(), edges.u.begin() + n_edges, indices.begin());

    thrust::gather(indices.begin(), indices.begin() + n_edges, 
        thrust::make_zip_iterator(thrust::make_tuple(
            edges.v.begin(), edges.w.begin(), edges.id.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(
            edges_temp.v.begin(), edges_temp.w.begin(), edges_temp.id.begin())));

    edges_temp.v.swap(edges.v);
    edges_temp.w.swap(edges.w);
    edges_temp.id.swap(edges.id);

    auto new_last = thrust::reduce_by_key(
        edges.u.begin(), edges.u.begin() + n_edges,
        thrust::make_zip_iterator(thrust::make_tuple(
            edges.w.begin(), edges.v.begin(), edges.id.begin())),
        edges_temp.u.begin(),
        thrust::make_zip_iterator(thrust::make_tuple(
            edges_temp.w.begin(), edges_temp.v.begin(), edges_temp.id.begin())),
        thrust::equal_to<int>(),
        binop_tuple_minimum());

    size_t n_min_edges = new_last.first - edges_temp.u.begin();

    //std::cout << "n_min_edges: " << n_min_edges << endl;

    thrust::sequence(succ_indices.begin(), succ_indices.begin() + n_vertices);
    thrust::scatter(
        thrust::make_zip_iterator(thrust::make_tuple(
            edges_temp.v.begin(), edges_temp.id.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(
            edges_temp.v.begin() + n_min_edges, edges_temp.id.begin() + n_min_edges)),
        edges_temp.u.begin(),
        thrust::make_zip_iterator(thrust::make_tuple(
            succ_indices.begin(), succ_id.begin())));

    // succ_tmp stores which succ are to be saved (1)/ dumped
    remove_circles<<<(n_vertices + BlockSize - 1) / BlockSize, BlockSize>>>
      (thrust::raw_pointer_cast(succ_indices.data()), n_vertices,
       thrust::raw_pointer_cast(succ.data()),
       thrust::raw_pointer_cast(succ_temp.data()));

    thrust::exclusive_scan(succ_temp.begin(), succ_temp.begin() + n_vertices, 
        succ_indices.begin());
    // save new mst edges
    thrust::scatter_if(succ_id.begin(), succ_id.begin() + n_vertices,
        succ_indices.begin(), succ_temp.begin(), mst_edges.begin() + n_mst);

    n_mst += succ_indices[n_vertices-1] + succ_temp[n_vertices-1];

    //std::cout << "n_mst: " << n_mst << endl;

    // generating super vertices (new vertices)
    thrust::sequence(succ_indices.begin(), succ_indices.begin() + n_vertices);
    merge_vertices<<<(n_vertices + BlockSize - 1) / BlockSize, BlockSize>>>
      (thrust::raw_pointer_cast(succ.data()), n_vertices);

    thrust::sort_by_key(succ.begin(), succ.begin() + n_vertices, succ_indices.begin());

    mark_segments<<<(n_vertices + BlockSize - 1) / BlockSize, BlockSize>>>
      (thrust::raw_pointer_cast(succ.data()),
       thrust::raw_pointer_cast(succ_temp.data()), n_vertices);

    // new_vertices stored for subsequent calls to do query about next-vertice id
    thrust::device_vector<int>& new_vertices = succ;
    thrust::exclusive_scan(succ_temp.begin(), succ_temp.begin() + n_vertices, 
        succ_id.begin());
    thrust::scatter(succ_id.begin(), succ_id.begin() + n_vertices, 
        succ_indices.begin(), new_vertices.begin());

    int new_vertice_size = succ_id[n_vertices-1] + succ_temp[n_vertices-1];

    //std::cout << "new_vertice_size: " << new_vertice_size << endl;

    // generating new edges
    mark_edges_to_keep<<<(n_edges + BlockSize - 1) / BlockSize, BlockSize>>>
      (thrust::raw_pointer_cast(edges.u.data()),
       thrust::raw_pointer_cast(edges.v.data()),
       thrust::raw_pointer_cast(new_vertices.data()),
       thrust::raw_pointer_cast(flags.data()), n_edges);
    thrust::exclusive_scan(flags.begin(), flags.begin() + n_edges, 
        indices.begin());

    int new_edge_size = indices[n_edges-1] + flags[n_edges-1];
    if (!new_edge_size) { return; }

    //std::cout << "new_edge_size: " << new_edge_size << endl;

    thrust::scatter_if(
        thrust::make_zip_iterator(thrust::make_tuple(
            edges.u.begin(), edges.v.begin(), edges.w.begin(), edges.id.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(
            edges.u.begin() + n_edges, edges.v.begin() + n_edges, edges.w.begin() + n_edges, edges.id.begin() + n_edges)),
        indices.begin(), flags.begin(), 
        thrust::make_zip_iterator(thrust::make_tuple(
            edges_temp.u.begin(), edges_temp.v.begin(), edges_temp.w.begin(), edges_temp.id.begin()))
        );

    update_edges_with_new_vertices<<<(new_edge_size + BlockSize - 1) / BlockSize, BlockSize>>>
      (thrust::raw_pointer_cast(edges_temp.v.data()),
       thrust::raw_pointer_cast(edges_temp.u.data()),
       thrust::raw_pointer_cast(new_vertices.data()), new_edge_size);

    edges.u.swap(edges_temp.u);
    edges.v.swap(edges_temp.v);
    edges.w.swap(edges_temp.w);
    edges.id.swap(edges_temp.id);

    n_vertices = new_vertice_size;
    n_edges = new_edge_size;
  }
}

//--------------------------------------------------------------------------------
// top level mst
//--------------------------------------------------------------------------------
std::pair<int*,int> mst(wghEdgeArray<int> G)
{
  startTime();

  Edges edges(G);
  thrust::device_vector<int> mst_edges(G.m);

  nextTime("prepare graph");

  int mst_size = 0;
  recursive_mst_loop(edges, mst_edges, mst_size);

  int *result_mst_edges = new int[mst_size];
  cudaMemcpy(result_mst_edges, thrust::raw_pointer_cast(mst_edges.data()),
      sizeof(int) * mst_size, cudaMemcpyDeviceToHost);

  return std::make_pair(result_mst_edges, mst_size);
}

// Input should be directed graph.
std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<float>> maximum_spanning_tree(const thrust::device_vector<int>& i, const thrust::device_vector<int>& j, const thrust::device_vector<float>& neg_costs)
{
  Edges edges(i, j, neg_costs);
  thrust::device_vector<int> mst_edges(edges.n_edges);

  // Invert costs to find maximum spanning tree.
  thrust::transform(edges.w.begin(), edges.w.end(), edges.w.begin(), thrust::negate<float>());
  
  int mst_size = 0;
  recursive_mst_loop(edges, mst_edges, mst_size);
  
  thrust::sort(mst_edges.begin(), mst_edges.begin() + mst_size);
  auto last_unique = thrust::unique(mst_edges.begin(), mst_edges.begin() + mst_size);
  mst_edges.resize(std::distance(mst_edges.begin(), last_unique));

  thrust::device_vector<int> mst_i(mst_edges.size());
  thrust::device_vector<int> mst_j(mst_edges.size());
  thrust::device_vector<float> mst_values(mst_edges.size());
  
  thrust::gather(mst_edges.begin(), mst_edges.end(), i.begin(), mst_i.begin());
  thrust::gather(mst_edges.begin(), mst_edges.end(), j.begin(), mst_j.begin());
  thrust::gather(mst_edges.begin(), mst_edges.end(), neg_costs.begin(), mst_values.begin());

  return {mst_i, mst_j, mst_values};
}
}