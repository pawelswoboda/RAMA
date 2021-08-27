#include <iostream>
#include <algorithm>
#include <cstdint>

#include "graph.cuh"
#include "gettime.h"
// #include "MST.h"
// #include "parallel.h"
#include <thrust/gather.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>

const int BlockSize = 256;

using namespace std;

namespace MST_kruskal
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

struct UndirectedEdges {
  thrust::device_vector<int> s;
  thrust::device_vector<int> t;
  thrust::device_vector<int> id; // this stores the id marked after split_graph
  thrust::device_vector<float> w;
  thrust::device_vector<int> result_id; // this stores the id for final result (the original id in the input file)

  int n_edges;
  int n_vertices;

  struct init_operator {
    typedef thrust::tuple<int, int, int, float> Tuple;

    __host__ __device__
    Tuple operator() (const wghEdge<int>& edge, const int idx) {
      return thrust::make_tuple(edge.u, edge.v, idx, edge.weight);
    }
  };

  UndirectedEdges() {}

  UndirectedEdges(int m, int n) : 
    s(m), t(m), id(m), w(m), result_id(m), n_edges(m), n_vertices(n) {}

  UndirectedEdges(const wghEdgeArray<int>& G):
    s(G.m), t(G.m), id(G.m), w(G.m), result_id(G.m), n_edges(G.m), n_vertices(G.n) { 
    thrust::device_vector<wghEdge<int>> E(G.E, G.E + G.m);
    thrust::transform(
        E.begin(), E.end(), thrust::make_counting_iterator(0), 
        thrust::make_zip_iterator(thrust::make_tuple(
            s.begin(), t.begin(), result_id.begin(), w.begin())),
        init_operator());
  }

  UndirectedEdges(const thrust::device_vector<int>& _u, const thrust::device_vector<int>& _v, const thrust::device_vector<float>& _w) 
  {
    assert(_u.size() == _v.size());
    assert(_u.size() == _w.size());
    s = thrust::device_vector<int>(_u.begin(), _u.end());
    t = thrust::device_vector<int>(_v.begin(), _v.end());
    w = thrust::device_vector<int>(_w.begin(), _w.end());
    id = thrust::device_vector<int>(_u.size());
    result_id = thrust::device_vector<int>(_u.size());
    thrust::sequence(result_id.begin(), result_id.end());
    n_edges = _u.size();
    n_vertices = *thrust::max_element(s.begin(), s.end()) + 1;
    assert(n_vertices == *thrust::max_element(t.begin(), t.end()) + 1); // should be undirected graph.
  }
};

struct Edges {
  thrust::device_vector<int> u;
  thrust::device_vector<int> v;
  thrust::device_vector<int> id;
  thrust::device_vector<float> w;

  int n_edges;
  int n_vertices;

  Edges() { }

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
void print_vector(const T& vec, string text, uint32_t size=100) {
  cout << text << endl;
  for (size_t i = 0; i < vec.size() && i < size; ++i) {
    cout << " " << vec[i];
  }
  cout << endl;
}

//--------------------------------------------------------------------------------
// kernels for mst
//--------------------------------------------------------------------------------
__global__
void remove_circles(int *input, size_t size, int* id, int* output, int *aux)
{
  const uint32_t pos = threadIdx.x + blockIdx.x * blockDim.x;
  if (pos < size) {
    int successor   = input[pos];
    int s_successor = input[successor];

    successor = ((successor > pos) && (s_successor == pos)) ? pos : successor;
    //if ((successor > pos) && (s_successor == pos)) {
    //  successor = pos;
    //}
    output[pos] = successor;
    if (aux) {
      aux[pos] = (successor != pos) && (id[pos] >= 0);
    }
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
    const int *new_vertices, int *output, size_t size)
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

//--------------------------------------------------------------------------------
// GPU MST
//--------------------------------------------------------------------------------

vector<pair<uint32_t, uint32_t>> split_graph(UndirectedEdges& edges)
{
  vector<pair<uint32_t, uint32_t>> result;

  UndirectedEdges edges_temp(edges.n_edges, edges.n_vertices);
  thrust::device_vector<int>  indices(edges.n_edges);

  thrust::sequence(indices.begin(), indices.begin() + edges.n_edges);
  thrust::sort_by_key(edges.w.begin(), edges.w.end(), indices.begin());
  thrust::gather(indices.begin(), indices.end(),
      thrust::make_zip_iterator(thrust::make_tuple(
          edges.s.begin(), edges.t.begin(), edges.result_id.begin())),
      thrust::make_zip_iterator(thrust::make_tuple(
          edges_temp.s.begin(), edges_temp.t.begin(), edges_temp.result_id.begin())));

  thrust::sequence(edges.id.begin(), edges.id.end());
  edges_temp.s.swap(edges.s);
  edges_temp.t.swap(edges.t);
  edges_temp.result_id.swap(edges.result_id);

  for (uint32_t i = 0, k = 1; i < edges.n_edges; ++k) {
    uint32_t size_k = min(int(pow(2, k-1)) * edges.n_vertices, edges.n_edges - i);
    result.push_back(make_pair(i, i+size_k));
    i += size_k;
  }
  return result;
}

void contract_and_build_subgraph(
    uint32_t begin, uint32_t end, 
    UndirectedEdges& edges, 
    const thrust::device_vector<int>& supervertices,
    Edges& output)
{
  uint32_t size = end - begin;

  thrust::gather(edges.s.begin()+begin, edges.s.begin()+end,
      supervertices.begin(),
      edges.s.begin()+begin);
  thrust::gather(edges.t.begin()+begin, edges.t.begin()+end,
      supervertices.begin(),
      edges.t.begin()+begin);

  // build subgraph in directed edge list style
  thrust::device_vector<int> flags(size, 0); 
  thrust::device_vector<int> indices(size); 

  mark_edges_to_keep<<<(size + BlockSize - 1) / BlockSize, BlockSize>>>
    (thrust::raw_pointer_cast(edges.s.data()) + begin,
     thrust::raw_pointer_cast(edges.t.data()) + begin,
     thrust::raw_pointer_cast(supervertices.data()),
     thrust::raw_pointer_cast(flags.data()), size);
  thrust::exclusive_scan(flags.begin(), flags.begin() + size, indices.begin());

  size = flags[size-1] + indices[size-1];
  output.u.resize(size*2);
  output.v.resize(size*2);
  output.w.resize(size*2);
  output.id.resize(size*2);
  output.n_edges = size*2;

  // parallel filtering edges
  thrust::scatter_if(
      thrust::make_zip_iterator(thrust::make_tuple(
          edges.s.begin()+begin, edges.t.begin()+begin, edges.w.begin()+begin, edges.id.begin()+begin)),
      thrust::make_zip_iterator(thrust::make_tuple(
          edges.s.begin()+end, edges.t.begin()+end, edges.w.begin()+end, edges.id.begin()+end)),
      indices.begin(), flags.begin(),
      thrust::make_zip_iterator(thrust::make_tuple(
          output.u.begin(), output.v.begin(), output.w.begin(), output.id.begin()))
      );

  thrust::copy(
      thrust::make_zip_iterator(thrust::make_tuple(
          output.u.begin(), output.v.begin(), output.w.begin(), output.id.begin())),
      thrust::make_zip_iterator(thrust::make_tuple(
          output.u.begin()+size, output.v.begin()+size, output.w.begin()+size, output.id.begin()+size)),
      thrust::make_zip_iterator(thrust::make_tuple(
          output.v.begin()+size, output.u.begin()+size, output.w.begin()+size, output.id.begin()+size))
      );
}

void boruvka_mst(
    Edges& edges,
    thrust::device_vector<int>& supervertices,
    thrust::device_vector<int>& mst_edges,
    int &n_mst)
{
  if (!edges.n_edges) return;

  assert(supervertices.size() == edges.n_vertices);

  size_t n_edges = edges.n_edges;
  size_t n_vertices = edges.n_vertices;

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

    // succ_indices is temporary for succ, the array of successors
    thrust::fill(succ_id.begin(), succ_id.end(), -1);
    thrust::scatter(
        thrust::make_zip_iterator(thrust::make_tuple(
            edges_temp.v.begin(), edges_temp.id.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(
            edges_temp.v.begin() + n_min_edges, edges_temp.id.begin() + n_min_edges)),
        edges_temp.u.begin(),
        thrust::make_zip_iterator(thrust::make_tuple(
            supervertices.begin(), succ_id.begin())));

    // succ_tmp stores which succ are to be saved (1)/ dumped
    // after this, succ_indices stores the array of successors
    // since we maintain a global supervertices, we need to use succ_id to know which vertices represents the newly generated mst edges
    remove_circles<<<(edges.n_vertices + BlockSize - 1) / BlockSize, BlockSize>>>
      (thrust::raw_pointer_cast(supervertices.data()), edges.n_vertices,
       thrust::raw_pointer_cast(succ_id.data()),
       thrust::raw_pointer_cast(succ_indices.data()),
       thrust::raw_pointer_cast(succ_temp.data()));
    supervertices.swap(succ_indices);

    thrust::exclusive_scan(succ_temp.begin(), succ_temp.begin() + n_vertices, 
        succ_indices.begin());
    // save new mst edges
    thrust::scatter_if(succ_id.begin(), succ_id.begin() + n_vertices,
        succ_indices.begin(), succ_temp.begin(), mst_edges.begin() + n_mst);

    n_mst += succ_indices[n_vertices-1] + succ_temp[n_vertices-1];

    // generating super vertices (new vertices)
    //thrust::sequence(succ_indices.begin(), succ_indices.begin() + n_vertices);
    merge_vertices<<<(edges.n_vertices + BlockSize - 1) / BlockSize, BlockSize>>>
      (thrust::raw_pointer_cast(supervertices.data()), edges.n_vertices);

    // generating new edges
    mark_edges_to_keep<<<(n_edges + BlockSize - 1) / BlockSize, BlockSize>>>
      (thrust::raw_pointer_cast(edges.u.data()),
       thrust::raw_pointer_cast(edges.v.data()),
       thrust::raw_pointer_cast(supervertices.data()),
       thrust::raw_pointer_cast(flags.data()), n_edges);
    thrust::exclusive_scan(flags.begin(), flags.begin() + n_edges, 
        indices.begin());

    int new_edge_size = indices[n_edges-1] + flags[n_edges-1];
    if (!new_edge_size) { return; }

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
       thrust::raw_pointer_cast(supervertices.data()), new_edge_size);

    edges.u.swap(edges_temp.u);
    edges.v.swap(edges_temp.v);
    edges.w.swap(edges_temp.w);
    edges.id.swap(edges_temp.id);

    assert(n_edges != new_edge_size);
    n_edges = new_edge_size;
  }
}

//--------------------------------------------------------------------------------
// top level mst
//--------------------------------------------------------------------------------
std::pair<int*,int> mst(wghEdgeArray<int> G)
{
  startTime();

  UndirectedEdges edges(G);

  nextTime("prepare graph");

  Edges subgraph;
  subgraph.n_vertices = G.n;

  thrust::device_vector<int> supervertices(G.n);
  thrust::device_vector<int> mst_edges(G.m);
  int n_mst = 0;

  thrust::sequence(supervertices.begin(), supervertices.end());

  auto split_indices = split_graph(edges);
  for (auto it = split_indices.begin(); ; ) {
    contract_and_build_subgraph(
        it->first, it->second, edges, supervertices, 
        subgraph);

    // this step, contrary to the paper, also includes connect components, by update the global super vertices
    boruvka_mst(subgraph, supervertices, mst_edges, n_mst);

    if (split_indices.end() == (++it)) break;
  }

  // fetch result ids, stored to edges.id temporarily
  thrust::gather(mst_edges.begin(), mst_edges.begin() + n_mst,
      edges.result_id.begin(), edges.id.begin());

  int *result_mst_edges = new int[n_mst];
  thrust::copy(edges.id.begin(), edges.id.begin() + n_mst, result_mst_edges);

  return make_pair(result_mst_edges, n_mst);
}

thrust::device_vector<int> maximum_spanning_tree(const thrust::device_vector<int>& i, const thrust::device_vector<int>& j, const thrust::device_vector<float>& neg_costs)
{
  UndirectedEdges edges(i, j, neg_costs);

  // Invert costs to find maximum spanning tree.
  thrust::transform(edges.w.begin(), edges.w.end(), edges.w.begin(), thrust::negate<float>());
  
  Edges subgraph;
  subgraph.n_vertices = edges.n_vertices;

  thrust::device_vector<int> supervertices(edges.n_vertices);
  thrust::device_vector<int> mst_edges(edges.n_edges);
  int n_mst = 0;

  thrust::sequence(supervertices.begin(), supervertices.end());

  auto split_indices = split_graph(edges);
  for (auto it = split_indices.begin(); ; ) {
    contract_and_build_subgraph(
        it->first, it->second, edges, supervertices, 
        subgraph);

    // this step, contrary to the paper, also includes connect components, by update the global super vertices
    boruvka_mst(subgraph, supervertices, mst_edges, n_mst);

    if (split_indices.end() == (++it)) break;
  }

  // fetch result ids, stored to edges.id temporarily
  thrust::gather(mst_edges.begin(), mst_edges.begin() + n_mst,
      edges.result_id.begin(), edges.id.begin());

  edges.id.resize(n_mst);
  // int *result_mst_edges = new int[n_mst];
  // thrust::copy(edges.id.begin(), edges.id.begin() + n_mst, result_mst_edges);

  return edges.id;
}
}