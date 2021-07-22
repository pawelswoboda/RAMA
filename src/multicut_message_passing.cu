#include "multicut_message_passing.h"
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <algorithm>
#include "parallel_gaec_utils.h"
#include "stdio.h"

__device__ int find_edge_index(const int i, const int j, const int* i_vec, const int * j_vec, const int* offsets)
{
    assert(i < j);
    assert(offsets[i] <= offsets[i+1]);
    if(offsets[i] == offsets[i+1])
        return -1;
    int lb = offsets[i];
    int ub = offsets[i+1]-1;
    if(i_vec[lb] != i)
        return -1;
    assert(lb <= ub);
    int middle = lb;

    while(lb != ub)
    {
        if(lb > ub)
            return -1;
        assert(lb <= ub);
        assert(i_vec[lb] == i);
        assert(i_vec[ub] == i);
        assert(j_vec[lb] <= j_vec[ub]);

        middle = (lb + ub)/2;

        if(j_vec[middle] == j)
            return middle;
        else if(j_vec[middle] < j)
            lb = middle + 1;
        else if(j_vec[middle] > j)
            ub = middle - 1;
    }

    return (lb + ub)/2;
} 

__device__ bool edge_present(const int i, const int j, const int* i_vec, const int * j_vec, const int* offsets)
{
    const int c = find_edge_index(i, j, i_vec, j_vec, offsets);
    if(i == -1)
        return false;
    return i_vec[c] == i && j_vec[c] == j;
}

struct edge_present_func {
    int* i;
    int* j;
    int* offsets;
    __device__ bool operator()(const thrust::tuple<int,int> t)
    {
        return edge_present(thrust::get<0>(t), thrust::get<1>(t), i, j, offsets); 
    }
};
struct edge_not_present_func {
    int* i;
    int* j;
    int* offsets;
    __device__ bool operator()(const thrust::tuple<int,int,float> t)
    {
        return !edge_present(thrust::get<0>(t), thrust::get<1>(t), i, j, offsets); 
    }
};

struct copy_edge_costs_func {
    const int* orig_i;
    const int* orig_j;
    const float* orig_edge_costs;
    const int* orig_offsets;
    
    __device__ void operator()(const thrust::tuple<int,int,float&> t)
    {
        const int i = thrust::get<0>(t);
        const int j = thrust::get<1>(t);
        assert(i < j);
        float& edge_cost = thrust::get<2>(t);
        if(!edge_present(i, j, orig_i, orig_j, orig_offsets))
            edge_cost = 0.0;
        else
        {
            const int orig_idx = find_edge_index(i, j, orig_i, orig_j, orig_offsets);
            assert(orig_i[orig_idx] == i && orig_j[orig_idx] == j);
            edge_cost = orig_edge_costs[orig_idx];
        }
    }
};

struct edge_index_binary_search_func
{
    int* i;
    int* j;
    int* offsets;
    int* edge_counter;

    __device__ int operator()(const thrust::tuple<int,int> t)
    {
        assert(edge_present(thrust::get<0>(t), thrust::get<1>(t), i, j, offsets));
        const int c = find_edge_index(thrust::get<0>(t), thrust::get<1>(t), i, j, offsets);
        atomicAdd(&edge_counter[c], 1);
        return c;
    }
};

struct sort_edge_nodes_func
{
    __host__ __device__
        void operator()(const thrust::tuple<int&,int&> t)
        {
            int& x = thrust::get<0>(t);
            int& y = thrust::get<1>(t);
            const int smallest = min(x, y);
            const int largest = max(x, y);
            assert(smallest < largest);
            x = smallest;
            y = largest;
        }
};

struct sort_triangle_nodes_func
{
    __host__ __device__
        void operator()(const thrust::tuple<int&,int&,int&> t)
        {
            int& x = thrust::get<0>(t);
            int& y = thrust::get<1>(t);
            int& z = thrust::get<2>(t);
            const int smallest = min(min(x, y), z);
            const int middle = max(min(x,y), min(max(x,y),z));
            const int largest = max(max(x, y), z);
            assert(smallest < middle && middle < largest);
            x = smallest;
            y = middle;
            z = largest;
        }
};


// return for each triangle the edge index of its three edges, plus number of triangles an edge is part of
multicut_message_passing::multicut_message_passing(
        thrust::device_vector<int>& orig_i, 
        thrust::device_vector<int>& orig_j,
        thrust::device_vector<float>& orig_edge_costs,
        thrust::device_vector<int>& _t1,
        thrust::device_vector<int>& _t2,
        thrust::device_vector<int>& _t3)
    : t1(_t1),
    t2(_t2),
    t3(_t3)
{
    std::cout << "triangle size = " << t1.size() << ", orig edges size = " << orig_i.size() << "\n";
    assert(orig_i.size() == orig_j.size());
    assert(orig_edge_costs.size() == orig_j.size());
    assert(t1.size() == t2.size() && t1.size() == t3.size()); 

    const int nr_nodes = std::max({
            *thrust::max_element(orig_i.begin(), orig_i.end()),
            *thrust::max_element(orig_j.begin(), orig_j.end()),
            *thrust::max_element(t1.begin(), t1.end()),
            *thrust::max_element(t2.begin(), t2.end()),
            *thrust::max_element(t3.begin(), t3.end())
            });

    // bring edges into normal form (first node < second node)
    {
        auto first = thrust::make_zip_iterator(thrust::make_tuple(i.begin(), j.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(i.end(), j.end()));
        thrust::for_each(first, last, sort_edge_nodes_func());
        coo_sorting(orig_i, orig_j, orig_edge_costs);
    }

    // bring triangles into normal form (first node < second node < third node)
    {
        auto first = thrust::make_zip_iterator(thrust::make_tuple(t1.begin(), t2.begin(), t3.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(t1.end(), t2.end(), t3.end()));
        thrust::for_each(first, last, sort_triangle_nodes_func());
    }

    // sort triangles and remove duplicates
    {
        coo_sorting(t1, t2, t3);
        assert(thrust::is_sorted(t1.begin(), t1.end()));
        auto first = thrust::make_zip_iterator(thrust::make_tuple(t1.begin(), t2.begin(), t3.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(t1.end(), t2.end(), t3.end()));
        auto new_last = thrust::unique(first, last);
        t1.resize(std::distance(first, new_last)); 
        t2.resize(std::distance(first, new_last)); 
        t3.resize(std::distance(first, new_last)); 
    }

    // edges that will participate in message passing are those in triangles. Hence, we use only these.
    i = thrust::device_vector<int>(3*t1.size());
    j = thrust::device_vector<int>(3*t1.size());
    thrust::copy(t1.begin(), t1.end(), i.begin());
    thrust::copy(t2.begin(), t2.end(), j.begin());
    thrust::copy(t1.begin(), t1.end(), i.begin() + t1.size());
    thrust::copy(t3.begin(), t3.end(), j.begin() + t1.size());
    thrust::copy(t2.begin(), t2.end(), i.begin() + 2*t1.size());
    thrust::copy(t3.begin(), t3.end(), j.begin() + 2*t1.size());

    // remove duplicate edges
    {
        coo_sorting(i,j);
        assert(thrust::is_sorted(i.begin(), i.end()));
        auto first = thrust::make_zip_iterator(thrust::make_tuple(i.begin(), j.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(i.end(), j.end()));
        auto new_last = thrust::unique(first, last);
        i.resize(std::distance(first, new_last));
        j.resize(std::distance(first, new_last));
    }

    // copy edge costs from given edges, todo: possibly use later
    {
        edge_costs = thrust::device_vector<float>(i.size());
        assert(thrust::is_sorted(orig_i.begin(), orig_i.end()));
        std::cout << "before computing orig offsets\n";
        thrust::device_vector<int> orig_offsets = compute_offsets_non_contiguous(nr_nodes+1, orig_i); // TODO: change name
        std::cout << "after computing orig offsets\n";
        auto first = thrust::make_zip_iterator(thrust::make_tuple(i.begin(), j.begin(), edge_costs.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(i.end(), j.end(), edge_costs.end()));
        copy_edge_costs_func func({
                thrust::raw_pointer_cast(orig_i.data()),
                thrust::raw_pointer_cast(orig_j.data()),
                thrust::raw_pointer_cast(orig_edge_costs.data()),
                thrust::raw_pointer_cast(orig_offsets.data())});
        thrust::for_each(first, last, func);
        std::cout << "after copying edge costs occurring in triangles\n";
    }

    const int nr_edges = i.size();
    const int nr_triangles = t1.size();

    // to which edge does first/second/third edge in triangle correspond to
    triangle_correspondence_12 = thrust::device_vector<int>(t1.size());
    triangle_correspondence_13 = thrust::device_vector<int>(t1.size());
    triangle_correspondence_23 = thrust::device_vector<int>(t1.size());
    edge_counter = thrust::device_vector<int>(nr_edges, 0);

    std::cout << "before computing offsets\n";
    thrust::device_vector<int> offsets = compute_offsets_non_contiguous(nr_nodes, i);
    std::cout << "after computing offsets\n";
    edge_index_binary_search_func func({
            thrust::raw_pointer_cast(i.data()),
            thrust::raw_pointer_cast(j.data()),
            thrust::raw_pointer_cast(offsets.data()),
            thrust::raw_pointer_cast(edge_counter.data()), 
            });

    {
        auto first = thrust::make_zip_iterator(thrust::make_tuple(t1.begin(), t2.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(t1.end(), t2.end()));
        thrust::transform(first, last, triangle_correspondence_12.begin(), func);
    }

    {
        auto first = thrust::make_zip_iterator(thrust::make_tuple(t1.begin(), t3.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(t1.end(), t3.end()));
        thrust::transform(first, last, triangle_correspondence_13.begin(), func);
    }

    {
        auto first = thrust::make_zip_iterator(thrust::make_tuple(t2.begin(), t3.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(t2.end(), t3.end()));
        thrust::transform(first, last, triangle_correspondence_23.begin(), func);
    }

    t12_costs = thrust::device_vector<float>(t1.size(), 0.0);
    t13_costs = thrust::device_vector<float>(t1.size(), 0.0);
    t23_costs = thrust::device_vector<float>(t1.size(), 0.0);

    // If some edges were not part of a triangle, append these edges and their original costs
    {
        coo_sorting(i, j, edge_costs);
        offsets = compute_offsets_non_contiguous(nr_nodes, i);
        auto orig_first = thrust::make_zip_iterator(thrust::make_tuple(orig_i.begin(), orig_j.begin(), orig_edge_costs.begin()));
        auto orig_last = thrust::make_zip_iterator(thrust::make_tuple(orig_i.end(), orig_j.end(), orig_edge_costs.end()));
        const int nr_triangle_edges = i.size();
        assert(thrust::is_sorted(i.begin(), i.end()));
        assert(offsets.back() == i.size());
        assert(i.size() == j.size());
        assert(i.size() == edge_costs.size());
        //i.resize(i.size() + orig_i.size());
        //j.resize(j.size() + orig_j.size());
        //edge_costs.resize(edge_costs.size() + orig_edge_costs.size());

        thrust::device_vector<int> i_not_covered(orig_i.size());;
        thrust::device_vector<int> j_not_covered(orig_i.size());;
        thrust::device_vector<float> edge_costs_not_covered(orig_i.size());;
        //auto new_first = thrust::make_zip_iterator(thrust::make_tuple(i.begin() + nr_triangle_edges, j.begin() + nr_triangle_edges, edge_costs.begin() + nr_triangle_edges));
        auto new_first = thrust::make_zip_iterator(thrust::make_tuple(i_not_covered.begin(), j_not_covered.begin(), edge_costs_not_covered.begin()));
        edge_not_present_func func({thrust::raw_pointer_cast(i.data()), thrust::raw_pointer_cast(j.data()), thrust::raw_pointer_cast(offsets.data())});
        std::cout << "before copy if\n";
        auto new_last = thrust::copy_if(orig_first, orig_last, new_first, func);
        std::cout << "after copy if\n";
        const int nr_new_edges = std::distance(new_first, new_last);
        std::cout << "nr triangle edges = " << nr_triangle_edges << ", nr additional edges = " << nr_new_edges << "\n";
        i_not_covered.resize(nr_new_edges);
        j_not_covered.resize(nr_new_edges);
        edge_costs_not_covered.resize(nr_new_edges);

        i.resize(i.size() + nr_new_edges);
        thrust::copy(i_not_covered.begin(), i_not_covered.end(), i.begin() + nr_triangle_edges);
        j.resize(j.size() + nr_new_edges);
        thrust::copy(j_not_covered.begin(), j_not_covered.end(), j.begin() + nr_triangle_edges);
        edge_costs.resize(edge_costs.size() + nr_new_edges);
        thrust::copy(edge_costs_not_covered.begin(), edge_costs_not_covered.end(), edge_costs.begin() + nr_triangle_edges);
        edge_counter.resize(i.size(), 0);
    } 
}

struct neg_part //: public unary_function<float,float>
{
      __host__ __device__ float operator()(const float x) const
      {
          return x < 0.0 ? x : 0.0;
      }
};

struct triangle_lb_func {
      __host__ __device__ float operator()(const thrust::tuple<float,float,float> x) const
      {
          const float c12 = thrust::get<0>(x);
          const float c13 = thrust::get<1>(x);
          const float c23 = thrust::get<2>(x);
          float lb = 0.0;
          lb = min(lb, c12 + c13);
          lb = min(lb, c12 + c23);
          lb = min(lb, c13 + c23);
          lb = min(lb, c12 + c13 + c23);
          return lb; 
      } 
};

double multicut_message_passing::edge_lower_bound()
{
    return thrust::transform_reduce(edge_costs.begin(), edge_costs.end(), neg_part(), 0.0, thrust::plus<float>());
}

double multicut_message_passing::triangle_lower_bound()
{
    auto first = thrust::make_zip_iterator(thrust::make_tuple(t12_costs.begin(), t13_costs.begin(), t23_costs.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(t12_costs.end(), t13_costs.end(), t23_costs.end()));
    return thrust::transform_reduce(first, last, triangle_lb_func(), 0.0, thrust::plus<float>());
}

double multicut_message_passing::lower_bound()
{
    return edge_lower_bound() + triangle_lower_bound(); 
}

struct increase_triangle_costs_func {
    float* edge_costs;
    int* edge_counter;
      __host__ __device__ void operator()(const thrust::tuple<int,float&> t) const
      {
          const int edge_idx = thrust::get<0>(t);
          float& triangle_cost = thrust::get<1>(t);
          assert(edge_counter[edge_idx] > 0);

          triangle_cost += edge_costs[edge_idx]/float(edge_counter[edge_idx]) ;
      } 
};

struct decrease_edge_costs_func {
      __host__ __device__ void operator()(const thrust::tuple<float&,int> x) const
      {
          float& cost = thrust::get<0>(x);
          int counter = thrust::get<1>(x);
          if(counter > 0)
              cost = 0.0;
      }
};

void multicut_message_passing::send_messages_to_triplets()
{
    // send costs to triangles
    {
        auto first = thrust::make_zip_iterator(thrust::make_tuple(triangle_correspondence_12.begin(), t12_costs.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(triangle_correspondence_12.end(), t12_costs.end()));
        increase_triangle_costs_func func({thrust::raw_pointer_cast(edge_costs.data()), thrust::raw_pointer_cast(edge_counter.data())});
        thrust::for_each(first, last, func);
    }
    {
        auto first = thrust::make_zip_iterator(thrust::make_tuple(triangle_correspondence_13.begin(), t13_costs.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(triangle_correspondence_13.end(), t13_costs.end()));
        thrust::for_each(first, last, increase_triangle_costs_func({thrust::raw_pointer_cast(edge_costs.data()), thrust::raw_pointer_cast(edge_counter.data())}));
    }
    {
        auto first = thrust::make_zip_iterator(thrust::make_tuple(triangle_correspondence_23.begin(), t23_costs.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(triangle_correspondence_23.end(), t23_costs.end()));
        thrust::for_each(first, last, increase_triangle_costs_func({thrust::raw_pointer_cast(edge_costs.data()), thrust::raw_pointer_cast(edge_counter.data())}));
    }

    // set costs of edges to zero (if edge participates in a triangle)
    {
        auto first = thrust::make_zip_iterator(thrust::make_tuple(edge_costs.begin(), edge_counter.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(edge_costs.end(), edge_counter.end())); 
        thrust::for_each(first, last, decrease_edge_costs_func());
    } 

    std::cout << "edge lb = " << edge_lower_bound() << " triangle lb = " << triangle_lower_bound() << "\n";
}

struct decrease_triangle_costs_func {
    float* edge_costs;

    __host__ __device__
        float min_marginal(const float x, const float y, const float z) const
        {
            float mm1 = min(x+y+z, min(x+y, x+z));
            float mm0 = min(0.0, y+z); 
            return mm1-mm0;
        }

    __device__ 
        void operator()(const thrust::tuple<int,int,int,float&,float&,float&,int,int,int> t) const
        {
            const int t12 = thrust::get<0>(t);
            const int t13 = thrust::get<1>(t);
            const int t23 = thrust::get<2>(t);
            float& t12_costs = thrust::get<3>(t);
            float& t13_costs = thrust::get<4>(t);
            float& t23_costs = thrust::get<5>(t);
            const int t12_correspondence = thrust::get<6>(t);
            const int t13_correspondence = thrust::get<7>(t);
            const int t23_correspondence = thrust::get<8>(t);

            float e12_diff = 0.0;
            float e13_diff = 0.0;
            float e23_diff = 0.0;
            {
                const float mm12 = min_marginal(t12_costs, t13_costs, t23_costs);
                t12_costs -= 1.0/3.0*mm12;
                e12_diff += 1.0/3.0*mm12;
            }
            {
                const float mm13 = min_marginal(t13_costs, t12_costs, t23_costs);
                t13_costs -= 1.0/2.0*mm13;
                e13_diff += 1.0/2.0*mm13;
            }
            {
                const float mm23 = min_marginal(t23_costs, t12_costs, t13_costs);
                t23_costs -= mm23;
                e23_diff += mm23;
            }
            {
                const float mm12 = min_marginal(t12_costs, t13_costs, t23_costs);
                t12_costs -= 1.0/2.0*mm12;
                e12_diff += 1.0/2.0*mm12;
            }
            {
                const float mm13 = min_marginal(t13_costs, t12_costs, t23_costs);
                t13_costs -= mm13;
                e13_diff += mm13;
            }
            {
                const float mm12 = min_marginal(t12_costs, t13_costs, t23_costs);
                t12_costs -= mm12;
                e12_diff += mm12;
            }

            atomicAdd(&edge_costs[t12_correspondence], e12_diff);
            atomicAdd(&edge_costs[t13_correspondence], e13_diff);
            atomicAdd(&edge_costs[t23_correspondence], e23_diff);
        }
};

void multicut_message_passing::send_messages_to_edges()
{
    auto first = thrust::make_zip_iterator(thrust::make_tuple(
                t1.begin(), t2.begin(), t3.begin(),
                t12_costs.begin(), t13_costs.begin(), t23_costs.begin(),
                triangle_correspondence_12.begin(), triangle_correspondence_13.begin(), triangle_correspondence_23.begin()
                ));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(
                t1.end(), t2.end(), t3.end(),
                t12_costs.end(), t13_costs.end(), t23_costs.end(),
                triangle_correspondence_12.end(), triangle_correspondence_13.end(), triangle_correspondence_23.end()
                ));
    thrust::for_each(first, last, decrease_triangle_costs_func({thrust::raw_pointer_cast(edge_costs.data())})); 
}

void multicut_message_passing::iteration()
{
    send_messages_to_triplets();
    send_messages_to_edges();
}

std::tuple<const thrust::device_vector<int>&, const thrust::device_vector<int>&, const thrust::device_vector<float>>
multicut_message_passing::reparametrized_edge_costs() const
{
    return {i, j, edge_costs};
}

