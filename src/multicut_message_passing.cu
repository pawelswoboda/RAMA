#include "multicut_message_passing.h"
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <algorithm>
#include "parallel_gaec_utils.h"
#include "time_measure_util.h"
#include "stdio.h"

void multicut_message_passing::compute_triangle_edge_correspondence(
    const thrust::device_vector<int>& ta, const thrust::device_vector<int>& tb, const thrust::device_vector<float>& t_packing_val, 
    thrust::device_vector<float>& edge_total_capacity, thrust::device_vector<int>& triangle_correspondence_ab)
{
    // Match edges of triangles with multicut edges by set intersection which requires the two sets to be sorted.
    thrust::device_vector<int> t_sort_order(ta.size());
    thrust::sequence(t_sort_order.begin(), t_sort_order.end());
    thrust::device_vector<int> ta_unique(ta.size());
    thrust::device_vector<int> tb_unique(tb.size());
    thrust::device_vector<int> t_counts(ta.size());
    thrust::device_vector<float> t_packing_val_sum(ta.size());

    // sort the triangle edges and remove duplicates. Store the sorting order to invert it after set intersection.
    {
        thrust::device_vector<int> ta_sorted = ta;
        thrust::device_vector<int> tb_sorted = tb;
        thrust::device_vector<float> t_packing_val_sorted = t_packing_val;
        auto first = thrust::make_zip_iterator(thrust::make_tuple(ta_sorted.begin(), tb_sorted.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(ta_sorted.end(), tb_sorted.end()));
        auto output = thrust::make_zip_iterator(thrust::make_tuple(t_packing_val_sorted.begin(), t_sort_order.begin()));
        thrust::sort_by_key(first, last, output);

        auto first_unique = thrust::make_zip_iterator(thrust::make_tuple(ta_unique.begin(), tb_unique.begin()));
        auto last_reduce = thrust::reduce_by_key(first, last, thrust::make_constant_iterator(1), first_unique, t_counts.begin());
        int num_unique = std::distance(first_unique, last_reduce.first);
        ta_unique.resize(num_unique);
        tb_unique.resize(num_unique);
        t_counts.resize(num_unique);

        // sum triangle packing values
        auto last_reduce_val = thrust::reduce_by_key(first, last, t_packing_val_sorted.begin(), thrust::make_discard_iterator(), t_packing_val_sum.begin());
        assert(std::distance(t_packing_val_sum.begin(), last_reduce_val.second) == num_unique);
        t_packing_val_sum.resize(num_unique);
    }

    // multicut edges are already sorted.
    auto first_edge = thrust::make_zip_iterator(thrust::make_tuple(i.begin(), j.begin()));
    auto last_edge = thrust::make_zip_iterator(thrust::make_tuple(i.end(), j.end()));
    assert(thrust::is_sorted(first_edge, last_edge));
    assert(std::distance(first_edge, thrust::unique(first_edge, last_edge)) == i.size());

    // perform intersection.
    auto first_unique = thrust::make_zip_iterator(thrust::make_tuple(ta_unique.begin(), tb_unique.begin()));
    auto last_unique = thrust::make_zip_iterator(thrust::make_tuple(ta_unique.end(), tb_unique.end()));
    thrust::device_vector<int> unique_int_edge_indices(ta_unique.size()); // stores the matched edge indices for each unique triangle edge.
    auto last_edge_int = thrust::set_intersection_by_key(first_edge, last_edge, first_unique, last_unique, 
                                    thrust::make_counting_iterator(0), thrust::make_discard_iterator(), 
                                    unique_int_edge_indices.begin());

    unique_int_edge_indices.resize(std::distance(unique_int_edge_indices.begin(), last_edge_int.second));
    assert(unique_int_edge_indices.size() == ta_unique.size()); // all triangle edges should be present.

    // Now invert the triangle sorting and duplicate removal operations.
    thrust::device_vector<int> int_edge_indices = invert_unique(unique_int_edge_indices, t_counts);
    thrust::scatter(int_edge_indices.begin(), int_edge_indices.end(), t_sort_order.begin(), triangle_correspondence_ab.begin());

    // Store the edge capacity (used to normalize outgoing message from edge.)
    thrust::device_vector<float> current_edge_cap(i.size(), 0);
    thrust::scatter(t_packing_val_sum.begin(), t_packing_val_sum.end(), unique_int_edge_indices.begin(), current_edge_cap.begin());
    thrust::transform(edge_total_capacity.begin(), edge_total_capacity.end(), current_edge_cap.begin(), edge_total_capacity.begin(), thrust::plus<float>());
}

// return for each triangle the edge index of its three edges, plus number of triangles an edge is part of
multicut_message_passing::multicut_message_passing(
        const dCOO& A,
        thrust::device_vector<int>&& _t1,
        thrust::device_vector<int>&& _t2,
        thrust::device_vector<int>&& _t3,
        thrust::device_vector<float>&& _t_val)
    : t1(std::move(_t1)),
    t2(std::move(_t2)),
    t3(std::move(_t3)),
    t_val(std::move(_t_val))
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
    std::cout << "triangle size = " << t1.size() << ", orig edges size = " << A.nnz() << "\n";
    assert(t1.size() == t2.size() && t1.size() == t3.size() && t1.size() == t_val.size()); 
    normalize_triangles(t1, t2, t3, t_val);
    
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
        coo_sorting(i, j);
        assert(thrust::is_sorted(i.begin(), i.end()));
        auto first = thrust::make_zip_iterator(thrust::make_tuple(i.begin(), j.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(i.end(), j.end()));
        auto new_last = thrust::unique(first, last);
        i.resize(std::distance(first, new_last));
        j.resize(std::distance(first, new_last));
    }

    const thrust::device_vector<int> orig_i = A.get_row_ids();
    const thrust::device_vector<int> orig_j = A.get_col_ids();
    const thrust::device_vector<float> orig_edge_costs = A.get_data();

    // copy edge costs from given edges, todo: possibly use later
    {
        edge_costs = thrust::device_vector<float>(i.size(), 0.0);
        auto first_edge = thrust::make_zip_iterator(thrust::make_tuple(i.begin(), j.begin()));
        auto last_edge = thrust::make_zip_iterator(thrust::make_tuple(i.end(), j.end()));

        auto first_orig = thrust::make_zip_iterator(thrust::make_tuple(orig_i.begin(), orig_j.begin()));
        auto last_orig = thrust::make_zip_iterator(thrust::make_tuple(orig_i.end(), orig_j.end()));

        thrust::device_vector<int> intersecting_indices_edge(i.size()); // edge indices to copy to.
        auto last_int_edge = thrust::set_intersection_by_key(first_edge, last_edge, first_orig, last_orig, 
                                        thrust::counting_iterator<int>(0), thrust::make_discard_iterator(), 
                                        intersecting_indices_edge.begin());
        intersecting_indices_edge.resize(std::distance(intersecting_indices_edge.begin(), last_int_edge.second));

        thrust::device_vector<float> orig_edge_costs_int(orig_edge_costs.size()); // costs to copy.
        auto last_int_orig = thrust::set_intersection_by_key(first_orig, last_orig, first_edge, last_edge, 
                                                            orig_edge_costs.begin(), thrust::make_discard_iterator(), 
                                                            orig_edge_costs_int.begin());
        orig_edge_costs_int.resize(std::distance(orig_edge_costs_int.begin(), last_int_orig.second));
        assert(intersecting_indices_edge.size() == orig_edge_costs_int.size());
        thrust::scatter(orig_edge_costs_int.begin(), orig_edge_costs_int.end(), intersecting_indices_edge.begin(), edge_costs.begin());
    }

    // If some edges were not part of a triangle, append these edges and their original costs
    {
        auto first_orig = thrust::make_zip_iterator(thrust::make_tuple(orig_i.begin(), orig_j.begin()));
        auto last_orig = thrust::make_zip_iterator(thrust::make_tuple(orig_i.end(), orig_j.end()));

        auto first_edge = thrust::make_zip_iterator(thrust::make_tuple(i.begin(), j.begin()));
        auto last_edge = thrust::make_zip_iterator(thrust::make_tuple(i.end(), j.end()));

        thrust::device_vector<int> merged_i(i.size() + orig_i.size());
        thrust::device_vector<int> merged_j(j.size() + orig_j.size());
        thrust::device_vector<float> merged_costs(edge_costs.size() + orig_edge_costs.size());

        auto first_merged = thrust::make_zip_iterator(thrust::make_tuple(merged_i.begin(), merged_j.begin()));

        auto merged_last = thrust::set_union_by_key(first_orig, last_orig, first_edge, last_edge, 
                                                    orig_edge_costs.begin(), edge_costs.begin(), 
                                                    first_merged, merged_costs.begin());
        int num_merged = std::distance(first_merged, merged_last.first);
        assert(std::distance(first_merged, thrust::unique(first_merged, merged_last.first)) == num_merged);
        merged_i.resize(num_merged);
        merged_j.resize(num_merged);
        merged_costs.resize(num_merged);
        thrust::swap(i, merged_i);
        thrust::swap(j, merged_j);
        thrust::swap(edge_costs, merged_costs);
        coo_sorting(i, j, edge_costs);

        // TODO: add covering numbers 0 for non-triangle edges
    }

    const int nr_edges = i.size();
    const int nr_triangles = t1.size();

    // to which edge does first/second/third edge in triangle correspond to
    triangle_correspondence_12 = thrust::device_vector<int>(t1.size());
    triangle_correspondence_13 = thrust::device_vector<int>(t1.size());
    triangle_correspondence_23 = thrust::device_vector<int>(t1.size());
    edge_total_capacity = thrust::device_vector<float>(nr_edges, 0.0);

    compute_triangle_edge_correspondence(t1, t2, t_val, edge_total_capacity, triangle_correspondence_12);
    compute_triangle_edge_correspondence(t1, t3, t_val, edge_total_capacity, triangle_correspondence_13);
    compute_triangle_edge_correspondence(t2, t3, t_val, edge_total_capacity, triangle_correspondence_23);

    t12_costs = thrust::device_vector<float>(t1.size(), 0.0);
    t13_costs = thrust::device_vector<float>(t1.size(), 0.0);
    t23_costs = thrust::device_vector<float>(t1.size(), 0.0);
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
    float* edge_total_capacity;
      __host__ __device__ void operator()(const thrust::tuple<int,float,float&> t) const
      {
          const int edge_idx = thrust::get<0>(t);
          const float triangle_packing_value = thrust::get<1>(t);
          float& triangle_cost = thrust::get<2>(t);
          assert(edge_total_capacity[edge_idx] > 0);
          assert(triangle_packing_value > 0);
          assert(triangle_packing_value <= edge_total_capacity[edge_idx]);

          triangle_cost += (triangle_packing_value * edge_costs[edge_idx]) / edge_total_capacity[edge_idx];
      } 
};

struct decrease_edge_costs_func {
      __host__ __device__ void operator()(const thrust::tuple<float&,float> x) const
      {
          float& cost = thrust::get<0>(x);
          const float cap = thrust::get<1>(x);
          if(cap > 0)
              cost = 0.0;
      }
};

void multicut_message_passing::send_messages_to_triplets()
{
    // send costs to triangles
    {
        auto first = thrust::make_zip_iterator(thrust::make_tuple(triangle_correspondence_12.begin(), t_val.begin(), t12_costs.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(triangle_correspondence_12.end(), t_val.end(), t12_costs.end()));
        increase_triangle_costs_func func({thrust::raw_pointer_cast(edge_costs.data()), thrust::raw_pointer_cast(edge_total_capacity.data())});
        thrust::for_each(first, last, func);
    }
    {
        auto first = thrust::make_zip_iterator(thrust::make_tuple(triangle_correspondence_13.begin(), t_val.begin(), t13_costs.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(triangle_correspondence_13.end(), t_val.end(), t13_costs.end()));
        increase_triangle_costs_func func({thrust::raw_pointer_cast(edge_costs.data()), thrust::raw_pointer_cast(edge_total_capacity.data())});
        thrust::for_each(first, last, func);
    }
    {
        auto first = thrust::make_zip_iterator(thrust::make_tuple(triangle_correspondence_23.begin(), t_val.begin(), t23_costs.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(triangle_correspondence_23.end(), t_val.end(), t23_costs.end()));
        increase_triangle_costs_func func({thrust::raw_pointer_cast(edge_costs.data()), thrust::raw_pointer_cast(edge_total_capacity.data())});
        thrust::for_each(first, last, func);
    }

    // set costs of edges to zero (if edge participates in a triangle)
    {
        auto first = thrust::make_zip_iterator(thrust::make_tuple(edge_costs.begin(), edge_total_capacity.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(edge_costs.end(), edge_total_capacity.end())); 
        thrust::for_each(first, last, decrease_edge_costs_func());
    } 

    // std::cout << "edge lb = " << edge_lower_bound() << " triangle lb = " << triangle_lower_bound() << "\n";
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

std::tuple<const thrust::device_vector<int>&, const thrust::device_vector<int>&, const thrust::device_vector<float>&>
multicut_message_passing::reparametrized_edge_costs() const
{
    return {i, j, edge_costs};
}