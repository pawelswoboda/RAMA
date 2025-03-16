#include "multicut_message_passing.h"
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/remove.h>
#include <thrust/unique.h>
#include <thrust/set_operations.h>
#include <algorithm>
#include "rama_utils.h"
#include "time_measure_util.h"
#include "stdio.h"
 
void multicut_message_passing::compute_triangle_edge_correspondence(
    const thrust::device_vector<int>& ta, const thrust::device_vector<int>& tb, 
    thrust::device_vector<int>& edge_counter, thrust::device_vector<int>& triangle_correspondence_ab)
{
    thrust::device_vector<int> t_sort_order(ta.size());
    thrust::sequence(t_sort_order.begin(), t_sort_order.end());
    thrust::device_vector<int> ta_unique(ta.size());
    thrust::device_vector<int> tb_unique(tb.size());
    thrust::device_vector<int> t_counts(ta.size());
    {
        thrust::device_vector<int> ta_sorted = ta;
        thrust::device_vector<int> tb_sorted = tb;
        auto first = thrust::make_zip_iterator(thrust::make_tuple(ta_sorted.begin(), tb_sorted.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(ta_sorted.end(), tb_sorted.end()));
        thrust::sort_by_key(first, last, t_sort_order.begin());

        auto first_unique = thrust::make_zip_iterator(thrust::make_tuple(ta_unique.begin(), tb_unique.begin()));
        auto last_reduce = thrust::reduce_by_key(first, last, thrust::make_constant_iterator(1), first_unique, t_counts.begin());
        int num_unique = std::distance(first_unique, last_reduce.first);
        ta_unique.resize(num_unique);
        tb_unique.resize(num_unique);
        t_counts.resize(num_unique);
    }

    auto first_edge = thrust::make_zip_iterator(thrust::make_tuple(i.begin(), j.begin()));
    auto last_edge = thrust::make_zip_iterator(thrust::make_tuple(i.end(), j.end()));
    assert(thrust::is_sorted(first_edge, last_edge));
    assert(std::distance(first_edge, thrust::unique(first_edge, last_edge)) == i.size());

    auto test_it = thrust::make_zip_iterator(thrust::make_tuple(i.begin(), j.begin()));

    auto first_unique = thrust::make_zip_iterator(thrust::make_tuple(ta_unique.begin(), tb_unique.begin()));
    auto last_unique = thrust::make_zip_iterator(thrust::make_tuple(ta_unique.end(), tb_unique.end()));
    thrust::device_vector<int> unique_correspondence(ta_unique.size());
    auto last_edge_int = thrust::set_intersection_by_key(first_edge, last_edge, first_unique, last_unique, 
                                    thrust::make_counting_iterator(0), thrust::make_discard_iterator(), 
                                    unique_correspondence.begin());

    unique_correspondence.resize(std::distance(unique_correspondence.begin(), last_edge_int.second));
    assert(unique_correspondence.size() == ta_unique.size()); // all triangle edges should be present.

    thrust::device_vector<int> correspondence_sorted = invert_unique(unique_correspondence, t_counts);
    thrust::scatter(correspondence_sorted.begin(), correspondence_sorted.end(), t_sort_order.begin(), triangle_correspondence_ab.begin());

    thrust::device_vector<int> edge_increment(i.size(), 0);
    thrust::scatter(t_counts.begin(), t_counts.end(), unique_correspondence.begin(), edge_increment.begin());
    thrust::transform(edge_counter.begin(), edge_counter.end(), edge_increment.begin(), edge_counter.begin(), thrust::plus<int>());
}

// return for each triangle the edge index of its three edges, plus number of triangles an edge is part of
multicut_message_passing::multicut_message_passing(
        const dCOO& A,
        thrust::device_vector<int>&& _t1,
        thrust::device_vector<int>&& _t2,
        thrust::device_vector<int>&& _t3,
        const bool verbose)
    : t1(std::move(_t1)),
    t2(std::move(_t2)),
    t3(std::move(_t3))
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
    if(verbose)
        std::cout << "triangle size = " << t1.size() << ", orig edges size = " << A.nnz() << "\n";
    assert(t1.size() == t2.size() && t1.size() == t3.size()); 
    normalize_triangles(t1, t2, t3);
    
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
    edge_counter = thrust::device_vector<int>(nr_edges, 0);

    compute_triangle_edge_correspondence(t1, t2, edge_counter, triangle_correspondence_12);
    compute_triangle_edge_correspondence(t1, t3, edge_counter, triangle_correspondence_13);
    compute_triangle_edge_correspondence(t2, t3, edge_counter, triangle_correspondence_23);

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

std::tuple<const thrust::device_vector<int>&, const thrust::device_vector<int>&, const thrust::device_vector<float>&>
multicut_message_passing::reparametrized_edge_costs() const
{
    return {i, j, edge_costs};
}


void multicut_message_passing::iteration(const bool use_nn) {
    if (use_nn) {
        update_lagrange_via_nn();
    } else {
        send_messages_to_triplets();
        send_messages_to_edges();

        thrust::host_vector<float> h_t12 = t12_costs;
        thrust::host_vector<float> h_t13 = t13_costs;
        thrust::host_vector<float> h_t23 = t23_costs;
    /*  for (int t = 0; t < std::min(10, (int)h_t12.size()); ++t) {
            std::cout << "[C++] Triangle " << t << ": "
                    << "λ12 = " << -h_t12[t] << ", "
                    << "λ13 = " << -h_t13[t] << ", "
                    << "λ23 = " << -h_t23[t] << std::endl;
        } */

    }
}

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h> 

namespace py = pybind11;

void multicut_message_passing::update_lagrange_via_nn() {

    static bool python_initialized = false;
    if (!python_initialized) {
        static py::scoped_interpreter guard{};
        python_initialized = true;
    }
    py::gil_scoped_acquire acquire;
    auto sys = py::module_::import("sys");
    sys.attr("path").attr("append")("/home/houraghene/RAMA/src/message_passing_nn");


    auto mod = py::module_::import("nn_message_passing");
    auto nn_func = mod.attr("nn_update");

    long edge_costs_ptr = reinterpret_cast<long>(thrust::raw_pointer_cast(edge_costs.data()));
    long t1_ptr = reinterpret_cast<long>(thrust::raw_pointer_cast(t1.data()));
    long t2_ptr = reinterpret_cast<long>(thrust::raw_pointer_cast(t2.data()));
    long t3_ptr = reinterpret_cast<long>(thrust::raw_pointer_cast(t3.data()));
    long i_ptr = reinterpret_cast<long>(thrust::raw_pointer_cast(i.data()));
    long j_ptr = reinterpret_cast<long>(thrust::raw_pointer_cast(j.data()));
    long t12_costs_ptr = reinterpret_cast<long>(thrust::raw_pointer_cast(t12_costs.data()));
    long t13_costs_ptr = reinterpret_cast<long>(thrust::raw_pointer_cast(t13_costs.data()));
    long t23_costs_ptr = reinterpret_cast<long>(thrust::raw_pointer_cast(t23_costs.data()));
    long triangle_corr_12_ptr = reinterpret_cast<long>(thrust::raw_pointer_cast(triangle_correspondence_12.data()));
    long triangle_corr_13_ptr = reinterpret_cast<long>(thrust::raw_pointer_cast(triangle_correspondence_13.data()));
    long triangle_corr_23_ptr = reinterpret_cast<long>(thrust::raw_pointer_cast(triangle_correspondence_23.data()));
    long edge_counter_ptr = reinterpret_cast<long>(thrust::raw_pointer_cast(edge_counter.data()));

    py::object result_obj = nn_func(
        edge_costs_ptr, t1_ptr, t2_ptr, t3_ptr, i_ptr, j_ptr,
        t12_costs_ptr, t13_costs_ptr, t23_costs_ptr,
        triangle_corr_12_ptr, triangle_corr_13_ptr, triangle_corr_23_ptr,
        edge_counter_ptr,
        edge_costs.size(), t1.size(), i.size()
    );

    py::tuple result = result_obj.cast<py::tuple>();

    py::array_t<float> updated_edge_costs = result[0].cast<py::array_t<float>>();
    py::array_t<float> updated_t12_costs = result[1].cast<py::array_t<float>>();
    py::array_t<float> updated_t13_costs = result[2].cast<py::array_t<float>>();
    py::array_t<float> updated_t23_costs = result[3].cast<py::array_t<float>>();

    auto buffer_edge = updated_edge_costs.request();
    float* edge_ptr = static_cast<float*>(buffer_edge.ptr);

    auto buffer_t12 = updated_t12_costs.request();
    float* t12_ptr_host = static_cast<float*>(buffer_t12.ptr);

    auto buffer_t13 = updated_t13_costs.request();
    float* t13_ptr_host = static_cast<float*>(buffer_t13.ptr);

    auto buffer_t23 = updated_t23_costs.request();
    float* t23_ptr_host = static_cast<float*>(buffer_t23.ptr);

    cudaMemcpy(thrust::raw_pointer_cast(edge_costs.data()), edge_ptr,
            sizeof(float) * edge_costs.size(), cudaMemcpyHostToDevice);

    cudaMemcpy(thrust::raw_pointer_cast(t12_costs.data()), t12_ptr_host,
            sizeof(float) * t12_costs.size(), cudaMemcpyHostToDevice);

    cudaMemcpy(thrust::raw_pointer_cast(t13_costs.data()), t13_ptr_host,
            sizeof(float) * t13_costs.size(), cudaMemcpyHostToDevice);

    cudaMemcpy(thrust::raw_pointer_cast(t23_costs.data()), t23_ptr_host,
            sizeof(float) * t23_costs.size(), cudaMemcpyHostToDevice);
}





