#include <ECLgraph.h>
#include "dCSR.h"
#include <set>

int get_cuda_device()
{
    return 0; 
}

void test_cc(const std::vector<int> row_offsets, const std::vector<int> col_ids, const int expected_nr_ccs)
{
    const int nnz = col_ids.size();
    const int num_rows = row_offsets.size()-1;

    int* d_row_offsets;
    int* d_col_ids;
    int* d_node_stat_out;

    if (cudaSuccess != cudaMalloc((void **)&d_row_offsets, (num_rows + 1) * sizeof(int))) {fprintf(stderr, "ERROR: could not allocate d_row_offsets\n\n");  exit(-1);}
    if (cudaSuccess != cudaMalloc((void **)&d_col_ids, nnz * sizeof(int))) {fprintf(stderr, "ERROR: could not allocate d_col_ids\n\n");  exit(-1);}
    if (cudaSuccess != cudaMalloc((void **)&d_node_stat_out, num_rows * sizeof(int))) {fprintf(stderr, "ERROR: could not allocate d_node_stat_out,\n\n");  exit(-1);}

    if (cudaSuccess != cudaMemcpy(d_row_offsets, row_offsets.data(), (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice)) {fprintf(stderr, "ERROR: copying to device failed\n\n");  exit(-1);}
    if (cudaSuccess != cudaMemcpy(d_col_ids, col_ids.data(), nnz * sizeof(int), cudaMemcpyHostToDevice)) {fprintf(stderr, "ERROR: copying to device failed\n\n");  exit(-1);}

    computeCC_gpu(num_rows, nnz, d_row_offsets, d_col_ids, d_node_stat_out, get_cuda_device());

    std::vector<int> node_stat_out(num_rows, -1);
    if (cudaSuccess != cudaMemcpy(node_stat_out.data(), d_node_stat_out, num_rows * sizeof(int), cudaMemcpyDeviceToHost)) {fprintf(stderr, "ERROR: copying from device failed\n\n");  exit(-1);}

    std::set<int> all_cc_ids;
    for (int n = 0; n < num_rows; n++)
    {
        int c_id = node_stat_out[n];
        std::cout<<"Node: "<<n<<" CC: "<<c_id<<std::endl;
        all_cc_ids.insert(c_id);
    }
    std::cout<<"# CCs: "<<all_cc_ids.size()<<std::endl;

    for (int n = 0; n < num_rows; n++)
    {
        int c_id = node_stat_out[n];
        assert(c_id >= 0);
        for (int neighbour_index = row_offsets[n]; neighbour_index < row_offsets[n + 1]; neighbour_index++)
        {   
            int neighbour = col_ids[neighbour_index];
            std::cout<<"Node: "<<n<<" Neighbour: "<<neighbour<<std::endl;
            assert(node_stat_out[neighbour] == c_id);
        }
    }
    if(all_cc_ids.size() != expected_nr_ccs)
        throw std::runtime_error("expected " + std::to_string(expected_nr_ccs) + " connected components");
}

int main(int argc, char** argv)
{
    {
        // CSR representation of [0, 1], [1, 0], [2, 3], [3, 2], [2, 4], [4, 2]:

        std::vector<int> row_offsets = {0, 1, 2, 4, 5, 6};
        std::vector<int> col_ids = {1, 0, 3, 4, 2, 2};

        test_cc(row_offsets, col_ids, 2);
    }

    {
        // CSR representation of [1, 2]

        std::vector<int> row_offsets = {0, 0, 1, 2};
        std::vector<int> col_ids = {2, 1};

        test_cc(row_offsets, col_ids, 2);
    }
}
