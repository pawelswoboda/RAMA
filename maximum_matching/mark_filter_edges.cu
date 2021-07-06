/**
 * @file markFilterEdge.cu
 * @date Spring 2020, revised Spring 2021
 * @author Hugo De Moraes
 */

__global__ void markFilterEdges_gpu(int * src, int * dst, int * matches, int * keepEdges, int numEdges) {
    // Get Thread ID
    const int NUM_THREADS = blockDim.x * gridDim.x;
    const int COL = blockIdx.x * blockDim.x + threadIdx.x;
    const int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    const int FIRST_T_ID = COL + ROW * NUM_THREADS;

    for(int curTID = FIRST_T_ID; curTID <= numEdges; curTID += NUM_THREADS) {
        if(matches[src[curTID]] != -1 || matches[dst[curTID]] != -1) {
            keepEdges[curTID] = 0;
        } 
        else {
            keepEdges[curTID] = 1;
        }
    }
}
