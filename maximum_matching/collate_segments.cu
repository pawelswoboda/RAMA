/**
 * @file collateSegments.cu
 * @date Spring 2020, revised Spring 2021
 * @author Hugo De Moraes
 */

/**
 * Scans input in parallel and collates the indecies with important data
 *
 * @param src the original unfiltered array
 * @param scanResult the output array of strongestNeighborScan.cu
 * @param output the array to be modified
 * @param numEdges the number of edges/elements in the above arrays
 */
__global__ void collateSegments_gpu(
        int * src, 
        int * scanResult, 
        int * output, 
        int numEdges
        ) {

    // Get Thread ID
    const int NUM_THREADS = blockDim.x * gridDim.x;
    const int COL = blockIdx.x * blockDim.x + threadIdx.x;
    const int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    const int FIRST_T_ID = COL + ROW * NUM_THREADS;

    // Each thread should handle numEdges / NUM_THREADS amount of work
    for(int curTID = FIRST_T_ID; curTID < numEdges; curTID += NUM_THREADS) {
        // compares src[i] with src[i+1], 
        // if they are not equal, then the i-th data element is the last one in its own segment
        if(src[curTID] != src[curTID+1]) {
            output[src[curTID]] = scanResult[curTID];
        }
    }
}
/**
 * After the previous step, the maximum-weight neighbors are placed as the last element within each segment in the output array(s). 
 * Most elements in these two arrays do not contain useful information. 
 * We want to collate the useful parts of this array, 
 * in order to produce an output array that has the same number of elements as the number of segments, 
 * and only store the maximum-weight neighbor information.
 */
