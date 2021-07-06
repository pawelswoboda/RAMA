/**
 * @file packGraph.cu
 * @date Spring 2020, revised Spring 2021
 * @author Hugo De Moraes
 */

__global__ void packGraph_gpu(int * newSrc, int * oldSrc, int * newDst, int * oldDst, int * newWeight, int * oldWeight, int * edgeMap, int numEdges) {
    // Get Thread ID
    const int NUM_THREADS = blockDim.x * gridDim.x;
    const int COL = blockIdx.x * blockDim.x + threadIdx.x;
    const int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    const int FIRST_T_ID = COL + ROW * NUM_THREADS;

    for(int curTID = FIRST_T_ID; curTID < numEdges; curTID += NUM_THREADS) {
        const int COMPARE_T_ID = curTID+1;
        if(edgeMap[curTID] != edgeMap[COMPARE_T_ID]) {
            newSrc[edgeMap[curTID]] = oldSrc[curTID];
            newDst[edgeMap[curTID]] = oldDst[curTID];
            newWeight[edgeMap[curTID]] = oldWeight[curTID];
        }
        // else {
        //  newSrc[edgeMap[curTID]] = -1;
        //  newDst[edgeMap[curTID]] = -1;
        //  newWeight[edgeMap[curTID]] = -1;
        // }
    }
}
