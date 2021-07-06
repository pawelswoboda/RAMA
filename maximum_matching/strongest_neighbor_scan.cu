/**
 * @file strongestNeighborScan.cu
 * @date Spring 2020, revised Spring 2021
 * @author Hugo De Moraes
 */

/**
 * Scans input in parallel picks two elements with a stride s, checks if these two elements are in the same segment; 
 * if so, it compares the two elements, store the maximum one in the appropriate location in the output array. 
 *
 * @param src input array that denotes each segment in the graph
 * @param oldDst input array that denotes the destination of each edge in src
 * @param newDst output array to be modified with new greatest destinatuon
 * @param oldWeight input array that denotes the weight of each edge in src
 * @param newWeight output array to be modified with new greatest edge weight
 * @param madeChanges integer flag for any changed made by function
 * @param distance stride distance
 * @param numEdges the number of edges/elements in the above arrays
 */
__global__ void strongestNeighborScan_gpu(
        int * src,
        int * oldDst, int * newDst,
        int * oldWeight, int * newWeight,
        int * madeChanges,
        int distance,
        int numEdges
        ) {

    const int NUM_THREADS = blockDim.x * gridDim.x;
    const int COL = blockIdx.x * blockDim.x + threadIdx.x;
    const int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    const int FIRST_T_ID = COL + ROW * NUM_THREADS;

    for(int curTID = FIRST_T_ID; curTID < numEdges; curTID += NUM_THREADS) {
        // get compare thread index, enforce 0 bound
        const int COMPARE_T_ID = curTID - distance > 0 ? curTID - distance : 0;

        // case : shared segment
        if( src[COMPARE_T_ID] == src[curTID]) {
            int strongerIndex;
            const int COMPARE_T_WEIGHT = oldWeight[COMPARE_T_ID];
            const int CUR_T_WEIGHT = oldWeight[curTID];

            if(COMPARE_T_WEIGHT > CUR_T_WEIGHT) {
                strongerIndex = COMPARE_T_ID;
            }
            else if(COMPARE_T_WEIGHT < CUR_T_WEIGHT) {
                strongerIndex = curTID;
            }
            // case: equal weights, take node with smaller vID
            else {
                const int COMPARE_T_D = oldDst[COMPARE_T_ID];
                const int CUR_T_D = oldDst[curTID];

                if(COMPARE_T_D < CUR_T_D) {
                    strongerIndex = COMPARE_T_ID;
                } else {
                    strongerIndex = curTID;
                };
            }

            //Set new destination
            newDst[curTID] = oldDst[strongerIndex];

            //Set new weight
            newWeight[curTID] = oldWeight[strongerIndex];

            if(newDst[curTID] != oldDst[curTID]) { *madeChanges = 1; };
        }
        // case : different segment
        else {
            // defaults to no change
            newDst[curTID] = oldDst[curTID];
            newWeight[curTID] = oldWeight[curTID];
        }
    }
}
/**
 * During each iteration of parallel segment-scan, each (independent) task picks two elements with a stride s, 
 * checks if these two elements are in the same segment; 
 * if so, it compares the two elements, store the maximum one in the appropriate location in the output array. 
 * A parallel segment-scan may involve multiple iterations, 
 * the first iteration uses stride s = 1 and the stride s doubles at every iteration.
 */
