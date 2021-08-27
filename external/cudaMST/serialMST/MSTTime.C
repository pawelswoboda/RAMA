// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <iostream>
#include <algorithm>
#include "gettime.h"
#include "utils.h"
#include "graph.h"
#include "parallel.h"
#include "IO.h"
#include "graphIO.h"
#include "parseCommandLine.h"
#include "MST.h"
using namespace std;
using namespace benchIO;

void timeMST(wghEdgeArray<intT> EA, int rounds, char* outFile) {
  intT m = EA.m;
  intT n = EA.n;
  wghEdge<intT> *InE = newA(wghEdge<intT>, m);
  parallel_for (intT i=0; i < m; i++) InE[i] = EA.E[i];
  pair<intT*,intT> Out = mst(wghEdgeArray<intT>(InE, n, m));
  for (int i=0; i < rounds; i++) {
    if (Out.first != NULL) free(Out.first);
    parallel_for (intT i=0; i < m; i++) InE[i] = EA.E[i];
    startTime();
    Out = mst(wghEdgeArray<intT>(InE,n,m));
    nextTimeN();
  }
  cout << endl;
  if (outFile != NULL) writeIntArrayToFile(Out.first, Out.second, outFile);
  free(InE);
  free(Out.first);
  EA.del();
}
    
int parallel_main(int argc, char* argv[]) {
  commandLine P(argc,argv,"[-o <outFile>] [-r <rounds>] <inFile>");
  char* iFile = P.getArgument(0);
  char* oFile = P.getOptionValue("-o");
  int rounds = P.getOptionIntValue("-r",1);
  wghEdgeArray<intT> EA = readWghEdgeArrayFromFile<intT>(iFile);
  timeMST(EA, rounds, oFile);
}
