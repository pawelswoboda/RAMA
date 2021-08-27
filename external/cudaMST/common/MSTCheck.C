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
#include <cstring>
#include "parallel.h"
#include "IO.h"
#include "graph.h"
#include "graphIO.h"
#include "parseCommandLine.h"

using namespace std;
using namespace benchIO;

pair<intT*,intT> mst(wghEdgeArray<intT> G);

int parallel_main(int argc, char* argv[]) { 
  commandLine P(argc,argv,"<inFile> <outfile>");
  pair<char*,char*> fnames = P.IOFileNames();
  char* iFile = fnames.first;
  char* oFile = fnames.second;
  wghEdgeArray<intT> In = readWghEdgeArrayFromFile<intT>(iFile);
  _seq<intT> Out = readIntArrayFromFile<intT>(oFile);
  intT n = Out.n;
  intT in_m = In.m;
  //check num edges
  pair<intT*,intT> serialMST = mst(In);
  if (n != serialMST.second){
    cout << "Wrong edge count: MST has " << serialMST.second << " edges but algorithm returned " << n << " edges\n";
    return (1);
    }

  //check for cycles
  bool* flags = newA(bool,in_m);
  parallel_for(intT i=0;i<in_m;i++) flags[i] = 0;
  parallel_for(intT i=0;i<n;i++)
    flags[Out.A[i]] = 1;

  Out.del();

  wghEdge<intT>* E = newA(wghEdge<intT>,in_m);
  intT m = sequence::pack(In.E,E,flags,in_m);
  wghEdgeArray<intT> EA(E,In.n,m); 


  pair<intT*,intT> check = mst(EA);
  if (m != check.second){
    cout << "Result is not a spanning tree " << endl; 
    return (1);
  }
  free(check.first);
  
  //check weights
  //weight from file
  double* weights = newA(double,m);
  parallel_for(intT i=0;i<m;i++) weights[i] = E[i].weight;
  double total = sequence::plusScan(weights,weights,m);
  
  //correct weight
  parallel_for(intT i=0;i<in_m;i++) flags[i] = 0;
  parallel_for(intT i=0;i<n;i++)
    flags[serialMST.first[i]] = 1;

  free(serialMST.first);

  m = sequence::pack(In.E,E,flags,in_m);

  In.del();

  free(flags);

  parallel_for(intT i=0;i<m;i++) weights[i] = E[i].weight;
 
  EA.del();
 
  double correctTotal = sequence::plusScan(weights,weights,m);

  free(weights);

  if((total - correctTotal) > 0.0000001) {
    cout << "MST has a weight of " << total << " but should have a weight of " << correctTotal << " " << total - correctTotal << endl;
    return (1);
  }

  return 0;
}
