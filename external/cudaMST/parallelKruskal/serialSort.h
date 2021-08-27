#ifndef A_STLSORT_INCLUDED
#define A_STLSORT_INCLUDED
#include <algorithm>

template <class E, class BinPred, class intT>
void compSort(E* A, intT n, BinPred f) { std::sort(A,A+n,f);}

#endif // _A_STLSORT_INCLUDED
