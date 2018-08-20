#include "openmptarget_insertsource.h"


// InsertSource: compute and insert source value at index iSource of arrays p and q


void OPENMPTARGET_InsertSource(float dt, int it, int iSource, 
			       float *p, float*q, float src) {
#pragma omp target map(to:src, iSource)
  {
     p[iSource]+=src;
     q[iSource]+=src;
  }
}
