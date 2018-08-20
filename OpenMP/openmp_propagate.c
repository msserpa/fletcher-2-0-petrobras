#include "openmp_propagate.h"
#include "../derivatives.h"
#include "../map.h"


// Propagate: using Fletcher's equations, propagate waves one dt,
//            either forward or backward in time


void OPENMP_Propagate(int sx, int sy, int sz, int bord,
	       float dx, float dy, float dz, float dt, int it, 
	       float * restrict ch1dxx, float * restrict ch1dyy, float * restrict ch1dzz, 
	       float * restrict ch1dxy, float * restrict ch1dyz, float * restrict ch1dxz, 
	       float * restrict v2px, float * restrict v2pz, float * restrict v2sz, float * restrict v2pn,
	       float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc) {


#define SAMPLE_PRE_LOOP
#include "../sample.h"
#undef SAMPLE_PRE_LOOP


#pragma omp parallel
  { // start omp

    // solve both equations in all internal grid points, 
    // including absortion zone
    
    
#pragma omp for
    for (int iz=bord; iz<sz-bord; iz++) {
      for (int iy=bord; iy<sy-bord; iy++) {
	for (int ix=bord; ix<sx-bord; ix++) {


#define SAMPLE_LOOP
#include "../sample.h"
#undef SAMPLE_LOOP


	}
      }
    }
  } // end omp
}
