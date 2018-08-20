#ifndef _OPENACC_PROPAGATE
#define _OPENACC_PROPAGATE

// Propagate: using Fletcher's equations, propagate waves one dt,
//            either forward or backward in time


void OPENACC_Propagate(int sx, int sy, int sz, int bord,
	       float dx, float dy, float dz, float dt, int it, 
	       float * restrict ch1dxx, float * restrict ch1dyy, float * restrict ch1dzz, 
	       float * restrict ch1dxy, float * restrict ch1dyz, float * restrict ch1dxz, 
	       float * restrict v2px, float * restrict v2pz, float * restrict v2sz, float * restrict v2pn,
	       float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc);

#endif
