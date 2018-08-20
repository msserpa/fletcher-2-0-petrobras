#include "openacc_defines.h"
#include "openacc_stuff.h"


void OPENACC_Initialize(const int rank, const int sx, const int sy, const int sz, const int bord,
	       float dx, float dy, float dz, float dt,
	       float * restrict ch1dxx, float * restrict ch1dyy, float * restrict ch1dzz, 
	       float * restrict ch1dxy, float * restrict ch1dyz, float * restrict ch1dxz, 
	       float * restrict v2px, float * restrict v2pz, float * restrict v2sz, float * restrict v2pn,
	       float * restrict vpz, float * restrict vsv, float * restrict epsilon, float * restrict delta,
	       float * restrict phi, float * restrict theta, float * restrict fatAbsorb,
	       float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc)
{

#pragma acc enter data copyin(ch1dxx[0:sx*sy*sz], ch1dyy[0:sx*sy*sz], ch1dzz[0:sx*sy*sz],     \
		              ch1dxy[0:sx*sy*sz], ch1dyz[0:sx*sy*sz], ch1dxz[0:sx*sy*sz],  \
		              v2px[0:sx*sy*sz], v2pz[0:sx*sy*sz], v2sz[0:sx*sy*sz], \
		              v2pn[0:sx*sy*sz], pc[0:sx*sy*sz], qc[0:sx*sy*sz], \
		              pp[0:sx*sy*sz], qp[0:sx*sy*sz], fatAbsorb[0:sx*sy*sz])


}


void OPENACC_Finalize()
{

}



void OPENACC_Update_pointers(const int sx, const int sy, const int sz, float *pc)
{

#pragma acc update host(pc[0:sx*sy*sz])

}
