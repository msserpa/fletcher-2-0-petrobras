#ifndef _OPENMPTARGET_STUFF
#define _OPENMPTARGET_STUFF

void OPENMPTARGET_Initialize(const int rank, const int sx, const int sy, const int sz, const int bord,
               float dx, float dy, float dz, float dt,
               float * restrict ch1dxx, float * restrict ch1dyy, float * restrict ch1dzz,
               float * restrict ch1dxy, float * restrict ch1dyz, float * restrict ch1dxz,
               float * restrict v2px, float * restrict v2pz, float * restrict v2sz, float * restrict v2pn,
               float * restrict vpz, float * restrict vsv, float * restrict epsilon, float * restrict delta,
               float * restrict phi, float * restrict theta, float * restrict fatAbsorb,
               float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc);


void OPENMPTARGET_Finalize();


void OPENMPTARGET_Update_pointers(const int sx, const int sy, const int sz, float *pc);

#endif

