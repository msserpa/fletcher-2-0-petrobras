#include "../driver.h"
#include "cuda_stuff.h"
#include "cuda_propagate.h"
#include "cuda_insertsource.h"
#include <stdio.h>

// Global device vars
float* dev_vpz=NULL;
float* dev_vsv=NULL;
float* dev_epsilon=NULL;
float* dev_delta=NULL;
float* dev_phi=NULL;
float* dev_theta=NULL;
float* dev_ch1dxx=NULL;
float* dev_ch1dyy=NULL;
float* dev_ch1dzz=NULL;
float* dev_ch1dxy=NULL;
float* dev_ch1dyz=NULL;
float* dev_ch1dxz=NULL;
float* dev_v2px=NULL;
float* dev_v2pz=NULL;
float* dev_v2sz=NULL;
float* dev_v2pn=NULL;
float* dev_pp=NULL;
float* dev_pc=NULL;
float* dev_qp=NULL;
float* dev_qc=NULL;
float* dev_fatAbsorb=NULL;


void DRIVER_Initialize(const int rank, const int sx, const int sy, const int sz, const int bord,
		       float dx, float dy, float dz, float dt,
		       float * restrict ch1dxx, float * restrict ch1dyy, float * restrict ch1dzz, 
		       float * restrict ch1dxy, float * restrict ch1dyz, float * restrict ch1dxz, 
		       float * restrict v2px, float * restrict v2pz, float * restrict v2sz, float * restrict v2pn,
		       float * restrict vpz, float * restrict vsv, float * restrict epsilon, float * restrict delta,
		       float * restrict phi, float * restrict theta, float * restrict fatAbsorb,
		       float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc)
{

	   CUDA_Initialize(rank, sx,   sy,   sz,   bord,
		  dx,  dy,  dz,  dt);
}



void DRIVER_Finalize()
{
	CUDA_Finalize();
}


void DRIVER_Update_pointers(const int sx, const int sy, const int sz, float *pc)
{
	// CUDA_Update_pointers(sx,sy,sz,pc);
}


void DRIVER_Propagate(const int sx, const int sy, const int sz, const int bord,
	       const float dx, const float dy, const float dz, const float dt, const int it, 
	       float * restrict ch1dxx, float * restrict ch1dyy, float * restrict ch1dzz, 
	       float * restrict ch1dxy, float * restrict ch1dyz, float * restrict ch1dxz, 
	       float * restrict v2px, float * restrict v2pz, float * restrict v2sz, float * restrict v2pn,
	       float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc)
{

	// CUDA_Propagate also does TimeForward
	   CUDA_Propagate(  sx,   sy,   sz,   bord,
	                    dx,   dy,   dz,   dt,   it);
	   
	   // CUDA_Propagate(  sx,   sy,   sz,   bord,
	   //                  dx,   dy,   dz,   dt,   it,
    //                         ch1dxx,    ch1dyy,    ch1dzz,
    //                         ch1dxy,    ch1dyz,    ch1dxz,
	   //                  v2px,    v2pz,    v2sz,    v2pn,
	   //                  pp,    pc,    qp,    qc);

}


void DRIVER_InsertSource(float dt, int it, int iSource, float *p, float*q, float src)
{
	CUDA_InsertSource(src, iSource, p, q);
}

