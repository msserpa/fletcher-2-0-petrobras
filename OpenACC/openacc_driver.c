#include "../driver.h"
#include "openacc_stuff.h"
#include "openacc_propagate.h"
#include "openacc_insertsource.h"
#include <stdio.h>


void DRIVER_Initialize(const int rank, const int sx, const int sy, const int sz, const int bord,
		       float dx, float dy, float dz, float dt,
		       float * restrict ch1dxx, float * restrict ch1dyy, float * restrict ch1dzz, 
		       float * restrict ch1dxy, float * restrict ch1dyz, float * restrict ch1dxz, 
		       float * restrict v2px, float * restrict v2pz, float * restrict v2sz, float * restrict v2pn,
		       float * restrict vpz, float * restrict vsv, float * restrict epsilon, float * restrict delta,
		       float * restrict phi, float * restrict theta, float * restrict fatAbsorb,
		       float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc)
{

	   OPENACC_Initialize(rank, sx,   sy,   sz,   bord,
		  dx,  dy,  dz,  dt,
	          ch1dxx,    ch1dyy,    ch1dzz, 
  	          ch1dxy,    ch1dyz,    ch1dxz, 
  	          v2px,    v2pz,    v2sz,    v2pn,
  	          vpz,    vsv,    epsilon,    delta,
  	          phi,    theta,    fatAbsorb,
  	          pp,    pc,    qp,    qc);

}



void DRIVER_Finalize()
{
	OPENACC_Finalize();
}


void DRIVER_Update_pointers(const int sx, const int sy, const int sz, float *pc)
{
	OPENACC_Update_pointers(sx,sy,sz,pc);
}


void DRIVER_Propagate(const int sx, const int sy, const int sz, const int bord,
	       const float dx, const float dy, const float dz, const float dt, const int it, 
	       float * restrict ch1dxx, float * restrict ch1dyy, float * restrict ch1dzz, 
	       float * restrict ch1dxy, float * restrict ch1dyz, float * restrict ch1dxz, 
	       float * restrict v2px, float * restrict v2pz, float * restrict v2sz, float * restrict v2pn,
	       float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc)
{

	   OPENACC_Propagate(  sx,   sy,   sz,   bord,
	                    dx,   dy,   dz,   dt,   it,
                            ch1dxx,    ch1dyy,    ch1dzz,
                            ch1dxy,    ch1dyz,    ch1dxz,
	                    v2px,    v2pz,    v2sz,    v2pn,
	                    pp,    pc,    qp,    qc);

}


void DRIVER_InsertSource(float dt, int it, int iSource, float *p, float*q, float src)
{

	        OPENACC_InsertSource(dt,it,iSource,p,q, src);

}

