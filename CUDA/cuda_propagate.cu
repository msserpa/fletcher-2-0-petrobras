#include "cuda_defines.h"
#include "cuda_propagate.h"
#include "../derivatives.h"
#include "../map.h"


__global__ void kernel_Propagate(const int sx, const int sy, const int sz, const int bord,
	       const float dx, const float dy, const float dz, const float dt, const int it, 
	       float * restrict ch1dxx, float * restrict ch1dyy, float * restrict ch1dzz, 
	       float * restrict ch1dxy, float * restrict ch1dyz, float * restrict ch1dxz, 
	       float * restrict v2px, float * restrict v2pz, float * restrict v2sz, float * restrict v2pn,
	       float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc)
{
  const int ix=blockIdx.x * blockDim.x + threadIdx.x;
  const int iy=blockIdx.y * blockDim.y + threadIdx.y;

#define SAMPLE_PRE_LOOP
#include "../sample.h"
#undef SAMPLE_PRE_LOOP

    // solve both equations in all internal grid points, 
    // including absortion zone
    
    for (int iz=bord+1; iz<sz-bord-1; iz++) {

#define SAMPLE_LOOP
#include "../sample.h"
#undef SAMPLE_LOOP

    }
}

// Propagate: using Fletcher's equations, propagate waves one dt,
//            either forward or backward in time
void CUDA_Propagate(const int sx, const int sy, const int sz, const int bord,
         const float dx, const float dy, const float dz, const float dt, const int it)
{

   extern float* vpz;
   extern float* vsv;
   extern float* epsilon;
   extern float* delta;
   extern float* phi;
   extern float* theta;
   extern float* ch1dxx;
   extern float* ch1dyy;
   extern float* ch1dzz;
   extern float* ch1dxy;
   extern float* ch1dyz;
   extern float* ch1dxz;
   extern float* v2px;
   extern float* v2pz;
   extern float* v2sz;
   extern float* v2pn;
   extern float* pp;
   extern float* pc;
   extern float* qp;
   extern float* qc;
   extern float* fatAbsorb;


  dim3 threadsPerBlock(BSIZE_X, BSIZE_Y);
  dim3 numBlocks(sx/threadsPerBlock.x, sy/threadsPerBlock.y);
  
  kernel_Propagate <<<numBlocks, threadsPerBlock>>> (  sx,   sy,   sz,   bord,
	         dx,   dy,   dz,   dt,   it, 
	        ch1dxx,  ch1dyy,  ch1dzz, 
	        ch1dxy,  ch1dyz,  ch1dxz, 
	        v2px,  v2pz,  v2sz,  v2pn,
	        pp,  pc,  qp,  qc);

  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaDeviceSynchronize());

  CUDA_SwapArrays(&pp, &pc, &qp, &qc);
}

// swap array pointers on time forward array propagation
void CUDA_SwapArrays(float **pp, float **pc, float **qp, float **qc) {
  float *tmp;

  tmp=*pp;
  *pp=*pc;
  *pc=tmp;

  tmp=*qp;
  *qp=*qc;
  *qc=tmp;
}
