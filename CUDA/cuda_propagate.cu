#include "cuda_defines.h"
#include "cuda_propagate.h"
#include "../derivatives.h"
#include "../map.h"

extern int deviceCount;


__global__ void kernel_Propagate(const int sx, const int sy, const int sz, const int bord,
	       const float dx, const float dy, const float dz, const float dt, const int it, 
	       float * restrict ch1dxx, float * restrict ch1dyy, float * restrict ch1dzz, 
	       float * restrict ch1dxy, float * restrict ch1dyz, float * restrict ch1dxz, 
	       float * restrict v2px, float * restrict v2pz, float * restrict v2sz, float * restrict v2pn,
	       float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc, const int dev)
{
  const int ix=(blockIdx.x * blockDim.x + threadIdx.x) + (dev * gridDim.x * blockDim.x);
  const int iy=(blockIdx.y * blockDim.y + threadIdx.y);

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
	       const float dx, const float dy, const float dz, const float dt, const int it, 
	       float * restrict ch1dxx, float * restrict ch1dyy, float * restrict ch1dzz, 
	       float * restrict ch1dxy, float * restrict ch1dyz, float * restrict ch1dxz, 
	       float * restrict v2px, float * restrict v2pz, float * restrict v2sz, float * restrict v2pn,
        #ifdef UNIFIED
	       float *pp, float *pc, float *qp, float *qc
        #else
         float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc
        #endif
        )

{
  static int print;
  #ifndef UNIFIED
   extern float* dev_vpz;
   extern float* dev_vsv;
   extern float* dev_epsilon;
   extern float* dev_delta;
   extern float* dev_phi;
   extern float* dev_theta;
   extern float* dev_ch1dxx;
   extern float* dev_ch1dyy;
   extern float* dev_ch1dzz;
   extern float* dev_ch1dxy;
   extern float* dev_ch1dyz;
   extern float* dev_ch1dxz;
   extern float* dev_v2px;
   extern float* dev_v2pz;
   extern float* dev_v2sz;
   extern float* dev_v2pn;
   extern float* dev_pp;
   extern float* dev_pc;
   extern float* dev_qp;
   extern float* dev_qc;
   extern float* dev_fatAbsorb;
  #endif


  dim3 threadsPerBlock(BSIZE_X, BSIZE_Y);
  #ifdef UNIFIED
  dim3 numBlocks(sx/threadsPerBlock.x / deviceCount, sy/threadsPerBlock.y);
  #else
  dim3 numBlocks(sx/threadsPerBlock.x, sy/threadsPerBlock.y);
  #endif

  if(!print){
    print = 1;
    printf("sx=%d sy=%d sz=%d\n", sx, sy, sz);
    printf("running kernel_Propagate with (%d,%d) blocks of (%d,%d) threads\n", numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);
  }
  #ifdef UNIFIED
  int d;
  for(d = deviceCount - 1; d >= 0; d--){
    cudaSetDevice(d);
  kernel_Propagate <<<numBlocks, threadsPerBlock>>> (  sx,   sy,   sz,   bord,
	         dx,   dy,   dz,   dt,   it, 
	        ch1dxx,  ch1dyy,  ch1dzz, 
	        ch1dxy,  ch1dyz,  ch1dxz, 
	        v2px,  v2pz,  v2sz,  v2pn,
	        pp,  pc,  qp,  qc, d);
      
      CUDA_CALL(cudaGetLastError());
      CUDA_CALL(cudaDeviceSynchronize());
    }
  #else
  kernel_Propagate <<<numBlocks, threadsPerBlock>>> (  sx,   sy,   sz,   bord,
           dx,   dy,   dz,   dt,   it, 
          dev_ch1dxx,  dev_ch1dyy,  dev_ch1dzz, 
          dev_ch1dxy,  dev_ch1dyz,  dev_ch1dxz, 
          dev_v2px,  dev_v2pz,  dev_v2sz,  dev_v2pn,
          dev_pp,  dev_pc,  dev_qp,  dev_qc);  
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());
  #endif

  #ifdef UNIFIED
    CUDA_SwapArrays(&pp, &pc, &qp, &qc);
  #else
    CUDA_SwapArrays(&dev_pp, &dev_pc, &dev_qp, &dev_qc);
  #endif
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
