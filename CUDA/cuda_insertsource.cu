#include "cuda_defines.h"
#include "cuda_insertsource.h"

__global__ void kernel_InsertSource(const float val, const int iSource,
	                            float * restrict qp, float * restrict qc)
{
  const int ix=blockIdx.x * blockDim.x + threadIdx.x;
  if (ix==0)
  {
    qp[iSource]+=val;
    qc[iSource]+=val;
  }
}


#ifdef UNIFIED
  void CUDA_InsertSource(const float val, const int iSource, float *p, float *q, float *pp, float *qp)
#else
  void CUDA_InsertSource(const float val, const int iSource, float *p, float *q)
#endif
{
  static int print;

  #ifndef UNIFIED
    extern float* dev_pp;
    extern float* dev_pc;
    extern float* dev_qp;
    extern float* dev_qc;
  #endif

#ifdef UNIFIED	
  if ((pp) && (qp))
#else
  if ((dev_pp) && (dev_qp))
#endif
  {
     dim3 threadsPerBlock(BSIZE_X, 1);
     dim3 numBlocks(1,1);

    if(!print){
      print = 1;
    printf("running kernel_InsertSource with (%d,%d) blocks of (%d,%d) threads\n", numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);
    }
    
    #ifdef UNIFIED    
     kernel_InsertSource<<<numBlocks, threadsPerBlock>>> (val, iSource, p, q);
    #else
     kernel_InsertSource<<<numBlocks, threadsPerBlock>>> (val, iSource, dev_pc, dev_qc);
    #endif
     CUDA_CALL(cudaGetLastError());
     CUDA_CALL(cudaDeviceSynchronize());
  }
}
