#include "cuda_defines.h"
#include "cuda_stuff.h"


void CUDA_Initialize(const int rank, const int sx, const int sy, const int sz, const int bord,
               float dx, float dy, float dz, float dt)
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

 
  // Set the device number based on rank
  int deviceCount;
  CUDA_CALL(cudaGetDeviceCount(&deviceCount));
  const int device=rank%deviceCount;
  cudaDeviceProp deviceProp;
  CUDA_CALL(cudaGetDeviceProperties(&deviceProp, device));
  printf("Using device(%d) %s with compute capability %d.%d.\n", device, deviceProp.name, deviceProp.major, deviceProp.minor);
  CUDA_CALL(cudaSetDevice(device));


  // Check sx,sy values
  if (sx%BSIZE_X != 0)
  {
     printf("sx(%d) must be multiple of BSIZE_X(%d)\n", sx, (int)BSIZE_X);
     exit(1);
  } 
  if (sy%BSIZE_Y != 0)
  {
     printf("sy(%d) must be multiple of BSIZE_Y(%d)\n", sy, (int)BSIZE_Y);
     exit(1);
  } 

   const size_t sxsysz=((size_t)sx*sy)*sz;
   const size_t msize_vol=sxsysz*sizeof(float);

   CUDA_CALL(cudaMallocManaged(&vpz, msize_vol));
   //dev_vpz, vpz, msize_vol, cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMallocManaged(&vsv, msize_vol));
   //dev_vsv, vsv, msize_vol, cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMallocManaged(&epsilon, msize_vol));
   //dev_epsilon, epsilon, msize_vol, cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMallocManaged(&delta, msize_vol));
   //dev_delta, delta, msize_vol, cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMallocManaged(&phi, msize_vol));
   //dev_phi, phi, msize_vol, cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMallocManaged(&theta, msize_vol));
   //dev_theta, theta, msize_vol, cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMallocManaged(&ch1dxx, msize_vol));
   //dev_ch1dxx, ch1dxx, msize_vol, cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMallocManaged(&ch1dyy, msize_vol));
   //dev_ch1dyy, ch1dyy, msize_vol, cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMallocManaged(&ch1dzz, msize_vol));
   //dev_ch1dzz, ch1dzz, msize_vol, cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMallocManaged(&ch1dxy, msize_vol));
   //dev_ch1dxy, ch1dxy, msize_vol, cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMallocManaged(&ch1dyz, msize_vol));
   //dev_ch1dyz, ch1dyz, msize_vol, cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMallocManaged(&ch1dxz, msize_vol));
   //dev_ch1dxz, ch1dxz, msize_vol, cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMallocManaged(&v2px, msize_vol));
   //dev_v2px, v2px, msize_vol, cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMallocManaged(&v2pz, msize_vol));
   //dev_v2pz, v2pz, msize_vol, cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMallocManaged(&v2sz, msize_vol));
   //dev_v2sz, v2sz, msize_vol, cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMallocManaged(&v2pn, msize_vol));
   //dev_v2pn, v2pn, msize_vol, cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMallocManaged(&pp, msize_vol));
   CUDA_CALL(cudaMemset(pp, 0, msize_vol));
   CUDA_CALL(cudaMallocManaged(&pc, msize_vol));
   CUDA_CALL(cudaMemset(pc, 0, msize_vol));
   CUDA_CALL(cudaMallocManaged(&qp, msize_vol));
   CUDA_CALL(cudaMemset(qp, 0, msize_vol));
   CUDA_CALL(cudaMallocManaged(&qc, msize_vol));
   CUDA_CALL(cudaMemset(qc, 0, msize_vol));
   CUDA_CALL(cudaMallocManaged(&fatAbsorb, msize_vol));
   if (!fatAbsorb) //dev_fatAbsorb, fatAbsorb, msize_vol, cudaMemcpyHostToDevice));
             CUDA_CALL(cudaMemset(fatAbsorb, 0, msize_vol));

  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaDeviceSynchronize());
  printf("GPU memory usage = %ld MiB\n", 21*msize_vol/1024/1024);

}


void CUDA_Finalize()
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

   CUDA_CALL(cudaFree(vpz));
   CUDA_CALL(cudaFree(vsv));
   CUDA_CALL(cudaFree(epsilon));
   CUDA_CALL(cudaFree(delta));
   CUDA_CALL(cudaFree(phi));
   CUDA_CALL(cudaFree(theta));
   CUDA_CALL(cudaFree(ch1dxx));
   CUDA_CALL(cudaFree(ch1dyy));
   CUDA_CALL(cudaFree(ch1dzz));
   CUDA_CALL(cudaFree(ch1dxy));
   CUDA_CALL(cudaFree(ch1dyz));
   CUDA_CALL(cudaFree(ch1dxz));
   CUDA_CALL(cudaFree(v2px));
   CUDA_CALL(cudaFree(v2pz));
   CUDA_CALL(cudaFree(v2sz));
   CUDA_CALL(cudaFree(v2pn));
   CUDA_CALL(cudaFree(pp));
   CUDA_CALL(cudaFree(pc));
   CUDA_CALL(cudaFree(qp));
   CUDA_CALL(cudaFree(qc));
   CUDA_CALL(cudaFree(fatAbsorb));

   printf("CUDA_Finalize: SUCCESS\n");
}



// void CUDA_Update_pointers(const int sx, const int sy, const int sz, float *pc)
// {
//    extern float* pc;
//    const size_t sxsysz=((size_t)sx*sy)*sz;
//    const size_t msize_vol=sxsysz*sizeof(float);
//    // if (pc) //pc, dev_pc, msize_vol, cudaMemcpyDeviceToHost));
// }
