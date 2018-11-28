#include "cuda_defines.h"
#include "cuda_stuff.h"
#ifdef UNIFIED
  #include "../fletcher.h"
  #include <string.h>
#endif

#ifdef UNIFIED
#include <omp.h>
extern int deviceCount;



__global__ void kernel_ArraysInit(float * restrict vpz, float * restrict vsv, float * restrict epsilon, float * restrict delta, float * restrict phi,
 float * restrict theta, float * restrict ch1dxx, float * restrict ch1dyy, float * restrict ch1dzz, float * restrict ch1dxy,
  float * restrict ch1dyz, float * restrict ch1dxz, float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc,
    float * restrict fatAbsorb, float * restrict v2px, float * restrict v2pz, float * restrict v2sz, float * restrict v2pn, const int N){
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  float sinTheta, cosTheta, sin2Theta, sinPhi, cosPhi, sin2Phi;
  if(i < N){
    vpz[i]=3000.0;
    epsilon[i]=0.24;
    delta[i]=0.1;
    phi[i]=0.0;
    theta[i]=atanf(1.0);
    if(SIGMA > MAX_SIGMA)
      vsv[i] = 0.0;
    else
      vsv[i] = vpz[i] * sqrtf(fabsf(epsilon[i] - delta[i]) / SIGMA);

    sinTheta=sin(theta[i]);
    cosTheta=cos(theta[i]);
    sin2Theta=sin(2.0*theta[i]);
    sinPhi=sin(phi[i]);
    cosPhi=cos(phi[i]);
    sin2Phi=sin(2.0*phi[i]);
    ch1dxx[i]=sinTheta*sinTheta * cosPhi*cosPhi;
    ch1dyy[i]=sinTheta*sinTheta * sinPhi*sinPhi;
    ch1dzz[i]=cosTheta*cosTheta;
    ch1dxy[i]=sinTheta*sinTheta * sin2Phi;
    ch1dyz[i]=sin2Theta         * sinPhi;
    ch1dxz[i]=sin2Theta         * cosPhi;

    pp[i] = 0.0f; pc[i] = 0.0f;
    qp[i] = 0.0f; qc[i] = 0.0f;

    v2sz[i] = vsv[i] * vsv[i];
    v2pz[i] = vpz[i] * vpz[i];
    v2px[i] = v2pz[i] * (1.0 + 2.0 * epsilon[i]);
    v2pn[i] = v2pz[i] * (1.0 + 2.0 * delta[i]);    
  }
}

void ArraysInit(float * restrict vpz, float * restrict vsv, float * restrict epsilon, float * restrict delta, float * restrict phi,
 float * restrict theta, float * restrict ch1dxx, float * restrict ch1dyy, float * restrict ch1dzz, float * restrict ch1dxy,
  float * restrict ch1dyz, float * restrict ch1dxz, float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc,
    float * restrict fatAbsorb, float * restrict v2px, float * restrict v2pz, float * restrict v2sz, float * restrict v2pn, const int N){
      int block = BSIZE_X;
      int numBlocks = (N + block - 1) / block;
      kernel_ArraysInit<<<numBlocks, block>>>(vpz, vsv, epsilon, delta, phi, theta, ch1dxx, ch1dyy, ch1dzz, ch1dxy, ch1dyz, ch1dxz,
        pp, pc, qp, qc, fatAbsorb, v2px, v2pz, v2sz, v2pn, N);
}


void GPU_Initialize(){
  int device;
  // CUDA_CALL(cudaGetDeviceCount(&deviceCount));
  deviceCount = 4;
  for(device = deviceCount - 1; device >= 0; device--){
    cudaDeviceProp deviceProp;
    CUDA_CALL(cudaGetDeviceProperties(&deviceProp, device));
    printf("Using device(%d) %s with compute capability %d.%d.\n", device, deviceProp.name, deviceProp.major, deviceProp.minor);
    CUDA_CALL(cudaSetDevice(device));
  }

}
#endif

void CUDA_Initialize(const int rank, const int sx, const int sy, const int sz, const int bord,
	       float dx, float dy, float dz, float dt,
	       float * restrict ch1dxx, float * restrict ch1dyy, float * restrict ch1dzz, 
	       float * restrict ch1dxy, float * restrict ch1dyz, float * restrict ch1dxz, 
	       float * restrict v2px, float * restrict v2pz, float * restrict v2sz, float * restrict v2pn,
	       float * restrict vpz, float * restrict vsv, float * restrict epsilon, float * restrict delta,
	       float * restrict phi, float * restrict theta, float * restrict fatAbsorb,
	       float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc)
{

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
 
  // Set the device number based on rank
  #ifndef UNIFIED
  CUDA_CALL(cudaGetDeviceCount(&deviceCount));
  const int device=rank%deviceCount;
  cudaDeviceProp deviceProp;
  CUDA_CALL(cudaGetDeviceProperties(&deviceProp, device));
  printf("Using device(%d) %s with compute capability %d.%d.\n", device, deviceProp.name, deviceProp.major, deviceProp.minor);
  CUDA_CALL(cudaSetDevice(device));
  #endif

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
   #ifdef UNIFIED
    memset(pp, 0, msize_vol);
    memset(pc, 0, msize_vol);
    memset(qp, 0, msize_vol);
    memset(qc, 0, msize_vol);
      #if defined(_ABSOR_SQUARE) || defined(_ABSOR_SPHERE)
       if (!fatAbsorb) 
        memset(fatAbsorb, 0, msize_vol);
      #endif
   #else
     CUDA_CALL(cudaMalloc(&dev_vpz, msize_vol));
     CUDA_CALL(cudaMemcpy(dev_vpz, vpz, msize_vol, cudaMemcpyHostToDevice));
     CUDA_CALL(cudaMalloc(&dev_vsv, msize_vol));
     CUDA_CALL(cudaMemcpy(dev_vsv, vsv, msize_vol, cudaMemcpyHostToDevice));
     CUDA_CALL(cudaMalloc(&dev_epsilon, msize_vol));
     CUDA_CALL(cudaMemcpy(dev_epsilon, epsilon, msize_vol, cudaMemcpyHostToDevice));
     CUDA_CALL(cudaMalloc(&dev_delta, msize_vol));
     CUDA_CALL(cudaMemcpy(dev_delta, delta, msize_vol, cudaMemcpyHostToDevice));
     CUDA_CALL(cudaMalloc(&dev_phi, msize_vol));
     CUDA_CALL(cudaMemcpy(dev_phi, phi, msize_vol, cudaMemcpyHostToDevice));
     CUDA_CALL(cudaMalloc(&dev_theta, msize_vol));
     CUDA_CALL(cudaMemcpy(dev_theta, theta, msize_vol, cudaMemcpyHostToDevice));
     CUDA_CALL(cudaMalloc(&dev_ch1dxx, msize_vol));
     CUDA_CALL(cudaMemcpy(dev_ch1dxx, ch1dxx, msize_vol, cudaMemcpyHostToDevice));
     CUDA_CALL(cudaMalloc(&dev_ch1dyy, msize_vol));
     CUDA_CALL(cudaMemcpy(dev_ch1dyy, ch1dyy, msize_vol, cudaMemcpyHostToDevice));
     CUDA_CALL(cudaMalloc(&dev_ch1dzz, msize_vol));
     CUDA_CALL(cudaMemcpy(dev_ch1dzz, ch1dzz, msize_vol, cudaMemcpyHostToDevice));
     CUDA_CALL(cudaMalloc(&dev_ch1dxy, msize_vol));
     CUDA_CALL(cudaMemcpy(dev_ch1dxy, ch1dxy, msize_vol, cudaMemcpyHostToDevice));
     CUDA_CALL(cudaMalloc(&dev_ch1dyz, msize_vol));
     CUDA_CALL(cudaMemcpy(dev_ch1dyz, ch1dyz, msize_vol, cudaMemcpyHostToDevice));
     CUDA_CALL(cudaMalloc(&dev_ch1dxz, msize_vol));
     CUDA_CALL(cudaMemcpy(dev_ch1dxz, ch1dxz, msize_vol, cudaMemcpyHostToDevice));
     CUDA_CALL(cudaMalloc(&dev_v2px, msize_vol));
     CUDA_CALL(cudaMemcpy(dev_v2px, v2px, msize_vol, cudaMemcpyHostToDevice));
     CUDA_CALL(cudaMalloc(&dev_v2pz, msize_vol));
     CUDA_CALL(cudaMemcpy(dev_v2pz, v2pz, msize_vol, cudaMemcpyHostToDevice));
     CUDA_CALL(cudaMalloc(&dev_v2sz, msize_vol));
     CUDA_CALL(cudaMemcpy(dev_v2sz, v2sz, msize_vol, cudaMemcpyHostToDevice));
     CUDA_CALL(cudaMalloc(&dev_v2pn, msize_vol));
     CUDA_CALL(cudaMemcpy(dev_v2pn, v2pn, msize_vol, cudaMemcpyHostToDevice));
     CUDA_CALL(cudaMalloc(&dev_pp, msize_vol));
     CUDA_CALL(cudaMemset(dev_pp, 0, msize_vol));
     CUDA_CALL(cudaMalloc(&dev_pc, msize_vol));
     CUDA_CALL(cudaMemset(dev_pc, 0, msize_vol));
     CUDA_CALL(cudaMalloc(&dev_qp, msize_vol));
     CUDA_CALL(cudaMemset(dev_qp, 0, msize_vol));
     CUDA_CALL(cudaMalloc(&dev_qc, msize_vol));
     CUDA_CALL(cudaMemset(dev_qc, 0, msize_vol));
     #if defined(_ABSOR_SQUARE) || defined(_ABSOR_SPHERE)
       CUDA_CALL(cudaMalloc(&dev_fatAbsorb, msize_vol));
       if (fatAbsorb) CUDA_CALL(cudaMemcpy(dev_fatAbsorb, fatAbsorb, msize_vol, cudaMemcpyHostToDevice));
       else           CUDA_CALL(cudaMemset(dev_fatAbsorb, 0, msize_vol));
     #endif
   #endif

  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaDeviceSynchronize());
  printf("GPU memory usage = %ld MiB\n", 21*msize_vol/1024/1024);

}


void CUDA_Finalize()
{

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

   CUDA_CALL(cudaFree(dev_vpz));
   CUDA_CALL(cudaFree(dev_vsv));
   CUDA_CALL(cudaFree(dev_epsilon));
   CUDA_CALL(cudaFree(dev_delta));
   CUDA_CALL(cudaFree(dev_phi));
   CUDA_CALL(cudaFree(dev_theta));
   CUDA_CALL(cudaFree(dev_ch1dxx));
   CUDA_CALL(cudaFree(dev_ch1dyy));
   CUDA_CALL(cudaFree(dev_ch1dzz));
   CUDA_CALL(cudaFree(dev_ch1dxy));
   CUDA_CALL(cudaFree(dev_ch1dyz));
   CUDA_CALL(cudaFree(dev_ch1dxz));
   CUDA_CALL(cudaFree(dev_v2px));
   CUDA_CALL(cudaFree(dev_v2pz));
   CUDA_CALL(cudaFree(dev_v2sz));
   CUDA_CALL(cudaFree(dev_v2pn));
   CUDA_CALL(cudaFree(dev_pp));
   CUDA_CALL(cudaFree(dev_pc));
   CUDA_CALL(cudaFree(dev_qp));
   CUDA_CALL(cudaFree(dev_qc));
   CUDA_CALL(cudaFree(dev_fatAbsorb));
  #endif

   printf("CUDA_Finalize: SUCCESS\n");
}



void CUDA_Update_pointers(const int sx, const int sy, const int sz, float *pc)
{
  #ifndef UNIFIED
   extern float* dev_pc;
   const size_t sxsysz=((size_t)sx*sy)*sz;
   const size_t msize_vol=sxsysz*sizeof(float);  
   if (pc) CUDA_CALL(cudaMemcpy(pc, dev_pc, msize_vol, cudaMemcpyDeviceToHost));
  #endif
}
