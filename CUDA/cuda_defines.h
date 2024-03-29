#ifndef __CUDA_DEFINES
#define __CUDA_DEFINES

#define restrict __restrict__
#define BSIZE_X 32
#define BSIZE_Y 8
#include <stdio.h>

#define CUDA_CALL(call) do{      \
   const cudaError_t err=call;         \
   if (err != cudaSuccess)       \
   {                             \
     fprintf(stderr, "CUDA ERROR: %s on %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__);\
     exit(1);                    \
   }}while(0)

#endif
