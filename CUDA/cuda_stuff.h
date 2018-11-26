#ifndef _CUDA_STUFF
#define _CUDA_STUFF

#ifdef __cplusplus
extern "C" {
#endif

void CUDA_Initialize(const int rank, const int sx, const int sy, const int sz, const int bord,
               float dx, float dy, float dz, float dt);

void CUDA_Finalize();

void CUDA_Update_pointers(const int sx, const int sy, const int sz, float *pc);

#ifdef __cplusplus
}
#endif
#endif

