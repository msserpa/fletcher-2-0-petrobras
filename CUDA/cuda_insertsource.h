#ifndef __CUDA_SOURCE
#define __CUDA_SOURCE

#ifdef __cplusplus
extern "C" {
#endif

#ifdef UNIFIED
	void CUDA_InsertSource(const float val, const int iSource, float *p, float *q, float *pp, float *qp);
#else
	void CUDA_InsertSource(const float val, const int iSource, float *p, float *q);
#endif

#ifdef __cplusplus
}
#endif

#endif
