CC=$(GCC)
CFLAGS= -L/usr/local/cuda-10.0/lib64/ -O3 -fopenmp
GPUCC=$(NVCC)
GPUCFLAGS=
LIBS=$(NVCC_LIBS) $(GCC_LIBS)
