CC=$(GCC)
CFLAGS=-I/usr/local/cuda/include -O3 -fopenmp -DUNIFIED -g -pg
GPUCC=$(NVCC)
GPUCFLAGS= -DUNIFIED -g -pg -Xcompiler -fopenmp
LIBS=$(NVCC_LIBS) $(GCC_LIBS)