# config include file for fletcher
# common for all backends

# Compilers
GCC=gcc
NVCC=nvcc
PGCC=pgcc
CLANG=clang

# Library paths
GCC_LIBS=-lm
NVCC_LIBS=-lcudart -lstdc++    # it may include CUDA lib64 path...
PGCC_LIBS=-lm
CLANG_LIBS=-lm

# Common flags
# Do not put compiler specific flag here (like those used for optimization)
COMMON_FLAGS=-DSAMPLE_ORIGINAL -DDERIVATIVES_ORIGINAL
