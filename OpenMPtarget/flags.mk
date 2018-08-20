# compilation with CLANG for OpenMP target
CC=$(CLANG)
CFLAGS=-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -g -nocudalib
LIBS=$(CLANG_LIBS)
