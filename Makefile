NVCC = nvcc
CC = clang

CUDAPATH = /usr/local/cuda

BUILD_DIR = build

main.out: src/*
	$(NVCC) src/*.cu -o main.out

clean:
	rm *.out
