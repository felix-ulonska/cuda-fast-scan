NVCC = nvcc
CC = clang

CUDAPATH = /usr/local/cuda

BUILD_DIR = build

main.out: src/main.cuh src/main.cu src/device.cu src/device.cuh
	$(NVCC) src/*.cu -o main.out

clean:
	rm *.out
