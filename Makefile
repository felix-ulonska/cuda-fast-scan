NVCC = nvcc
CC = clang

CUDAPATH = /usr/local/cuda

BUILD_DIR = build

.phony = run

main.out: src/*
	$(NVCC) src/*.cu -o main.out

run: main.out
	./main.out

clean:
	rm *.out
