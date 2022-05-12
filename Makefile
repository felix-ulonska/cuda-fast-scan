NVCC = nvcc
CUDA_OPTIONS = --relocatable-device-code=true
CC = clang

CUDAPATH = /usr/local/cuda

BUILD_DIR = build
BUILD_DIR_BASELINE = $(BUILD_DIR)/baseline
BUILD_DIR_SHARED = $(BUILD_DIR)/shared
BIN_DIR = bin

.phony = clean runBaseline

$(BUILD_DIR_BASELINE)/%.o: baseline/%.cu* 
	mkdir -p $(BUILD_DIR_BASELINE)
	$(NVCC) -c $< -o $@ $(CUDA_OPTIONS)

$(BUILD_DIR_SHARED)/%.o: shared/shared.cu shared/shared.cuh
	mkdir -p $(BUILD_DIR_SHARED)
	$(NVCC) -c $< -o $@ $(CUDA_OPTIONS)

$(BIN_DIR)/baseline: $(BUILD_DIR_BASELINE)/main.o $(BUILD_DIR_BASELINE)/device.o $(BUILD_DIR_SHARED)/shared.o
	mkdir -p bin
	$(NVCC) $^ -o $@ $(CUDA_OPTIONS)

runBaseline: ${BIN_DIR}/baseline
	./$^

clean:
	rm -rf build
	rm -rf bin
