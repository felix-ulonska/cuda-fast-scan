NVCC = nvcc
CUDA_OPTIONS = --relocatable-device-code=true
CC = clang

CUDAPATH = /usr/local/cuda

BUILD_DIR = build
BUILD_DIR_BASELINE = $(BUILD_DIR)/baseline
BUILD_DIR_SHARED = $(BUILD_DIR)/shared
BIN_DIR = bin

.phony = clean runBaseline runCheck format

.DEFAULT_GOAL := $(BIN_DIR)/baseline 

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

runCheck: ${BIN_DIR}/baseline
	@echo "Running memcheck"
	cuda-memcheck ${BIN_DIR}/baseline
	@echo "Running racecheck"
	cuda-memcheck ${BIN_DIR}/baseline --tool racecheck
	@echo "Running synccheck"
	cuda-memcheck ${BIN_DIR}/baseline --tool synccheck
	@echo "Running initcheck"
	cuda-memcheck ${BIN_DIR}/baseline --tool initcheck

format:
	find -iname *.cuh -o -iname *.cu | xargs clang-format -i

clean:
	rm -rf build
	rm -rf bin
