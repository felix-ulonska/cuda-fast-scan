NVCC = nvcc
CUDA_OPTIONS = --relocatable-device-code=true

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
	@echo "[+] Running memcheck"
	compute-sanitizer ${BIN_DIR}/baseline | grep 'ERROR SUMMARY: 0 errors'
	@echo "[+] Running racecheck"
	compute-sanitizer ${BIN_DIR}/baseline --tool racecheck | grep 'ERROR SUMMARY: 0 errors'
	@echo "[+] Running synccheck"
	compute-sanitizer ${BIN_DIR}/baseline --tool synccheck | grep 'ERROR SUMMARY: 0 errors'
	@echo "[+] Running initcheck"
	compute-sanitizer ${BIN_DIR}/baseline --tool initcheck | grep 'ERROR SUMMARY: 0 errors'
	@echo "[!] Check for Makefile error code!"

format:
	find -iname *.cuh -o -iname *.cu | xargs clang-format -i -style=Google

clean:
	rm -rf build
	rm -rf bin
