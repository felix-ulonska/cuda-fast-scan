#include "../shared/shared.cuh"
#include <iostream>
#include "device.cuh"

#define VERSION "Baseline"

void init() {
}


int main() {
    print_motd();

    int block_size = 128; 
    int amount_blocks = 50;
    int amount_elems = block_size * amount_blocks;
    int size_of_input = sizeof(float) * amount_elems;
    int size_of_partition_descriptiors = sizeof(PartitionDescriptor) * amount_blocks;

    float *input;
    PartitionDescriptor *partition_descriptiors;

    if (cudaError error = cudaMallocManaged(&input, size_of_input)) {
	std::cerr << "[!] Cuda Malloc Managed for input failed with error" << cudaGetErrorName(error) << std::endl;
	return EXIT_FAILURE;
    }

    if (cudaError error = cudaMallocManaged(&partition_descriptiors, size_of_partition_descriptiors)) {
	std::cerr << "[!] Cuda Malloc Managed for state failed with error" << cudaGetErrorName(error) << std::endl;
	cudaFree(input);
	return EXIT_FAILURE;
    }


    init_array(input, amount_elems);
    init_state_arr(partition_descriptiors, amount_blocks);

    // I will assume that malloc will not fail.
    float *gold = (float *) malloc(size_of_input);
    scan_host(gold, input, amount_elems);

    std::cout << "[+] Starting kernel..." << std::endl;
    scan_lookback<<<amount_blocks, block_size, block_size * sizeof(float)>>>(input, partition_descriptiors);

    if (cudaError error = cudaDeviceSynchronize()) {
	std::cerr << "[!] Execution failed" << cudaGetErrorName(error) << std::endl;
	cudaFree(input);
	cudaFree(partition_descriptiors);
	free(gold);
	return EXIT_FAILURE;
    }

    if (arr_equal(gold, input, amount_elems)) {
	std::cout << "[+] result is correct" << std::endl;
    } else {
	std::cerr << "[!] Result is not correct" << std::endl;
	cudaFree(input);
	cudaFree(partition_descriptiors);
	free(gold);
	return EXIT_FAILURE;
    }


    cudaFree(input);
    cudaFree(partition_descriptiors);
    free(gold);
    return EXIT_SUCCESS;
}
