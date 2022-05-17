#include <iostream>

#include "../shared/shared.cuh"
#include "device.cuh"
#include "main.cuh"

#define VERSION "Baseline"

Result exec(int block_size, int amount_blocks) {
  int amount_elems = block_size * amount_blocks;
  int size_of_input = sizeof(float) * amount_elems;
  int size_of_partition_descriptiors =
      sizeof(PartitionDescriptor) * amount_blocks;

  float *input;
  PartitionDescriptor *partition_descriptiors;

  if (cudaError error = cudaMallocManaged(&input, size_of_input)) {
    std::cerr << "[!] Cuda Malloc Managed for input failed with error"
              << cudaGetErrorName(error) << std::endl;
    exit(EXIT_FAILURE);
  }

  if (cudaError error = cudaMallocManaged(&partition_descriptiors,
                                          size_of_partition_descriptiors)) {
    std::cerr << "[!] Cuda Malloc Managed for state failed with error"
              << cudaGetErrorName(error) << std::endl;
    cudaFree(input);
    exit(EXIT_FAILURE);
  }

  init_array(input, amount_elems);
  init_state_arr(partition_descriptiors, amount_blocks);

  // I will assume that malloc will not fail.
  float *gold = (float *)malloc(size_of_input);
  scan_host(gold, input, amount_elems);

  // std::cout << "[+] Starting kernel..." << std::endl;

  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  scan_lookback<<<amount_blocks, block_size, block_size * sizeof(float) * 2>>>(
      input, partition_descriptiors);
  cudaEventRecord(stop);

  if (cudaError error = cudaDeviceSynchronize()) {
    std::cerr << "[!] Execution failed" << cudaGetErrorName(error) << std::endl;
    cudaFree(input);
    cudaFree(partition_descriptiors);
    free(gold);
    exit(EXIT_FAILURE);
  }

  if (cudaError error = cudaEventSynchronize(stop)) {
    std::cerr << "[!] Event Sync failed" << cudaGetErrorName(error)
              << std::endl;
    cudaFree(input);
    cudaFree(partition_descriptiors);
    free(gold);
    exit(EXIT_FAILURE);
  };

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  // printf("took %f ms\n", milliseconds);

  if (arr_equal(gold, input, amount_elems)) {
    // std::cout << "[+] result is correct" << std::endl;
  } else {
    std::cerr << "[!] Result is not correct" << std::endl;

    for (int i = 0; i < amount_blocks; i++) {

      printf("State %d got state %d and inclusive_prefix %f and agg %f\n", i,
             partition_descriptiors[i].flag,
             partition_descriptiors[i].inclusive_prefix,
             partition_descriptiors[i].aggregate);
      if (partition_descriptiors[i].aggregate != 128) {
        printf("BAD AGG");
        exit(1);
      }

      if (partition_descriptiors[i].inclusive_prefix != (i + 1) * 128) {
        printf("BAD inc");
        exit(1);
      }
    }

    cudaFree(input);
    cudaFree(partition_descriptiors);
    free(gold);
    exit(EXIT_FAILURE);
  }

  cudaFree(input);
  cudaFree(partition_descriptiors);
  free(gold);
  return Result{.time = milliseconds};
}

int main() {
  int iters = 200000;

  Result results[iters];
  for (int i = 0; i < iters; i++) {
    results[i] = exec(128, 512);
    printf("----\n");
  }
  printf("\n");

  float sum = 0;
  for (int i = 0; i < iters; i++) {
    float time = results[i].time;
    sum += time;

    // printf("time: %f\n", time);
  }

  printf("Avg: %f", sum / iters);
}
