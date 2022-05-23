#include <iostream>

#include "../shared/shared.cuh"
#include "device.cuh"
#include "main.cuh"

#define VERSION "Baseline"

Result exec() {
  int *input;
  PartitionDescriptor *partition_descriptiors;

  if (cudaError error = cudaMallocManaged(&input, SIZE_OF_INPUT)) {
    std::cerr << "[!] Cuda Malloc Managed for input failed with error"
              << cudaGetErrorName(error) << std::endl;
    exit(EXIT_FAILURE);
  }

  if (cudaError error = cudaMallocManaged(&partition_descriptiors, SIZE_OF_PARTITION_DESCRIPTIORS)) {
    std::cerr << "[!] Cuda Malloc Managed for state failed with error"
              << cudaGetErrorName(error) << std::endl;
    cudaFree(input);
    exit(EXIT_FAILURE);
  }

  init_array(input, AMOUNT_ELEMS);
  init_state_arr(partition_descriptiors, AMOUNT_BLOCKS);

  // I will assume that malloc will not fail.
  int *gold = (int *)malloc(SIZE_OF_INPUT);
  scan_host(gold, input, AMOUNT_ELEMS);

  // std::cout << "[+] Starting kernel..." << std::endl;

  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  scan_kernel<<<AMOUNT_BLOCKS, THREADS_PER_BLOCK,
                  sizeof(int) * (THREADS_PER_BLOCK + ITEMS_PER_BLOCK)>>>(input,
                                                        partition_descriptiors);
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

  if (arr_equal(gold, input, AMOUNT_ELEMS)) {
    // std::cout << "[+] result is correct" << std::endl;
  } else {
    std::cerr << "[!] Result is not correct" << std::endl;

    for (int i = 0; i < AMOUNT_BLOCKS; i++) {
      printf("State %d got state %d and inclusive_prefix %d and agg %d\n", i,
             partition_descriptiors[i].flag,
             partition_descriptiors[i].inclusive_prefix,
             partition_descriptiors[i].aggregate);
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
  int iters = 1;

  Result results[iters];
  for (int i = 0; i < iters; i++) {
    results[i] = exec();
    printf(".");
  }
  printf("\n");

  int sum = 0;
  for (int i = 0; i < iters; i++) {
    int time = results[i].time;
    sum += time;

    // printf("time: %f\n", time);
  }

  printf("Avg: %d", sum / iters);
}
