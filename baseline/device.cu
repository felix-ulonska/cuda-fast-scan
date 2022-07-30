#include "../shared/shared.cuh"
#include <stdio.h>
#include "main.cuh"
#include "thread.cuh"
#include "block.cuh"
#include "params.cuh"

__global__ void scan_kernel(int *g_input, PartitionDescriptor volatile *states) {

  extern __shared__ int s[];

  // Parfor block
  {
    int* b_ptr_input = &g_input[ITEMS_PER_BLOCK * blockIdx.x];
    int* b_ptr_shared_reduction = &s[0];
    int* b_ptr_shared_input_copy = &s[blockDim.x];
    volatile PartitionDescriptor* partDesc = &states[blockIdx.x];

    for (int i = 0; i < ITEMS_PER_BLOCK; i += blockDim.x) {
      // Thread level
      if (b_ptr_input[i + threadIdx.x] != 1) {
        printf("BAD INPUT");
      };
      b_ptr_shared_input_copy[i + threadIdx.x] = b_ptr_input[i + threadIdx.x];
    }
    __syncthreads();


    // Parfor thread in block
    int* t_ptr_input = &b_ptr_input[threadIdx.x * ITEMS_PER_THREAD];
    int* t_ptr_shared_reduction = &b_ptr_shared_reduction[threadIdx.x];
    int* t_ptr_shared_input = &b_ptr_shared_input_copy[threadIdx.x * ITEMS_PER_THREAD];
    // PARFOR THREAD
    {
      // t_mem_cpy(t_ptr_shared_input, t_ptr_input);
      *t_ptr_shared_reduction = t_tree_reduction(t_ptr_shared_input);
    }

    __syncthreads();

    // Block Wide Reduction
    b_tree_reduction(b_ptr_shared_reduction);

    // Run Code only on one threadx 
    if (threadIdx.x == 0) {
      b_set_partition_descriptor(partDesc, b_ptr_shared_reduction[0]);
    }
    __syncthreads();

    if (threadIdx.x == 0 && blockIdx.x != 0) {
      b_get_exclusive_prefix(states, &b_ptr_shared_reduction[0]);
    }
    __syncthreads();

    if (!threadIdx.x && blockIdx.x != 0) {
      // TODO move to function
      partDesc->inclusive_prefix = bin_op(b_ptr_shared_reduction[0], partDesc->aggregate);       
      __threadfence();
      partDesc->flag = FLAG_INCLUSIVE_PREFIX;
    }
    __syncthreads();

    b_scan(b_ptr_shared_input_copy);
    __syncthreads();

    // Parfor thread
    {
      if (blockIdx.x == 0) {
        t_mem_cpy(t_ptr_input, t_ptr_shared_input);
      } else {
        // TODO fix global colleasing
        t_bin_op(t_ptr_input, t_ptr_shared_input, b_ptr_shared_reduction[0]);
      }
    }
  }
}
