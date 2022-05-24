#include "../shared/shared.cuh"
#include <stdio.h>
#include "main.cuh"
#include "thread.cuh"
#include "block.cuh"

__global__ void scan_kernel(int *a, PartitionDescriptor *states) {

  extern __shared__ int s[];

  // Parfor block
  {
    int* b_ptr_input = &a[ITEMS_PER_BLOCK * blockIdx.x];
    int* b_ptr_shared_reduction = &s[0];
    int* b_ptr_shared_input_copy = &s[blockDim.x];
    PartitionDescriptor* partDesc = &states[blockIdx.x];

    // Parfor thread copy into shared memory
    {
      int* t_ptr_input = &b_ptr_input[threadIdx.x * ITEMS_PER_THREAD];
      int* t_ptr_shared_reduction = &b_ptr_shared_reduction[threadIdx.x];
      int* t_ptr_shared_input = &b_ptr_shared_input_copy[threadIdx.x * ITEMS_PER_THREAD];

      t_mem_cpy(t_ptr_shared_input, t_ptr_input);
    }
    __syncthreads();

    // PARFOR for reduction
    {
      int* t_ptr_input = &b_ptr_input[threadIdx.x * ITEMS_PER_THREAD];
      int* t_ptr_shared_reduction = &b_ptr_shared_reduction[threadIdx.x];
      int* t_ptr_shared_input = &b_ptr_shared_input_copy[threadIdx.x * ITEMS_PER_THREAD];

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
      partDesc->inclusive_prefix = bin_op(b_ptr_shared_reduction[0], partDesc->aggregate);       
      __threadfence();
      partDesc->flag = FLAG_INCLUSIVE_PREFIX;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
      b_scan(b_ptr_shared_input_copy);
    }

    // Parfor binOp
    {
      int* t_ptr_input = &b_ptr_input[threadIdx.x * ITEMS_PER_THREAD];
      int* t_ptr_shared_reduction = &b_ptr_shared_reduction[threadIdx.x];
      int* t_ptr_shared_input = &b_ptr_shared_input_copy[threadIdx.x * ITEMS_PER_THREAD];

      t_bin_op(t_ptr_input, t_ptr_shared_input, b_ptr_shared_reduction[0]);
    }
  };
}
