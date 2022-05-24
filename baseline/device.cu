#include "../shared/shared.cuh"
#include "parfor.cuh"
#include <stdio.h>
#include "main.cuh"


__device__ void t_mem_cpy(int* dest, int* src) {
    // Copy into shared
    // TODO do stride things for perfomance
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      dest[i] = src[i];
    } 
}

__device__ int t_tree_reduction(int* a) {
  int sum = a[0];
  for (int i = 1; i < ITEMS_PER_THREAD; i++) {
    sum = bin_op(sum, a[i]);
  } 
  return sum;
}

__device__ void b_tree_reduction(int* a) {
    for (int d = 1; d < blockDim.x; d *= 2) {
      if (threadIdx.x % (d * 2) == 0 && threadIdx.x + d < blockDim.x) {
        a[threadIdx.x] =
            bin_op(a[threadIdx.x + d], a[threadIdx.x]);
      }
      __syncthreads();
    }
}

__device__ void b_set_partition_descriptor(PartitionDescriptor* partDesc, int aggregate) {
  if (blockIdx.x != 0) {
    partDesc->aggregate = aggregate;       
    __threadfence();
    partDesc->flag = FLAG_AGGREGATE;
  } else if (threadIdx.x == 0 && blockIdx.x == 0) {
    partDesc->inclusive_prefix = aggregate;       
    // Technically the aggregate is not needed
    partDesc->aggregate = aggregate;
    __threadfence();
    partDesc->flag = FLAG_INCLUSIVE_PREFIX;
  }
}

__device__ void b_get_exclusive_prefix(PartitionDescriptor* states, int* exclusive_prefix_location) {
  int exclusive_prefix = 0;
  int end_index = blockIdx.x - WINDOW;
  if (end_index < 0) end_index = 0;
  PartitionDescriptor *end = &states[end_index];
  bool done = false;
  while (!done) {
    exclusive_prefix = 0;
    PartitionDescriptor *currState = &states[blockIdx.x];

    while (currState != end) {
      currState--;
      int flag = currState->flag;
      __threadfence();

      if (flag == FLAG_BLOCK) {
        break;
      } else if (flag == FLAG_AGGREGATE) {
        exclusive_prefix = bin_op(currState->aggregate, exclusive_prefix);
      } else if (flag == FLAG_INCLUSIVE_PREFIX) {
        exclusive_prefix =
            bin_op(currState->inclusive_prefix, exclusive_prefix);
        done = true;
        break;
      }
    }
  }
  *exclusive_prefix_location = exclusive_prefix;
}

__device__ void b_scan(int* a) {
  int *currDest = a;
  int *currSrc = a;

  // Setting first value
  *currDest = *currSrc;

  // Moving the pointers through the array and using the last value to calc
  // the next value
  do {
    int nextVal = bin_op(*(++currSrc), *currDest);
    *(++currDest) = nextVal;
  } while (currDest != &a[ITEMS_PER_BLOCK - 1]);
}

__device__ void t_bin_op(int* dest, int* src, int addValue) {
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    if (blockIdx.x > 0) {
      dest[i] = bin_op(src[i], addValue);
    } else {
      dest[i] = src[i];
    }
  }
}

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
