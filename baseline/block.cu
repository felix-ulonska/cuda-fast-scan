#include "../shared/shared.cuh"
#include "main.cuh"
#include "thread.cuh"
#include <stdio.h>
#include "params.cuh"

__device__ void b_tree_reduction(int* a) {
  for (int d = 1; d < blockDim.x; d *= 2) {
    // TODO move to thread level
    // TODO remove second part of if
    if (threadIdx.x % (d * 2) == 0) {
      a[threadIdx.x] =
          bin_op(a[threadIdx.x + d], a[threadIdx.x]);
    }
    __syncthreads();
  }
}

__device__ void b_set_partition_descriptor(volatile PartitionDescriptor* partDesc, int aggregate) {
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

__device__ void b_get_exclusive_prefix(volatile PartitionDescriptor* states, int* exclusive_prefix_location) {
  int exclusive_prefix = 0;
  int end_index = blockIdx.x - WINDOW;
  if (end_index < 0) end_index = 0;
  volatile PartitionDescriptor *end = &states[end_index];
  bool done = false;
  while (!done) {
    exclusive_prefix = 0;
    volatile PartitionDescriptor *currState = &states[blockIdx.x];

    while (currState != end) {
      currState--;
      int flag = currState->flag;
      __threadfence();
      int agg = currState->aggregate;
      
      if (flag == FLAG_BLOCK) {
        break;
      } else if (flag == FLAG_AGGREGATE) {
        if (agg != 1024) {
          printf("bad agg readed\n"); 
        }
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

// Sklansky scan
__device__ void b_scan(int* a) {
  t_scan(&a[threadIdx.x * ITEMS_PER_THREAD], ITEMS_PER_THREAD);
  __syncthreads();

  for (int k = 2; k <= blockDim.x; k *= 2) {
    int add_val_index = ((threadIdx.x / k) * k) + ((k / 2) - 1);
    // printf("tid: %d, k: %d, add_val %d\n", threadIdx.x, k, add_val);
    if (threadIdx.x % k >= k / 2) {
      t_bin_op(&a[threadIdx.x * ITEMS_PER_THREAD], &a[threadIdx.x * ITEMS_PER_THREAD], a[(add_val_index + 1) * ITEMS_PER_THREAD - 1]);
    }
    __syncthreads();
  }
}
