#include "../shared/shared.cuh"
#include "main.cuh"

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
