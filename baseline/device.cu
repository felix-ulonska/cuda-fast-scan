#include "../shared/shared.cuh"
#include <stdio.h>
#define WINDOW 3

__device__ void debug_print(int i) {
  return;

  if (threadIdx.x == 0) {
    printf("%d: %d\n", i, blockIdx.x);
  }
}

/// Output will be written to base_ptr[0]
__device__ void compute_partition_wide_aggregate(float *base_ptr) {
  for (int d = 1; d <= blockDim.x; d *= 2) {
    if (threadIdx.x % (d * 2) == 0) {
      base_ptr[threadIdx.x] =
          bin_op(base_ptr[threadIdx.x + d], base_ptr[threadIdx.x]);
    }
    __syncthreads();
  }
}

/**
 * Each trehad copies one item into the shared memory
 */
__device__ void copy_one_item_per_thread(float *dest, float *src) {
  dest[threadIdx.x] = src[threadIdx.x];
}

__device__ void record_partition_wide_aggregate(PartitionDescriptor *state,
                                                float reduction_result) {
  // Only let first thread of block do this
  if (threadIdx.x != 0)
    return;

  state->aggregate = reduction_result;
  __threadfence();
  state->flag = FLAG_AGGREGATE;

  if (blockIdx.x == 0) {
    state->inclusive_prefix = reduction_result;
    __threadfence();
    state->flag = FLAG_INCLUSIVE_PREFIX;
  }
}

/**
 * Implementing the
 * https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda*
 * A Sum Scan Algorithm That Is Not Work-Efficient
 */
__device__ void partition_wide_scan(float *x) {
  int k = threadIdx.x;
  for (int d = 1; d < blockDim.x; d *= 2) {
    if (k >= d) {
      x[k] = bin_op(x[k - d], x[k]);
    }
    __syncthreads();
  }
}

__device__ void partition_wide_bin_op(float *x, float b) {
  x[threadIdx.x] = bin_op(x[threadIdx.x], b);
}

// THIS SHOULD ONLY BE CALLED BY ONE THREAD!
__device__ float
determine_partitions_exclusive_prefix(PartitionDescriptor *states) {
  // Only let first thread of block do this
  if (threadIdx.x != 0)
    return -1;

  // By definition the first block does needs the neutral element
  if (blockIdx.x == 0)
    return NEUTRAL_ELEMENT;

  int exclusive_prefix = 0;
  int end_index = blockIdx.x - WINDOW;
  if (end_index < 0)
    end_index = 0;
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
  return exclusive_prefix;
}

__device__ void
record_partition_wide_inclusive_prefix(PartitionDescriptor *state,
                                       float partition_wide_aggregate,
                                       float exclusive_prefix) {
  state->inclusive_prefix = partition_wide_aggregate + exclusive_prefix;
  __threadfence();
  state->flag = FLAG_INCLUSIVE_PREFIX;
}

// Shared memory is two parts, first for inplace reduction, second for input
// values for inplace scan
__global__ void scan_lookback(float *a, PartitionDescriptor *states) {
  // This will be the size of a partition
  extern __shared__ float s[];
  float *reduction_inplace = s;
  float *shared_input = &s[blockDim.x + 1];

  float *global_base_ptr = &a[blockIdx.x * blockDim.x];
  PartitionDescriptor *own_partition_descriptor = &states[blockIdx.x];

  copy_one_item_per_thread(reduction_inplace, global_base_ptr);
  copy_one_item_per_thread(shared_input, global_base_ptr);

  __syncthreads();
  compute_partition_wide_aggregate(reduction_inplace);
  __syncthreads();
  float aggregate = s[0];

  // We are using shared[1] for propagating the exclusive_prefix inside the
  // block
  if (threadIdx.x == 0) {
    record_partition_wide_aggregate(own_partition_descriptor, aggregate);
    s[1] = determine_partitions_exclusive_prefix(states);
    record_partition_wide_inclusive_prefix(own_partition_descriptor, aggregate,
                                           s[1]);
  }
  __syncthreads();
  float exclusive_prefix = s[1];
  partition_wide_scan(shared_input);
  __syncthreads();
  // Because every thread copies the value it wrote, there is no need to
  // syncthreads
  partition_wide_bin_op(shared_input, exclusive_prefix);
  copy_one_item_per_thread(global_base_ptr, shared_input);
}
