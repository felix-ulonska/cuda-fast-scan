#include "../shared/shared.cuh"
#include "main.cuh"
#include "thread.cuh"
#include <stdio.h>
#include "params.cuh"

// __device__ void b_tree_reduction(int* a) {
//   for (int d = 1; d < blockDim.x; d *= 2) {
//     // TODO move to thread level
//     // TODO remove second part of if
//     if (threadIdx.x % (d * 2) == 0) {
//       a[threadIdx.x] =
//           bin_op(a[threadIdx.x + d], a[threadIdx.x]);
//     }
//     __syncthreads();
//   }
// }

__device__ void b_tree_reduction(int* a) {
  for (std::size_t k = (THREADS_PER_BLOCK / 2); k > 0; k = k / 2) {
  
    if (threadIdx.x < k) {
      a[(threadIdx.x - 0)] =
          a[(threadIdx.x - 0)] +
          a[((threadIdx.x - 0) + k)];
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

__device__ void b_get_exclusive_prefix_new(volatile PartitionDescriptor* states, int* exclusive_prefix_location) {
  if (blockIdx.x > 0) {
    *exclusive_prefix_location = 0;
    int exclusive_prefix = 0;
    auto not_done = true;
    while (not_done) {
      *exclusive_prefix_location = 0;
      exclusive_prefix = 0;
      auto i = 1;
      auto flag = 0;
      auto agg = 0;
      auto prefix = 0;
      auto not_break_loop = true;
      while (i <= WINDOW && blockIdx.x - i >= 0 &&
             not_break_loop) {
        volatile PartitionDescriptor *currState = &states[blockIdx.x - i];
        // unsafe
        {
          flag = currState->flag;
          __threadfence();
          agg = currState->aggregate;
          prefix = currState->inclusive_prefix;
        }
        if (flag == FLAG_BLOCK) {
          not_break_loop = false;
        }
        if (flag == FLAG_AGGREGATE) {
          exclusive_prefix += agg;
        }
        if (flag == FLAG_INCLUSIVE_PREFIX) {
          exclusive_prefix += prefix;
          not_break_loop = false;
          not_done = false;
        }
        i = i - 1;
      }
    }
    *exclusive_prefix_location = exclusive_prefix;
  }
}

// Sklansky scan
// __device__ void b_scan(int* a) {
//   t_scan(&a[threadIdx.x * ITEMS_PER_THREAD], ITEMS_PER_THREAD);
//   __syncthreads();
// 
//   for (int k = 2; k <= blockDim.x; k *= 2) {
//     int add_val_index = ((threadIdx.x / k) * k) + ((k / 2) - 1);
//     // printf("tid: %d, k: %d, add_val %d\n", threadIdx.x, k, add_val);
//     if (threadIdx.x % k >= k / 2) {
//       t_bin_op(&a[threadIdx.x * ITEMS_PER_THREAD], &a[threadIdx.x * ITEMS_PER_THREAD], a[(add_val_index + 1) * ITEMS_PER_THREAD - 1]);
//     }
//     __syncthreads();
//   }
// }

__device__ void b_scan(int* a) {
  const auto foo = a;
  for (std::size_t d = THREADS_PER_BLOCK; d > 0; d = d / 2) {
    if (threadIdx.x < d) {
      (&(*foo))[(((threadIdx.x - 0) * ((THREADS_PER_BLOCK * ITEMS_PER_THREAD) / d)) +
                 (((THREADS_PER_BLOCK * ITEMS_PER_THREAD) / d) - 1))] =
          (&(*foo))[(
              ((threadIdx.x - 0) * ((THREADS_PER_BLOCK * ITEMS_PER_THREAD) / d)) +
              (((THREADS_PER_BLOCK * ITEMS_PER_THREAD) / d) - 1))] +
          (&(*foo))[(
              ((threadIdx.x - 0) * ((THREADS_PER_BLOCK * ITEMS_PER_THREAD) / d)) +
              ((THREADS_PER_BLOCK / d) - 1))];
    }
    __syncthreads();
  }
  
  if (threadIdx.x < 1) {
    foo[((threadIdx.x - 0) + ((THREADS_PER_BLOCK * ITEMS_PER_THREAD) - 1))] = 0;
  }
  __syncthreads();

  for (std::size_t d = 1; d <= THREADS_PER_BLOCK; d = d * 2) {

    if (threadIdx.x < d) {
      const auto t = (&(
          *foo))[(((threadIdx.x - 0) * ((THREADS_PER_BLOCK * ITEMS_PER_THREAD) / d)) +
                  ((THREADS_PER_BLOCK / d) - 1))];
      (&(*foo))[(((threadIdx.x - 0) * ((THREADS_PER_BLOCK * ITEMS_PER_THREAD) / d)) +
                 ((THREADS_PER_BLOCK / d) - 1))] =
          (&(*foo))[(
              ((threadIdx.x - 0) * ((THREADS_PER_BLOCK * ITEMS_PER_THREAD) / d)) +
              (((THREADS_PER_BLOCK * ITEMS_PER_THREAD) / d) - 1))];
      (&(*foo))[(((threadIdx.x - 0) * ((THREADS_PER_BLOCK * ITEMS_PER_THREAD) / d)) +
                 (((THREADS_PER_BLOCK * ITEMS_PER_THREAD) / d) - 1))] =
          (&(*foo))[(
              ((threadIdx.x - 0) * ((THREADS_PER_BLOCK * ITEMS_PER_THREAD) / d)) +
              (((THREADS_PER_BLOCK * ITEMS_PER_THREAD) / d) - 1))] +
          t;
    }
    __syncthreads();
  }
}
