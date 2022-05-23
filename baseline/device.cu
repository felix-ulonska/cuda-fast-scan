#include "../shared/shared.cuh"
#include "parfor.cuh"
#include <stdio.h>
#include "main.cuh"

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

      // Copy into shared
      // TODO do stride things for perfomance
      for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        t_ptr_shared_input[i] = t_ptr_input[i];
      } 
    }
    __syncthreads();

    // PARFOR for reduction
    {
      int* t_ptr_input = &b_ptr_input[threadIdx.x * ITEMS_PER_THREAD];
      int* t_ptr_shared_reduction = &b_ptr_shared_reduction[threadIdx.x];
      int* t_ptr_shared_input = &b_ptr_shared_input_copy[threadIdx.x * ITEMS_PER_THREAD];

      int sum = t_ptr_shared_input[0];
      for (int i = 1; i < ITEMS_PER_THREAD; i++) {
        sum = bin_op(sum, t_ptr_shared_input[i]);
      } 
      *t_ptr_shared_reduction = sum;
    }

    __syncthreads();

    // Block Wide Reduction

    for (int d = 1; d < blockDim.x; d *= 2) {
      if (threadIdx.x % (d * 2) == 0 && threadIdx.x + d < blockDim.x) {
        b_ptr_shared_reduction[threadIdx.x] =
            bin_op(b_ptr_shared_reduction[threadIdx.x + d], b_ptr_shared_reduction[threadIdx.x]);
      }
      __syncthreads();
    }

    // Run Code only on one threadx 
    if (threadIdx.x == 0 && blockIdx.x != 0) {
      partDesc->aggregate = b_ptr_shared_reduction[0];       
      __threadfence();
      partDesc->flag = FLAG_AGGREGATE;
    } else if (threadIdx.x == 0 && blockIdx.x == 0) {
      partDesc->inclusive_prefix = b_ptr_shared_reduction[0];       
      // Technically the aggregate is not needed
      partDesc->aggregate = b_ptr_shared_reduction[0];       
      __threadfence();
      partDesc->flag = FLAG_INCLUSIVE_PREFIX;
    }

    __syncthreads();

    if (threadIdx.x == 0 && blockIdx.x != 0) {
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
      b_ptr_shared_reduction[0] = exclusive_prefix;
    }

    __syncthreads();

    if (!threadIdx.x && blockIdx.x != 0) {
      partDesc->inclusive_prefix = bin_op(b_ptr_shared_reduction[0], partDesc->aggregate);       
      __threadfence();
      partDesc->flag = FLAG_INCLUSIVE_PREFIX;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
      int *currDest = b_ptr_shared_input_copy;
      int *currSrc = b_ptr_shared_input_copy;

      // Setting first value
      *currDest = *currSrc;

      // Moving the pointers through the array and using the last value to calc
      // the next value
      do {
        int nextVal = bin_op(*(++currSrc), *currDest);
        *(++currDest) = nextVal;
      } while (currDest != &b_ptr_shared_input_copy[ITEMS_PER_BLOCK - 1]);
    }

    // Parfor binOp
    {
      int* t_ptr_input = &b_ptr_input[threadIdx.x * ITEMS_PER_THREAD];
      int* t_ptr_shared_reduction = &b_ptr_shared_reduction[threadIdx.x];
      int* t_ptr_shared_input = &b_ptr_shared_input_copy[threadIdx.x * ITEMS_PER_THREAD];
      for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (blockIdx.x > 0) {
          t_ptr_input[i] = bin_op(t_ptr_shared_input[i], b_ptr_shared_reduction[0]);
        } else {
          t_ptr_input[i] = t_ptr_shared_input[i];
        }
      }
    }
  };
}
