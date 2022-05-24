#include "main.cuh"
#include "thread.cuh"
#include "../shared/shared.cuh"

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

__device__ void t_bin_op(int* dest, int* src, int addValue) {
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    if (blockIdx.x > 0) {
      dest[i] = bin_op(src[i], addValue);
    } else {
      dest[i] = src[i];
    }
  }
}
