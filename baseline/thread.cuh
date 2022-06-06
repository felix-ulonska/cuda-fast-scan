#ifndef H_THREAD
#define H_THREAD _

/**
  * Executes ITEMS_PER_THREAD times from src to dest bin_op with addValue as second value
  */
__device__ void t_bin_op(int* dest, int* src, int addValue);

/**
  * Executes on ITEMS_PER_THREAD items on a a reduction with bin_op
  * 
  * @return The reduction output
  */
__device__ int t_tree_reduction(int* a);
/**
  * Copies ITEMS_PER_THREAD items from src to dest
  */
__device__ void t_mem_cpy(int* dest, int* src);
__device__ void t_scan(int* i, int len);

#endif
