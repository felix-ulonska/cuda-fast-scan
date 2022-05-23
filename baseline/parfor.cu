#include "../shared/shared.cuh"

template <typename TLambda>
__device__ void parfor_block(
    int* input, int itemsPerBlock, PartitionDescriptor* partDescs,
    TLambda lambda) {
  int blockId = blockIdx.x;
  int* base_ptr = &input[itemsPerBlock * blockId];
  lambda(blockId, base_ptr, itemsPerBlock, &partDescs[blockId]);
}

template <typename TLambda>
__device__ void execute_on_one_thread_per_block(
    TLambda lambda
  ) {
  if (threadIdx.x == 0) {
    lambda();
  }
}

template <typename TLambda>
__device__ void parfor_thread(
    int* a, int items_per_thread, TLambda lambda
    ) {
  lambda();
}

__global__ void scan_lookback(int* a, PartitionDescriptor* states) {
  parfor_block(a, 10, states,
    [] __device__(
      int blockId,
      int* base_ptr,
      int len,
      PartitionDescriptor* partDesc) {
    });
}
