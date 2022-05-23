#ifndef H_PARFOR
#define H_PARFOR _

#define PAR_FOR_BLOCK(input, itemsPerBlock, partDescs, lambda) { \
    int blockId = blockIdx.x; \
    int* base_ptr = &input[itemsPerBlock * blockId]; \
    lambda(base_ptr, itemsPerBlock, &partDescs[blockId]); \
}

#define PAR_FOR_THREAD(input, itemsPerThread, lambda) { \
    int threadId = threadIdx.x; \
    int* base_ptr = &input[threadIdx.x * itemsPerThread]; \
    lambda(base_ptr, itemsPerThread); \
}

#endif
// #endif
// template <typename TLambda>
// __device__ void parfor_block(
//     int* input, int itemsPerBlock, PartitionDescriptor* partDescs,
//     // lambda Function should be (int blockId, int* base_ptr, int items_per_block, PartitionDescriptor state)
//     // , PartitionDescriptor*
//     TLambda lambda);
// 
// template <typename TLambda>
// __device__ void execute_on_one_thread_per_block(TLambda lambda);
// 
// template <typename TLambda>
// __device__ void parfor_thread(int* a, int items_per_thread, TLambda lambda);
// 
// __global__ void scan_lookback(int* a, PartitionDescriptor* states);
