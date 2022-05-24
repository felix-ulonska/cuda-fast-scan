#ifndef H_BLOCK
#define H_BLOCK _

__device__ void b_tree_reduction(int* a);
__device__ void b_set_partition_descriptor(PartitionDescriptor* partDesc, int aggregate);
__device__ void b_get_exclusive_prefix(PartitionDescriptor* states, int* exclusive_prefix_location);
__device__ void b_scan(int* a);

#endif
