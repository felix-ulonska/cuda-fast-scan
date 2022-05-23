#ifndef H_DEVICE
#define H_DEVICE _
#include "../shared/shared.cuh"

__global__ void scan_kernel(int *a, PartitionDescriptor *states);

#endif
