#ifndef H_DEVICE
#define H_DEVICE _
#include "../shared/shared.cuh"

__global__ void scan_lookback(float *a, PartitionDescriptor *states);

#endif
