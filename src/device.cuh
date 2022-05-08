#ifndef H_DEVICE
#define H_DEVICE  _

#include "main.cuh"

__global__ void scan_lookback(float *a, partition_state *state);

#endif
