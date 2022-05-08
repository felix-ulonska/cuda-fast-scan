#ifndef H_DEVICE
#define H_DEVICE  _

#include "main.cuh"

__global__ void scan_lookback(float *a, volatile partition_state *state);

#endif
