#ifndef H_HOST
#define H_HOST  _
#include "main.cuh"

void fillArray(float* arr, int n);
void fillStateArr(partition_state* states, int n);
void runKernel(float* arr, partition_state* states, int n);
void output(float *arr, partition_state *states, int n);
#endif
