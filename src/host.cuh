#ifndef H_HOST
#define H_HOST  _
#include "main.cuh"

void fillArray(float* arr, int n);
void fillStateArr(partition_state* states, int n);
cudaError runKernel(float* arr, partition_state* states, int n);
void output(float *arr, partition_state *states, int n);
float hostBinOp(float a, float b);
void scanHost(float* dest, float* src, int n);
bool verifyResult(float* result, float* input, int n);
bool arr_equal(float* a, float* b, int n);
#endif
