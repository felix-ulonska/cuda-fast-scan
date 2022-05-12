#ifndef H_MAIN
#define H_MAIN  _

#include <iostream>

#define N  1024 * 50
#define WARP_SIZE 32
#define WINDOW 100

#define FLAG_BLOCK 0 << 0
#define FLAG_AGGREGATE (1 << 1)
#define FLAG_INCLUSIVEP_PREFIX  (1 << 2)

// MAYBE Optimize Struct to enable collalcasing
typedef struct  {
    int flag;
    float aggregate;
    float inclusive_prefix;
    float prefix;
} partition_state;

__device__ float binOp(float a, float b);

#endif
