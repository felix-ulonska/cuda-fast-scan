#ifndef H_MAIN
#define H_MAIN  _

#include <iostream>

#define N 1024 * 10
#define WARP_SIZE 32
#define WINDOW 3

#define FLAG_BLOCK 1 << 0
#define FLAG_AGGREGATE (1 << 1)
#define FLAG_INCLUSIVEP_PREFIX  (1 << 2)

typedef struct  {
    int flag;
    float aggregate;
    float inclusive_prefix;
    float prefix;
} partition_state;

__device__ float binOp(float a, float b);

#endif
