#ifndef H_MAIN
#define H_MAIN _
#include "params.cuh"
#define ITEMS_PER_BLOCK THREADS_PER_BLOCK * ITEMS_PER_THREAD
// #define WINDOW 1
#define AMOUNT_ELEMS ITEMS_PER_BLOCK * AMOUNT_BLOCKS
#define SIZE_OF_INPUT sizeof(int) * AMOUNT_BLOCKS * ITEMS_PER_BLOCK
#define SIZE_OF_PARTITION_DESCRIPTIORS sizeof(PartitionDescriptor) * AMOUNT_BLOCKS
#define SIZE_OF_STATUS_ARRS sizeof(int) * AMOUNT_BLOCKS

typedef struct {
  float time;
} Result;

#endif
