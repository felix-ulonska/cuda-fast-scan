#ifndef H_MAIN
#define H_MAIN _
#define THREADS_PER_BLOCK 64
#define ITEMS_PER_THREAD 32
#define ITEMS_PER_BLOCK THREADS_PER_BLOCK * ITEMS_PER_THREAD
#define AMOUNT_BLOCKS 64
#define WINDOW 3
#define AMOUNT_ELEMS ITEMS_PER_BLOCK * AMOUNT_BLOCKS
#define SIZE_OF_INPUT sizeof(int) * AMOUNT_BLOCKS * ITEMS_PER_BLOCK
#define SIZE_OF_PARTITION_DESCRIPTIORS sizeof(PartitionDescriptor) * AMOUNT_BLOCKS

typedef struct {
  float time;
} Result;

#endif
