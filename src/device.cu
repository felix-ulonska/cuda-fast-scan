#include "main.cuh"
#include "device.cuh"

// #define DEBUG_PRINT(i) if (threadIdx.x == 0) printf("%d:  %d\n", i, blockIdx.x);
#define DEBUG_PRINT(i)

__device__ float binOp(float a, float b) {
    return a + b;
}

// base_ptr should point to the first element of the array which inside the partition
__device__ int reduction(float *base_ptr, int lane_id) {
    float sum = base_ptr[lane_id];
    // see https://people.maths.ox.ac.uk/gilesm/cuda/lecs/lec4.pdf
    for (int i=1; i< WARP_SIZE; i*=2)
        sum = binOp(sum, __shfl_xor_sync(0xffffff, sum, i));
    return sum;
}

__device__ void scan(float *base_ptr, int lane_id) {
    // see https://people.maths.ox.ac.uk/gilesm/cuda/lecs/lec4.pdf

    int temp1 = base_ptr[lane_id];
    for (int d=1; d<32; d<<=1) {
        int temp2 = __shfl_up_sync(0xffffff, temp1, d);
        if (lane_id >= d) temp1 = binOp(temp1, temp2); 
    }

    base_ptr[lane_id] = temp1;
}

__global__ void scan_lookback(float *a, partition_state *state) {
    // Setup indexes
    DEBUG_PRINT(0);
    int partition_index = ((blockIdx.x * 1024) + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % 32;
    bool partition_head = lane_id == 0;

    // Setup Pointers
    float* base_ptr = &a[partition_index * WARP_SIZE];

    // Compute  and  record the  partition-wide aggregate
    int sum = reduction(base_ptr, lane_id);
    DEBUG_PRINT(1);

    if (partition_head) {
        state[partition_index].aggregate = sum;
        __threadfence();
        state[partition_index].flag = FLAG_AGGREGATE;
        
        if (partition_index == 0) {
            state[partition_index].inclusive_prefix = sum;
            __threadfence();
            state[partition_index].flag = FLAG_INCLUSIVEP_PREFIX;
        } 
    }

    // Determine the partition’s exclusive prefix using decoupledlook-back
    float prefix = 0;
    DEBUG_PRINT(2);
    if (partition_head)
        while (state[partition_index].flag == FLAG_AGGREGATE) {
            prefix = 0;
            for (int i = partition_index - 1; i > -1 && i > partition_index - WINDOW - 1; i--) {
                int flag = state[i].flag;
                __threadfence();
                if (flag == FLAG_BLOCK)
                    break;
                else if (flag & FLAG_AGGREGATE) {
                    prefix += state[i].aggregate;
                }
                else if (flag & FLAG_INCLUSIVEP_PREFIX) {
                    prefix = prefix + state[i].inclusive_prefix;
                    
                    // Compute and record the partition-wide inclusive prefixes.

                    state[partition_index].prefix = prefix;
                    state[partition_index].inclusive_prefix = prefix + state[partition_index].aggregate;
                    __threadfence();
                    state[partition_index].flag = FLAG_INCLUSIVEP_PREFIX;
                    break;
                }
                break;
            }
        }

    __syncwarp();
    DEBUG_PRINT(3);
    // printf("finished partition_index %d\n", partition_index);
    scan(base_ptr, lane_id);
    base_ptr[lane_id] = binOp(base_ptr[lane_id], state[partition_index].prefix);
}
