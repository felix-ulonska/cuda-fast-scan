#include "main.cuh"
#include "device.cuh"

__device__ float binOp(float a, float b) {
    return a + b;
}

// base_ptr should point to the first element of the array which inside the partition
__device__ int reduction(float *base_ptr, int warp_i) {
    float sum = base_ptr[warp_i];
    // see https://people.maths.ox.ac.uk/gilesm/cuda/lecs/lec4.pdf
    for (int i=1; i< WARP_SIZE; i*=2)
        sum = binOp(sum, __shfl_xor_sync(0xffffff, sum, i));
    return sum;
}

__device__ void scan(float *base_ptr, int warp_i) {
    // see https://people.maths.ox.ac.uk/gilesm/cuda/lecs/lec4.pdf

    int temp1 = base_ptr[warp_i];
    for (int d=1; d<32; d<<=1) {
        int temp2 = __shfl_up_sync(0xffffff, temp1, d);
        if (warp_i >= d) temp1 = binOp(temp1, temp2); 
    }

    base_ptr[warp_i] = temp1;
}

__global__ void scan_lookback(float *a, volatile partition_state *state) {
    // Setup indexes
    int partition_index = ((blockIdx.x * 1024) + threadIdx.x) / WARP_SIZE;
    int warp_i = threadIdx.x % 32;
    bool partition_head = warp_i == 0;

    // Setup Pointers
    float* base_ptr = &a[partition_index * WARP_SIZE];

    // Compute  and  record the  partition-wide aggregate
    int sum = reduction(base_ptr, warp_i);

    if (partition_head) {
        state[partition_index].aggregate = sum;
        __threadfence_system();
        state[partition_index].flag = FLAG_AGGREGATE;
        
        if (partition_index == 0) {
            state[partition_index].inclusive_prefix = sum;
            __threadfence_system();
            state[partition_index].flag = FLAG_INCLUSIVEP_PREFIX;
        } 
    }

    // Determine the partition’s exclusive prefix using decoupledlook-back
    float prefix = 0;
    if (!partition_head)
        while (state[partition_index].flag == FLAG_AGGREGATE) {
            prefix = 0;
            for (int i = partition_index - 1; i > -1 && i > partition_index - WINDOW - 1; i--) {
                int flag = state[i].flag;
                __threadfence_system();
                if (flag == FLAG_BLOCK)
                    break;
                else if (flag & FLAG_AGGREGATE) {
                    __threadfence_system();
                    prefix += state[i].aggregate;
                }
                else {
                    prefix = prefix + state[i].inclusive_prefix;
                    
                    // Compute and record the partition-wide inclusive prefixes.

                    if (prefix == 0) {
                        printf("p: %d, i: %d, prefix: %f\n", partition_index, i, state[i].inclusive_prefix);
                        break;
                    }
                    state[partition_index].prefix = prefix;
                    state[partition_index].inclusive_prefix = prefix + state[partition_index].aggregate;
                    __threadfence_system();
                    state[partition_index].flag = FLAG_INCLUSIVEP_PREFIX;
                    break;
                }
            }
        }

    __syncwarp();
    // printf("finished partition_index %d\n", partition_index);
    scan(base_ptr, warp_i);
    base_ptr[warp_i] = binOp(base_ptr[warp_i], state[partition_index].prefix);
}
