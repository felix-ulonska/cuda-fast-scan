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

__global__ void vector_add(float *a, partition_state *state) {
    // Setup indexes
    int partition_index = (blockIdx.x * 1024) + threadIdx.x / WARP_SIZE;
    int warp_i = threadIdx.x % 32;
    bool partition_head = warp_i == 0;

    // Setup Pointers
    float* base_ptr = &a[partition_index * WARP_SIZE];

    // Compute  and  record the  partition-wide aggregate
    int sum = reduction(base_ptr, warp_i);

    if (partition_head) {

        state[partition_index].aggregate = sum;
        state[partition_index].flag = FLAG_AGGREGATE;
        
        if (partition_index == 0) {
            state[partition_index].inclusive_prefix = sum;
            state[partition_index].flag = FLAG_INCLUSIVEP_PREFIX;
        } 
    }

    // Determine the partition’s exclusive prefix using decoupledlook-back
    float prefix = 0;
    if (!partition_head)
        while (state[partition_index].flag & FLAG_AGGREGATE) {
            prefix = 0;
            for (int i = partition_index - 1; i > -1 && i > partition_index - WINDOW - 1; i--) {
                if (state[i].flag == FLAG_BLOCK)
                    break;
                else if (state[i].flag & FLAG_AGGREGATE) {
                    prefix += state[i].aggregate;
                }
                else {
                    prefix = prefix + state[i].inclusive_prefix;
                    
                    // Compute and record the partition-wide inclusive prefixes.

                    state[partition_index].prefix = prefix;
                    state[partition_index].inclusive_prefix = prefix + state[partition_index].aggregate;
                    __threadfence();
                    state[partition_index].flag = FLAG_INCLUSIVEP_PREFIX;
                    break;
                }
            }
        }

    __syncwarp();
    scan(base_ptr, warp_i);
    base_ptr[warp_i] = binOp(base_ptr[warp_i], state[partition_index].prefix);
}

int main(){
    float *a;
    partition_state *state;

    int num_partition = N / WARP_SIZE;

    // Allocate memory
    auto error = cudaMallocManaged(&state, sizeof(partition_state) * num_partition);
    cudaMallocManaged(&a, sizeof(float) * N );

    // Initialize array
    for(int i = 0; i < N; i++){
        a[i] = i; 
    }

    for(int i = 0; i < num_partition; i++) 
        // all other values are not read on startup
        state[i] = {.flag = FLAG_BLOCK};

    int blocks = N / 1024; 
    std::cout << "Start kernel with " << blocks << " blocks "<< std::endl;
    vector_add<<<blocks, 1024>>>(a,state);

    error = cudaDeviceSynchronize();
    
    if (error)
        std::cout << "Invocation falied with " << cudaGetErrorString(error) << std::endl;

    for (int i = N - 100 ; i < N; i++) {
        std::cout << "" << a[i] << ",";
    }
    std::cout << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << "Sturct Number " << i << " Got Flag " << state[i].flag << " and output " << state[i].inclusive_prefix << std::endl;
    }

    cudaFree(a);
    cudaFree(state);
}
