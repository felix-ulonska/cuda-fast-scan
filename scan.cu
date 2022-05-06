#include <iostream>

#define N 4000
#define PARTITION_SIZE 4
#define WINDOW 3

#define FLAG_BLOCK 1 << 0
#define FLAG_AGGREGATE (1 << 1)
#define FLAG_INCLUSIVEP_PREFIX  (1 << 2)

typedef struct  {
    int flag;
    float aggregate;
    float inclusive_prefix;
} partition_state;

__global__ void vector_add(volatile float *a, volatile partition_state *state) {
    int t = threadIdx.x;
    volatile float* base_ptr = &a[t * PARTITION_SIZE];

    float sum = 0;
    for (int i = 0; i < PARTITION_SIZE; i++) 
        sum = sum + base_ptr[i];

    state[t].aggregate = sum;
    state[t].flag = FLAG_AGGREGATE;
    
    if (t == 0) {
        state[t].inclusive_prefix = sum;
        state[t].flag = FLAG_INCLUSIVEP_PREFIX;
    } 

    float prefix = 0;
    while (state[t].flag & FLAG_AGGREGATE) {
        // float withPrefix = 0;
        prefix = 0;
        for (int i = t - 1; i > -1 && i > t - WINDOW - 1; i--) {
            if (state[i].flag == FLAG_BLOCK)
                break;
            else if (state[i].flag & FLAG_AGGREGATE)
                prefix += state[i].aggregate;
            else {
                prefix = prefix + state[i].inclusive_prefix;
                state[t].inclusive_prefix = prefix + state[t].aggregate;
                // memfence
                state[t].flag = FLAG_INCLUSIVEP_PREFIX;
                break;
            }
        }
    }

    base_ptr[0] = base_ptr[0] + prefix;
    for (int i = 1; i < PARTITION_SIZE; i++) 
        base_ptr[i] = base_ptr[i] + base_ptr[i - 1];
    __syncthreads();
}

int main(){
    float *a;
    partition_state *state;

    int num_partition = N / PARTITION_SIZE;

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

    // for (int i = 0; i < N; i++) {
    //     std::cout << "" << a[i] << ",";
    // }
    // Main function
    std::cout << "Start kernel"<< std::endl;
    vector_add<<<1,num_partition>>>(a,state);
    error = cudaDeviceSynchronize();
    std::cout << cudaGetErrorString(error);

    for (int i = 0; i < 100; i++) {
        std::cout << "" << a[i] << ",";
    }
    std::cout << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << "Sturct Number " << i << " Got Flag " << state[i].flag << " and output " << state[i].inclusive_prefix << std::endl;
    }
    // for (int i = num_partition - 10; i < num_partition; i++) {
    //     std::cout << "Sturct Number " << i << " Got Flag " << state[i].flag << " and output " << state[i].aggregate << std::endl;
    // }

    cudaFree(a);
    cudaFree(state);
}
