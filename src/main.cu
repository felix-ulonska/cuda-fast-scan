#include "main.cuh"
#include "device.cuh"

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
    scan_lookback<<<blocks, 1024>>>(a,state);

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
