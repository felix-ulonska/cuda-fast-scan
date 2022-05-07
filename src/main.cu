#include "main.cuh"
#include "device.cuh"
#include "host.cuh"

int main(){
    std::cout << "Starting..." << std::endl;

    float *a;
    partition_state *state;

    int num_partition = N / WARP_SIZE;

    // Allocate memory
    auto error = cudaMallocManaged(&state, sizeof(partition_state) * num_partition);
    if (error) 
        std::cerr << "Failed allocate States" << std::endl;
    error = cudaMallocManaged(&a, sizeof(float) * N );
        std::cerr << "Failed allocate Input" << std::endl;
    
    fillArray(a, N);
    fillStateArr(state, num_partition);
    
    runKernel(a, state, N);

    output(a, state, N);

    cudaFree(a);
    cudaFree(state);
}
