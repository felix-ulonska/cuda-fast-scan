#include "main.cuh"
#include "device.cuh"
#include "host.cuh"

int main(){
    std::cout << "Starting..." << std::endl;

    float *a;
    float *copy_input = (float *) malloc(sizeof(float) * N);
    partition_state *state;

    int num_partition = N / WARP_SIZE;

    // Allocate memory
    auto error = cudaMallocManaged(&state, sizeof(partition_state) * num_partition);
    if (error) 
        std::cerr << "Failed allocate States" << std::endl;
    error = cudaMallocManaged(&a, sizeof(float) * N );
    if (error) 
        std::cerr << "Failed allocate Input" << std::endl;
    
    fillArray(a, N);
    memcpy(copy_input, a, N * sizeof(float));
    fillStateArr(state, num_partition);

    
    cudaError_t c_error = runKernel(a, state, N);
    if (c_error) {
        cudaFree(a);
        cudaFree(state);
        exit(EXIT_FAILURE);
    }

    bool valid = verifyResult(a, copy_input, N);
    cudaFree(a);
    cudaFree(state);

    if (valid) {
        std::cout << "Result is correct" << std::endl;
        exit(EXIT_SUCCESS);
    } else {
        std::cerr << "Result is not correct!" << std::endl;
        exit(EXIT_FAILURE);
    }

}
