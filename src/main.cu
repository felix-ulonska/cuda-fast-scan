#include "main.cuh"
#include "device.cuh"
#include "host.cuh"

#define LOOP_TILL_ERROR false

int main(){
    std::cout << "Starting..." << std::endl;

    float *a;
    float *correct = (float *) malloc(sizeof(float) * N);
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
    scanHost(correct, copy_input, N);
    
    do {
        memcpy(a, copy_input, N * sizeof(float));
        fillStateArr(state, num_partition);
        
        cudaError_t c_error = runKernel(a, state, N);
        std::cout << "Done running" << std::endl;
        if (c_error) {
            cudaFree(a);
            cudaFree(state);
            exit(EXIT_FAILURE);
        }

        bool valid = arr_equal(a, correct, N);
        if (valid) {
            std::cout << "Result is correct" << std::endl;
            // exit(EXIT_SUCCESS);
        } else {
            std::cerr << "Result is not correct!" << std::endl;
            exit(EXIT_FAILURE);
        }
    } while (LOOP_TILL_ERROR);

    free(correct);
    free(copy_input);

    cudaFree(a);
    cudaFree(state);


}
