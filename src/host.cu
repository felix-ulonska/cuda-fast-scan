#include "main.cuh"
#include "device.cuh"

void fillArray(float* arr, int n) {
    std::cout << "fill array" << std::endl;
    // Initialize array
    for(int i = 0; i < n; i++){
        arr[i] = i; 
    }
}

void fillStateArr(partition_state* states, int n) {
    std::cout << "Setting states Mem" << std::endl;
    memset(states, 0, sizeof(partition_state) * n);
}

void runKernel(float *arr, partition_state *states, int n) {
    int blocks = N / 1024; 
    std::cout << "Start kernel with " << blocks << " blocks "<< std::endl;
    scan_lookback<<<blocks, 1024>>>(arr, states);
    auto error = cudaDeviceSynchronize();
    if (error) {
        std::cout << "Invocation falied with " << cudaGetErrorString(error) << std::endl;
    }
}

void output(float *arr, partition_state *states, int n) {
    for (int i = N - 100 ; i < N; i++) {
        std::cout << "" << arr[i] << ",";
    }

    std::cout << std::endl;

    for (int i = 0; i < 10; i++) {
        std::cout << "Sturct Number " << i << " Got Flag " << states[i].flag << " and output " << states[i].inclusive_prefix << std::endl;
    }
}
