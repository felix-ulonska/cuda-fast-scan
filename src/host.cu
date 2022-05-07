#include "main.cuh"
#include "device.cuh"

void fillArray(float* arr, int n) {
    std::cout << "fill array" << std::endl;
    // Initialize array
    for(int i = 0; i < n; i++){
        arr[i] = i % 1024; 
    }
}

void fillStateArr(partition_state* states, int n) {
    std::cout << "Setting states Mem" << std::endl;
    memset(states, 0, sizeof(partition_state) * n);
}

cudaError runKernel(float *arr, partition_state *states, int n) {
    int blocks = N / 1024; 
    std::cout << "Start kernel with " << blocks << " blocks "<< std::endl;
    scan_lookback<<<blocks, 1024>>>(arr, states);
    auto error = cudaDeviceSynchronize();
    if (error) {
        std::cout << "Invocation falied with " << cudaGetErrorString(error) << std::endl;
    }
    return error;
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

float hostBinOp(float a, float b) {
    return a + b;
}

void scanHost(float* dest, float* src, int n) {
    float* currDest = dest;
    float* currSrc = src;

    // Setting first value
    *currDest = *currSrc; 

    // Moving the pointers through the array and using the last value to calc the next value
    do {
        float nextVal = hostBinOp(*(++currSrc), *currDest);
        *(++currDest) = nextVal;
    } while (currDest != &dest[n]);
}

bool arr_equal(float* a, float* b, int n) {
    for (int i = 0; i < n; i++) {
        // Floats are funny
        if (floor(a[i]) != floor(b[i])) {
           std::cout << "I: " << i << " First val: " << a[i] << " Second Val" << b[i] << std::endl;
           return false;
        }
    }
    return true;
}

bool verifyResult(float* result, float* input, int n) {
    float* output_arr = (float*) malloc(sizeof(float) * N);
    scanHost(output_arr, input, N);

    bool is_equal = arr_equal(result, output_arr, n);

    free(output_arr);

    return is_equal;
}
