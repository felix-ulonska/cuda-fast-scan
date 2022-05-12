#include "shared.cuh"
#include <iostream>


void print_motd() {
#ifdef VERSION
	std::cout << "Starting " << VERSION;
#else
	std::cout << "Starting unknown Version";
#endif
}

/**
	The binary operation, this is shared between all versions
*/
__device__ __host__ float bin_op(float a, float b) {
    return a + b;
}

void init_array(float* arr, int n) {
		for(int i = 0; i < n; i++){
		#ifdef RANDOM_INIT
		#else
					// TODO change to random values
					arr[i] = 1;// i % 10; 
		#endif
		}
}

void init_state_arr(PartitionDescriptor* state, int n) {
    memset(state, 0, sizeof(PartitionDescriptor) * n);
}

/**
	------------------------------
	# Functions to verify Result #
	------------------------------
*/
void scan_host(float* dest, float* src, int n) {
    float* currDest = dest;
    float* currSrc = src;

    // Setting first value
    *currDest = *currSrc; 

    // Moving the pointers through the array and using the last value to calc the next value
    do {
        float nextVal = bin_op(*(++currSrc), *currDest);
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

void output_arr(float* a, int n) {
    for (int i = 0; i < n; i++) {
	printf("[?] i: %d: %f\n", i, a[i]);
    }
}

bool verifyResult(float* result, float* input, int n) {
    float* output_arr = (float*) malloc(sizeof(float) * n);
    scan_host(output_arr, input, n);

    bool is_equal = arr_equal(result, output_arr, n);

    free(output_arr);

    return is_equal;
}
