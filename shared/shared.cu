#include <iostream>

#include "shared.cuh"

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
__device__ __host__ int bin_op(int a, int b) { return a + b; }

void init_array(int *arr, int n) {
  for (int i = 0; i < n; i++) {
#ifdef RANDOM_INIT
#else
    // TODO change to random values
    arr[i] = 1;  // i % 10;
#endif
  }
}

void init_state_arr(PartitionDescriptor *state, int n) {
  for (int i = 0; i < n; i++) {
    state[0] = PartitionDescriptor {
      .flag = FLAG_BLOCK,
      .aggregate = 0,
      .inclusive_prefix = 0,
    };
  }
  //memset(state, 0, sizeof(PartitionDescriptor) * n);
}

/**
        ------------------------------
        # Functions to verify Result #
        ------------------------------
*/
void scan_host(int *dest, int *src, int n) {
  int *currDest = dest;
  int *currSrc = src;

  // Setting first value
  *currDest = *currSrc;

  // Moving the pointers through the array and using the last value to calc the
  // next value
  do {
    int nextVal = bin_op(*(++currSrc), *currDest);
    *(++currDest) = nextVal;
  } while (currDest != &dest[n]);
}

bool arr_equal(int *a, int *b, int n) {
  bool bad = false;
  for (int i = 0; i < n; i++) {
    if (abs(a[i] - b[i]) > 0.3) {
      std::cout << "I: " << i << " First val: " << a[i] << " Second Val" << b[i]
                << std::endl;
      bad = true;
    }
  }
  return !bad;
}

void output_arr(int *a, int n) {
  for (int i = 0; i < n; i++) {
    printf("[?] i: %d: %d\n", i, a[i]);
  }
}

bool verifyResult(int *result, int *input, int n) {
  int *output_arr = (int *)malloc(sizeof(int) * n);
  scan_host(output_arr, input, n);

  bool is_equal = arr_equal(result, output_arr, n);

  free(output_arr);

  return is_equal;
}
