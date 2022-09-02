#include "main.cuh"
#include "params.cuh"
#include <cstdio>
#include <iostream>

// CURSED ONE-File
void scan_host(int *dest, int *src, int n) {
  int *currDest = dest;
  int *currSrc = src;

  // Setting first value
  *currDest = *currSrc;

  // Moving the pointers through the array and using the last value to calc the
  // next value
  do {
    int nextVal = *(++currSrc) + *currDest;
    *(++currDest) = nextVal;
  } while (currDest != &dest[n]);
}

bool arr_equal(int *a, int *b, int n) {
  bool bad = false;
  for (int i = 0; i < n; i++) {
    if (abs(a[i] - b[i]) > 0.3) {
      std::cout << "I: " << i << " First val: " << a[i] << " Second Val" << b[i]
                << std::endl;
      return false;
      bad = true;
    }
  }
  return !bad;
}

__global__ void scan_kernel(int *const input, int *const flag, int *const agg,
                            int *const prefix) {
  __shared__ int b_shared_input_bkp[sizeof(int) * ITEMS_PER_BLOCK];
  __shared__ int s[sizeof(int) * (THREADS_PER_BLOCK + ITEMS_PER_BLOCK + 100)];

  // Parfor block
  {
    int *const b_ptr_input = &input[ITEMS_PER_BLOCK * blockIdx.x];
    int *const b_ptr_shared_reduction = &s[0];
    int *const b_ptr_shared_input_copy = &s[THREADS_PER_BLOCK];

    // for (int i = 0; i < ITEMS_PER_BLOCK; i += blockDim.x) {
    //   b_ptr_shared_input_copy[i + threadIdx.x] = b_ptr_input[i +
    //   threadIdx.x];
    // }
    for (int i = 0; i < ITEMS_PER_THREAD; i += 1) {
      b_ptr_shared_input_copy[i + (threadIdx.x * ITEMS_PER_THREAD)] =
          b_ptr_input[i + (threadIdx.x * ITEMS_PER_THREAD)];
      b_shared_input_bkp[i + (threadIdx.x * ITEMS_PER_THREAD)] =
          b_ptr_input[i + (threadIdx.x * ITEMS_PER_THREAD)];
    }
    __syncthreads();

    // Parfor thread in block
    int *const t_ptr_input = &b_ptr_input[threadIdx.x * ITEMS_PER_THREAD];
    int *const t_ptr_shared_reduction = &b_ptr_shared_reduction[threadIdx.x];
    int *const t_ptr_shared_input =
        &b_ptr_shared_input_copy[threadIdx.x * ITEMS_PER_THREAD];
    // PARFOR THREAD
    {
      // t_mem_cpy(t_ptr_shared_input, t_ptr_input);
      int sum = t_ptr_shared_input[0];
      for (int i = 1; i < ITEMS_PER_THREAD; i++) {
        sum = sum + t_ptr_shared_input[i];
      }
      *t_ptr_shared_reduction = sum;
    }
    __syncthreads();

    for (std::size_t k = (THREADS_PER_BLOCK / 2); k > 0; k = k / 2) {
      if (threadIdx.x < k) {
        b_ptr_shared_reduction[(threadIdx.x - 0)] =
            b_ptr_shared_reduction[(threadIdx.x - 0)] +
            b_ptr_shared_reduction[((threadIdx.x - 0) + k)];
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      if (blockIdx.x != 0) {
        agg[blockIdx.x] = b_ptr_shared_reduction[0];
        __threadfence();
        flag[blockIdx.x] = 1;
      } else if (threadIdx.x == 0 && blockIdx.x == 0) {
        prefix[blockIdx.x] = b_ptr_shared_reduction[0];
        // Technically the aggregate is not needed
        agg[blockIdx.x] = b_ptr_shared_reduction[0];
        __threadfence();
        flag[blockIdx.x] = 2;
      }
    }
    __syncthreads();

    int *const exclusive_prefix_location = &b_ptr_shared_reduction[0];
    const auto blockId = blockIdx.x;
    if (blockIdx.x > 0 && threadIdx.x == 0) {
      *exclusive_prefix_location = 0;
      int exclusive_prefix = 0;
      auto not_done = true;
      while (not_done) {
        *exclusive_prefix_location = 0;
        exclusive_prefix = 0;
        auto i = 1;
        auto t_flag = 0;
        auto t_agg = 0;
        auto t_prefix = 0;
        auto not_break_loop = true;
        while (i <= WINDOW && blockId - i >= 0 && not_break_loop) {
          // unsafe
          {
            t_flag = flag[blockId - i];
            __threadfence();
            t_agg = agg[blockId - i];
            t_prefix = prefix[blockId - i];
          }
          if (t_flag == 0) {
            not_break_loop = false;
          }
          if (t_flag == 1) {
            exclusive_prefix += t_agg;
          }
          if (t_flag == 2) {
            exclusive_prefix += t_prefix;
            not_break_loop = false;
            not_done = false;
          }
          i = i + 1;
        }
      }
      *exclusive_prefix_location = exclusive_prefix;
    }

    if (!threadIdx.x && blockIdx.x != 0) {
      // TODO move to function
      prefix[blockIdx.x] = b_ptr_shared_reduction[0] + agg[blockIdx.x];
      __threadfence();
      flag[blockIdx.x] = 2;
    }
    __syncthreads();

    const auto foo = b_ptr_shared_input_copy;
    for (std::size_t d = THREADS_PER_BLOCK; d > 0; d = d / 2) {
      if (threadIdx.x < d) {
        const auto baseThread = &foo[threadIdx.x * ((THREADS_PER_BLOCK * ITEMS_PER_THREAD) / d)];
        //const auto t_ptr = &baseThread[(threadIdx.x * ((THREADS_PER_BLOCK * ITEMS_PER_THREAD) / d))];
        baseThread[((THREADS_PER_BLOCK * ITEMS_PER_THREAD) / d) - 1] +=
            baseThread[((THREADS_PER_BLOCK / d) - 1)];
        baseThread[0] = 100;
      }
      __syncthreads();
    }

    if (threadIdx.x < 1) {
      foo[((threadIdx.x - 0) + ((THREADS_PER_BLOCK * ITEMS_PER_THREAD) - 1))] =
          0;
    }
    __syncthreads();

    for (std::size_t d = 1; d <= THREADS_PER_BLOCK; d = d * 2) {

      if (threadIdx.x < d) {
        const auto t = (&(*foo))[(
            ((threadIdx.x - 0) * ((THREADS_PER_BLOCK * ITEMS_PER_THREAD) / d)) +
            ((THREADS_PER_BLOCK / d) - 1))];
        (&(*foo))[(
            ((threadIdx.x - 0) * ((THREADS_PER_BLOCK * ITEMS_PER_THREAD) / d)) +
            ((THREADS_PER_BLOCK / d) - 1))] =
            (&(*foo))[(((threadIdx.x - 0) *
                        ((THREADS_PER_BLOCK * ITEMS_PER_THREAD) / d)) +
                       (((THREADS_PER_BLOCK * ITEMS_PER_THREAD) / d) - 1))];
        (&(*foo))[(
            ((threadIdx.x - 0) * ((THREADS_PER_BLOCK * ITEMS_PER_THREAD) / d)) +
            (((THREADS_PER_BLOCK * ITEMS_PER_THREAD) / d) - 1))] =
            (&(*foo))[(((threadIdx.x - 0) *
                        ((THREADS_PER_BLOCK * ITEMS_PER_THREAD) / d)) +
                       (((THREADS_PER_BLOCK * ITEMS_PER_THREAD) / d) - 1))] +
            t;
      }
      __syncthreads();
    }

    // Parfor thread
    {
      if (blockIdx.x == 0) {
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
          t_ptr_input[i] = t_ptr_input[i] + t_ptr_shared_input[i];
        }
      } else {
        // TODO fix global colleasing
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
          t_ptr_input[i] =
              b_shared_input_bkp[(threadIdx.x * ITEMS_PER_THREAD) + i] +
              t_ptr_shared_input[i] + b_ptr_shared_reduction[0];
        }
      }
    }
  }
}

Result exec() {
  int *c_input = (int *)malloc(SIZE_OF_INPUT);
  int *c_flag = (int *)malloc(SIZE_OF_STATUS_ARRS);
  int *c_agg = (int *)malloc(SIZE_OF_STATUS_ARRS);
  int *c_prefix = (int *)malloc(SIZE_OF_STATUS_ARRS);

  int *g_input;
  int *g_flag;
  int *g_agg;
  int *g_prefix;

  for (int i = 0; i < AMOUNT_BLOCKS; i++) {
    c_flag[i] = 0;
    c_prefix[i] = 0;
    c_agg[i] = 0;
  }

  for (int i = 0; i < AMOUNT_ELEMS; i++) {
    c_input[i] = i % 11 + 1; //% 3;
  }

  int *gold = (int *)malloc(SIZE_OF_INPUT);
  scan_host(gold, c_input, AMOUNT_ELEMS);

  if (cudaMalloc(&g_input, SIZE_OF_INPUT) |
      cudaMalloc(&g_flag, SIZE_OF_STATUS_ARRS) |
      cudaMalloc(&g_agg, SIZE_OF_STATUS_ARRS) |
      cudaMalloc(&g_prefix, SIZE_OF_STATUS_ARRS)) {
    printf("cannot alloc on gpu");
    exit(-1);
  }

  if (cudaMemcpy(g_input, c_input, SIZE_OF_INPUT, cudaMemcpyHostToDevice) |
      cudaMemcpy(g_flag, c_flag, SIZE_OF_STATUS_ARRS, cudaMemcpyHostToDevice) |
      cudaMemcpy(g_prefix, c_prefix, SIZE_OF_STATUS_ARRS,
                 cudaMemcpyHostToDevice) |
      cudaMemcpy(g_agg, c_agg, SIZE_OF_STATUS_ARRS, cudaMemcpyHostToDevice)) {
    printf("cannot copy on gpu");
    exit(-1);
  }

  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  scan_kernel<<<AMOUNT_BLOCKS, THREADS_PER_BLOCK>>>(g_input, g_flag, g_agg,
                                                    g_prefix);
  cudaEventRecord(stop);

  if (cudaError error = cudaDeviceSynchronize()) {
    std::cerr << "[!] Execution failed" << cudaGetErrorName(error) << std::endl;
    exit(-1);
  }

  if (cudaMemcpy(c_input, g_input, SIZE_OF_INPUT, cudaMemcpyDeviceToHost) |
      cudaMemcpy(c_flag, g_flag, SIZE_OF_STATUS_ARRS, cudaMemcpyDeviceToHost) |
      cudaMemcpy(c_prefix, g_prefix, SIZE_OF_STATUS_ARRS,
                 cudaMemcpyDeviceToHost) |
      cudaMemcpy(c_agg, g_agg, SIZE_OF_STATUS_ARRS, cudaMemcpyDeviceToHost)) {
    std::cerr << "[!] Memcpy failed" << std::endl;
    exit(-1);
  }

  if (arr_equal(gold, c_input, AMOUNT_ELEMS)) {
    std::cout << "[+] result is correct" << std::endl;
  } else {
    std::cerr << "[!] Result is not correct" << std::endl;
    for (int i = 0; i < AMOUNT_BLOCKS; i++) {
      printf("State %d got state %d and inclusive_prefix %d and agg %d\n", i,
             c_flag[i], c_prefix[i], c_agg[i]);
    }
    exit(-1);
  }

  cudaFree(g_flag);
  cudaFree(g_agg);
  cudaFree(g_prefix);
  cudaFree(g_input);

  free(c_flag);
  free(c_input);
  free(c_prefix);
  free(c_agg);
  free(gold);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  return Result{.time = milliseconds};
}

int main() {
  int iters = 250;

  Result results[iters];
  for (int i = 0; i < iters; i++) {
    results[i] = exec();
  }

  int sum = 0;
  FILE *fp;
  if ((fp = fopen(CSV_OUTPUT_PATH, "w")) == NULL) {
    printf("cannot open.\n");
    exit(1);
  }

  for (int i = 0; i < iters; i++) {
    float time = results[i].time;
    sum += time;
    std::fprintf(fp, "%f,\n", time);
    // printf("time: %f\n", time);
  }
  std::fclose(fp);

  printf("success\n");
}
