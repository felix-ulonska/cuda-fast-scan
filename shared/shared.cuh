/**
        Utils which are used by all versions of the code.
        This contains utils as verifing the result or genereting the correct
   result.
*/
#ifndef H_SHARED
#define H_SHARED _

#define FLAG_BLOCK 0 << 0
#define FLAG_AGGREGATE (1 << 1)
#define FLAG_INCLUSIVE_PREFIX (1 << 2)

#define WARP_SIZE 32
#define NEUTRAL_ELEMENT 0

// MAYBE Optimize Struct to enable collalcasing
typedef struct {
  // Allowed values are FLAG_BLOCK, FLAG_INCLUSIVE_PREFIX, FLAG_AGGREGATE
  int flag;
  float aggregate;
  float inclusive_prefix;
} PartitionDescriptor;

void print_motd();

__device__ __host__ float bin_op(float a, float b);

void init_array(float *arr, int n);
void init_state_arr(PartitionDescriptor *states, int n);

void scan_host(float *dest, float *src, int n);
bool verify_result(float *result, float *input, int n);
bool arr_equal(float *a, float *b, int n);

void output_arr(float *a, int n);

#endif
