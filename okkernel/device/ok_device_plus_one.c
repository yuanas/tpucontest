#include "okk.h"
#ifndef NULL
#define NULL 0
#endif
#define DIV_UP(a, b) (((a) - 1) / (b) + 1)

typedef struct {
    unsigned long long output_addr;
    unsigned long long input_addr;
    int N, C, H, W;
} __attribute__((packed)) param_t;

void plus_one_0(const void *args) {
    param_t *param = (param_t *)args;
    dim4 shape = {.n = param->N, .c = param->C, .h = param->H, .w = param->W};
    dim4 stride;
    // The output and input are in the aligned layout.
    okk_128_byte_aligned_stride_for_32bit(&stride, 0, &shape);
    unsigned int tensor_size = stride.n * shape.n * sizeof(float);
    OKKERNEL_ASSERT(tensor_size * 2 <= okk_local_mem_size_per_npu());
    // Determine addresses of output and input.
    local_addr_t output_addr = 0;
    local_addr_t input_addr = tensor_size;
    // Initialize.
    okk_initialize();
    // Copy input from global to local.
    okk_gdma_32bit_cpy_S2L(input_addr, param->input_addr, &shape, NULL, NULL);
    // Calculate output = input + 1.
    okk_bdc_add_C(output_addr, input_addr, 1.f, &shape, NULL, NULL);
    // Copy output from local to global.
    okk_gdma_32bit_cpy_L2S(param->output_addr, output_addr, &shape, NULL, NULL);
    // Synchronize.
    okk_poll();
}
OKKERNEL_FUNC_REGISTER(plus_one_0);

void plus_one_1(const void *args) {
    param_t *param = (param_t *)args;
    dim4 shape = {.n = param->N / 2, .c = param->C, .h = param->H, .w = param->W};
    dim4 stride;
    // The output and input are in the aligned layout.
    okk_128_byte_aligned_stride_for_32bit(&stride, 0, &shape);
    unsigned int tensor_size_local = stride.n * shape.n * sizeof(float);
    unsigned int tensor_size_global = shape.n * shape.c * shape.h * shape.w * sizeof(float);
    OKKERNEL_ASSERT(tensor_size_local * 4 <= okk_local_mem_size_per_npu());
    // Determine addresses of output and input (ping-pong buffers).
    local_addr_t output_addr[2] = {0, tensor_size_local};
    local_addr_t input_addr[2] = {tensor_size_local * 2, tensor_size_local * 3};
    // Initialize.
    okk_initialize();
    ////////////////////////////////////////////////////////////////////
    // Step 0
    // Copy the first part of input from global to local.
    okk_gdma_32bit_cpy_S2L(input_addr[0], param->input_addr, &shape, NULL, NULL);
    ////////////////////////////////////////////////////////////////////
    // Step 1
    // Start parallel.
    okk_parallel_start();
    // Copy the second part of input from global to local.
    okk_gdma_32bit_cpy_S2L(input_addr[1], param->input_addr + tensor_size_global, &shape, NULL, NULL);
    // Calculate output = input + 1 for the first part.
    okk_bdc_add_C(output_addr[0], input_addr[0], 1.f, &shape, NULL, NULL);
    // End parallel.
    okk_parallel_end();
    ////////////////////////////////////////////////////////////////////
    // Step 2
    // Start parallel.
    okk_parallel_start();
    // Calculate output = input + 1 for the second part.
    okk_bdc_add_C(output_addr[1], input_addr[1], 1.f, &shape, NULL, NULL);
    // Copy the first part of output from local to global.
    okk_gdma_32bit_cpy_L2S(param->output_addr, output_addr[0], &shape, NULL, NULL);
    // End parallel.
    okk_parallel_end();
    ////////////////////////////////////////////////////////////////////
    // Step 3
    // Copy the second part of output from local to global.
    okk_gdma_32bit_cpy_L2S(param->output_addr + tensor_size_global, output_addr[1], &shape, NULL, NULL);
    // Synchronize.
    okk_poll();
}

OKKERNEL_FUNC_REGISTER(plus_one_1);

void plus_one_2(const void *args) {
    param_t *param = (param_t *)args;
    dim4 shape_one_batch = {.n = 1, .c = param->C, .h = param->H, .w = param->W};
    dim4 stride_one_batch, stride;
    // Calculate number of working batches.
    okk_128_byte_aligned_stride_for_32bit(&stride_one_batch, 0, &shape_one_batch);
    unsigned int tensor_size_one_batch_local = stride_one_batch.n * sizeof(float);
    OKKERNEL_ASSERT(tensor_size_one_batch_local * 4 <= okk_local_mem_size_per_npu());
    int M = okk_local_mem_size_per_npu() / 4 / tensor_size_one_batch_local;
    dim4 shape = {.n = M, .c = param->C, .h = param->H, .w = param->W};
    // The output and input are in the aligned layout.
    okk_128_byte_aligned_stride_for_32bit(&stride, 0, &shape);
    unsigned int tensor_size_local = shape.n * stride.n * sizeof(float);
    unsigned int tensor_size_global = shape.n * shape.c * shape.h * shape.w * sizeof(float);
    // Determine addresses of output and input (ping-pong buffers).
    local_addr_t output_addr[2] = {0, tensor_size_local};
    local_addr_t input_addr[2] = {tensor_size_local * 2, tensor_size_local * 3};
    // Get the number of parts and shape of the last part.
    int S = DIV_UP(param->N, M);
    dim4 shape_last = {.n = param->N - (S - 1) * M, .c = param->C, .h = param->H, .w = param->W};
    // Initialize.
    okk_initialize();
    // Step 0 ~ Step S + 1
    for (int i = 0; i < S + 2; ++i) {
        // Start parallel.
        okk_parallel_start();
        // Copy part i of input from global to local.
        if (i < S)
            okk_gdma_32bit_cpy_S2L(input_addr[i % 2], param->input_addr + i * tensor_size_global, i == S - 1 ? &shape_last : &shape, NULL, NULL);
        // Calculate output = input + 1 for part i - 1.
        if (i > 0 && i < S + 1)
            okk_bdc_add_C(output_addr[(i - 1) % 2], input_addr[(i - 1) % 2], 1.f, i - 1 == S - 1 ? &shape_last : &shape, NULL, NULL);
        // Copy part i - 2 of output from local to global.
        if (i > 1)
            okk_gdma_32bit_cpy_L2S(param->output_addr + (i - 2) * tensor_size_global, output_addr[(i - 2) % 2], i - 2 == S - 1 ? &shape_last : &shape, NULL, NULL);
        // End parallel.
        okk_parallel_end();
    }
    // Synchronize.
    okk_poll();
}

OKKERNEL_FUNC_REGISTER(plus_one_2);

void plus_one_3(const void *args) {
    param_t *param = (param_t *)args;
    // Calculate the length of input.
    unsigned long long len = param->N * param->C * param->H * param->W;
    if (len < okk_npu_num())
        plus_one_0(args);
    else {
        unsigned long long L = len;
        param_t param_reshape = {.output_addr = param->output_addr, .input_addr = param->input_addr};
        // Reshape.
        param_reshape.C = okk_npu_num();
        L /= param_reshape.C;
        param_reshape.H = 1;
        param_reshape.W = L < 32 ? L : 32;
        L /= param_reshape.W;
        param_reshape.N = L;
        plus_one_2(&param_reshape);
        // Deal with the tail if it exists.
        L = param_reshape.N * param_reshape.C * param_reshape.H * param_reshape.W;
        if (L < len) {
            param_reshape.output_addr += (len - L) * sizeof(float);
            param_reshape.input_addr += (len - L) * sizeof(float);
            param_reshape.N = len - L;
            param_reshape.C = 1;
            param_reshape.H = 1;
            param_reshape.W = 1;
            plus_one_3(&param_reshape);
        }
    }
}

OKKERNEL_FUNC_REGISTER(plus_one_3);
