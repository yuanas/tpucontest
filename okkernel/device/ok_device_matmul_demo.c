#include "okk.h"
#ifndef NULL
#define NULL 0
#endif
#define DIV_UP(a, b) (((a) - 1) / (b) + 1)
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define LOCAL_MEM_SIZE okk_local_mem_size_per_npu()
#define NPU_NUM okk_npu_num()
#define NO_USE 0
typedef struct {
    int left_rows, left_cols, right_cols;
    unsigned long long output_addr;
    unsigned long long left_addr;
    unsigned long long right_addr;
} __attribute__((packed)) param_t;

void matmul_demo(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;
    dim4 output_stride, left_stride, right_stride;
    int left_cols_per_channel = DIV_UP(param->left_cols, NPU_NUM);
    if (left_cols_per_channel > 128)
        left_cols_per_channel = 128;
    int right_cols_per_channel = DIV_UP(param->right_cols, NPU_NUM);
    if (right_cols_per_channel > 128)
        right_cols_per_channel = 128;
    // Local left matrix tensor.
    local_addr_t left_addr = 0;
    dim4 left_shape = {
        .n = param->left_rows, .c = DIV_UP(param->left_cols, left_cols_per_channel),
        .h = 1, .w = left_cols_per_channel
    };
    okk_128_byte_aligned_stride_for_32bit(&left_stride, 0, &left_shape);
    // Local right matrix tensor.
    local_addr_t right_addr = left_addr + left_stride.n * left_shape.n * sizeof(float);
    dim4 right_shape = {
        .n = param->left_cols, .c = DIV_UP(param->right_cols, right_cols_per_channel),
        .h = 1, .w = right_cols_per_channel
    };
    okk_128_byte_aligned_stride_for_32bit(&right_stride, 0, &right_shape);
    // Local output matrix tensor.
    local_addr_t output_addr = right_addr + right_stride.n * right_shape.n * sizeof(float);
    dim4 output_shape = {
        .n = param->left_rows, .c = DIV_UP(param->right_cols, right_cols_per_channel),
        .h = 1, .w = right_cols_per_channel
    };
    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
    // Safe checking.
    OKKERNEL_ASSERT(output_addr + output_stride.n * output_shape.n * sizeof(float) <= LOCAL_MEM_SIZE);
    // Copy global left matrix tensor to local left matrix tensor.
    okk_gdma_32bit_matrix_S2L(
        left_addr,
        param->left_addr,
        param->left_rows,
        param->left_cols,
        left_cols_per_channel,
        param->left_cols);
    // Copy global right matrix tensor to local right matrix tensor.
    okk_gdma_32bit_matrix_S2L(
        right_addr,
        param->right_addr,
        param->left_cols,
        param->right_cols,
        right_cols_per_channel,
        param->right_cols);
    // Matrix multiplication.
    okk_bdc_matmul(
        output_addr,
        left_addr,
        right_addr,
        NO_USE,
        param->left_rows,
        param->left_cols,
        param->right_cols,
        left_cols_per_channel,
        right_cols_per_channel,
        false,
        false);
    // Copy local output matrix tensor to global output matrix tensor.
    okk_gdma_32bit_matrix_L2S(
        param->output_addr,
        output_addr,
        param->left_rows,
        param->right_cols,
        right_cols_per_channel,
        param->right_cols);
    okk_poll();
}
OKKERNEL_FUNC_REGISTER(matmul_demo);
