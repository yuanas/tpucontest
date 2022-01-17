#include "okk.h"
#ifndef NULL
#define NULL 0
#endif
#define DIV_UP(a, b) (((a) - 1) / (b) + 1)
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define LOCAL_MEM_SIZE okk_local_mem_size_per_npu()
#define NO_USE 0
typedef struct {
    int N, C, H, W;
    int kernel_h, kernel_w;
    int pad_top, pad_bottom, pad_left, pad_right;
    int stride_h, stride_w;
    int dilation_h, dilation_w;
    unsigned long long output_addr;
    unsigned long long input_addr;
    unsigned long long kernel_addr;
} __attribute__((packed)) param_t;

void depthwise_demo(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;
    const int kernel_h_ext = (param->kernel_h - 1) * param->dilation_h + 1;
    const int kernel_w_ext = (param->kernel_w - 1) * param->dilation_w + 1;
    const int output_h = (param->H + param->pad_top + param->pad_bottom - kernel_h_ext) / param->stride_h + 1;
    const int output_w = (param->W + param->pad_left + param->pad_right - kernel_w_ext) / param->stride_w + 1;
    dim4 output_shape = {.n = param->N, .c = param->C, .h = output_h, .w = output_w};
    dim4 input_shape = {.n = param->N, .c = param->C, .h = param->H, .w = param->W};
    dim4 kernel_shape = {.n = 1, .c = param->C, .h = param->kernel_h, .w = param->kernel_w};
    dim4 output_stride, input_stride, kernel_stride;
    // output is 64-byte aligned layout
    local_addr_t output_addr = 0;
    okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &output_shape);
    // input is 64-byte aligned layout
    local_addr_t input_addr = output_addr + output_shape.n * output_stride.n * sizeof(float);
    okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &input_shape);
    // kernel is compact layout
    local_addr_t kernel_addr = input_addr + input_shape.n * input_stride.n * sizeof(float);
    okk_compact_stride(&kernel_stride, 0, &kernel_shape);
    // check local memory exceeded
    OKKERNEL_ASSERT(kernel_addr + kernel_shape.n * kernel_stride.n * sizeof(float) <= LOCAL_MEM_SIZE);
    // copy input from global memory to local memory
    okk_gdma_32bit_cpy_S2L(
        input_addr,
        param->input_addr,
        &input_shape,
        NULL,
        NULL);
    // copy kernel from global memory to local memory
    okk_gdma_32bit_cpy_S2L(
        kernel_addr,
        param->kernel_addr,
        &kernel_shape,
        &kernel_stride,
        NULL);
    // depthwise
    Padding padding = {
        .top = param->pad_top, .bottom = param->pad_bottom,
        .left = param->pad_left, .right = param->pad_right
    };
    dim2 stride = {.h = param->stride_h, .w = param->stride_w};
    dim2 dilation = {.h = param->dilation_h, .w = param->dilation_w};
    okk_bdc_depthwise2d(
        output_addr,
        input_addr,
        kernel_addr,
        NO_USE,
        &input_shape,
        param->kernel_h,
        param->kernel_w,
        false,
        &padding,
        &stride,
        &dilation);
    // copy output from local memory to global memory
    okk_gdma_32bit_cpy_L2S(
        param->output_addr,
        output_addr,
        &output_shape,
        NULL,
        NULL);
    okk_poll();
}
OKKERNEL_FUNC_REGISTER(depthwise_demo);
