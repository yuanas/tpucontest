#include "okk.h"
#ifndef NULL
#define NULL 0
#endif
#define DIV_UP(a, b) (((a) - 1) / (b) + 1)
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

typedef struct {
    unsigned long long output_addr;
    unsigned long long index_addr;
    unsigned long long input_addr;
    int N, C, H, W;
    int kernel_h, kernel_w;
    int pad_top, pad_bottom, pad_left, pad_right;
    int stride_h, stride_w;
    int ceil_mode;
} __attribute__((packed)) param_t;

void max_pool_0(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;
    dim2 pool_stride = {.h = param->stride_h, .w = param->stride_w};
    const dim4 input_shape = {.n = 1, .c = param->C * param->N, .h = param->H, .w = param->W};
    int output_h = input_shape.h + param->pad_top + param->pad_bottom - param->kernel_h;
    int output_w = input_shape.w + param->pad_left + param->pad_right - param->kernel_w;
    if (param->ceil_mode) {
        output_h = DIV_UP(output_h, param->stride_h) + 1;
        output_w = DIV_UP(output_w, param->stride_w) + 1;
    } else {
        output_h = output_h / param->stride_h + 1;
        output_w = output_w / param->stride_w + 1;
    }
    const dim4 output_shape = {.n = 1, .c = param->C * param->N, .h = output_h, .w = output_w};
    local_addr_t output_addr = 0, input_addr;
    dim4 max_output_shape = output_shape, max_input_shape = input_shape;
    for (;;) {
        bool done = false;
        for (;;) {
            dim4 output_stride, input_stride;
            okk_128_byte_aligned_stride_for_32bit(&output_stride, 0, &max_output_shape);
            unsigned int output_size = output_stride.n * sizeof(float);
            input_addr = output_addr + output_size;
            if (input_addr >= okk_local_mem_size_per_npu())
                break;
            okk_128_byte_aligned_stride_for_32bit(&input_stride, 0, &max_input_shape);
            unsigned int input_size = input_stride.n * sizeof(float);
            if (input_addr + input_size > okk_local_mem_size_per_npu())
                break;
            done = true;
            break;
        }
        if (!done) {
            if (max_output_shape.h > 1) {
                --max_output_shape.h;
                max_input_shape.h = MIN(input_shape.h, param->kernel_h + (max_output_shape.h - 1) * param->stride_h);
            } else if (max_output_shape.c > okk_npu_num()) {
                if (max_output_shape.c % okk_npu_num() == 0)
                    max_output_shape.c -= okk_npu_num();
                else
                    max_output_shape.c -= max_output_shape.c % okk_npu_num();
                max_input_shape.c = max_output_shape.c;
            } else
                OKKERNEL_ASSERT(0);
        } else
            break;
    }
    int remained_output_c = output_shape.c;
    int done_output_c = 0;
    dim4 work_input_shape = {.n = 1, .w = input_shape.w};
    dim4 work_output_shape = {.n = 1, .w = output_shape.w};
    dim4 input_global_stride = {.n = 0, .c = input_shape.h * input_shape.w, .h = input_shape.w, .w = 1};
    dim4 output_global_stride = {.n = 0, .c = output_shape.h * output_shape.w, .h = output_shape.w, .w = 1};
    Padding pad = {.left = param->pad_left, .right = param->pad_right};
    while ((input_shape.h + pad.top + pad.bottom - param->kernel_h) / param->stride_h + 1 != output_shape.h)
        ++pad.bottom;
    while ((input_shape.w + pad.left + pad.right - param->kernel_w) / param->stride_w + 1 != output_shape.w)
        ++pad.right;
    while (remained_output_c > 0) {
        work_output_shape.c = MIN(remained_output_c, max_output_shape.c);
        work_input_shape.c = work_output_shape.c;
        int remained_output_h = output_shape.h;
        int done_output_h = 0;
        while (remained_output_h > 0) {
            work_output_shape.h = MIN(remained_output_h, max_output_shape.h);
            int input_h_start = done_output_h * param->stride_h;
            int input_h_end = input_h_start + param->kernel_h + (work_output_shape.h - 1) * param->stride_h;
            work_input_shape.h = MIN(input_shape.h + param->pad_top, input_h_end) - MAX(param->pad_top, input_h_start);
            OKKERNEL_ASSERT(work_input_shape.h > 0);
            okk_gdma_32bit_cpy_S2L(
                input_addr,
                param->input_addr + (done_output_c * input_shape.h * input_shape.w + MAX(0, input_h_start - param->pad_top) * input_shape.w) * sizeof(float),
                &work_input_shape,
                NULL,
                &input_global_stride);
            pad.top = MAX(param->pad_top - input_h_start, 0);
            pad.bottom = MAX(input_h_end - input_shape.h - param->pad_top, 0);
            okk_bdc_max_pool2d(
                output_addr,
                input_addr,
                &work_input_shape,
                param->kernel_h,
                param->kernel_w,
                &pad,
                &pool_stride);
            okk_gdma_32bit_cpy_L2S(
                param->output_addr + (done_output_c * output_shape.h * output_shape.w + done_output_h * output_shape.w) * sizeof(float),
                output_addr,
                &work_output_shape,
                &output_global_stride,
                NULL);
            remained_output_h -= work_output_shape.h;
            done_output_h += work_output_shape.h;
        }
        remained_output_c -= work_output_shape.c;
        done_output_c += work_output_shape.c;
    }
    okk_poll();
}

OKKERNEL_FUNC_REGISTER(max_pool_0);
