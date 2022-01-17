#include <assert.h>
#include <iostream>
#include <random>
#include <sys/time.h>
#include "bmlib_runtime.h"
#define BMLIB_SAFE_CALL(cmd) assert(cmd == BM_SUCCESS)
#define DIV_UP(a, b) (((a) - 1) / (b) + 1)
#ifdef USING_CMODEL
#define MAXIT (1)
#else
#define MAXIT (100)
#endif
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

static inline void depthwise_reference(float *output, const float *input, const float *kernel, const param_t &param) {
    const int kernel_h_ext = (param.kernel_h - 1) * param.dilation_h + 1;
    const int kernel_w_ext = (param.kernel_w - 1) * param.dilation_w + 1;
    const int output_h = (param.H + param.pad_top + param.pad_bottom - kernel_h_ext) / param.stride_h + 1;
    const int output_w = (param.W + param.pad_left + param.pad_right - kernel_w_ext) / param.stride_w + 1;
    for (int n = 0; n < param.N; ++n) {
        for (int c = 0; c < param.C; ++c) {
            for (int oh = 0; oh < output_h; ++oh) {
                for (int ow = 0; ow < output_w; ++ow) {
                    float acc = 0.f;
                    for (int kh = 0; kh < param.kernel_h; ++kh) {
                        for (int kw = 0; kw < param.kernel_w; ++kw) {
                            int ih = oh * param.stride_h + kh * param.dilation_h - param.pad_top;
                            int iw = ow * param.stride_w + kw * param.dilation_w - param.pad_left;
                            if (ih >= 0 && ih < param.H && iw >= 0 && iw < param.W) {
                                float ival = input[n * param.C * param.H * param.W + c * param.H * param.W + ih * param.W + iw];
                                float kval = kernel[c * param.kernel_h * param.kernel_w + kh * param.kernel_w + kw];
                                acc += ival * kval;
                            }
                        }
                    }
                    output[n * param.C * output_h * output_w + c * output_h * output_w + oh * output_w + ow] = acc;
                }
            }
        }
    }
}

int depthwise(bm_handle_t &handle, param_t &param, const char *device_func_name) {
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_real_distribution<float> dist_value{-1.f, 1.f};
    float *output_host = nullptr, *input_host = nullptr, *kernel_host = nullptr, *output_ref = nullptr;
    const int kernel_h_ext = (param.kernel_h - 1) * param.dilation_h + 1;
    const int kernel_w_ext = (param.kernel_w - 1) * param.dilation_w + 1;
    const int output_h = (param.H + param.pad_top + param.pad_bottom - kernel_h_ext) / param.stride_h + 1;
    const int output_w = (param.W + param.pad_left + param.pad_right - kernel_w_ext) / param.stride_w + 1;
    long long input_len = (long long)param.N * param.C * param.H * param.W;
    long long kernel_len = (long long)param.C * param.kernel_h * param.kernel_w;
    long long output_len = (long long)param.N * param.C * output_h * output_w;
    // alloc device memory
    bm_device_mem_t output_dev, input_dev, kernel_dev;
    BMLIB_SAFE_CALL(bm_malloc_device_byte(handle, &output_dev, output_len * sizeof(float)));
    BMLIB_SAFE_CALL(bm_malloc_device_byte(handle, &input_dev, input_len * sizeof(float)));
    BMLIB_SAFE_CALL(bm_malloc_device_byte(handle, &kernel_dev, kernel_len * sizeof(float)));
    param.output_addr = bm_mem_get_device_addr(output_dev);
    param.input_addr = bm_mem_get_device_addr(input_dev);
    param.kernel_addr = bm_mem_get_device_addr(kernel_dev);
    // alloc host memory
    output_host = new float[output_len];
    output_ref = new float[output_len];
    input_host = new float[input_len];
    kernel_host = new float[kernel_len];
    // random input and kernel values
    for (int i = 0; i < input_len; ++i)
        input_host[i] = dist_value(rng);
    for (int i = 0; i < kernel_len; ++i)
        kernel_host[i] = dist_value(rng);
    // reference
    depthwise_reference(output_ref, input_host, kernel_host, param);
    // copy input and kernel from host to device
    BMLIB_SAFE_CALL(bm_memcpy_s2d(handle, input_dev, input_host));
    BMLIB_SAFE_CALL(bm_memcpy_s2d(handle, kernel_dev, kernel_host));
    // launch kernel function
    struct timeval start_time, end_time;
    long long elapsed_time = 0;
    for (int i = 0; i < MAXIT; ++i) {
        gettimeofday(&start_time, NULL);
        BMLIB_SAFE_CALL(okkernel_launch_sync(handle, device_func_name, &param, sizeof(param)));
        gettimeofday(&end_time, NULL);
        elapsed_time += (end_time.tv_sec - start_time.tv_sec) * 1000000 + (end_time.tv_usec - start_time.tv_usec);
    }
    // copy output from device to host
    BMLIB_SAFE_CALL(bm_memcpy_d2s(handle, output_host, output_dev));
    bool pass = true;
    for (long long i = 0; i < output_len; ++i) {
        if (!std::isfinite(output_host[i]) && !std::isfinite(output_ref[i]))
            continue;
        float max_val = std::max(std::fabs(output_host[i]), std::fabs(output_ref[i]));
        if (!(std::fabs(output_host[i] - output_ref[i]) < 1e-4 * std::max(max_val, 1.f))) {
            pass = false;
            break;
        }
    }
    int res = -1;
    if (pass) {
        res = std::round(elapsed_time / (double)MAXIT);
        std::cout << "elapsed time: " << res << "(us)" << std::endl;
    }
    // free
    bm_free_device(handle, output_dev);
    bm_free_device(handle, input_dev);
    bm_free_device(handle, kernel_dev);
    delete [] output_host;
    delete [] output_ref;
    delete [] input_host;
    delete [] kernel_host;
    return res;
}

int main() {
    bm_handle_t handle;
    // initialize
    BMLIB_SAFE_CALL(bm_dev_request(&handle, 0));
    // demo
    param_t param;
    param.N = 4;
    param.C = 64;
    param.W = 16;
    param.H = 16;
    param.kernel_h = 3;
    param.kernel_w = 3;
    param.stride_h = 2;
    param.stride_w = 2;
    param.dilation_h = 2;
    param.dilation_w = 2;
    param.pad_top = 1;
    param.pad_bottom = 1;
    param.pad_left = 1;
    param.pad_right = 1;
    if (depthwise(handle, param, "depthwise_demo") >= 0)
        std::cout << "depthwise_demo pass" << std::endl;
    else
        std::cout << "depthwise_demo fail" << std::endl;
    ////////////////////////////////////////////////////////////////////////
    /// CONTEST CASES
    /// ////////////////////////////////////////////////////////////////////
    param_t params[] = {
        {.N = 4, .C = 3,    .H = 224, .W = 224, .kernel_h = 11, .kernel_w = 11, .pad_top = 2, .pad_bottom = 2, .pad_left = 2, .pad_right = 2, .stride_h = 4, .stride_w = 4, .dilation_h = 1, .dilation_w = 1}, // 0
        {.N = 4, .C = 3,    .H = 256, .W = 256, .kernel_h = 7,  .kernel_w = 7,  .pad_top = 3, .pad_bottom = 3, .pad_left = 3, .pad_right = 3, .stride_h = 2, .stride_w = 2, .dilation_h = 1, .dilation_w = 1}, // 1
        {.N = 4, .C = 3,    .H = 640, .W = 640, .kernel_h = 3,  .kernel_w = 3,  .pad_top = 1, .pad_bottom = 1, .pad_left = 1, .pad_right = 1, .stride_h = 2, .stride_w = 2, .dilation_h = 1, .dilation_w = 1}, // 2
        {.N = 4, .C = 96,   .H = 150, .W = 150, .kernel_h = 3,  .kernel_w = 3,  .pad_top = 1, .pad_bottom = 1, .pad_left = 1, .pad_right = 1, .stride_h = 2, .stride_w = 2, .dilation_h = 1, .dilation_w = 1}, // 3
        {.N = 4, .C = 144,  .H = 75,  .W = 75,  .kernel_h = 3,  .kernel_w = 3,  .pad_top = 1, .pad_bottom = 1, .pad_left = 1, .pad_right = 1, .stride_h = 1, .stride_w = 1, .dilation_h = 1, .dilation_w = 1}, // 4
        {.N = 4, .C = 192,  .H = 38,  .W = 38,  .kernel_h = 3,  .kernel_w = 3,  .pad_top = 1, .pad_bottom = 1, .pad_left = 1, .pad_right = 1, .stride_h = 2, .stride_w = 2, .dilation_h = 1, .dilation_w = 1}, // 5
        {.N = 4, .C = 336,  .H = 29,  .W = 29,  .kernel_h = 5,  .kernel_w = 5,  .pad_top = 2, .pad_bottom = 2, .pad_left = 2, .pad_right = 2, .stride_h = 2, .stride_w = 2, .dilation_h = 1, .dilation_w = 1}, // 6
        {.N = 4, .C = 512,  .H = 14,  .W = 14,  .kernel_h = 3,  .kernel_w = 3,  .pad_top = 0, .pad_bottom = 1, .pad_left = 0, .pad_right = 1, .stride_h = 2, .stride_w = 2, .dilation_h = 1, .dilation_w = 1}, // 7
        {.N = 4, .C = 960,  .H = 28,  .W = 28,  .kernel_h = 3,  .kernel_w = 3,  .pad_top = 4, .pad_bottom = 4, .pad_left = 4, .pad_right = 4, .stride_h = 1, .stride_w = 1, .dilation_h = 4, .dilation_w = 4}, // 8
        {.N = 4, .C = 2048, .H = 33,  .W = 33,  .kernel_h = 3,  .kernel_w = 3,  .pad_top = 6, .pad_bottom = 6, .pad_left = 6, .pad_right = 6, .stride_h = 1, .stride_w = 1, .dilation_h = 6, .dilation_w = 6}, // 9
    };
    int results[sizeof(params) / sizeof(param_t)];
    for (unsigned int i = 0; i < sizeof(params) / sizeof(param_t); ++i) {
        int res = depthwise(handle, params[i], "depthwise_contest");
        if (res >= 0)
            std::cout << "case " << i << " pass" << std::endl;
        else
            std::cout << "case " << i << " fail" << std::endl;
        results[i] = res;
    }
    (void)(results);
    // deinitialize
    bm_dev_free(handle);
    return 0;
}

