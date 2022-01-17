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
    unsigned long long output_addr;
    unsigned long long input_addr;
} __attribute__((packed)) param_t;

static inline void softmax_reference(float *output, const float *input, const param_t &param) {
    for (int n = 0; n < param.N; ++n) {
        for (int h = 0; h < param.H; ++h) {
            for (int w = 0; w < param.W; ++w) {
                float max_cval = -std::numeric_limits<float>::max();
                for (int c = 0; c < param.C; ++c)
                    max_cval = std::max(max_cval, input[n * param.C * param.H * param.W + c * param.H * param.W + h * param.W + w]);
                float sum = 0.f;
                for (int c = 0; c < param.C; ++c)
                    sum += std::exp(input[n * param.C * param.H * param.W + c * param.H * param.W + h * param.W + w] - max_cval);
                for (int c = 0; c < param.C; ++c)
                    output[n * param.C * param.H * param.W + c * param.H * param.W + h * param.W + w] =
                        std::exp(input[n * param.C * param.H * param.W + c * param.H * param.W + h * param.W + w] - max_cval) /
                        sum;
            }
        }
    }
}

int softmax(bm_handle_t &handle, param_t &param, const char *device_func_name) {
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_real_distribution<float> dist_value{-5.f, 5.f};
    float *output_host = nullptr, *input_host = nullptr, *output_ref = nullptr;
    long long len = (long long)param.N * param.C * param.H * param.W;
    // alloc device memory
    bm_device_mem_t output_dev, input_dev;
    BMLIB_SAFE_CALL(bm_malloc_device_byte(handle, &output_dev, len * sizeof(float)));
    BMLIB_SAFE_CALL(bm_malloc_device_byte(handle, &input_dev, len * sizeof(float)));
    param.output_addr = bm_mem_get_device_addr(output_dev);
    param.input_addr = bm_mem_get_device_addr(input_dev);
    // alloc host memory
    output_host = new float[len];
    output_ref = new float[len];
    input_host = new float[len];
    // random input and kernel values
    for (int i = 0; i < len; ++i)
        input_host[i] = dist_value(rng);
    // reference
    softmax_reference(output_ref, input_host, param);
    // copy input from host to device
    BMLIB_SAFE_CALL(bm_memcpy_s2d(handle, input_dev, input_host));
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
    for (long long i = 0; i < len; ++i) {
        if (!std::isfinite(output_host[i]) && !std::isfinite(output_ref[i]))
            continue;
        float max_val = std::max(std::fabs(output_host[i]), std::fabs(output_ref[i]));
        if (!(std::fabs(output_host[i] - output_ref[i]) < 1e-3 * std::max(max_val, 1.f))) {
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
    delete [] output_host;
    delete [] output_ref;
    delete [] input_host;
    return res;
}

int main() {
    bm_handle_t handle;
    // initialize
    BMLIB_SAFE_CALL(bm_dev_request(&handle, 0));
    ////////////////////////////////////////////////////////////////////////
    /// CONTEST CASES
    /// ////////////////////////////////////////////////////////////////////
    param_t params[] = {
        {.N = 1,     .C = 370,  .H = 13,  .W = 13 }, // 0
        {.N = 1,     .C = 1000, .H = 1,   .W = 1  }, // 1
        {.N = 4,     .C = 2,    .H = 157, .W = 283}, // 2
        {.N = 79,    .C = 4090, .H = 1,   .W = 1  }, // 3
        {.N = 6132,  .C = 21,   .H = 1,   .W = 1  }, // 4
    };
    int results[sizeof(params) / sizeof(param_t)];
    for (unsigned int i = 0; i < sizeof(params) / sizeof(param_t); ++i) {
        int res = softmax(handle, params[i], "softmax_contest");
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

