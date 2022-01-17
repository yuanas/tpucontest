#include <assert.h>
#include <iostream>
#include <random>
#include <sys/time.h>
#include "okk_param.h"
#include "bmlib_runtime.h"
#define BMLIB_SAFE_CALL(cmd) assert(cmd == BM_SUCCESS)
#define DIV_UP(a, b) (((a) - 1) / (b) + 1)
#ifdef USING_CMODEL
#define MAXIT (1)
#else
#define MAXIT (100)
#endif
typedef struct {
    int left_rows, left_cols, right_cols;
    unsigned long long output_addr;
    unsigned long long left_addr;
    unsigned long long right_addr;
} __attribute__((packed)) param_t;

static inline void matmul_reference(float *output, const float *left, const float *right, const param_t &param) {
    for (int m = 0; m < param.left_rows; ++m) {
        for (int n = 0; n < param.right_cols; ++n) {
            float acc = 0.f;
            for (int k = 0; k < param.left_cols; ++k) {
                float lval = left[m * param.left_cols + k];
                float rval = right[k * param.right_cols + n];
                acc += lval * rval;
            }
            output[m * param.right_cols + n] = acc;
        }
    }
}

int matmul(bm_handle_t &handle, param_t &param, const char *device_func_name) {
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_real_distribution<float> dist_value{-1.f, 1.f};
    float *output_host = nullptr, *left_host = nullptr, *right_host = nullptr, *output_ref = nullptr;
    int output_len = param.left_rows * param.right_cols;
    int left_len = param.left_rows * param.left_cols;
    int right_len = param.left_cols * param.right_cols;
    // alloc device memory
    bm_device_mem_t output_dev, left_dev, right_dev;
    BMLIB_SAFE_CALL(bm_malloc_device_byte(handle, &output_dev, output_len * sizeof(float)));
    BMLIB_SAFE_CALL(bm_malloc_device_byte(handle, &left_dev, left_len * sizeof(float)));
    BMLIB_SAFE_CALL(bm_malloc_device_byte(handle, &right_dev, right_len * sizeof(float)));
    param.output_addr = bm_mem_get_device_addr(output_dev);
    param.left_addr = bm_mem_get_device_addr(left_dev);
    param.right_addr = bm_mem_get_device_addr(right_dev);
    // alloc host memory
    output_host = new float[output_len];
    output_ref = new float[output_len];
    left_host = new float[left_len];
    right_host = new float[right_len];
    // random left matrix and right matrix values
    for (int i = 0; i < left_len; ++i)
        left_host[i] = dist_value(rng);
    for (int i = 0; i < right_len; ++i)
        right_host[i] = dist_value(rng);
    // reference
    matmul_reference(output_ref, left_host, right_host, param);
    // copy left matrix and right matrix from host to device
    BMLIB_SAFE_CALL(bm_memcpy_s2d(handle, left_dev, left_host));
    BMLIB_SAFE_CALL(bm_memcpy_s2d(handle, right_dev, right_host));
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
    for (int i = 0; i < output_len; ++i) {
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
    bm_free_device(handle, left_dev);
    bm_free_device(handle, right_dev);
    delete [] output_host;
    delete [] output_ref;
    delete [] left_host;
    delete [] right_host;
    return res;
}

int main() {
    bm_handle_t handle;
    // Initialize.
    BMLIB_SAFE_CALL(bm_dev_request(&handle, 0));
    // demo
    param_t param;
    param.left_rows = 100;
    param.left_cols = 200;
    param.right_cols = 150;
    if (matmul(handle, param, "matmul_demo") >= 0)
        std::cout << "matmul_demo pass" << std::endl;
    else
        std::cout << "matmul_demo fail" << std::endl;
    ////////////////////////////////////////////////////////////////////////
    /// CONTEST CASES
    /// ////////////////////////////////////////////////////////////////////
    param_t params[] = {
        {.left_rows = 2,      .left_cols = 100352, .right_cols = 2048 }, // 0
        {.left_rows = 2,      .left_cols = 1280,   .right_cols = 1000 }, // 1
        {.left_rows = 2,      .left_cols = 25088,  .right_cols = 4096 }, // 2
        {.left_rows = 4,      .left_cols = 1024,   .right_cols = 25088}, // 3
        {.left_rows = 32,     .left_cols = 2048,   .right_cols = 36   }, // 4
        {.left_rows = 64,     .left_cols = 9216,   .right_cols = 4096 }, // 5
        {.left_rows = 79,     .left_cols = 256,    .right_cols = 4090 }, // 6
        {.left_rows = 200,    .left_cols = 4096,   .right_cols = 324  }, // 7
        {.left_rows = 256,    .left_cols = 768,    .right_cols = 3072 }, // 8
        {.left_rows = 256,    .left_cols = 3072,   .right_cols = 768  }, // 9
        {.left_rows = 300,    .left_cols = 2048,   .right_cols = 80   }, // 10
        {.left_rows = 1024,   .left_cols = 1024,   .right_cols = 1024 }, // 11
        {.left_rows = 2048,   .left_cols = 4,      .right_cols = 1024 }, // 12
        {.left_rows = 12544,  .left_cols = 2,      .right_cols = 1024 }, // 13
        {.left_rows = 100352, .left_cols = 1024,   .right_cols = 1    }, // 14
    };
    int results[sizeof(params) / sizeof(param_t)];
    for (unsigned int i = 0; i < sizeof(params) / sizeof(param_t); ++i) {
        int res = matmul(handle, params[i], "matmul_contest");
        if (res >= 0)
            std::cout << "case " << i << " pass" << std::endl;
        else
            std::cout << "case " << i << " fail" << std::endl;
        results[i] = res;
    }
    (void)(results);
    // Deinitialize.
    bm_dev_free(handle);
    return 0;
}

