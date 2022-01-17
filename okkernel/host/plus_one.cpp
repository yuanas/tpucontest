#include "bmlib_runtime.h"
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <random>
#define BMLIB_SAFE_CALL(cmd) assert(cmd == BM_SUCCESS)

typedef struct {
    unsigned long long output_addr;
    unsigned long long input_addr;
    int N, C, H, W;
} __attribute__((packed)) param_t;

int main() {
    bm_handle_t handle;
    // Initialize.
    BMLIB_SAFE_CALL(bm_dev_request(&handle, 0));
    param_t param;
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_real_distribution<float> dist_value;
    std::uniform_int_distribution<int> dist_n(1, 16);
    std::uniform_int_distribution<int> dist_c(1, 128);
    std::uniform_int_distribution<int> dist_h(1, 64);
    std::uniform_int_distribution<int> dist_w(1, 64);
    bm_device_mem_t output_dev, input_dev;
    float *output_host = nullptr, *input_host = nullptr;
    param.N = dist_n(rng);
    param.C = dist_c(rng);
    param.H = dist_h(rng);
    param.W = dist_w(rng);
    int length = param.N * param.C * param.H * param.W;
    int size = length * 4;
    BMLIB_SAFE_CALL(bm_malloc_device_byte(handle, &output_dev, size));
    BMLIB_SAFE_CALL(bm_malloc_device_byte(handle, &input_dev, size));
    param.output_addr = bm_mem_get_device_addr(output_dev);
    param.input_addr = bm_mem_get_device_addr(input_dev);
    output_host = new float[length];
    input_host = new float[length];
    for (int i = 0; i < length; ++i)
        input_host[i] = dist_value(rng);
    BMLIB_SAFE_CALL(bm_memcpy_s2d(handle, input_dev, input_host));
    // Launch kernel plus_one_0.
    BMLIB_SAFE_CALL(okkernel_launch_sync(handle, "plus_one_0", &param, sizeof(param)));
    BMLIB_SAFE_CALL(bm_memcpy_d2s(handle, output_host, output_dev));
    for (int i = 0; i < length; ++i)
        assert(std::fabs(output_host[i] - (input_host[i] + 1.f)) < 1e-5);
    std::cout << "plus_one_0 succeeded." << std::endl;
    // Launch kernel plus_one_1.
    if (param.N % 2 == 0) {
        BMLIB_SAFE_CALL(okkernel_launch_sync(handle, "plus_one_1", &param, sizeof(param)));
        BMLIB_SAFE_CALL(bm_memcpy_d2s(handle, output_host, output_dev));
        for (int i = 0; i < length; ++i)
            assert(std::fabs(output_host[i] - (input_host[i] + 1.f)) < 1e-5);
        std::cout << "plus_one_1 succeeded." << std::endl;
    }
    // Launch kernel plus_one_2.
    BMLIB_SAFE_CALL(okkernel_launch_sync(handle, "plus_one_2", &param, sizeof(param)));
    BMLIB_SAFE_CALL(bm_memcpy_d2s(handle, output_host, output_dev));
    for (int i = 0; i < length; ++i)
        assert(std::fabs(output_host[i] - (input_host[i] + 1.f)) < 1e-5);
    std::cout << "plus_one_2 succeeded." << std::endl;
    // Launch kernel plus_one_3.
    BMLIB_SAFE_CALL(okkernel_launch_sync(handle, "plus_one_3", &param, sizeof(param)));
    BMLIB_SAFE_CALL(bm_memcpy_d2s(handle, output_host, output_dev));
    for (int i = 0; i < length; ++i)
        assert(std::fabs(output_host[i] - (input_host[i] + 1.f)) < 1e-5);
    std::cout << "plus_one_3 succeeded." << std::endl;
    bm_free_device(handle, output_dev);
    bm_free_device(handle, input_dev);
    delete [] output_host;
    delete [] input_host;
    // Deinitialize.
    bm_dev_free(handle);
    return 0;
}
