#include "bmlib_runtime.h"
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <fstream>

#define BMLIB_SAFE_CALL(cmd) assert(cmd == BM_SUCCESS)
#define LOCAL_MEM_SIZE (1 << (10 + 9))
#define NPU_NUM (1 << 6)

typedef struct {
    unsigned long long addr;
} __attribute__((packed)) date_t;

int main(int argc, char *argv[]) {
    std::string path = "./";
    if (argc > 1)
        path = std::string(argv[1]) + "/";
    bm_handle_t handle = nullptr;
    date_t param;
    bm_device_mem_t devMem;
    // Initialize.
    BMLIB_SAFE_CALL(bm_dev_request(&handle, 0));
    char *hostMem = new char[LOCAL_MEM_SIZE * NPU_NUM];
    BMLIB_SAFE_CALL(bm_malloc_device_byte(handle, &devMem, LOCAL_MEM_SIZE * NPU_NUM));
    param.addr = bm_mem_get_device_addr(devMem);
    // Launch kernel function.
    BMLIB_SAFE_CALL(okkernel_launch_sync(handle, "get_local_memory", &param, sizeof(param)));
    BMLIB_SAFE_CALL(bm_memcpy_d2s(handle, hostMem, devMem));
    // Write file
    for (int i = 0; i < NPU_NUM; ++i) {
        std::string fname = path + "local_mem_" + std::to_string(i) + ".bin";
        std::ofstream f(fname, std::ios::out | std::ios::binary);
        if (!f) {
            std::cout << "Failed to create file " << fname << std::endl;
            break;
        }
        f.write(hostMem + i * LOCAL_MEM_SIZE, LOCAL_MEM_SIZE);
        f.close();
    }
    delete [] hostMem;
    bm_free_device(handle, devMem);
    // Deinitialize.
    bm_dev_free(handle);
    return 0;
}
