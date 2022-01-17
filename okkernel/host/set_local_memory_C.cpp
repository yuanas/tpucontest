#include "bmlib_runtime.h"
#include <iostream>

typedef struct {
    unsigned int val;
} __attribute__((packed)) date_t;

int main(int argc, char *argv[]) {
    bm_handle_t handle;
    int val = 0;
    if (argc > 1)
        val = atoi(argv[1]);
    date_t param = {.val = (unsigned int &)val};
    // Initialize.
    bm_dev_request(&handle, 0);
    // Launch kernel function.
    okkernel_launch_sync(handle, "set_local_memory_C", &param, sizeof(param));
    // Deinitialize.
    bm_dev_free(handle);
    return 0;
}
