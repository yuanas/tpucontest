#include "bmlib_runtime.h"
typedef struct {
    int year;
    int month;
    int day;
} __attribute__((packed)) date_t;
int main() {
    bm_handle_t handle;
    date_t param = {.year = 2021, .month = 1, .day = 1};
    // Initialize.
    bm_dev_request(&handle, 0);
    // Launch kernel function.
    okkernel_launch_sync(handle, "hello_world", &param, sizeof(param));
    // Deinitialize.
    bm_dev_free(handle);
    return 0;
}
