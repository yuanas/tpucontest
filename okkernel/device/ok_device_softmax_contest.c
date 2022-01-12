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
    unsigned long long output_addr;
    unsigned long long input_addr;
    int N, C, H, W;
} __attribute__((packed)) param_t;

void softmax_contest(const void *args) {
    okk_initialize();
    param_t *param = (param_t *)args;
    (void)(param);
    // TODO
    okk_poll();
}
OKKERNEL_FUNC_REGISTER(softmax_contest);
