#include "okk.h"
#define DIV_UP(a, b) (((a) - 1) / (b) + 1)

typedef struct {
    unsigned int val;
} __attribute__((packed)) param_t;

void set_local_memory_C(const void *args) {
    param_t *p = (param_t *)args;
    okk_initialize();
    x32 C = {.u32 = p->val};
    int local_mem_size = okk_local_mem_size_per_npu();
    dim4 shape = {.n = 64, .c = okk_npu_num(), .h = 64, .w = local_mem_size / 64 / 64 / 4};
    okk_bdc_32bit_set_C(0, C, &shape, 0);
    okk_poll();
}

OKKERNEL_FUNC_REGISTER(set_local_memory_C);
