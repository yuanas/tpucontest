#include "okk.h"
#define DIV_UP(a, b) (((a) - 1) / (b) + 1)

typedef struct {
    unsigned long long addr;
} __attribute__((packed)) param_t;

void get_local_memory(const void *args) {
    param_t *p = (param_t *)args;
    okk_initialize();
    int local_mem_size = okk_local_mem_size_per_npu();
    dim4 shape = {.n = 1, .c = okk_npu_num(), .h = 256, .w = local_mem_size / 256 / 4};
    okk_gdma_32bit_cpy_L2S(p->addr, 0, &shape, 0, 0);
    okk_poll();
}

OKKERNEL_FUNC_REGISTER(get_local_memory);
