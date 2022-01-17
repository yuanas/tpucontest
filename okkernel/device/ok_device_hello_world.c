#include "okk.h"
typedef struct {
    int year;
    int month;
    int day;
} __attribute__((packed)) date_t;

void hello_world(const void *args) {
    date_t *param = (date_t *)args;
    OKKERNEL_LOG("Hello World! Today is %d/%d/%d.\n", param->month, param->day, param->year);
}

OKKERNEL_FUNC_REGISTER(hello_world);
