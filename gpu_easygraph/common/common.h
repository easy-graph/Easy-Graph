#pragma once

#define EG_FLOAT_INF 1e10
#define EG_FLOAT_EPSILON 1e-6

#define EXIT_IF_CUDA_FAILED(condition)              \
        cuda_ret = condition;                       \
        if (cuda_ret != cudaSuccess) {              \
            goto exit;                              \
        }                                           \

#define IS_EQUAL(x, y)                              \
        ((-EG_FLOAT_EPSILON <= ((x) - (y)))          \
        && (((x) - (y)) <= EG_FLOAT_EPSILON))       \

#ifndef _IN_
#define _IN_
#endif

#ifndef _OUT_
#define _OUT_
#endif

typedef enum {
    EG_GPU_SUCC = 0,
    EG_GPU_FAILED_TO_ALLOCATE_HOST_MEM,
    EG_GPU_FAILED_TO_ALLOCATE_DEVICE_MEM,
    EG_GPU_DEVICE_ERR,
    EG_GPU_UNKNOW_ERROR
} EG_GPU_STATUS_CODE;

typedef enum {
    EG_GPU_NODE_ACTIVE,
    EG_GPU_NODE_INACTIVE
} EG_GPU_NODE_STATUS;