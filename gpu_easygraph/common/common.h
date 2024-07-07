#pragma once

#include "err.h"

#define EG_DOUBLE_INF 1e100

#define EXIT_IF_CUDA_FAILED(condition)              \
        cuda_ret = condition;                       \
        if (cuda_ret != cudaSuccess) {              \
            goto exit;                              \
        }                                           \

#ifndef _IN_
#define _IN_
#endif

#ifndef _OUT_
#define _OUT_
#endif

#ifndef _BUFFER_
#define _BUFFER_
#endif