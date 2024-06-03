#pragma once

#include <string>

namespace gpu_easygraph {

typedef enum {
    EG_GPU_SUCC = 0,
    EG_GPU_FAILED_TO_ALLOCATE_HOST_MEM,
    EG_GPU_FAILED_TO_ALLOCATE_DEVICE_MEM,
    EG_GPU_DEVICE_ERR,
    EG_GPU_UNKNOW_ERROR,
    EG_UNSUPPORTED_GRAPH
} EG_GPU_STATUS_CODE;

std::string err_code_detail(
    int status
);

} // namespace gpu_easygraph