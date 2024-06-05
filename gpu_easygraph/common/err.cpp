#include <string>

#include "err.h"

namespace gpu_easygraph {

using std::string;

std::string err_code_detail(
    int status
) {
    switch (status) {
        case EG_GPU_SUCC:
            return "EasyGraph GPU: success";
        case EG_GPU_FAILED_TO_ALLOCATE_HOST_MEM:
            return "EasyGraph GPU: failed to allocate host mem";
        case EG_GPU_FAILED_TO_ALLOCATE_DEVICE_MEM:
            return "EasyGraph GPU: failed to allocate gpu mem";
        case EG_GPU_DEVICE_ERR:
            return "EasyGraph GPU: gpu error occurred";
        case EG_GPU_UNKNOW_ERROR:
            return "EasyGraph GPU: gpu unkonw error";
        case EG_UNSUPPORTED_GRAPH:
            return "EasyGraph GPU: unsupported graph type";
        default:
            break;
    }
    return "EasyGraph GPU: not a valid err_code";
}

} // namespace gpu_easygraph