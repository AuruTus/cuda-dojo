#include <torch/extension.h>
#include "reduce.h"

namespace reduce {
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("reduce", &reduce::reduce_cuda);
        m.def("test_shfl_down_sync", &reduce::test_shfl_down_sync);
    }
}
