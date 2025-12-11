#include <torch/extension.h>
#include "reduce_max.hpp"

namespace reduce_max {
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("reduce_max", &reduce_max::reduce_max_cuda);
    }
}
