#include <torch/extension.h>
#include "add.h"

namespace add_ops {
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("add_cpu", &add_ops::add_cpu);
        m.def("add_cuda", &add_ops::add_cuda);
    }
}
