#include <torch/extension.h>
#include "add.h"

namespace day_01 {
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("add_cpu", &day_01::add_cpu);
        m.def("add_cuda", &day_01::add_cuda);
    }
}
