#include <torch/extension.h>
#include "matmul.h"

namespace matmul_tile {
	PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
		m.def("matmul_tile", &matmul_tile::matmul_tile_cuda);
	}
}
