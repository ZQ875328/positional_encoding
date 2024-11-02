#include <torch/extension.h>

torch::Tensor pe(torch::Tensor input, torch::Tensor vec);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("pe", torch::wrap_pybind_function(pe), "pe");
}