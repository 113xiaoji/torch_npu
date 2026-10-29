// Compile this against real PyTorch headers. This checks the generated entry
// bridge type surface; it is not a CANN or device-overlap qualification test.
#include "torch_npu/csrc/core/npu/eager_parallel/EntryGuard.h"
#include <ATen/core/IListRef.h>

namespace c10_npu::eager_parallel::test {
void CompileEntryBridge(at::Tensor& tensor, at::TensorList list, const at::ITensorListRef& list_ref,
                        const c10::optional<at::Tensor>& optional,
                        const c10::List<c10::optional<at::Tensor>>& indices)
{
    Invoke("test::void", "compile", [] {}, tensor, list, list_ref, optional, indices);
    auto result = Invoke("test::embedding", "compile", [&] {
        return std::make_tuple(tensor, tensor, tensor, tensor);
    }, std::vector<at::Tensor>{tensor}, false, int64_t{0});
    at::Tensor& reference = Invoke("test::reference", "compile", [&]() -> at::Tensor& {
        return tensor;
    }, result);
    auto scalar = Invoke("test::scalar", "compile", [] { return int64_t{7}; }, reference);
    (void)scalar;
}
} // namespace c10_npu::eager_parallel::test
