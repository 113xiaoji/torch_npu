// Real ATen containers: verify identity, optional holes and failure propagation.
#include "torch_npu/csrc/core/npu/eager_parallel/TensorVisitor.h"
#include <ATen/Functions.h>
#include <array>
#include <stdexcept>

int main()
{
    using c10_npu::eager_parallel::detail::VisitTensors;
    const auto first = at::empty({1});
    const auto second = at::empty({2});
    const std::vector<at::Tensor> inputs{first, at::Tensor(), second};
    const at::ITensorListRef unboxed{at::TensorList(inputs)};
    c10::List<at::Tensor> boxed_inputs;
    for (const auto& item : inputs) boxed_inputs.push_back(item);
    const at::ITensorListRef boxed{boxed_inputs};
    c10::List<std::optional<at::Tensor>> optional;
    optional.push_back(std::nullopt);
    optional.push_back(first);
    optional.push_back(at::Tensor());
    std::array<const c10::TensorImpl*, 7> actual{};
    std::size_t count = 0;
    auto visitor = [&](const at::Tensor& value) {
        if (count == actual.size()) throw std::runtime_error("unexpected tensor");
        actual[count++] = value.unsafeGetTensorImpl();
    };
    VisitTensors(visitor, unboxed, boxed, optional, std::make_tuple(second, 5, true),
                 std::optional<at::Tensor>{first}, std::nullopt);
    const std::array<const c10::TensorImpl*, 7> expected{
        first.unsafeGetTensorImpl(), second.unsafeGetTensorImpl(),
        first.unsafeGetTensorImpl(), second.unsafeGetTensorImpl(),
        first.unsafeGetTensorImpl(), second.unsafeGetTensorImpl(), first.unsafeGetTensorImpl()};
    if (count != expected.size() || actual != expected) return 1;
    auto fail = [](const at::Tensor&) { throw std::runtime_error("visitor failure"); };
    try { VisitTensors(fail, boxed); }
    catch (const std::runtime_error& error) {
        return std::string(error.what()) == "visitor failure" ? 0 : 2;
    }
    return 3;
}
