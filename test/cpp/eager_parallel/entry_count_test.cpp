// Diagnostic counting contract only. No SDK, kernel, or scheduler is exercised.
#include "torch_npu/csrc/core/npu/eager_parallel/EntryGuard.h"
#include <ATen/core/IListRef.h>
#include <ATen/Functions.h>
#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>

static std::atomic<std::size_t> allocations{0};
void* operator new(std::size_t size)
{
    allocations.fetch_add(1, std::memory_order_relaxed);
    if (void* pointer = std::malloc(size ? size : 1)) return pointer;
    throw std::bad_alloc();
}
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }

struct ThrowsDuringCount {};
struct HugeCount {};
struct TwoHandles {};
namespace c10_npu::eager_parallel::detail {
template <> struct TensorCounter<ThrowsDuringCount> {
    static std::size_t count(const ThrowsDuringCount&) { throw std::bad_alloc(); }
};
template <> struct TensorCounter<HugeCount> {
    static std::size_t count(const HugeCount&) { return IncompleteTensorCount - 1; }
};
template <> struct TensorCounter<TwoHandles> {
    static std::size_t count(const TwoHandles&) { return 2; }
};
} // namespace c10_npu::eager_parallel::detail

int main()
{
    using namespace c10_npu::eager_parallel;
    static_assert(noexcept(EntryGuard("test", "diagnostic", 0)));
    static_assert(noexcept(std::declval<EntryGuard&>().complete(0)));
    static_assert(noexcept(detail::NoThrowTensorCount(ThrowsDuringCount{})));
    if (detail::NoThrowTensorCount(ThrowsDuringCount{}) != detail::IncompleteTensorCount) return 1;
    if (detail::NoThrowTensorCount(HugeCount{}, TwoHandles{}) != detail::IncompleteTensorCount) return 2;
    if (detail::NoThrowTensorCount(int64_t{5}, true, std::optional<int>{3}) != 0) return 3;
    if (detail::NoThrowTensorCount(std::make_tuple(TwoHandles{}, TwoHandles{})) != 4) return 4;
    const at::Tensor defined = at::empty({1});
    const std::vector<at::Tensor> inputs{defined, at::Tensor(), defined};
    const at::ITensorListRef unboxed{at::TensorList(inputs)};
    c10::List<at::Tensor> boxed_inputs;
    for (const auto& value : inputs) boxed_inputs.push_back(value);
    const at::ITensorListRef boxed{boxed_inputs};
    const at::ITensorListRef empty{at::TensorList{}};
    allocations.store(0, std::memory_order_relaxed);
    const auto unboxed_count = detail::NoThrowTensorCount(unboxed);
    const auto boxed_count = detail::NoThrowTensorCount(boxed);
    const auto empty_count = detail::NoThrowTensorCount(empty);
    const auto count_allocations = allocations.load(std::memory_order_relaxed);
    if (unboxed_count != 2 || boxed_count != 2 || empty_count != 0) {
        std::fprintf(stderr, "ITensorListRef defined counts expected 2/2/0, got %zu/%zu/%zu\n",
                     unboxed_count, boxed_count, empty_count);
        return 5;
    }
    if (count_allocations != 0) {
        std::fprintf(stderr, "ITensorListRef counting allocated %zu times\n", count_allocations);
        return 6;
    }
    return 0;
}
