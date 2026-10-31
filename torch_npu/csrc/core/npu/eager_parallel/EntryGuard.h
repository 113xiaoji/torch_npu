#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/core/List.h>
#include <ATen/core/IListRef.h>
#include <c10/util/ArrayRef.h>
#include <exception>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include "torch_npu/csrc/core/npu/NPUMacros.h"
#include "torch_npu/csrc/core/npu/eager_parallel/ImmediateInvoke.h"

namespace c10_npu::eager_parallel {

// Shadow diagnostics only count handles; no owner is captured or retained here.
// Counts do not qualify scalar arguments, allocation lifetime, or SDK effects.
C10_NPU_API bool Enabled() noexcept;
C10_NPU_API void Boundary(const char* reason);

class C10_NPU_API EntryGuard {
public:
    EntryGuard(const char* schema, const char* route, std::size_t input_count) noexcept;
    ~EntryGuard() noexcept;
    EntryGuard(const EntryGuard&) = delete;
    EntryGuard& operator=(const EntryGuard&) = delete;
    void complete(std::size_t output_count) noexcept;
    void fail(std::exception_ptr error) noexcept;

private:
    std::uint64_t token_ = 0;
    bool finished_ = false;
};

namespace detail {
inline constexpr std::size_t IncompleteTensorCount = std::numeric_limits<std::size_t>::max();
inline std::size_t add_count(std::size_t first, std::size_t second) noexcept
{
    return second >= IncompleteTensorCount - first ? IncompleteTensorCount : first + second;
}
// Only these explicit container types are counted. Unknown classes take the
// scalar path; this diagnostic count is not evidence of full effect coverage.
template <class T> struct TensorCounter {
    static std::size_t count(const T&) { return 0; }
};
template <> struct TensorCounter<at::Tensor> {
    static std::size_t count(const at::Tensor& value)
    {
        return value.defined() ? 1 : 0;
    }
};
template <class T> std::size_t tensor_count(const T& value)
{
    return TensorCounter<std::decay_t<T>>::count(value);
}
template <> struct TensorCounter<at::ITensorListRef> {
    static std::size_t count(const at::ITensorListRef& values)
    {
        std::size_t result = 0;
        for (const auto& value : values) result = add_count(result, tensor_count(value));
        return result;
    }
};
template <class T> struct TensorCounter<std::optional<T>> {
    static std::size_t count(const std::optional<T>& value)
    {
        return value ? tensor_count(*value) : 0;
    }
};
template <class T> struct TensorCounter<c10::ArrayRef<T>> {
    static std::size_t count(c10::ArrayRef<T> values)
    {
        std::size_t result = 0;
        for (const auto& value : values) result = add_count(result, tensor_count(value));
        return result;
    }
};
template <class T> struct TensorCounter<c10::List<T>> {
    static std::size_t count(const c10::List<T>& values)
    {
        std::size_t result = 0;
        for (const T value : values) result = add_count(result, tensor_count(value));
        return result;
    }
};
template <class T, class A> struct TensorCounter<std::vector<T, A>> {
    static std::size_t count(const std::vector<T, A>& values)
    {
        std::size_t result = 0;
        for (const auto& value : values) result = add_count(result, tensor_count(value));
        return result;
    }
};
template <class... T> struct TensorCounter<std::tuple<T...>> {
    static std::size_t count(const std::tuple<T...>& values)
    {
        std::size_t result = 0;
        std::apply([&](const auto&... value) {
            ((result = add_count(result, tensor_count(value))), ...);
        }, values);
        return result;
    }
};
template <class... Args> std::size_t NoThrowTensorCount(const Args&... args) noexcept
{
    try {
        std::size_t result = 0;
        ((result = add_count(result, tensor_count(args))), ...);
        return result;
    } catch (...) {
        return IncompleteTensorCount;
    }
}
} // namespace detail

template <class F, class... Args>
decltype(auto) Invoke(const char* schema, const char* route, F&& body, const Args&... args)
{
    if (RuntimeEnabled()) return InvokeImmediate(schema, route, std::forward<F>(body), args...);
    if (!Enabled()) return std::forward<F>(body)();
    EntryGuard guard(schema, route, detail::NoThrowTensorCount(args...));
    try {
        using Result = std::invoke_result_t<F>;
        if constexpr (std::is_void_v<Result>) {
            std::forward<F>(body)();
            guard.complete(0);
            return;
        } else {
            Result result = std::forward<F>(body)();
            guard.complete(detail::NoThrowTensorCount(result));
            return result;
        }
    } catch (...) {
        guard.fail(std::current_exception());
        throw;
    }
}

} // namespace c10_npu::eager_parallel
