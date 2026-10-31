#pragma once

#include "torch_npu/csrc/core/npu/eager_parallel/ImmediateRuntime.h"
#include "torch_npu/csrc/core/npu/eager_parallel/TensorVisitor.h"
#include <cstring>
#include <exception>
#include <type_traits>
#include <utility>

namespace c10_npu::eager_parallel {
namespace detail {
template <class... Args> bool ImmediateScalarContract(const char* schema, const Args&... args)
{
    if (std::strcmp(schema, "aten::_embedding_bag") != 0 &&
        std::strcmp(schema, "aten::_embedding_bag_forward_only") != 0) return true;
    if constexpr (sizeof...(Args) == 9) {
        using Types = std::tuple<std::decay_t<Args>...>;
        if constexpr (std::is_same_v<std::tuple_element_t<3, Types>, bool> &&
                      std::is_same_v<std::tuple_element_t<4, Types>, int64_t> &&
                      std::is_same_v<std::tuple_element_t<6, Types>, std::optional<at::Tensor>>) {
            const auto values = std::tie(args...);
            const auto& weights = std::get<6>(values);
            return !std::get<3>(values) && std::get<4>(values) == 0 &&
                   (!weights || !weights->defined());
        }
    }
    return false;
}
} // namespace detail

// The original body runs immediately on this caller. Its captures never escape.
// All effects/lifetime accounting is handled by the native begin/complete pair.
template <class F, class... Args>
decltype(auto) InvokeImmediate(const char* schema, const char* route, F&& body, const Args&... args)
{
    ImmediateGuard guard(schema, route);
    try {
        if (!detail::ImmediateScalarContract(schema, args...)) guard.disqualify();
        auto input = [&](const at::Tensor& value) { guard.add_input(value); };
        detail::VisitTensors(input, args...);
        guard.begin();
        using Result = std::invoke_result_t<F>;
        if constexpr (std::is_void_v<Result>) {
            std::forward<F>(body)();
            guard.complete();
            return;
        } else {
            Result result = std::forward<F>(body)();
            try {
                auto output = [&](const at::Tensor& value) { guard.add_output(value); };
                detail::VisitTensors(output, result);
                guard.complete();
            } catch (...) {
                // Recovery must run while every original result is still alive,
                // including outputs not yet visited when a visitor fails.
                guard.fail(std::current_exception());
                throw;
            }
            return result;
        }
    } catch (...) {
        guard.fail(std::current_exception());
        throw;
    }
}
} // namespace c10_npu::eager_parallel
