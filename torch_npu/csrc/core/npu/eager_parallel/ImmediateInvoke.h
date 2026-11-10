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
        if constexpr (std::is_same_v<std::tuple_element_t<0, Types>, at::Tensor> &&
                      std::is_same_v<std::tuple_element_t<1, Types>, at::Tensor> &&
                      std::is_same_v<std::tuple_element_t<2, Types>, at::Tensor> &&
                      std::is_same_v<std::tuple_element_t<5, Types>, bool> &&
                      std::is_same_v<std::tuple_element_t<7, Types>, bool> &&
                      std::is_same_v<std::tuple_element_t<8, Types>, int64_t> &&
                      std::is_same_v<std::tuple_element_t<3, Types>, bool> &&
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
template <class... Args>
bool BeginImmediateCaller(ImmediateCallerGuard& guard, const char* schema, const Args&... args)
{
    if constexpr (sizeof...(Args)==9) {
        using Types=std::tuple<std::decay_t<Args>...>;
        if constexpr (std::is_same_v<std::tuple_element_t<0,Types>,at::Tensor> &&
                      std::is_same_v<std::tuple_element_t<1,Types>,at::Tensor> &&
                      std::is_same_v<std::tuple_element_t<2,Types>,at::Tensor> &&
                      std::is_same_v<std::tuple_element_t<7,Types>,bool>) {
            if (ImmediateScalarContract(schema,args...)) {
                const auto values=std::tie(args...);
                return guard.begin(schema,std::get<0>(values),std::get<1>(values),
                                   std::get<2>(values),std::get<7>(values));
            }
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
    ImmediateCallerGuard caller;
    try {
        if (detail::BeginImmediateCaller(caller,schema,args...)) {
            using Result=std::invoke_result_t<F>;
            if constexpr (std::is_void_v<Result>) {
                std::forward<F>(body)(); caller.complete(); return;
            } else {
                Result result=std::forward<F>(body)();
                try { caller.complete(); }
                catch (...) {
                    // Exceptional-only capture while every original result is alive.
                    auto output=[&](const at::Tensor& value) { caller.add_failed_output(value); };
                    detail::VisitTensors(output,result);
                    caller.fail(std::current_exception()); throw;
                }
                return result;
            }
        }
    } catch (...) { caller.fail(std::current_exception()); throw; }
    ImmediateGuard guard(schema, route);
    try {
        if (caller.disqualified() || !detail::ImmediateScalarContract(schema, args...)) guard.disqualify();
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
