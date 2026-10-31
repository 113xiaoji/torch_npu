#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/core/List.h>
#include <ATen/core/IListRef.h>
#include <c10/util/ArrayRef.h>
#include <optional>
#include <tuple>
#include <type_traits>
#include <vector>

namespace c10_npu::eager_parallel::detail {

// Visit the actual native handles, without materializing tensor lists. Unlike
// diagnostic counting, failures must propagate before a launch can be admitted.
template <class T> struct TensorVisitor {
    template <class F> static void visit(F&, const T&) {}
};
template <class F, class T> void VisitTensorValue(F& visitor, const T& value)
{
    TensorVisitor<std::decay_t<T>>::visit(visitor, value);
}
template <> struct TensorVisitor<at::Tensor> {
    template <class F> static void visit(F& visitor, const at::Tensor& value)
    { if (value.defined()) visitor(value); }
};
template <> struct TensorVisitor<at::ITensorListRef> {
    template <class F> static void visit(F& visitor, const at::ITensorListRef& values)
    { for (const auto& value : values) VisitTensorValue(visitor, value); }
};
template <class T> struct TensorVisitor<std::optional<T>> {
    template <class F> static void visit(F& visitor, const std::optional<T>& value)
    { if (value) VisitTensorValue(visitor, *value); }
};
template <class T> struct TensorVisitor<c10::ArrayRef<T>> {
    template <class F> static void visit(F& visitor, c10::ArrayRef<T> values)
    { for (const auto& value : values) VisitTensorValue(visitor, value); }
};
template <class T> struct TensorVisitor<c10::List<T>> {
    template <class F> static void visit(F& visitor, const c10::List<T>& values)
    { for (const T value : values) VisitTensorValue(visitor, value); }
};
template <class T, class A> struct TensorVisitor<std::vector<T, A>> {
    template <class F> static void visit(F& visitor, const std::vector<T, A>& values)
    { for (const auto& value : values) VisitTensorValue(visitor, value); }
};
template <class... T> struct TensorVisitor<std::tuple<T...>> {
    template <class F> static void visit(F& visitor, const std::tuple<T...>& values)
    { std::apply([&](const auto&... value) { (VisitTensorValue(visitor, value), ...); }, values); }
};
template <class F, class... T> void VisitTensors(F& visitor, const T&... values)
{ (VisitTensorValue(visitor, values), ...); }

} // namespace c10_npu::eager_parallel::detail
