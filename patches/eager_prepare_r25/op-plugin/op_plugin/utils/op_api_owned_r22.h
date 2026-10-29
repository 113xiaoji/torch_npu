#pragma once

#include "op_api_common_base.h"
#include "op_api_owned_pack_r22.h"
#include <array>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace hdb_opapi {

using NativeOwners = UniqueOwners<c10::Storage>;
struct OwnedTensor {
    // Each logical view has its own metadata. NativeOwners holds one Storage
    // reference per identity; no TensorImpl/PyObject enters the delayed command.
    TensorStructPtr metadata;
};
TORCH_NPU_API void CheckOwnedSdkProfile(const char* operation);
TORCH_NPU_API void PoisonOwnedSdkCleanup() noexcept;

TORCH_NPU_API OwnedTensor SnapshotOwnedTensor(NativeOwners& owners, const at::Tensor& tensor,
    c10::optional<aclDataType> forced_type = c10::nullopt);

struct OwnedSnapshotter {
    NativeOwners& owners;

    OwnedTensor operator()(const at::Tensor& value) const
    {
        return SnapshotOwnedTensor(owners, value);
    }
    OwnedTensor operator()(const TensorWrapper& value) const
    {
        return SnapshotOwnedTensor(owners, value.tensor_, value.dtype);
    }
    std::vector<OwnedTensor> operator()(const TensorListWrapper& value) const
    {
        std::vector<OwnedTensor> result;
        result.reserve(value.tensor_list_.size());
        for (const auto& tensor : value.tensor_list_)
            result.emplace_back(SnapshotOwnedTensor(owners, tensor, value.dtype));
        return result;
    }
    at::Scalar operator()(const at::Scalar& value) const
    {
        TORCH_CHECK(!value.isSymbolic(), "owned OpAPI does not admit symbolic scalar callbacks");
        return value;
    }
    std::string operator()(const std::string& value) const { return value; }
    std::string operator()(const char* value) const
    {
        if (!value) throw std::invalid_argument("owned OpAPI string is null");
        return value;
    }
    std::string operator()(char* value) const { return (*this)(static_cast<const char*>(value)); }
    template<size_t N> std::string operator()(const char (&value)[N]) const
    {
        return (*this)(static_cast<const char*>(value));
    }
    template<class T> std::decay_t<T> operator()(const T& value) const
    {
        if constexpr (std::is_arithmetic<T>::value || std::is_enum<T>::value || std::is_same<T, std::nullptr_t>::value) {
            return value;
        } else {
            // Inactive ordinary macros still instantiate this template. Reject
            // unknown borrowed objects only when this real capture path runs.
            throw std::invalid_argument("owned OpAPI argument type has no ownership adapter");
        }
    }
    template<size_t N> std::array<bool, N> operator()(const std::array<bool, N>& value) const { return value; }

    std::vector<int64_t> operator()(const at::ArrayRef<c10::SymInt>& value) const
    {
        for (const auto& item : value)
            TORCH_CHECK(!item.is_symbolic(), "owned OpAPI does not admit symbolic dimension callbacks");
        const auto concrete = c10::asIntArrayRefUnchecked(value);
        return {concrete.begin(), concrete.end()};
    }
    template<class T> auto operator()(const at::ArrayRef<T>& value) const
    {
        using Item = std::decay_t<decltype((*this)(std::declval<const T&>()))>;
        std::vector<Item> result;
        result.reserve(value.size());
        for (const auto& item : value) result.emplace_back((*this)(item));
        return result;
    }
    template<class T> auto operator()(const std::vector<T>& value) const
    {
        using Item = std::decay_t<decltype((*this)(std::declval<const T&>()))>;
        std::vector<Item> result;
        result.reserve(value.size());
        for (size_t i = 0; i < value.size(); ++i) {
            if constexpr (std::is_same<T, bool>::value) result.emplace_back((*this)(bool(value[i])));
            else result.emplace_back((*this)(value[i]));
        }
        return result;
    }
    template<class T> auto operator()(const c10::optional<T>& value) const
    {
        using Item = std::decay_t<decltype((*this)(std::declval<const T&>()))>;
        if (!value.has_value()) return c10::optional<Item>{};
        return c10::optional<Item>{(*this)(value.value())};
    }
    template<class T> auto operator()(const c10::OptionalArrayRef<T>& value) const
    {
        using Item = std::decay_t<decltype((*this)(std::declval<const at::ArrayRef<T>&>()))>;
        if (!value.has_value()) return c10::optional<Item>{};
        return c10::optional<Item>{(*this)(value.value())};
    }
};

struct OwnedConvertPolicy {
    // V1 represents an empty TensorList by null; V2 constructs an empty list.
    bool v1 = true;
    bool* cleanup_failed = nullptr;
    std::exception_ptr* cleanup_failure = nullptr;
    TORCH_NPU_API aclTensor* Convert(const OwnedTensor&) const;
    TORCH_NPU_API aclScalar* Convert(const at::Scalar&) const;
    TORCH_NPU_API aclIntArray* Convert(const std::vector<int64_t>&) const;
    TORCH_NPU_API aclBoolArray* Convert(const std::vector<bool>&) const;
    TORCH_NPU_API aclTensorList* Convert(const std::vector<OwnedTensor>&) const;
    TORCH_NPU_API aclScalarList* Convert(const std::vector<at::Scalar>&) const;
    char* Convert(const std::string& value) const noexcept { return const_cast<char*>(value.c_str()); }
    template<class T> auto Convert(const T& value) const -> std::conditional_t<
        std::is_arithmetic<T>::value || std::is_enum<T>::value || std::is_same<T, std::nullptr_t>::value,
        std::decay_t<decltype(::ConvertTypeV2(value))>, void*>
    {
        if constexpr (std::is_arithmetic<T>::value || std::is_enum<T>::value || std::is_same<T, std::nullptr_t>::value)
            return ::ConvertTypeV2(value);
        else throw std::invalid_argument("owned OpAPI argument has no SDK conversion adapter");
    }
    template<size_t N> aclBoolArray* Convert(const std::array<bool, N>& value) const
    {
        auto* result = ::ConvertTypeV2(value);
        if (!result) throw std::bad_alloc();
        return result;
    }
    template<class T, class Converted = std::decay_t<decltype(
        std::declval<const OwnedConvertPolicy&>().Convert(std::declval<const T&>()))>>
    auto Convert(const c10::optional<T>& value) const ->
        std::conditional_t<std::is_pointer<Converted>::value, Converted, void*>
    {
        if constexpr (std::is_pointer<Converted>::value) return value.has_value() ? Convert(value.value()) : nullptr;
        else throw std::invalid_argument("owned OpAPI optional scalar needs an explicit SDK schema adapter");
    }
    TORCH_NPU_API void Release(aclTensor* value) const;
    TORCH_NPU_API void Release(aclScalar* value) const;
    TORCH_NPU_API void Release(aclIntArray* value) const;
    TORCH_NPU_API void Release(aclBoolArray* value) const;
    TORCH_NPU_API void Release(aclTensorList* value) const;
    TORCH_NPU_API void Release(aclScalarList* value) const;
    template<class T> void Release(T value) const { ::Release(value); }
private:
    template<class Handle, class Destroy>
    void ReleaseChecked(Handle value, Destroy destroy) const
    {
        if (!value) return;
        try { TORCH_CHECK(destroy && destroy(value) == 0, "owned OpAPI descriptor cleanup failed"); }
        catch (...) {
            if (cleanup_failed) *cleanup_failed = true;
            if (cleanup_failure && !*cleanup_failure) *cleanup_failure = std::current_exception();
            PoisonOwnedSdkCleanup();
            throw;
        }
    }
};
} // namespace hdb_opapi
