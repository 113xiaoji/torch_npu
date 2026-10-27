#pragma once

#include <typeinfo>
#include <c10/core/impl/TorchDispatchModeTLS.h>
#include "torch_npu/csrc/core/NPUTensorImpl.h"
#include "torch_npu/csrc/core/npu/HostRegionCapture.h"
#include "torch_npu/csrc/core/npu/HostRegionStorageOwner.h"

namespace c10_npu::host_regions {
// Recheck the original source before EVERY lookup, even a cache hit. The
// original Tensor PyObject/autograd metadata are deliberately not copied.
inline c10::StorageImpl* CapturedStorageSourceKey(const at::Tensor& value, bool plain_acl) noexcept
{
    if (!plain_acl || !value.defined()) return nullptr;
    try {
        auto* impl = value.unsafeGetTensorImpl();
        if ((typeid(*impl) != typeid(torch_npu::NPUTensorImpl) && typeid(*impl) != typeid(c10::TensorImpl)) ||
            impl->device_type() != c10::DeviceType::PrivateUse1 || impl->key_set().has_any(c10::python_ks) ||
            impl->get_backend_meta() || impl->fake_tensor_mode() || impl->has_symbolic_sizes_strides() ||
            !impl->has_storage() || c10::impl::TorchDispatchModeTLS::stack_len() != 0) return nullptr;
        return impl->storage().unsafeGetStorageImpl();
    } catch (...) { return nullptr; }
}
// Only added buffer owners use this shell. Exported hostMem and ACL ownership
// are unchanged; the shell shares storage and copies metadata with an empty PyObjectSlot.
inline at::Tensor RetainBufferOwnerFallback(const at::Tensor& value, bool plain_acl)
{
    if (!plain_acl || !value.defined()) return value;
    auto* impl = value.unsafeGetTensorImpl();
    if (impl->key_set().has_any(c10::python_ks) || impl->get_backend_meta() ||
        impl->fake_tensor_mode() || impl->has_symbolic_sizes_strides()) return value;
    const bool generic_npu = typeid(*impl) == typeid(c10::TensorImpl) &&
        impl->device_type() == c10::DeviceType::PrivateUse1 &&
        c10::impl::TorchDispatchModeTLS::stack_len() == 0;
    if (typeid(*impl) != typeid(torch_npu::NPUTensorImpl) && !generic_npu) return value;
    return at::Tensor(impl->shallow_copy_and_detach(impl->version_counter(), false));
}
inline at::Tensor RetainBufferOwner(const at::Tensor& value, bool plain_acl)
{
    if (IsCapturing()) {
        auto owner = TryRetainCapturedStorageOwner(value, plain_acl);
        if (owner.defined()) return owner;
    }
    // This is the exact r12 shell/copy fallback, including custom handlers.
    return RetainBufferOwnerFallback(value, plain_acl);
}
} // namespace c10_npu::host_regions
