#pragma once
#include <ATen/Tensor.h>
#include <c10/core/TensorImpl.h>
#include <atomic>
#include <optional>
#include <typeinfo>

namespace c10_npu::host_regions {
struct StorageOwnerPacketSnapshot {
    uint64_t requests, hits, misses, source_rejects, factory_rejects;
    uint64_t created, live, cache_entries_live, cache_clears;
};
namespace storage_owner_detail {
struct Counters {
    std::atomic<uint64_t> requests{0}, hits{0}, misses{0}, source_rejects{0}, factory_rejects{0};
    std::atomic<uint64_t> created{0}, live{0}, cache_entries_live{0}, cache_clears{0};
};
inline Counters counters;

// Never exposed as a Python tensor or passed to an operator. The base is an
// VIEW-tag TensorImpl with an empty keyset owns only the retained Storage.
// No source dtype/shape/stride/offset/version/PyObject/autograd state is copied.
class StorageOwnerImpl final : public c10::TensorImpl {
public:
    explicit StorageOwnerImpl(c10::Storage storage)
        : TensorImpl(ImplType::VIEW, checked_storage(std::move(storage)),
                     c10::DispatchKeySet{}, caffe2::TypeMeta{})
    {
        counters.created.fetch_add(1, std::memory_order_relaxed);
        counters.live.fetch_add(1, std::memory_order_relaxed);
    }
    ~StorageOwnerImpl() override { release_owned(); }
    void release_resources() override
    {
        release_owned();
    }
    bool certified() const noexcept { return live_ && !pyobj_slot()->load_pyobj(); }
    c10::StorageImpl* storage_identity() const noexcept { return storage().unsafeGetStorageImpl(); }
private:
    static c10::Storage checked_storage(c10::Storage storage)
    {
        TORCH_CHECK(storage.unsafeGetStorageImpl(), "private storage owner requires live Storage");
        return storage;
    }
    void release_owned() noexcept
    {
        if (!live_) return;
        live_ = false; // Reentrant release cannot decrement twice.
        TensorImpl::release_resources();
        // Count actual resource retirement, not removal of a cache entry.
        counters.live.fetch_sub(1, std::memory_order_release);
    }
    bool live_ = true;
};
inline at::Tensor Make(c10::Storage storage)
{
    c10::intrusive_ptr<c10::TensorImpl> impl = c10::make_intrusive<StorageOwnerImpl>(std::move(storage));
    return at::Tensor(std::move(impl));
}
} // namespace storage_owner_detail

inline bool IsCapturedStorageOwner(const at::Tensor& value) noexcept
{
    if (!value.defined()) return false;
    auto* impl = value.unsafeGetTensorImpl();
    return typeid(*impl) == typeid(storage_owner_detail::StorageOwnerImpl) &&
           static_cast<storage_owner_detail::StorageOwnerImpl*>(impl)->certified();
}
inline StorageOwnerPacketSnapshot GetStorageOwnerPacketSnapshot() noexcept
{
    auto& c = storage_owner_detail::counters;
    return {c.requests.load(std::memory_order_relaxed), c.hits.load(std::memory_order_relaxed),
        c.misses.load(std::memory_order_relaxed), c.source_rejects.load(std::memory_order_relaxed),
        c.factory_rejects.load(std::memory_order_relaxed), c.created.load(std::memory_order_relaxed),
        c.live.load(std::memory_order_acquire), c.cache_entries_live.load(std::memory_order_acquire),
        c.cache_clears.load(std::memory_order_relaxed)};
}
} // namespace c10_npu::host_regions
