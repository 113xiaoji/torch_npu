// Links the production EntryGuard.cpp. Counts allocations around its first
// disabled invocation before any Configure/Snapshot call initializes diagnostics.
#include "torch_npu/csrc/core/npu/eager_parallel/EntryGuard.h"
#include <atomic>
#include <cstdlib>
#include <new>

static std::atomic<std::size_t> allocations{0};
void* operator new(std::size_t size)
{
    allocations.fetch_add(1, std::memory_order_relaxed);
    if (void* pointer = std::malloc(size ? size : 1)) return pointer;
    throw std::bad_alloc();
}
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }

int main()
{
    allocations.store(0, std::memory_order_relaxed);
    const int value = c10_npu::eager_parallel::Invoke("probe::off", "native_probe", [] { return 41; }, 0);
    c10_npu::eager_parallel::Boundary("probe::off_boundary");
    return value == 41 && allocations.load(std::memory_order_relaxed) == 0 ? 0 : 1;
}
