// Exercise production InvokeImmediate exception ordering with a faulting guard.
// The fake guard intentionally retains nothing: all four original outputs must
// still be alive when fail() begins recovery after complete() throws.
#include "torch_npu/csrc/core/npu/eager_parallel/ImmediateInvoke.h"
#include <ATen/Functions.h>
#include <array>
#include <cstdio>
#include <stdexcept>

namespace {
int live = 0;
bool recovered_alive = false;
at::Tensor output() {
    ++live;
    return at::from_blob(new float{1}, {1}, [](void* value) {
        --live; delete static_cast<float*>(value);
    }, at::TensorOptions().dtype(at::kFloat));
}
}
namespace c10_npu::eager_parallel {
struct ImmediateGuardImpl { bool failed = false; };
ImmediateGuard::ImmediateGuard(const char*, const char*) : impl_(new ImmediateGuardImpl) {}
ImmediateGuard::~ImmediateGuard() noexcept = default;
void ImmediateGuard::disqualify() noexcept {}
void ImmediateGuard::add_input(const at::Tensor&) {}
void ImmediateGuard::begin() {}
void ImmediateGuard::add_output(const at::Tensor&) {}
void ImmediateGuard::complete() { throw std::runtime_error("event record fault"); }
void ImmediateGuard::fail(std::exception_ptr) noexcept {
    if (!impl_->failed) { recovered_alive = live == 4; impl_->failed = true; }
}
}
int main() {
    try {
        c10_npu::eager_parallel::InvokeImmediate("test::four_outputs", "test", [] {
            return std::make_tuple(output(), output(), output(), output());
        });
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()) != "event record fault" || live != 0 || !recovered_alive) return 1;
        std::puts("PASS original four outputs alive during failure recovery");
        return 0;
    }
    return 2;
}
