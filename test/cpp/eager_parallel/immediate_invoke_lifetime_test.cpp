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
bool caller_path = false;
unsigned caller_failures = 0, caller_visited_outputs = 0;
bool caller_complete_throws = true;
at::Tensor output() {
    ++live;
    return at::from_blob(new float{1}, {1}, [](void* value) {
        --live; delete static_cast<float*>(value);
    }, at::TensorOptions().dtype(at::kFloat));
}
}
namespace c10_npu::eager_parallel {
bool ImmediateCallerGuard::begin(const char*,const at::Tensor&,const at::Tensor&,const at::Tensor&,bool) { return caller_path; }
ImmediateCallerGuard::~ImmediateCallerGuard() noexcept = default;
void ImmediateCallerGuard::complete() { if(caller_complete_throws) throw std::runtime_error("event record fault"); finished_=true; }
void ImmediateCallerGuard::add_failed_output(const at::Tensor&) noexcept { ++caller_visited_outputs; }
void ImmediateCallerGuard::fail(std::exception_ptr) noexcept { if(!finished_) {recovered_alive=live==4; ++caller_failures; finished_=true;} }
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
        recovered_alive=false;
        caller_path=true;
        auto input=at::zeros({1});
        const std::optional<at::Tensor> sample;
        try {
            c10_npu::eager_parallel::InvokeImmediate("aten::_embedding_bag", "test", [] {
                return std::make_tuple(output(), output(), output(), output());
            },input,input,input,false,int64_t(0),false,sample,false,int64_t(-1));
        } catch (const std::runtime_error& caller_error) {
            if (std::string(caller_error.what())!="event record fault" || live!=0 || !recovered_alive || caller_failures!=1 || caller_visited_outputs!=4) return 3;
            caller_visited_outputs=0; caller_complete_throws=false;
            {
                auto result=c10_npu::eager_parallel::InvokeImmediate("aten::_embedding_bag", "test", [] {
                    return std::make_tuple(output(), output(), output(), output());
                },input,input,input,false,int64_t(0),false,sample,false,int64_t(-1));
                (void)result;
                if(live!=4 || caller_visited_outputs!=0 || caller_failures!=1) return 5;
            }
            if(live!=0) return 6;
            std::puts("PASS full/selective outputs alive during failure; zero successful caller output visits");
            return 0;
        }
        return 4;
    }
    return 2;
}
