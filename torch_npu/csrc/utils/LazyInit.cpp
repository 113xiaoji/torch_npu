#include <atomic>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/Exceptions.h>
#include "torch_npu/csrc/utils/LazyInit.h"

namespace torch_npu {
namespace utils {


static std::atomic<bool> npu_run_yet{false};
static thread_local bool npu_is_initializing = false;

void npu_lazy_init()
{
    if (npu_run_yet.load(std::memory_order_acquire) || npu_is_initializing) {
        return;
    }

    pybind11::gil_scoped_acquire g;
    if (npu_run_yet.load(std::memory_order_acquire) || npu_is_initializing) {
        return;
    }

    npu_is_initializing = true;
    try {
        auto module = THPObjectPtr(PyImport_ImportModule("torch_npu.npu"));
        if (!module) {
            throw python_error();
        }
        auto res = THPObjectPtr(PyObject_CallMethod(module.get(), "_lazy_init", ""));
        if (!res) {
            throw python_error();
        }
        npu_run_yet.store(true, std::memory_order_release);
    } catch (...) {
        npu_is_initializing = false;
        throw;
    }
    npu_is_initializing = false;
}

void npu_set_run_yet_variable_to_false()
{
    npu_is_initializing = false;
    npu_run_yet.store(false, std::memory_order_release);
}

}
}
