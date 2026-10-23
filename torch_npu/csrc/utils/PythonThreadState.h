#pragma once

#ifndef BUILD_LIBTORCH
#include <Python.h>

namespace torch_npu {
namespace utils {

inline bool has_attached_python_thread_state() noexcept
{
#if PY_VERSION_HEX >= 0x030D0000
    return PyThreadState_GetUnchecked() != nullptr;
#else
    return _PyThreadState_UncheckedGet() != nullptr;
#endif
}

class PythonThreadStateDetachGuard final {
public:
    explicit PythonThreadStateDetachGuard(bool detach_requested = true) noexcept
        : state_(detach_requested && has_attached_python_thread_state() ? PyEval_SaveThread() : nullptr)
    {}

    ~PythonThreadStateDetachGuard()
    {
        if (state_ != nullptr) {
            PyEval_RestoreThread(state_);
        }
    }

    PythonThreadStateDetachGuard(const PythonThreadStateDetachGuard &) = delete;
    PythonThreadStateDetachGuard &operator=(const PythonThreadStateDetachGuard &) = delete;
    PythonThreadStateDetachGuard(PythonThreadStateDetachGuard &&) = delete;
    PythonThreadStateDetachGuard &operator=(PythonThreadStateDetachGuard &&) = delete;

private:
    PyThreadState *state_ = nullptr;
};

}  // namespace utils
}  // namespace torch_npu
#endif
