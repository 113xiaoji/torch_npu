// Private smoke-test module. The three configuration methods are the unchanged
// production bindings; probe methods exercise actual CPU ATen bodies only.
#include "torch_npu/csrc/npu/EagerParallelPython.h"
#include "torch_npu/csrc/core/npu/eager_parallel/EntryGuard.h"
#include <ATen/Functions.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_variable.h>
#include <stdexcept>

PyObject* invoke_tensors(PyObject*, PyObject* args)
{
    HANDLE_TH_ERRORS
    PyObject *left = nullptr, *right = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &left, &right)) return nullptr;
    TORCH_CHECK(THPVariable_Check(left) && THPVariable_Check(right), "probe requires tensors");
    const auto& a = THPVariable_Unpack(left);
    const auto& b = THPVariable_Unpack(right);
    auto output = c10_npu::eager_parallel::Invoke("probe::outer", "native_probe", [&] {
        auto product = c10_npu::eager_parallel::Invoke("aten::mm", "native_probe",
            [&] { return at::mm(a, b); }, a, b);
        const std::vector<at::Tensor> inputs{product, product};
        return c10_npu::eager_parallel::Invoke("aten::cat", "native_probe",
            [&] { return at::cat(inputs, 0); }, inputs);
    }, a, b);
    return THPVariable_Wrap(output);
    END_HANDLE_TH_ERRORS
}

PyObject* invoke_failure(PyObject*, PyObject*)
{
    HANDLE_TH_ERRORS
    c10_npu::eager_parallel::Invoke("probe::body_failure", "native_probe", []() -> void {
        throw std::runtime_error("original operator body failure");
    });
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

PyMethodDef methods[] = {
    {"_npu_eager_parallel_configure", THNPModule_eagerParallelConfigure, METH_VARARGS, nullptr},
    {"_npu_eager_parallel_snapshot", THNPModule_eagerParallelSnapshot, METH_NOARGS, nullptr},
    {"_npu_eager_parallel_drain_close", THNPModule_eagerParallelDrainClose, METH_NOARGS, nullptr},
    {"invoke_tensors", invoke_tensors, METH_VARARGS, nullptr},
    {"invoke_failure", invoke_failure, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};
PyModuleDef_Slot slots[] = {
#ifdef Py_GIL_DISABLED
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {0, nullptr},
};
PyModuleDef definition = {
    PyModuleDef_HEAD_INIT, "_r26_native_entry_probe", nullptr, 0,
    methods, slots, nullptr, nullptr, nullptr,
};
PyMODINIT_FUNC PyInit__r26_native_entry_probe()
{
    return PyModuleDef_Init(&definition);
}
