#include "torch_npu/csrc/npu/EagerParallelPython.h"
#include "torch_npu/csrc/core/npu/eager_parallel/Runtime.h"
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/python_strings.h>

PyObject* THNPModule_eagerParallelConfigure(PyObject*,PyObject* arguments)
{
    HANDLE_TH_ERRORS
    PyObject *enabled=nullptr,*mode=nullptr;
    unsigned long long streams=0,capacity=0,bytes=0;
    if(!PyArg_ParseTuple(arguments,"OOKKK",&enabled,&mode,&streams,&capacity,&bytes)) return nullptr;
    TORCH_CHECK(PyBool_Check(enabled) && PyUnicode_CheckExact(mode),"native eager-parallel requires bool and plain str");
    const char* name=PyUnicode_AsUTF8(mode);
    if(!name) return nullptr;
    return THPUtils_packString(c10_npu::eager_parallel::Configure(enabled==Py_True,name,streams,capacity,bytes));
    END_HANDLE_TH_ERRORS
}
PyObject* THNPModule_eagerParallelSnapshot(PyObject*,PyObject*)
{
    HANDLE_TH_ERRORS
    return THPUtils_packString(c10_npu::eager_parallel::Snapshot());
    END_HANDLE_TH_ERRORS
}
PyObject* THNPModule_eagerParallelDrainClose(PyObject*,PyObject*)
{
    HANDLE_TH_ERRORS
    return THPUtils_packString(c10_npu::eager_parallel::DrainClose());
    END_HANDLE_TH_ERRORS
}
