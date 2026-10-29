#include "torch_npu/csrc/npu/EagerPreparePython.h"
#include "torch_npu/csrc/core/npu/EagerPrepare.h"
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/python_strings.h>

PyObject* THNPModule_eagerPrepareConfigure(PyObject*,PyObject* arguments)
{
    HANDLE_TH_ERRORS
    PyObject *enabled=nullptr,*operators=nullptr,*trace=nullptr;
    unsigned long long workers=0,capacity=0,bytes=0,limit=0;
    if (!PyArg_ParseTuple(arguments,"OKKKOOK",&enabled,&workers,&capacity,&bytes,&operators,&trace,&limit)) return nullptr;
    TORCH_CHECK(PyBool_Check(enabled) && PyTuple_CheckExact(operators) && PyBool_Check(trace),"eager Prepare requires bools and a plain operator tuple");
    c10_npu::eager_prepare::Configuration config;
    config.enabled=enabled==Py_True; config.workers=workers; config.capacity=capacity; config.byte_limit=bytes;
    config.diagnostic_trace=trace==Py_True; config.trace_limit=limit;
    for (Py_ssize_t i=0;i<PyTuple_GET_SIZE(operators);++i) {
        auto* value=PyTuple_GET_ITEM(operators,i); TORCH_CHECK(PyUnicode_CheckExact(value),"eager Prepare operator must be str");
        const char* name=PyUnicode_AsUTF8(value); if (!name) return nullptr;
        config.allowed_ops.emplace_back(name);
    }
    return THPUtils_packString(c10_npu::eager_prepare::Configure(std::move(config)));
    END_HANDLE_TH_ERRORS
}
PyObject* THNPModule_eagerPrepareSnapshot(PyObject*,PyObject*)
{
    HANDLE_TH_ERRORS
    return THPUtils_packString(c10_npu::eager_prepare::Snapshot());
    END_HANDLE_TH_ERRORS
}
PyObject* THNPModule_eagerPrepareDrainClose(PyObject*,PyObject*)
{
    HANDLE_TH_ERRORS
    return THPUtils_packString(c10_npu::eager_prepare::DrainClose());
    END_HANDLE_TH_ERRORS
}
