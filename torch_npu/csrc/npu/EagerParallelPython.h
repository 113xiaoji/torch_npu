#pragma once
#include <Python.h>
PyObject* THNPModule_eagerParallelConfigure(PyObject*,PyObject*);
PyObject* THNPModule_eagerParallelSnapshot(PyObject*,PyObject*);
PyObject* THNPModule_eagerParallelDrainClose(PyObject*,PyObject*);
