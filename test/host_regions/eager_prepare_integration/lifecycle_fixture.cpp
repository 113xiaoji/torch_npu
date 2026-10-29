#include "torch_npu/csrc/core/npu/EagerPrepareLifecycle.h"
#include <condition_variable>
#include <stdexcept>
using namespace c10_npu::eager_prepare;
namespace {
std::mutex lifecycle,control;
std::condition_variable changed;
bool released=false,fail_close=false;
std::atomic<int> calls{0},configured{0};
ColdCloseOnce once;
#ifdef R24_REPRODUCE_OLD_LOCK_ORDER
class Guard {
    std::unique_lock<std::mutex> lock_; InterpreterDetach detach_;
public: explicit Guard(std::mutex& m) : lock_(m) {} // Old inversion: lock BEFORE detach.
};
#else
using Guard=LifecycleGuard;
#endif
std::pair<int,int> native_snapshot() {
    InterpreterDetach detached;
    std::lock_guard<std::mutex> lock(control);
    return {calls.load(),configured.load()};
}
void native_close() {
    Guard guard(lifecycle);
    once.Run([] {
        ++calls;
        { std::unique_lock<std::mutex> lock(control); changed.wait(lock,[] { return released; }); }
        if (fail_close) throw std::runtime_error("unique-cold-ack-failure");
    });
    // Production Configure/DrainClose return Snapshot while lifecycle is held.
    // Its separate short state lock and nested detach must not self-lock or
    // reattach the interpreter before releasing the outer lifecycle guard.
    (void)native_snapshot();
    if (PyThreadState_GetUnchecked()!=nullptr) throw std::runtime_error("premature interpreter reattachment");
}
PyObject* reset(PyObject*,PyObject* args) {
    int fail=0; if (!PyArg_ParseTuple(args,"p",&fail)) return nullptr;
    once.Reset(); released=false; fail_close=fail!=0; calls=0; configured=0; Py_RETURN_NONE;
}
PyObject* release(PyObject*,PyObject*) {
    { std::lock_guard<std::mutex> lock(control); released=true; changed.notify_all(); } Py_RETURN_NONE;
}
PyObject* close_call(PyObject*,PyObject*) {
    try { native_close(); } catch (const std::exception& e) { PyErr_SetString(PyExc_RuntimeError,e.what()); return nullptr; }
    Py_RETURN_NONE;
}
PyObject* configure_call(PyObject*,PyObject*) {
    { Guard guard(lifecycle); ++configured; (void)native_snapshot(); } Py_RETURN_NONE;
}
PyObject* finalize_call(PyObject*,PyObject*) {
    try { native_close(); }
    catch (...) {
        std::fprintf(stderr,"SDK_CALLBACK_CALLS=%d\n",calls.load()); std::fflush(stderr);
        FailStopFinalize(control,changed);
    }
    std::fprintf(stderr,"ORIGINAL_FINALIZE_REACHED\n"); std::fflush(stderr);
    Py_RETURN_NONE;
}
PyObject* snapshot(PyObject*,PyObject*) { const auto value=native_snapshot(); return Py_BuildValue("ii",value.first,value.second); }
PyObject* nested(PyObject*,PyObject*) {
    bool okay=false;
    { InterpreterDetach first;
      okay=PyThreadState_GetUnchecked()==nullptr;
      { InterpreterDetach second; okay=okay && PyThreadState_GetUnchecked()==nullptr; }
      okay=okay && PyThreadState_GetUnchecked()==nullptr; }
    return PyBool_FromLong(okay && PyThreadState_GetUnchecked()!=nullptr);
}
PyMethodDef methods[]={{"reset",reset,METH_VARARGS,nullptr},{"release",release,METH_NOARGS,nullptr},
    {"close",close_call,METH_NOARGS,nullptr},{"configure",configure_call,METH_NOARGS,nullptr},
    {"snapshot",snapshot,METH_NOARGS,nullptr},{"nested",nested,METH_NOARGS,nullptr},
    {"finalize",finalize_call,METH_NOARGS,nullptr},{nullptr,nullptr,0,nullptr}};
PyModuleDef definition={PyModuleDef_HEAD_INIT,"_r24_lifecycle_fixture",nullptr,-1,methods,nullptr,nullptr,nullptr,nullptr};
}
PyMODINIT_FUNC PyInit__r24_lifecycle_fixture() { return PyModule_Create(&definition); }
