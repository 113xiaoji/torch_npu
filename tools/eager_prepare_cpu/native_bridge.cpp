#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "torch_npu/csrc/core/npu/EagerPrepareQueueCore.h"
#include "build_info.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <deque>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#endif

namespace {
using Core = c10_npu::queue::EagerPrepareQueueCore;
uint64_t NowNs()
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
uint64_t Tid()
{
#ifdef _WIN32
    return GetCurrentThreadId();
#else
    return static_cast<uint64_t>(syscall(SYS_gettid));
#endif
}
uint64_t ThreadCpuNs()
{
#ifdef _WIN32
    FILETIME created, exited, kernel, user;
    if (!GetThreadTimes(GetCurrentThread(), &created, &exited, &kernel, &user))
        throw std::runtime_error("GetThreadTimes failed");
    const uint64_t k = (static_cast<uint64_t>(kernel.dwHighDateTime) << 32) | kernel.dwLowDateTime;
    const uint64_t u = (static_cast<uint64_t>(user.dwHighDateTime) << 32) | user.dwLowDateTime;
    return (k + u) * 100;
#else
    timespec value{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value)) throw std::runtime_error("thread CPU clock failed");
    return static_cast<uint64_t>(value.tv_sec) * 1000000000ULL + static_cast<uint64_t>(value.tv_nsec);
#endif
}
uint64_t ThreadCycles()
{
#ifdef _WIN32
    ULONG64 cycles = 0;
    if (!QueryThreadCycleTime(GetCurrentThread(), &cycles)) throw std::runtime_error("QueryThreadCycleTime failed");
    return cycles;
#else
    return 0; // Unsupported here; exposed as None, never converted into ns.
#endif
}

// Exactly the same function is called by inline and worker paths. It has no
// Python/Torch/SDK references, and its output depends only on copied metadata
// and the common iteration count. This is synthetic CPU work, never CANN.
uint64_t RunPrepareWork(const std::vector<int64_t>& metadata, uint64_t iterations)
{
    uint64_t value = 1469598103934665603ULL;
    for (const int64_t item : metadata) { value ^= static_cast<uint64_t>(item); value *= 1099511628211ULL; }
    double floating = static_cast<double>(value & 0xffffU) / 65536.0;
    for (uint64_t i = 0; i < iterations; ++i) {
        value ^= value >> 12; value ^= value << 25; value ^= value >> 27;
        value *= 2685821657736338717ULL;
        floating = floating * 1.00000011920928955078125 + static_cast<double>(value & 255U) * 0.000003814697265625;
        if (floating > 1048576.0) floating -= 1048576.0;
    }
    uint64_t bits = 0;
    std::memcpy(&bits, &floating, sizeof(bits));
    return value ^ bits;
}

struct Record {
    int64_t seq;
    uint64_t tid, start_ns, end_ns, thread_cpu_ns, thread_cycles, checksum;
};
struct Metrics {
    std::mutex mu;
    std::condition_variable cv;
    const bool trace;
    const size_t trace_limit;
    const int64_t gate_seq, fail_seq;
    bool gate_entered = false, gate_released = false, aborted = false;
    std::string reason;
    uint64_t prepare_calls = 0, checksum_xor = 0, execute_markers = 0;
    uint64_t cpu_completed = 0, cancelled = 0, trace_dropped = 0;
    std::set<uint64_t> worker_tids;
    std::vector<Record> records;
    Metrics(bool enabled, size_t limit, int64_t gate, int64_t fail)
        : trace(enabled), trace_limit(limit), gate_seq(gate), fail_seq(fail)
    { if (trace) records.reserve(trace_limit); }
};

struct CpuJob final : Core::Job {
    const int64_t seq;
    std::vector<int64_t> metadata;
    const uint64_t iterations;
    const std::shared_ptr<Metrics> metrics;
    std::atomic<bool> prepared{false};
    CpuJob(int64_t id, std::vector<int64_t> values, uint64_t work, std::shared_ptr<Metrics> counters)
        : seq(id), metadata(std::move(values)), iterations(work), metrics(std::move(counters)) {}
    void Prepare() override
    {
        {
            std::unique_lock<std::mutex> lock(metrics->mu);
            if (seq == metrics->gate_seq) {
                metrics->gate_entered = true; metrics->cv.notify_all();
                metrics->cv.wait(lock, [&] { return metrics->gate_released || metrics->aborted; });
            }
            if (metrics->aborted) throw std::runtime_error(metrics->reason);
        }
        if (seq == metrics->fail_seq) throw std::runtime_error("injected Prepare failure");
        const uint64_t tid = Tid();
        const uint64_t start = metrics->trace ? NowNs() : 0;
        const uint64_t cpu_start = metrics->trace ? ThreadCpuNs() : 0;
        const uint64_t cycle_start = metrics->trace ? ThreadCycles() : 0;
        const uint64_t checksum = RunPrepareWork(metadata, iterations);
        const uint64_t cpu = metrics->trace ? ThreadCpuNs() - cpu_start : 0;
        const uint64_t cycles = metrics->trace ? ThreadCycles() - cycle_start : 0;
        const uint64_t end = metrics->trace ? NowNs() : 0;
        {
            std::lock_guard<std::mutex> lock(metrics->mu);
            ++metrics->prepare_calls; metrics->checksum_xor ^= checksum; metrics->worker_tids.insert(tid);
            if (metrics->trace) {
                if (metrics->records.size() < metrics->trace_limit) metrics->records.push_back({seq, tid, start, end, cpu, cycles, checksum});
                else ++metrics->trace_dropped;
            }
            prepared.store(true, std::memory_order_release); metrics->cv.notify_all();
        }
    }
    void Execute() override
    {
        // Only a native marker. The caller subsequently executes the real CPU
        // Torch operator in Python and acknowledges it through Session.retire.
        std::lock_guard<std::mutex> lock(metrics->mu); ++metrics->execute_markers;
    }
    void Cancel() noexcept override { std::vector<int64_t>().swap(metadata); }
    size_t RetainedBytes() const noexcept override { return metadata.capacity() * sizeof(int64_t); }
};

struct Snapshot {
    Core::SnapshotData core;
    uint64_t prepare_calls, checksum_xor, execute_markers, cpu_completed, cancelled, trace_dropped;
    bool closed, aborted, stopped, trace;
    std::string reason;
    std::vector<uint64_t> worker_tids;
};

class Session final {
    struct Item { int64_t seq; Core::TicketPtr ticket; std::shared_ptr<CpuJob> job; };
    const std::shared_ptr<Metrics> metrics_;
    Core core_;
    const uint64_t iterations_;
    std::mutex submit_mu_, fifo_mu_, lifecycle_mu_;
    std::condition_variable fifo_cv_;
    std::deque<std::shared_ptr<Item>> fifo_;
    std::unordered_map<int64_t, std::shared_ptr<Item>> owners_;
    int64_t last_submitted_ = -1, last_retired_ = -1, inflight_seq_ = -1;
    uint64_t consumer_tid_ = 0;
    bool closed_ = false, aborted_ = false, stopped_ = false, waiting_next_ = false;
    std::string reason_;

    void SignalAbort(const std::string& reason)
    {
        {
            std::lock_guard<std::mutex> lock(metrics_->mu);
            if (!metrics_->aborted) metrics_->reason = reason;
            metrics_->aborted = true; metrics_->gate_released = true; metrics_->cv.notify_all();
        }
        {
            std::lock_guard<std::mutex> lock(fifo_mu_);
            if (!aborted_) reason_ = reason;
            aborted_ = true; closed_ = true; fifo_cv_.notify_all();
        }
        core_.Close();
    }
    std::string AbortReason()
    {
        std::lock_guard<std::mutex> lock(fifo_mu_);
        return aborted_ ? reason_ : "CPU simulation session closed";
    }
public:
    Session(size_t workers, size_t capacity, size_t bytes, uint64_t work, bool trace,
            size_t trace_limit, int64_t gate, int64_t fail)
        : metrics_(std::make_shared<Metrics>(trace, trace_limit, gate, fail)),
          core_({capacity, bytes, workers}), iterations_(work) {}
    ~Session() { try { Abort("CPU simulation session destroyed"); } catch (...) {} }

    void Submit(int64_t seq, std::vector<int64_t> metadata)
    {
        std::lock_guard<std::mutex> submission(submit_mu_);
        {
            std::lock_guard<std::mutex> lock(fifo_mu_);
            if (aborted_ || closed_) throw std::runtime_error(aborted_ ? reason_ : "CPU simulation session closed");
            if (seq < 0 || seq <= last_submitted_) throw std::invalid_argument("seq must be nonnegative and strictly increasing per Session");
        }
        auto job = std::make_shared<CpuJob>(seq, std::move(metadata), iterations_, metrics_);
        Core::TicketPtr ticket;
        try {
            ticket = core_.Submit(job, [&](const Core::TicketPtr& accepted) {
                std::lock_guard<std::mutex> lock(fifo_mu_);
                if (closed_ || aborted_) return Core::PublishReceipt{Core::PublishState::NotAccepted,
                    std::make_exception_ptr(std::runtime_error(aborted_ ? reason_ : "CPU simulation session closed"))};
                try {
                    auto item = std::make_shared<Item>(Item{seq, accepted, job});
                    owners_.emplace(seq, item);
                    try { fifo_.push_back(item); } catch (...) { owners_.erase(seq); throw; }
                    fifo_cv_.notify_all();
                    return Core::PublishReceipt{Core::PublishState::Committed, nullptr};
                } catch (...) { return Core::PublishReceipt{Core::PublishState::NotAccepted, std::current_exception()}; }
            });
            if (core_.Publication(ticket) != Core::PublishState::Committed) throw std::runtime_error(AbortReason());
        } catch (const std::exception& error) {
            std::string reason = error.what();
            try { core_.RethrowIfFailed(); } catch (const std::exception& first) { reason = first.what(); }
            {
                std::lock_guard<std::mutex> lock(fifo_mu_);
                if (aborted_) reason = reason_;
            }
            SignalAbort(reason); throw std::runtime_error(reason);
        }
        { std::lock_guard<std::mutex> lock(fifo_mu_); last_submitted_ = seq; }
    }

    std::optional<int64_t> Next()
    {
        std::shared_ptr<Item> item;
        {
            std::unique_lock<std::mutex> lock(fifo_mu_);
            if (aborted_) throw std::runtime_error(reason_);
            if (inflight_seq_ >= 0) throw std::runtime_error("retire the previous CPU operation before calling next");
            if (waiting_next_) throw std::runtime_error("only one CPU consumer may wait in next");
            waiting_next_ = true;
            fifo_cv_.wait(lock, [&] { return aborted_ || closed_ || !fifo_.empty(); });
            waiting_next_ = false;
            if (aborted_) throw std::runtime_error(reason_);
            if (fifo_.empty()) return std::nullopt;
            item = fifo_.front(); fifo_.pop_front(); inflight_seq_ = item->seq; consumer_tid_ = Tid();
        }
        try { core_.Consume(item->ticket); }
        catch (const std::exception& error) { SignalAbort(error.what()); throw; }
        return item->seq;
    }

    void Retire(int64_t seq)
    {
        std::shared_ptr<Item> item;
        {
            std::lock_guard<std::mutex> lock(fifo_mu_);
            if (aborted_) throw std::runtime_error(reason_);
            if (seq >= 0 && seq <= last_retired_) return;
            auto found = owners_.find(seq);
            if (found == owners_.end() || inflight_seq_ != seq) throw std::invalid_argument("retire requires the ticket returned by next");
            if (consumer_tid_ != Tid()) throw std::runtime_error("retire must run on the CPU consumer thread");
            item = found->second;
        }
        core_.Retire(item->ticket);
        {
            std::lock_guard<std::mutex> lock(fifo_mu_);
            owners_.erase(seq); last_retired_ = seq; inflight_seq_ = -1; fifo_cv_.notify_all();
        }
        { std::lock_guard<std::mutex> lock(metrics_->mu); ++metrics_->cpu_completed; }
    }

    void Close()
    {
        core_.Close();
        std::lock_guard<std::mutex> lock(fifo_mu_); closed_ = true; fifo_cv_.notify_all();
    }
    void Abort(const std::string& reason)
    {
        std::lock_guard<std::mutex> lifecycle(lifecycle_mu_);
        { std::lock_guard<std::mutex> lock(fifo_mu_); if (stopped_) return; }
        SignalAbort(reason);
        std::lock_guard<std::mutex> submissions(submit_mu_);
        std::vector<std::shared_ptr<Item>> remaining;
        {
            std::lock_guard<std::mutex> lock(fifo_mu_);
            for (const auto& pair : owners_) remaining.push_back(pair.second);
            fifo_.clear();
        }
        // This bridge owns copied integers only. Abort cancels those native
        // simulation resources; it never owns or frees a Python/Torch tensor.
        for (const auto& item : remaining) {
            core_.CancelAndWait(item->ticket); core_.Retire(item->ticket);
        }
        {
            std::lock_guard<std::mutex> lock(fifo_mu_);
            owners_.clear(); inflight_seq_ = -1; fifo_cv_.notify_all();
        }
        { std::lock_guard<std::mutex> lock(metrics_->mu); metrics_->cancelled += remaining.size(); }
        core_.Shutdown();
        { std::lock_guard<std::mutex> lock(fifo_mu_); stopped_ = true; }
    }
    void Shutdown()
    {
        { std::lock_guard<std::mutex> lock(fifo_mu_); if (stopped_) return; }
        bool aborted;
        { std::lock_guard<std::mutex> lock(fifo_mu_); aborted = aborted_; }
        if (aborted) { Abort(AbortReason()); return; }
        std::lock_guard<std::mutex> lifecycle(lifecycle_mu_);
        Close();
        std::lock_guard<std::mutex> submissions(submit_mu_);
        {
            std::lock_guard<std::mutex> lock(fifo_mu_);
            if (!owners_.empty()) throw std::runtime_error("retire or abort outstanding CPU tickets before shutdown");
        }
        core_.Shutdown();
        { std::lock_guard<std::mutex> lock(fifo_mu_); stopped_ = true; }
    }
    Snapshot Inspect()
    {
        Snapshot result{}; result.core = core_.Snapshot();
        {
            std::lock_guard<std::mutex> lock(metrics_->mu);
            result.prepare_calls = metrics_->prepare_calls; result.checksum_xor = metrics_->checksum_xor;
            result.execute_markers = metrics_->execute_markers; result.cpu_completed = metrics_->cpu_completed;
            result.cancelled = metrics_->cancelled; result.trace_dropped = metrics_->trace_dropped;
            result.trace = metrics_->trace; result.worker_tids.assign(metrics_->worker_tids.begin(), metrics_->worker_tids.end());
        }
        {
            std::lock_guard<std::mutex> lock(fifo_mu_);
            result.closed = closed_; result.aborted = aborted_; result.stopped = stopped_; result.reason = reason_;
        }
        return result;
    }
    std::vector<Record> Trace()
    { std::lock_guard<std::mutex> lock(metrics_->mu); return metrics_->records; }
    void WaitGate()
    {
        std::unique_lock<std::mutex> lock(metrics_->mu);
        if (metrics_->gate_seq < 0) throw std::invalid_argument("test_gate_seq is not configured");
        metrics_->cv.wait(lock, [&] { return metrics_->gate_entered || metrics_->aborted; });
        if (metrics_->aborted) throw std::runtime_error(metrics_->reason);
    }
    void ReleaseGate()
    { std::lock_guard<std::mutex> lock(metrics_->mu); metrics_->gate_released = true; metrics_->cv.notify_all(); }
    void WaitPrepared(int64_t seq)
    {
        std::shared_ptr<CpuJob> job;
        { std::lock_guard<std::mutex> lock(fifo_mu_);
            auto found = owners_.find(seq);
            if (found == owners_.end()) throw std::invalid_argument("unknown preparation ticket");
            job = found->second->job; }
        std::unique_lock<std::mutex> lock(metrics_->mu);
        metrics_->cv.wait(lock, [&] { return job->prepared.load(std::memory_order_acquire) || metrics_->aborted; });
        if (metrics_->aborted) throw std::runtime_error(metrics_->reason);
    }
};

// Every potentially blocking native entry detaches BEFORE any core, FIFO, or
// lifecycle mutex is acquired. All Python objects are built after reattachment.
struct Detached {
    PyThreadState* thread = PyEval_SaveThread();
    ~Detached() { PyEval_RestoreThread(thread); }
};
PyObject* TranslateException()
{
    try { throw; }
    catch (const std::invalid_argument& error) { PyErr_SetString(PyExc_ValueError, error.what()); }
    catch (const std::bad_alloc&) { PyErr_NoMemory(); }
    catch (const std::exception& error) { PyErr_SetString(PyExc_RuntimeError, error.what()); }
    catch (...) { PyErr_SetString(PyExc_RuntimeError, "unknown native CPU simulation failure"); }
    return nullptr;
}
template <typename Fn> PyObject* NativeVoid(Fn&& function)
{
    try { { Detached detached; function(); } Py_RETURN_NONE; }
    catch (...) { return TranslateException(); }
}
bool Metadata(PyObject* source, std::vector<int64_t>& result)
{
    PyObject* sequence = PySequence_Fast(source, "metadata must be a sequence of signed int64 values");
    if (!sequence) return false;
    const Py_ssize_t count = PySequence_Fast_GET_SIZE(sequence);
    try { result.reserve(static_cast<size_t>(count)); }
    catch (...) { Py_DECREF(sequence); TranslateException(); return false; }
    for (Py_ssize_t i = 0; i < count; ++i) {
        const long long value = PyLong_AsLongLong(PySequence_Fast_GET_ITEM(sequence, i));
        if (PyErr_Occurred()) { Py_DECREF(sequence); return false; }
        result.push_back(static_cast<int64_t>(value));
    }
    Py_DECREF(sequence); return true;
}
void PutU(PyObject* dictionary, const char* key, uint64_t value)
{
    PyObject* number = PyLong_FromUnsignedLongLong(value);
    if (number) { PyDict_SetItemString(dictionary, key, number); Py_DECREF(number); }
}
PyObject* SnapshotObject(const Snapshot& view)
{
    PyObject* result = PyDict_New(); if (!result) return nullptr;
    PutU(result, "slots_in_use", view.core.slots_in_use); PutU(result, "retained_bytes", view.core.retained_bytes);
    PutU(result, "queued", view.core.queued); PutU(result, "preparing", view.core.preparing); PutU(result, "ready", view.core.ready);
    PutU(result, "admitted", view.core.admitted); PutU(result, "retired", view.core.retired);
    PutU(result, "workers_started", view.core.workers_started); PutU(result, "live_workers", view.stopped ? 0 : view.core.workers_started);
    PutU(result, "peak_slots_in_use", view.core.peak_slots_in_use); PutU(result, "peak_retained_bytes", view.core.peak_retained_bytes);
    PutU(result, "prepare_calls", view.prepare_calls); PutU(result, "checksum_xor", view.checksum_xor);
    PutU(result, "execute_markers", view.execute_markers); PutU(result, "cpu_completed", view.cpu_completed);
    PutU(result, "cancelled", view.cancelled); PutU(result, "trace_dropped", view.trace_dropped);
    PyDict_SetItemString(result, "real_npu", Py_False); PyDict_SetItemString(result, "simulated_prepare", Py_True);
    PyDict_SetItemString(result, "closed", view.closed ? Py_True : Py_False);
    PyDict_SetItemString(result, "aborted", view.aborted ? Py_True : Py_False);
    PyDict_SetItemString(result, "trace_enabled", view.trace ? Py_True : Py_False);
    PyObject* tids = PyList_New(static_cast<Py_ssize_t>(view.worker_tids.size()));
    if (!tids) { Py_DECREF(result); return nullptr; }
    for (size_t i = 0; i < view.worker_tids.size(); ++i) PyList_SET_ITEM(tids, static_cast<Py_ssize_t>(i), PyLong_FromUnsignedLongLong(view.worker_tids[i]));
    PyDict_SetItemString(result, "worker_tids", tids); Py_DECREF(tids);
    if (PyErr_Occurred()) { Py_DECREF(result); return nullptr; }
    return result;
}

struct PySession { PyObject_HEAD Session* native; };
PyObject* SessionNew(PyTypeObject* type, PyObject*, PyObject*)
{
    auto* self = reinterpret_cast<PySession*>(type->tp_alloc(type, 0));
    if (self) self->native = nullptr;
    return reinterpret_cast<PyObject*>(self);
}
int SessionInit(PySession* self, PyObject* args, PyObject* keywords)
{
    static const char* names[] = {"workers", "capacity", "byte_limit", "work_iterations", "trace", "trace_limit", "test_gate_seq", "fail_prepare_seq", nullptr};
    Py_ssize_t workers, capacity, bytes, limit = 4096; long long iterations; int trace = 0;
    PyObject *gate_object = Py_None, *fail_object = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, keywords, "nnnL|pnOO:Session", const_cast<char**>(names),
        &workers, &capacity, &bytes, &iterations, &trace, &limit, &gate_object, &fail_object)) return -1;
    if (self->native) { PyErr_SetString(PyExc_RuntimeError, "Session cannot be reinitialized"); return -1; }
    const long long gate = gate_object == Py_None ? -1 : PyLong_AsLongLong(gate_object);
    const long long fail = fail_object == Py_None ? -1 : PyLong_AsLongLong(fail_object);
    if (PyErr_Occurred()) return -1;
    if (workers < 1 || workers > 256 || capacity < 1 || bytes < 1 || iterations < 0 || limit < 0 || limit > 1000000) {
        PyErr_SetString(PyExc_ValueError, "positive workers/capacity/byte_limit, nonnegative iterations, and bounded trace_limit required"); return -1;
    }
    try { Detached detached; self->native = new Session(static_cast<size_t>(workers), static_cast<size_t>(capacity),
        static_cast<size_t>(bytes), static_cast<uint64_t>(iterations), trace != 0, static_cast<size_t>(limit), gate, fail); }
    catch (...) { TranslateException(); return -1; }
    return 0;
}
void SessionDealloc(PySession* self)
{
    { Detached detached; delete self->native; self->native = nullptr; }
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}
Session& Get(PySession* self)
{
    if (!self->native) throw std::runtime_error("Session is not initialized");
    return *self->native;
}
PyObject* SessionSubmit(PySession* self, PyObject* args)
{
    long long seq; PyObject* values;
    if (!PyArg_ParseTuple(args, "LO:submit", &seq, &values)) return nullptr;
    std::vector<int64_t> metadata; if (!Metadata(values, metadata)) return nullptr;
    return NativeVoid([&] { Get(self).Submit(seq, std::move(metadata)); });
}
PyObject* SessionNext(PySession* self, PyObject*)
{
    try { std::optional<int64_t> seq; { Detached detached; seq = Get(self).Next(); }
        if (!seq) Py_RETURN_NONE; return PyLong_FromLongLong(*seq); }
    catch (...) { return TranslateException(); }
}
PyObject* SessionRetire(PySession* self, PyObject* value)
{
    const long long seq = PyLong_AsLongLong(value); if (PyErr_Occurred()) return nullptr;
    return NativeVoid([&] { Get(self).Retire(seq); });
}
PyObject* SessionClose(PySession* self, PyObject*) { return NativeVoid([&] { Get(self).Close(); }); }
PyObject* SessionAbort(PySession* self, PyObject* args)
{
    const char* text = "CPU simulation aborted"; if (!PyArg_ParseTuple(args, "|s:abort", &text)) return nullptr;
    const std::string reason(text); return NativeVoid([&] { Get(self).Abort(reason); });
}
PyObject* SessionSnapshot(PySession* self, PyObject*)
{
    try { Snapshot view; { Detached detached; view = Get(self).Inspect(); } return SnapshotObject(view); }
    catch (...) { return TranslateException(); }
}
PyObject* SessionShutdown(PySession* self, PyObject*)
{
    try { Snapshot view; { Detached detached; Get(self).Shutdown(); view = Get(self).Inspect(); } return SnapshotObject(view); }
    catch (...) { return TranslateException(); }
}
PyObject* SessionTrace(PySession* self, PyObject*)
{
    try {
        std::vector<Record> records; { Detached detached; records = Get(self).Trace(); }
        PyObject* rows = PyList_New(static_cast<Py_ssize_t>(records.size())); if (!rows) return nullptr;
        for (size_t i = 0; i < records.size(); ++i) {
            const auto& row = records[i];
            PyObject* entry = Py_BuildValue("{sL,sK,sK,sK,sK,sK}", "seq", static_cast<long long>(row.seq),
                "tid", row.tid, "start_ns", row.start_ns, "end_ns", row.end_ns,
                "thread_cpu_ns", row.thread_cpu_ns, "checksum", row.checksum);
            if (!entry) { Py_DECREF(rows); return nullptr; }
#ifdef _WIN32
            PutU(entry, "thread_cycles", row.thread_cycles);
#else
            PyDict_SetItemString(entry, "thread_cycles", Py_None);
#endif
            PyList_SET_ITEM(rows, static_cast<Py_ssize_t>(i), entry);
        }
        return rows;
    } catch (...) { return TranslateException(); }
}
PyObject* SessionWaitGate(PySession* self, PyObject*) { return NativeVoid([&] { Get(self).WaitGate(); }); }
PyObject* SessionReleaseGate(PySession* self, PyObject*) { return NativeVoid([&] { Get(self).ReleaseGate(); }); }
PyObject* SessionWaitPrepared(PySession* self, PyObject* value)
{
    const long long seq = PyLong_AsLongLong(value); if (PyErr_Occurred()) return nullptr;
    return NativeVoid([&] { Get(self).WaitPrepared(seq); });
}
PyObject* PrepareInline(PyObject*, PyObject* args, PyObject* keywords)
{
    static const char* names[] = {"metadata", "iterations", "trace", nullptr};
    PyObject* values; long long iterations; int trace = 0;
    if (!PyArg_ParseTupleAndKeywords(args, keywords, "OL|p:prepare_inline", const_cast<char**>(names), &values, &iterations, &trace)) return nullptr;
    if (iterations < 0) { PyErr_SetString(PyExc_ValueError, "iterations must be nonnegative"); return nullptr; }
    std::vector<int64_t> metadata; if (!Metadata(values, metadata)) return nullptr;
    try {
        uint64_t checksum, wall = 0, cpu = 0, cycles = 0, tid;
        { Detached detached; tid = Tid(); const uint64_t start = trace ? NowNs() : 0;
            const uint64_t cpu_start = trace ? ThreadCpuNs() : 0;
            const uint64_t cycles_start = trace ? ThreadCycles() : 0;
            checksum = RunPrepareWork(metadata, static_cast<uint64_t>(iterations));
            if (trace) { cpu = ThreadCpuNs() - cpu_start; cycles = ThreadCycles() - cycles_start; wall = NowNs() - start; } }
        PyObject* result = PyDict_New(); if (!result) return nullptr;
        PutU(result, "checksum", checksum); PutU(result, "tid", tid);
        if (trace) { PutU(result, "wall_ns", wall); PutU(result, "thread_cpu_ns", cpu); }
        else { PyDict_SetItemString(result, "wall_ns", Py_None); PyDict_SetItemString(result, "thread_cpu_ns", Py_None); }
#ifdef _WIN32
        if (trace) PutU(result, "thread_cycles", cycles);
        else PyDict_SetItemString(result, "thread_cycles", Py_None);
#else
        (void)cycles;
        PyDict_SetItemString(result, "thread_cycles", Py_None);
#endif
        return result;
    } catch (...) { return TranslateException(); }
}
PyObject* CoreInfo(PyObject*, PyObject*)
{
    return Py_BuildValue("{ss,ss,sO,sO,ss,ss}", "core_path", kCorePath, "core_sha256", kCoreSha256,
        "real_npu", Py_False, "simulated_prepare", Py_True,
        "execute_kind", "native_marker_then_caller_CPU_operator", "clock", "steady_clock_ns");
}
PyObject* NativeNow(PyObject*, PyObject*) { return PyLong_FromUnsignedLongLong(NowNs()); }
#define METHOD(fn) reinterpret_cast<PyCFunction>(reinterpret_cast<void(*)(void)>(fn))
PyMethodDef session_methods[] = {
    {"submit", METHOD(SessionSubmit), METH_VARARGS, nullptr}, {"next", METHOD(SessionNext), METH_NOARGS, nullptr},
    {"retire", METHOD(SessionRetire), METH_O, nullptr}, {"close", METHOD(SessionClose), METH_NOARGS, nullptr},
    {"abort", METHOD(SessionAbort), METH_VARARGS, nullptr}, {"shutdown", METHOD(SessionShutdown), METH_NOARGS, nullptr},
    {"snapshot", METHOD(SessionSnapshot), METH_NOARGS, nullptr}, {"trace_records", METHOD(SessionTrace), METH_NOARGS, nullptr},
    {"wait_gate_entered", METHOD(SessionWaitGate), METH_NOARGS, nullptr},
    {"release_test_gate", METHOD(SessionReleaseGate), METH_NOARGS, nullptr},
    {"wait_prepared", METHOD(SessionWaitPrepared), METH_O, nullptr}, {nullptr, nullptr, 0, nullptr}
};
PyType_Slot session_slots[] = {
    {Py_tp_new, reinterpret_cast<void*>(SessionNew)}, {Py_tp_init, reinterpret_cast<void*>(SessionInit)},
    {Py_tp_dealloc, reinterpret_cast<void*>(SessionDealloc)}, {Py_tp_methods, session_methods}, {0, nullptr}
};
PyType_Spec session_spec = {"_eager_prepare_cpu.Session", sizeof(PySession), 0, Py_TPFLAGS_DEFAULT, session_slots};
PyMethodDef module_methods[] = {
    {"prepare_inline", METHOD(PrepareInline), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"core_info", CoreInfo, METH_NOARGS, nullptr}, {"now_ns", NativeNow, METH_NOARGS, nullptr}, {nullptr, nullptr, 0, nullptr}
};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_eager_prepare_cpu", "CPU-only simulation using the production queue core",
    -1, module_methods, nullptr, nullptr, nullptr, nullptr};
} // namespace

PyMODINIT_FUNC PyInit__eager_prepare_cpu()
{
    PyObject* result = PyModule_Create(&module); if (!result) return nullptr;
    PyObject* type = PyType_FromSpec(&session_spec);
    if (!type || PyModule_AddObject(result, "Session", type) < 0) { Py_XDECREF(type); Py_DECREF(result); return nullptr; }
    return result;
}
