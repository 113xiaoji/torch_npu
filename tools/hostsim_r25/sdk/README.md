# HOST_SIM_R25 typed OpAPI provider

This library simulates descriptor conversion, unique executor preparation,
consumption/cancellation and host lifecycle bookkeeping. It performs no tensor
arithmetic. All receipts explicitly report `numeric_supported=false`,
`real_sdk_qualified=false`, and `pta_cache_supported=false`. It has no Torch,
CPython, real CANN, or device runtime dependency.

Build and test from the repository root:

```text
python tools/hostsim_r25/sdk/build_provider.py --output-dir /isolated/run/sdk --test
python -m unittest discover -s test/hostsim_r25_sdk -v
```

The same helper uses MSVC on Windows and C++17 GCC/Clang on Linux. Linux emits
`libopapi.so`; Windows emits `opapi.dll`. The build manifest records source and
binary SHA256 values. An explicit output directory is reused only when both
source and binary hashes match its manifest. No SDK or existing runtime files
are changed. For the isolated Linux benchmark, prepend this new output directory
to that process's `LD_LIBRARY_PATH`, allowing the standard OpAPI `dlopen` resolver
to load it. Do not install it globally or use `LD_PRELOAD`.

The standalone header defines the integer-compatible ACL enums and opaque
descriptor types needed here. A direct Linux build can instead define
`R25_HOSTSIM_USE_ACL_HEADERS` and pass the installed ACL include directory. A
minimal standalone command is:

```text
c++ -std=c++17 -O2 -shared -fPIC -pthread tools/hostsim_r25/sdk/r25_hostsim_opapi.cpp -o /isolated/run/sdk/libopapi.so
```

The typed schemas follow the pinned OpAPI tuples:

| Operator | Arguments before workspace size and executor output pointers |
| --- | --- |
| `aclnnEmbeddingBag` | weight, indices, offsets, bool scale, int64 mode, bool sparse, optional sample weights, bool last offset, int64 padding, four output tensors |
| `aclnnAdd` | self, other, scalar alpha, output |
| `aclnnRelu` | self, output |
| `aclnnMm` | self, mat2, output, int8 cube math type |

EmbeddingBag's schema was checked against `ops-nn` commit
`814ce56c09d6bac1a99364a71a421ef7c1abf519`,
`index/embedding_bag/op_api/aclnn_embedding_bag.h`. Relu and Mm were checked
against the corresponding fixed R24 audited headers. Add's tuple and scalar
type were checked against the pinned op-plugin `AddKernelNpuOpApi.cpp` callsite.
These are simulation schemas; their presence does not qualify any real SDK path.

Each creation copies the supplied metadata arrays or scalar bytes. Opaque
handles are monotonic integer tokens stored in a mutex-protected live registry;
they are never dereferenced and never reused, including across diagnostic resets.
Their backing allocations are reclaimed on destroy or executor consumption.
Lists own their children after successful creation and destroy them recursively.
Executors borrow descriptors and reject their premature destruction at Execute.
Execute consumes a recognized executor once even on a validation failure;
Destroy cancels it once. Unknown, stale, wrong-kind, null required output, and
unsupported dtype arguments produce nonzero statuses. C++ exceptions are caught
at the C ABI and converted to a failure/null return.

Workspace is always zero. There are no sleeps, delay knobs, artificial CPU
loops, kernel work, or PTA executor caching. HugeMem uses one bounded active flag
per thread and a global initialized/released lease count, supporting the R24
worker-UnInit/consumer-Release ordering. Cache permission always returns false.
Actual cache init/hash/find stubs fail explicitly if called. The legacy macros'
unconditional inactive-cache cleanup is a harmless separately counted no-op.

`R25HostSimAbi()` returns `1`; `R25HostSimBackendKind()` returns `HOST_SIM_R25`.
`R25HostSimDiagnosticsJson()` returns a calling-thread UTF-8 string. Diagnostics
include descriptor/executor counters, per-operation counts, HugeMem leases,
native preparation and execution TIDs, and errors. `R25HostSimResetDiagnostics()`
rejects a reset while descriptors, executors or HugeMem leases remain live.
`R25HostSimLastError()` gives the calling thread's last error message.

Tracing is off by default. Enable `R25HostSimSetTracing(true)` only in separate
cold receipt runs. It records at most 4096 prepare/execute/cancel events with
sequence, unique executor ID, operator, native TID, Prepare TID and status;
`trace_dropped` reports overflow. Use R24 caller command sequence numbers to prove
caller ordering. Provider executor IDs follow worker preparation order, so their
raw ordering is not a cross-run equivalence hash. Tensor/executor metadata JSON
accessors exist only for explicit diagnostics and are never used by preparation.
