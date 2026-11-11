"""Generate typed, process-local CANN timing wrappers from installed headers.

This diagnostic library forwards every argument/result unchanged. Its disabled
and enabled overhead must be measured before interpreting phase times. It also
intercepts explicit dlsym lookups, which otherwise bypass ELF interposition.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

RUNTIME = {
    'event': ['aclrtRecordEvent', 'aclrtStreamWaitEvent', 'aclrtResetEvent',
              'aclrtCreateEvent', 'aclrtCreateEventWithFlag', 'aclrtCreateEventExWithFlag',
              'aclrtDestroyEvent', 'aclrtQueryEventStatus', 'aclrtQueryEventWaitStatus'],
    'sync': ['aclrtSynchronizeStream', 'aclrtSynchronizeStreamWithTimeout',
             'aclrtSynchronizeDevice', 'aclrtSynchronizeDeviceWithTimeout',
             'aclrtSynchronizeEvent', 'aclrtStreamQuery'],
    'copy_sync': ['aclrtMemcpy'],
    'copy_async': ['aclrtMemcpyAsync'],
    'stream': ['aclrtGetStreamAttribute', 'aclrtGetCurrentContext', 'aclrtGetDevice'],
    'descriptor': ['aclCreateTensor', 'aclDestroyTensor', 'aclCreateTensorList',
                   'aclDestroyTensorList', 'aclCreateScalar', 'aclDestroyScalar',
                   'aclCreateIntArray', 'aclDestroyIntArray'],
}
OPERATORS = ['aclnnEmbeddingBag', 'aclnnCat', 'aclnnBatchMatMul', 'aclnnIndex',
             'aclnnAddmm', 'aclnnRelu', 'aclnnSigmoid', 'aclnnInplaceZero']

PREAMBLE = r'''
#define _GNU_SOURCE
#include <dlfcn.h>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <time.h>
using Lookup = void*(*)(void*,const char*);
static Lookup lookup() {
    static auto p=reinterpret_cast<Lookup>(dlvsym(RTLD_NEXT,"dlsym","GLIBC_2.17"));
    if (!p) std::abort();
    return p;
}
'''
ENGINE = r'''
constexpr int SITE_COUNT=16;
struct Site { uint64_t caller=0,count=0,exclusive_ns=0; };
struct Bucket { uint64_t count=0, inclusive_ns=0, exclusive_ns=0, caller=0, overflow=0; Site sites[SITE_COUNT]; };
struct Frame;
struct ThreadData { int phase=-1; Frame* frame=nullptr; Bucket data[2][TAG_COUNT]; };
static thread_local ThreadData tls;
static std::atomic<void*> original[TAG_COUNT]{};
static std::atomic<uint64_t> next_lookups{0};
extern "C" uint64_t r33_phase_next_lookups() { return next_lookups.load(); }
#if defined(__aarch64__)
static uint64_t frequency() { uint64_t f; asm volatile("mrs %0, cntfrq_el0":"=r"(f)); return f; }
static const uint64_t tick_ns=[] { auto f=frequency(); if (!f || 1000000000ULL%f) std::abort(); return 1000000000ULL/f; }();
static uint64_t now() { uint64_t t; asm volatile("isb; mrs %0, cntvct_el0":"=r"(t)::"memory"); return t*tick_ns; }
#else
static uint64_t frequency() { return 1000000000ULL; }
static uint64_t now() { timespec t{}; clock_gettime(CLOCK_MONOTONIC_RAW,&t); return uint64_t(t.tv_sec)*1000000000ULL+t.tv_nsec; }
#endif
extern "C" uint64_t r33_phase_frequency() { return frequency(); }
struct Frame {
    bool enabled; int phase,tag; uint64_t start=0, child=0,caller=0; Frame* parent=nullptr;
    Frame(int t,void* address):enabled(tls.phase>=0),phase(tls.phase),tag(t),caller(reinterpret_cast<uintptr_t>(address)) {
        if (!enabled) return;
        parent=tls.frame; tls.frame=this;
        if (!tls.data[phase][tag].caller) tls.data[phase][tag].caller=caller;
        start=now();
    }
    ~Frame() {
        if (!enabled) return;
        auto elapsed=now()-start; auto& b=tls.data[phase][tag];
        ++b.count; b.inclusive_ns+=elapsed; b.exclusive_ns+=elapsed-child;
        bool found=false;
        for (auto& site:b.sites) if (site.caller==caller || !site.caller) {
            site.caller=caller; ++site.count; site.exclusive_ns+=elapsed-child; found=true; break;
        }
        if (!found) ++b.overflow;
        if (parent) parent->child+=elapsed;
        tls.frame=parent;
    }
};
static void* resolve(int tag,const char* name) {
    if (auto p=original[tag].load(std::memory_order_relaxed)) return p;
    auto p=lookup()(RTLD_NEXT,name);
    if (!p) {
        const char* libs[]={"libascendcl.so","libopapi_nn.so","libopapi_math.so","libopapi.so"};
        for (auto lib:libs) {
            auto h=dlopen(lib,RTLD_LAZY|RTLD_NOLOAD);
            if (!h) continue;
            p=lookup()(h,name); dlclose(h);
            if (p) break;
        }
    }
    if (!p) { std::fprintf(stderr,"R33 cannot resolve %s\n",name); std::abort(); }
    original[tag].store(p,std::memory_order_relaxed); return p;
}
template<class Fn,int Tag> struct Call;
template<class R,class... A,int Tag> struct Call<R(*)(A...),Tag> {
    static R run(void* caller,const char* name,A... args) {
        auto fn=reinterpret_cast<R(*)(A...)>(resolve(Tag,name)); Frame frame(Tag,caller);
        return fn(args...);
    }
};
template<class R,class... A,int Tag> struct Call<R(*)(A...) noexcept,Tag>:Call<R(*)(A...),Tag> {};
extern "C" void r33_phase_reset() { tls.phase=-1; tls.frame=nullptr; for (auto& row:tls.data) for (auto& b:row) b=Bucket{}; }
extern "C" void r33_phase_select(int phase) { if (phase < -1 || phase>1 || tls.frame) std::abort(); tls.phase=phase; }
extern "C" int r33_phase_tags() { return TAG_COUNT; }
extern "C" int r33_phase_version() { return 3; }
extern "C" const char* r33_phase_name(int tag) { return names[tag]; }
extern "C" const char* r33_phase_category(int tag) { return categories[tag]; }
extern "C" const Bucket* r33_phase_data() { return &tls.data[0][0]; }
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--include', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    desired = [(n, c) for c, names in RUNTIME.items() for n in names]
    desired += [(n+'GetWorkspaceSize', 'workspace') for n in OPERATORS]
    desired += [(n, 'launch') for n in OPERATORS]
    texts = {}
    for directory in ('acl', 'aclnn', 'aclnnop'):
        for path in (args.include/directory).glob('*.h'):
            text = path.read_text(errors='strict')
            texts[path] = re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.S)
    rows, missing, includes = [], [], set()
    for name, category in desired:
        found = []
        for path, text in texts.items():
            pattern = (r'(?:ACLNN_API|ACL_FUNC_VISIBILITY)\s+'
                       r'([\w:]+\s*\*?)\s*'+re.escape(name)+r'\s*\(([^;{}]*)\)\s*;')
            for match in re.finditer(pattern, text):
                found.append((path, match.group(1).strip(), match.group(2).strip()))
        if not found:
            if name == 'aclrtStreamQuery':
                rows.append(dict(name=name, category=category, return_type='int',
                                 parameters='void* stream, int* status', private_abi=True))
            else:
                missing.append(name)
            continue
        signatures = {(ret, ' '.join(params.split())) for _, ret, params in found}
        assert len(signatures) == 1, (name, signatures)
        path, ret, params = found[0]
        includes.add(path.relative_to(args.include).as_posix())
        rows.append(dict(name=name, category=category, return_type=ret,
                         parameters=' '.join(params.split()),
                         header=str(path), header_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    rows.append(dict(name='nanosleep', category='poll_sleep', return_type='int',
                     parameters='const struct timespec* request, struct timespec* remaining'))
    for name in OPERATORS:
        assert all(any(r['name'] == n for r in rows) for n in (name, name+'GetWorkspaceSize')), name
    source = PREAMBLE + ''.join('#include <'+n+'>\n' for n in sorted(includes))
    for row in rows:
        if row.get('private_abi'):
            source += f'extern "C" {row["return_type"]} {row["name"]}({row["parameters"]});\n'
    source += 'constexpr int TAG_COUNT='+str(len(rows))+';\n'
    source += 'static const char* names[]={'+','.join(json.dumps(r['name']) for r in rows)+'};\n'
    source += 'static const char* categories[]={'+','.join(json.dumps(r['category']) for r in rows)+'};\n'
    source += ENGINE
    for tag, row in enumerate(rows):
        params = row['parameters']
        names = [] if params in ('', 'void') else [re.search(r'(\w+)\s*$', s).group(1) for s in params.split(',')]
        source += (f'extern "C" {row["return_type"]} {row["name"]}({params}) {{ '
                   f'return Call<decltype(&{row["name"]}),{tag}>::run(__builtin_return_address(0),'
                   f'"{row["name"]}"'+(' ,'+','.join(names) if names else '')+'); }\n')
    source += 'extern "C" void* dlsym(void* handle,const char* name) noexcept { auto p=lookup()(handle,name); if (handle==RTLD_NEXT) ++next_lookups; if (!p) return p;\n'
    for tag, row in enumerate(rows):
        name = row['name']
        source += (f'if (!std::strcmp(name,"{name}")) {{ auto wrapper=reinterpret_cast<void*>(&{name}); '
                   f'if (p!=wrapper) original[{tag}].store(p,std::memory_order_relaxed); return wrapper; }}\n')
    source += 'return p; }\n'
    cpp = args.output/'phase_api_tap.cpp'
    cpp.write_text(source)
    command = ['c++', '-shared', '-fPIC', '-O2', '-std=c++17', '-I'+str(args.include),
               str(cpp), '-ldl', '-o', str(args.output/'libphase_api_tap.so')]
    (args.output/'manifest.json').write_text(json.dumps(dict(rows=rows, missing_optional=missing,
                                                           compile=command), indent=2)+'\n')
    subprocess.run(command, check=True)
    print(json.dumps(dict(tags=len(rows), missing_optional=missing)), flush=True)


if __name__ == '__main__':
    main()
