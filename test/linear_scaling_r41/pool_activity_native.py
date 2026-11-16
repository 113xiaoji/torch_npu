"""Compile real allocator pool mutation bodies with explicit CPU dependencies."""
from pathlib import Path
import os
import subprocess
import tempfile
from event_wait_native import compiler_environment

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'torch_npu/csrc/core/npu/NPUCachingAllocator.cpp'

def method(source,signature):
    start=source.index(signature);opening=source.index('{',start);level=1;cursor=opening+1
    while level:
        level+=(source[cursor]=='{')-(source[cursor]=='}');cursor+=1
    return source[start:cursor]

PREFIX=r'''
#include <atomic>
#include <algorithm>
#include <functional>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>
#include "torch_npu/csrc/core/npu/CapturePoolActivity.h"
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("pool check"); } while (false)
#define TORCH_NPU_MEMORY_LOGD(...) ((void)0)
using aclrtStream=void*;
using MempoolId_t=std::pair<unsigned long,unsigned long>;
class PoolFixture {
    std::recursive_mutex mutex;
    std::vector<std::pair<MempoolId_t,std::function<bool(aclrtStream)>>> captures_underway;
    c10_npu::detail::CapturePoolActivity capture_pool_activity_;
    void create_or_incref_pool(MempoolId_t) {}
public:
'''

SUFFIX=r'''
};
void require(bool condition,const char* message) {
    if (!condition) throw std::runtime_error(message);
}
int main() {
    PoolFixture fixture;
    require(!fixture.hasCapturesUnderwayFast(),"fresh allocator has no pool");
    auto filter=[](aclrtStream){return true;};
    fixture.beginAllocateToPool({1,0},filter);
    fixture.beginAllocateToPool({2,0},filter);
    require(fixture.hasCapturesUnderwayFast(),"nested active pools must bypass");
    fixture.endAllocateToPool({1,0});
    require(fixture.hasCapturesUnderwayFast(),"remaining pool still active");
    bool duplicate=false;
    try { fixture.beginAllocateToPool({2,0},filter); } catch (...) { duplicate=true; }
    require(duplicate && fixture.hasCapturesUnderwayFast(),"duplicate error preserves active flag");
    fixture.endAllocateToPool({2,0});
    require(!fixture.hasCapturesUnderwayFast(),"last pool leaves flag inactive");
    bool missing=false;
    try { fixture.endAllocateToPool({99,0}); } catch (...) { missing=true; }
    require(missing && !fixture.hasCapturesUnderwayFast(),"invalid end preserves inactive flag");
    std::atomic<unsigned> failures{0};
    std::vector<std::thread> workers;
    for (unsigned worker=0;worker<4;++worker) {
        workers.emplace_back([&,worker] {
            for (unsigned iteration=0;iteration<20000;++iteration) {
                const MempoolId_t id{worker+10,iteration};
                fixture.beginAllocateToPool(id,filter);
                if (!fixture.hasCapturesUnderwayFast()) ++failures;
                std::this_thread::yield();
                if (!fixture.hasCapturesUnderwayFast()) ++failures;
                fixture.endAllocateToPool(id);
            }
        });
    }
    for (auto& worker:workers) worker.join();
    require(failures.load()==0,"a caller inside its own active scope observed false eligibility");
    require(!fixture.hasCapturesUnderwayFast(),"all joined scopes must publish inactive");
    std::cout << "PASS actual pool begin/end, nested/error paths, 80000 concurrent scopes\n";
}
'''

def main():
    compiler,environment=compiler_environment()
    source=SOURCE.read_text(encoding='utf-8')
    names=('void beginAllocateToPool(MempoolId_t mempool_id, std::function<bool(aclrtStream)> filter)',
           'void endAllocateToPool(MempoolId_t mempool_id)',
           'bool hasCapturesUnderwayFast() const')
    bodies='\n'.join(method(source,name) for name in names)
    with tempfile.TemporaryDirectory(prefix='r42_pool_activity_') as tmp:
        build=Path(tmp);unit=build/'pool.cpp';unit.write_text(PREFIX+bodies+SUFFIX,encoding='utf-8')
        binary=build/('pool.exe' if os.name=='nt' else 'pool')
        if os.name=='nt':
            # C4324 only describes the intentional cache-line padding here.
            command=[compiler,'/nologo','/std:c++17','/EHsc','/W4','/WX','/wd4324','/O2',f'/I{ROOT}',str(unit),f'/Fe:{binary}',f'/Fo:{build}{os.sep}']
        else:
            command=[compiler,'-std=c++17','-pthread','-Wall','-Wextra','-Werror','-O2',f'-I{ROOT}',str(unit),'-o',str(binary)]
        subprocess.run(command,cwd=build,env=environment,check=True,timeout=120)
        subprocess.run([str(binary)],env=environment,check=True,timeout=30)

if __name__=='__main__':main()
