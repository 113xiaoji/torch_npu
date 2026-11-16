"""Verify the real locked/atomic option dispatch without an NPU dependency."""
from pathlib import Path
import os
import subprocess
import tempfile
from event_wait_native import compiler_environment
from pool_activity_native import method

ROOT=Path(__file__).resolve().parents[2]
PREFIX=r'''
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#define TORCH_NPU_API
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("option check"); } while (false)
namespace c10 { using DeviceIndex = int; }
unsigned locked_calls=0,fast_calls=0;
bool active=false;
namespace c10_npu { namespace NPUCachingAllocator {
bool hasCapturesUnderway(int) { ++locked_calls; return active; }
bool hasCapturesUnderwayFast(int) { ++fast_calls; return active; }
}}
namespace {
'''
SUFFIX=r'''
void require(bool ok,const char* message) { if (!ok) throw std::runtime_error(message); }
int main(int argc,char** argv) {
    require(argc==2,"expected mode argument");
    const std::string expected=argv[1];
    require(OpApiWorkspacePoolQueryR42()==expected,"effective mode differs");
    if (expected=="invalid") {
        bool rejected=false;
        try { (void)WorkspacePoolActiveR42(0); } catch (...) { rejected=true; }
        require(rejected && locked_calls==0 && fast_calls==0,"invalid option must fail closed");
    } else {
        require(!WorkspacePoolActiveR42(0),"inactive eligibility differs");
        active=true;
        require(WorkspacePoolActiveR42(0),"active pool must bypass in both modes");
        require(expected=="atomic" ? fast_calls==2 && locked_calls==0 : locked_calls==2 && fast_calls==0,
            "atomic treatment unexpectedly calls locked allocator query");
    }
    std::cout << "PASS actual pool option mode=" << expected << '\n';
}
'''

def main():
    compiler,environment=compiler_environment()
    source=(ROOT/'patches/eager_mpsc_r34/op-plugin/op_plugin/utils/op_api_common.cpp').read_text(encoding='utf-8')
    start=source.index('enum class WorkspacePoolQueryR42')
    end=source.index('bool WorkspaceCacheEnabledR41()',start)
    getter=method(source,'extern "C" TORCH_NPU_API const char* OpApiWorkspacePoolQueryR42() noexcept')
    with tempfile.TemporaryDirectory(prefix='r42_pool_query_') as tmp:
        build=Path(tmp);unit=build/'query.cpp';unit.write_text(PREFIX+source[start:end]+'}\n'+getter+SUFFIX,encoding='utf-8')
        binary=build/('query.exe' if os.name=='nt' else 'query')
        if os.name=='nt':
            command=[compiler,'/nologo','/std:c++17','/EHsc','/W4','/WX','/D_CRT_SECURE_NO_WARNINGS','/O2',str(unit),f'/Fe:{binary}',f'/Fo:{build}{os.sep}']
        else:
            command=[compiler,'-std=c++17','-Wall','-Wextra','-Werror','-O2',str(unit),'-o',str(binary)]
        subprocess.run(command,cwd=build,env=environment,check=True,timeout=120)
        for value,expected in ((None,'locked'),('locked','locked'),('atomic','atomic'),('','invalid'),('wrong','invalid')):
            env=environment.copy()
            if value is None:env.pop('TORCH_NPU_WORKSPACE_POOL_QUERY',None)
            else:env['TORCH_NPU_WORKSPACE_POOL_QUERY']=value
            subprocess.run([str(binary),expected],env=env,check=True,timeout=10)

if __name__=='__main__':main()
