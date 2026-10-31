"""Run isolated Linux immediate-adapter protocol/lifetime/sanitizer evidence.

Copy this runner and its explicit source files into a new directory on the test
host. This never modifies the frozen PTA build source passed with --frozen.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import time

PRODUCTION = [
    'torch_npu/csrc/core/npu/eager_parallel/ImmediateRuntime.cpp',
    'torch_npu/csrc/core/npu/eager_parallel/ImmediateRuntime.h',
    'torch_npu/csrc/core/npu/eager_parallel/ImmediateProtocol.h',
    'torch_npu/csrc/core/npu/eager_parallel/ImmediateInvoke.h',
    'torch_npu/csrc/core/npu/eager_parallel/AllocatorEventScope.h',
    'torch_npu/csrc/core/npu/NPUCachingAllocator.cpp',
    'torch_npu/csrc/core/npu/CachingHostAllocator.cpp',
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(source, signature):
    start = source.index(signature)
    brace = source.index('{', start)
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--entry-probe', type=Path)
    parser.add_argument('--case', action='append', choices=('protocol','output_lifetime','scalar_contract','destructor','allocator_event_scope'))
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / 'source'
    logs = root / 'logs'
    logs.mkdir(exist_ok=True)
    generated = root / 'generated'
    generated.mkdir(exist_ok=True)
    binaries = root / 'bin'
    binaries.mkdir(exist_ok=True)
    before = {name: digest(args.frozen / name) for name in PRODUCTION}
    copied = {name: digest(source / name) for name in PRODUCTION}
    if before != copied:
        raise RuntimeError('isolated production source does not match frozen candidate')
    report = {
        'schema': 1, 'started_unix': time.time(), 'host': platform.node(),
        'machine': platform.machine(), 'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'test_affinity': [184, 186],
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'production_before': before, 'source_hashes': copied,
        'scope': 'single-caller protocol, output lifetime and scalar qualification; not SDK concurrency qualification',
        'commands': [], 'results': [],
    }
    runtime = (source / PRODUCTION[0]).read_text()
    fixture = (source / 'test/cpp/eager_parallel/immediate_outputs_fixture.cpp.in').read_text()
    for marker, signature in (
        ('ADD_OUTPUT', 'void ImmediateGuard::add_output('),
        ('COMPLETE', 'void ImmediateGuard::complete('),
        ('FAIL', 'void ImmediateGuard::fail('),
    ):
        fixture = fixture.replace('/*' + marker + '*/', extract(runtime, signature))
    output_source = generated / 'immediate_outputs.cpp'
    output_source.write_text(fixture)
    helper = extract((source / PRODUCTION[3]).read_text(),
                     'template <class... Args> bool ImmediateScalarContract(')
    scalar_source = generated / 'immediate_scalar_contract.cpp'
    scalar_source.write_text(r'''
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <tuple>
#include <type_traits>
namespace at { struct Tensor { bool present=false; bool defined() const {return present;} }; }
''' + helper + r'''
int main() {
    at::Tensor tensor{true};
    std::optional<at::Tensor> none, undefined{at::Tensor{false}}, defined{tensor};
    unsigned cases=0;
    for(const auto* schema: {"aten::_embedding_bag", "aten::_embedding_bag_forward_only"}) {
        auto call=[&](bool scale,int64_t mode,const std::optional<at::Tensor>& sample) {
            ++cases;
            return ImmediateScalarContract(schema,tensor,tensor,tensor,scale,mode,false,sample,false,int64_t(-1));
        };
        assert(call(false,0,none));
        assert(call(false,0,undefined));
        assert(!call(false,0,defined));
        assert(!call(false,1,none));
        assert(!call(false,2,none));
        assert(!call(true,0,none));
        assert(!ImmediateScalarContract(schema,tensor)); ++cases;
        assert(!ImmediateScalarContract(schema,tensor,tensor,tensor,false,int32_t(0),false,none,false,int64_t(-1))); ++cases;
    }
    assert(ImmediateScalarContract("aten::cat",tensor)); ++cases;
    std::cout << "PASS actual ImmediateScalarContract: " << cases << " cases\n";
}
''')
    destructor_source=generated/'immediate_destructor.cpp'
    destructor_fixture=(source/'test/cpp/eager_parallel/immediate_destructor_fixture.cpp.in').read_text()
    destructor_source.write_text(destructor_fixture.replace('/*DESTRUCTOR*/',extract(runtime,'ImmediateGuard::~ImmediateGuard() noexcept')))
    spec=importlib.util.spec_from_file_location('r27_allocator_scope_fixture',source/'test/eager_parallel/test_allocator_event_scope.py')
    fixture_module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture_module)
    allocator_source=generated/'allocator_event_scope.cpp'
    allocator_source.write_text(fixture_module.generated_fixture())
    cases = {
        'protocol': source / 'test/cpp/eager_parallel/immediate_protocol_test.cpp',
        'output_lifetime': output_source,
        'scalar_contract': scalar_source,
        'destructor': destructor_source,
        'allocator_event_scope': allocator_source,
    }
    if args.case:
        cases = {name: path for name, path in cases.items() if name in args.case}
    runtime_dir = '/usr/lib64/clang/17/lib/aarch64-openEuler-linux-gnu'
    modes = {
        'release': ['g++', '-O2'],
        'asan': ['clang++', '-O1', '-g', '-fsanitize=address', '-fno-omit-frame-pointer',
                 '-shared-libsan', '-Wl,-rpath,' + runtime_dir],
        'tsan': ['clang++', '-O1', '-g', '-fsanitize=thread', '-fno-omit-frame-pointer',
                 '-shared-libsan', '-Wl,-rpath,' + runtime_dir],
    }
    environment = os.environ.copy()
    environment['ASAN_OPTIONS'] = 'detect_leaks=1:halt_on_error=1'
    environment['TSAN_OPTIONS'] = 'halt_on_error=1'
    environment['OMP_NUM_THREADS'] = '1'
    environment['OPENBLAS_NUM_THREADS'] = '1'

    def run(name, command, timeout=90):
        command = ['taskset', '-c', '184,186'] + [str(value) for value in command]
        report['commands'].append({'name': name, 'argv': command})
        started = time.time()
        try:
            result = subprocess.run(command, cwd=root, env=environment,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
            output, code = result.stdout, result.returncode
        except subprocess.TimeoutExpired as error:
            output, code = (error.stdout or b'') + b'\nTEST WATCHDOG EXPIRED\n', 124
        path = logs / (name + '.log')
        path.write_bytes(output)
        row = {'name': name, 'returncode': code, 'seconds': time.time()-started,
               'log': str(path), 'log_sha256': digest(path)}
        report['results'].append(row)
        print(json.dumps(row), flush=True)
        return code

    run('gcc_version', ['g++', '--version'])
    run('clang_version', ['clang++', '--version'])
    for mode, compiler in modes.items():
        for name, test_source in cases.items():
            executable = binaries / (mode + '_' + name)
            command = compiler + ['-std=c++17', '-pthread', '-Wall', '-Wextra', '-Werror',
                                  '-I' + str(source), str(test_source), '-o', str(executable)]
            if run(mode + '_' + name + '_compile', command) == 0:
                run(mode + '_' + name + '_run', [executable], timeout=30)
    if args.entry_probe:
        for name in ('tensor_visitor_test', 'immediate_invoke_lifetime_test'):
            run('existing_' + name, [args.entry_probe / name], timeout=30)
    report['production_after'] = {name: digest(args.frozen / name) for name in PRODUCTION}
    report['production_unchanged'] = before == report['production_after']
    report['generated_hashes'] = {str(path.relative_to(root)): digest(path) for path in generated.iterdir()}
    report['binaries_hashes'] = {str(path.relative_to(root)): digest(path) for path in binaries.iterdir()}
    report['finished_unix'] = time.time()
    report['passed'] = report['production_unchanged'] and all(row['returncode'] == 0 for row in report['results'])
    (root / 'results.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'passed': report['passed'], 'results': str(root / 'results.json')}), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
