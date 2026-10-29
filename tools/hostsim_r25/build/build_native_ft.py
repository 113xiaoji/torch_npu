"""Build a prepared R25 source tree entirely inside a new remote attempt directory.

This performs a full TorchNPU native build; it never reuses parent TorchNPU
object files. Unchanged vendor archives are retained and hashed. Source and SDK
preparation are separate so their reviewed receipts remain visible. Run with
the remote system Python, not through setup.py installation.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/home/r25-eager-prepare-hostsim-20261029')
OLD = Path('/home/z00482291/results/qwen-dispatch-20260914/freethread-gil-20260928')
PYTORCH = Path('/root/ft-pytorch-repro/envs/py314t/lib/python3.14t/site-packages/torch')
PYINC = Path('/root/ft-pytorch-repro/prefixes/py314t/include/python3.14t')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def owned(path):
    path = path.resolve(strict=True)
    if not path.is_relative_to(ROOT.resolve(strict=True)) or path == ROOT:
        raise ValueError('write target must be inside the new isolated task root')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', required=True, type=Path)
    parser.add_argument('--provider', required=True, type=Path)
    parser.add_argument('--package-base', required=True, type=Path)
    parser.add_argument('--cpus', default='160,162,164,166,168,170,172,174,176,178,180,182')
    parser.add_argument('--jobs', type=int, default=12)
    args = parser.parse_args()
    attempt = owned(args.attempt)
    source = owned(attempt / 'src' / 'torch-npu')
    provider = owned(args.provider)
    if not provider.is_file() or not (1 <= args.jobs <= 12):
        raise ValueError('regular new provider and 1..12 jobs are required')
    cpus = {int(value) for value in args.cpus.split(',')}
    if not cpus <= os.sched_getaffinity(0):
        raise ValueError('build CPU selection exceeds the permitted affinity')
    if cpus & set(range(144, 159)):
        raise ValueError('build CPUs overlap reserved measurement CPUs')
    os.sched_setaffinity(0, cpus)
    build = attempt / 'build-ft'
    package = attempt / 'packages-ft' / 'torch_npu'
    if build.exists() or package.exists():
        raise ValueError('fresh build/package directories required; use a new attempt')
    build.mkdir()
    package.parent.mkdir()
    shutil.copytree(args.package_base.resolve(strict=True), package, symlinks=True)
    if (package / 'lib').is_symlink():
        raise ValueError('copied package lib directory must be physical')
    # Keep generated package-only files, while making every delivered Python
    # source agree with the prepared and hashed candidate tree.
    for path in (source / 'torch_npu').rglob('*'):
        if path.is_file() and path.suffix in {'.py', '.pyi'}:
            dest = package / path.relative_to(source / 'torch_npu')
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.is_symlink():
                raise ValueError('package Python destination must be physical')
            shutil.copyfile(path, dest)
    # These two native outputs are rebuilt below, not accepted as cached output.
    (package / 'lib' / 'libtorch_npu.so').unlink(missing_ok=True)
    for path in package.glob('_C*.so'):
        path.unlink()
    evidence = attempt / 'build-evidence'
    evidence.mkdir()
    shutil.copyfile(Path(__file__), evidence / 'build-driver-source.py')
    commands = []
    env = dict(os.environ, MAX_JOBS=str(args.jobs), OMP_NUM_THREADS='1')
    env.pop('PYTHONPATH', None)

    def run(argv, name):
        commands.append({'name': name, 'argv': [str(x) for x in argv], 'cwd': str(build)})
        (evidence / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        print(name, flush=True)
        with (evidence / (name + '.log')).open('w') as log:
            result = subprocess.run([str(x) for x in argv], cwd=build, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        print(name, 'exit', result.returncode, flush=True)
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, argv)

    tracked = {}
    for path in sorted(source.rglob('*')):
        if path.is_file() and not path.is_symlink() and (
            path.suffix in {'.h', '.hpp', '.cpp', '.c', '.cmake', '.in', '.py', '.pyi'}
            or path.name == 'CMakeLists.txt'
        ):
            tracked[str(path.relative_to(source))] = sha(path)
    inputs = {
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source': str(source), 'provider': str(provider), 'provider_sha256': sha(provider),
        'package_base': str(args.package_base), 'cpus': sorted(cpus), 'jobs': args.jobs,
        'source_files': tracked, 'full_compile': True, 'reused_native_objects': 0,
        'driver_sha256': sha(evidence / 'build-driver-source.py'),
        'vendor_archives': {str(p): sha(p.resolve()) for p in sorted((source / 'third_party').rglob('*.a'))},
        'sdk_libraries': {str(p): sha(p.resolve()) for p in sorted((source / 'third_party/acl/libs').glob('*.so'))},
    }
    (evidence / 'inputs.json').write_text(json.dumps(inputs, indent=2) + '\n')
    cache = {}
    for line in (OLD / 'build-ft' / 'CMakeCache.txt').read_text().splitlines():
        if line and not line.startswith(('#', '//')) and '=' in line:
            key, value = line.split('=', 1)
            cache[key.split(':')[0]] = value
    configure = ['cmake', '-S', source, '-B', build, '-G', 'Ninja',
                 '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CXX_COMPILER=/usr/bin/c++',
                 '-DCMAKE_C_COMPILER=/usr/bin/cc',
                 '-DCMAKE_CXX_FLAGS=' + cache['CMAKE_CXX_FLAGS'],
                 '-DTORCH_VERSION=2.13.0', '-DGLIBCXX_USE_CXX11_ABI=1',
                 '-DPYTORCH_INSTALL_DIR=' + str(PYTORCH),
                 '-DPYTHON_INCLUDE_DIR=' + str(PYINC),
                 '-DTORCHNPU_INSTALL_LIBDIR=' + str(package / 'lib'),
                 '-DCMAKE_INSTALL_PREFIX=' + str(package / 'lib'),
                 '-DTORCH_NPU_EXPERIMENTAL_FREE_THREADED=ON',
                 '-DTORCH_NPU_EAGER_PREPARE_HOST_SIM=ON',
                 '-DTORCH_NPU_EAGER_PREPARE_HOST_SIM_LIBRARY=' + str(provider)]
    run(configure, 'configure')
    run(['cmake', '--build', build, '--target', 'torch_npu', '--parallel', str(args.jobs)], 'native-build')
    # _C is built outside CMake in the pinned FT recipe. Preserve every flag and
    # library, replacing only source/output paths and adding the HostSim profile.
    recipe = json.loads((OLD / 'build-commands-ft.json').read_text())['commands'][-1]
    old_source = str(OLD / 'src' / 'torch-npu')
    old_package = str(OLD / 'packages-ft' / 'torch_npu')
    binding = [word.replace(old_source, str(source)).replace(old_package, str(package)) for word in recipe]
    binding[1:1] = ['-DTORCH_NPU_EAGER_PREPARE_HOST_SIM=1', '-I' + str(build / 'r25_hostsim')]
    run(binding, 'python-bindings')
    outputs = [package / 'lib' / 'libtorch_npu.so', *package.glob('_C*.so')]
    if len(outputs) != 2 or any(not path.is_file() for path in outputs):
        raise RuntimeError('expected rebuilt native library and exactly one _C extension')
    identities = {}
    for path in outputs:
        identities[str(path)] = {'sha256': sha(path), 'bytes': path.stat().st_size,
                                'elf_notes': subprocess.check_output(['readelf', '-n', str(path)], text=True)}
    if sha(provider) != inputs['provider_sha256']:
        raise RuntimeError('provider changed while building')
    after = {name: sha(source / name) for name in tracked}
    if after != tracked:
        raise RuntimeError('source changed while building')
    result = {'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'outputs': identities, 'provider_sha256': sha(provider),
              'pythonpath': str(package.parent),
              'ld_library_path': ':'.join([str(package / 'lib'), str(provider.parent),
                                           str(source / 'third_party/acl/libs'), str(PYTORCH / 'lib')]),
              'full_compile': True, 'reused_native_objects': 0}
    (evidence / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
