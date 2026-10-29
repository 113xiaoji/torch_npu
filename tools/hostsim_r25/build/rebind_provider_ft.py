"""Rebind a successful isolated FT build to a new provider, with audited reuse.

Requires a new prepared source tree and a completed parent attempt. Compiler
dependency records decide what is rebuilt; unrelated objects are independently
copied and every reused dependency and object is hashed. No model is executed.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess

ROOT = Path('/home/r25-eager-prepare-hostsim-20261029')
PROFILE = Path('r25_hostsim/EagerPrepareHostSimProfile.h')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def owned(path):
    path = path.resolve(strict=True)
    if path == ROOT or not path.is_relative_to(ROOT.resolve(strict=True)):
        raise ValueError('all write targets must remain in the isolated task root')
    return path


def verify_source_dependency(dep, oldsource, source, historical_pins, observed):
    relative = dep.relative_to(oldsource)
    key = relative.as_posix()
    if key not in historical_pins:
        raise ValueError('completed full-build source pin missing: ' + key)
    expected = historical_pins[key]
    if key not in observed:
        if sha(dep) != expected or sha(source / relative) != expected:
            raise ValueError('reused source differs from completed full-build pin: ' + key)
        observed[key] = expected
    elif observed[key] != expected:
        raise ValueError('recorded source dependency differs from historical pin: ' + key)


def verify_source_postcheck(observed, oldsource, source, historical_pins):
    for name, h in observed.items():
        if name not in historical_pins or h != historical_pins[name]:
            raise ValueError('source dependency lost its completed full-build pin: ' + name)
        expected = historical_pins[name]
        if sha(source / name) != expected or sha(oldsource / name) != expected:
            raise ValueError('source dependency changed from historical pin during rebind: ' + name)


def link_argv(build):
    line = subprocess.check_output(['ninja', '-C', str(build), '-t', 'commands', 'torch_npu'], text=True).splitlines()[-1]
    argv = shlex.split(line)
    if argv[:2] != [':', '&&'] or argv[-2:] != ['&&', ':']:
        raise ValueError('unexpected CMake link wrapper')
    # POSIX shlex retains backslashes before '$' inside double quotes, whereas
    # Ninja's shell removes that escape. Preserve the effective linker RPATH.
    result = []
    for word in argv[2:-2]:
        if word.startswith('-Wl,-rpath,'):
            allowed = ('-Wl,-rpath,\\$ORIGIN/lib/:\\$ORIGIN/',
                       '-Wl,-rpath,$ORIGIN/lib/:$ORIGIN/')
            if word not in allowed:
                raise ValueError('unexpected RPATH token; refusing general shell unescaping')
            word = word.replace('\\$ORIGIN', '$ORIGIN')
        result.append(word)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-attempt', type=Path, required=True)
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--provider', type=Path, required=True)
    args = parser.parse_args()
    parent, attempt, provider = map(owned, [args.parent_attempt, args.attempt, args.provider])
    if parent == attempt or not provider.is_file():
        raise ValueError('distinct new attempt and regular new provider required')
    oldbuild, oldsource = parent / 'build-ft', parent / 'src/torch-npu'
    build, source = attempt / 'build-ft', owned(attempt / 'src/torch-npu')
    oldpackage, package = parent / 'packages-ft/torch_npu', attempt / 'packages-ft/torch_npu'
    if build.exists() or package.exists():
        raise ValueError('fresh build/package directories required')
    previous = json.loads((parent / 'build-evidence/result.json').read_text())
    for name, row in previous['outputs'].items():
        if sha(Path(name)) != row['sha256']:
            raise ValueError('completed parent output changed: ' + name)
    oldinputs_path = parent / 'build-evidence/inputs.json'
    oldinputs = json.loads(oldinputs_path.read_text())
    historical_source_pins = oldinputs.get('source_files')
    if (oldinputs.get('source') != str(oldsource) or
            not isinstance(historical_source_pins, dict) or not historical_source_pins):
        raise ValueError('completed parent full-build source manifest required')
    historical_inputs_sha = sha(oldinputs_path)
    os.sched_setaffinity(0, set(range(160, 184, 2)))
    build.mkdir()
    package.parent.mkdir()
    shutil.copytree(oldpackage, package, symlinks=True)
    if (package / 'lib').is_symlink():
        raise ValueError('physical package lib directory required')
    (package / 'lib/libtorch_npu.so').unlink()
    for path in package.glob('_C*.so'):
        path.unlink()
    for path in (source / 'torch_npu').rglob('*'):
        if path.is_file() and path.suffix in {'.py', '.pyi'}:
            dest = package / path.relative_to(source / 'torch_npu')
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.is_symlink():
                raise ValueError('physical Python package destination required')
            shutil.copyfile(path, dest)
    evidence = attempt / 'build-evidence'
    evidence.mkdir()
    shutil.copyfile(Path(__file__), evidence / 'rebind-driver-source.py')
    commands = []
    env = dict(os.environ, OMP_NUM_THREADS='1', TMPDIR=str(build))
    env.pop('PYTHONPATH', None)

    def run(argv, name):
        commands.append({'name': name, 'argv': [str(x) for x in argv], 'cwd': str(build)})
        (evidence / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        print(name, flush=True)
        with (evidence / (name + '.log')).open('w') as log:
            result = subprocess.run(argv, cwd=build, env=env, stdout=log, stderr=subprocess.STDOUT)
        print(name, 'exit', result.returncode, flush=True)
        result.check_returncode()

    def remap(word):
        return word.replace(str(parent), str(attempt))

    oldcommands = json.loads((parent / 'build-evidence/commands.json').read_text())
    configure = [remap(word) for word in oldcommands[0]['argv']]
    key = '-DTORCH_NPU_EAGER_PREPARE_HOST_SIM_LIBRARY='
    configure = [key + str(provider) if word.startswith(key) else word for word in configure]
    run(configure, 'configure')
    rawdeps = subprocess.check_output(['ninja', '-C', str(oldbuild), '-t', 'deps'], text=True)
    (evidence / 'parent-ninja-deps.txt').write_text(rawdeps)
    dependencies = {}
    current = None
    for line in rawdeps.splitlines():
        match = re.match(r'^(.*\.o): #deps \d+,', line)
        if match:
            current = match[1]
            dependencies[current] = []
        elif line.startswith('    ') and current:
            dep = Path(line.strip())
            dependencies[current].append((oldbuild / dep).resolve() if not dep.is_absolute() else dep)
        elif not line:
            current = None
    oldlink = link_argv(oldbuild)
    objects = [word for word in oldlink if word.endswith('.o')]
    if len(objects) != len(set(objects)) or not objects:
        raise ValueError('unexpected parent object list')
    affected = [name for name in objects if oldbuild / PROFILE in dependencies.get(name, [])]
    expected = 'CMakeFiles/torch_npu.dir/third_party/op-plugin/op_plugin/utils/op_api_common.cpp.o'
    if affected != [expected]:
        raise ValueError('unexpected profile closure: ' + repr(affected))
    oldrecipe_path, recipe_path = oldbuild / 'compile_commands.json', build / 'compile_commands.json'
    oldrecipes = {row['output']: row for row in json.loads(oldrecipe_path.read_text())}
    recipes = {row['output']: row for row in json.loads(recipe_path.read_text())}
    command_hashes = {}
    for name in objects:
        expected_argv = [remap(word) for word in shlex.split(oldrecipes[name]['command'])]
        actual_argv = shlex.split(recipes[name]['command'])
        if expected_argv != actual_argv:
            raise ValueError('effective compiler argv changed beyond path remap: ' + name)
        command_hashes[name] = hashlib.sha256(json.dumps(actual_argv, separators=(',', ':')).encode()).hexdigest()
    comparison = {'compared_objects': len(objects), 'reused_objects': len(objects) - len(affected),
                  'recompiled_objects': affected, 'rule': 'new argv equals token-wise parent-to-attempt remap',
                  'parent_compile_commands_sha256': sha(oldrecipe_path),
                  'new_compile_commands_sha256': sha(recipe_path), 'per_object_command_sha256': command_hashes}
    comparison['comparison_sha256'] = hashlib.sha256(
        json.dumps(command_hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    (evidence / 'compile-command-comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
    source_hashes, external_hashes, reused, rebuilt = {}, {}, {}, {}
    for name in objects:
        path = oldbuild / name
        if path.is_symlink() or not path.is_file() or name not in dependencies:
            raise ValueError('unproven parent object: ' + name)
        dest = build / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        origin_sha = sha(path)
        if name in affected:
            rebuilt[name] = {'parent_sha256': origin_sha}
            continue
        for dep in dependencies[name]:
            if dep.is_relative_to(oldsource):
                verify_source_dependency(dep, oldsource, source,
                                         historical_source_pins, source_hashes)
            elif dep.is_relative_to(oldbuild):
                raise ValueError('unhandled generated dependency of reused object: ' + str(dep))
            else:
                if str(dep) not in external_hashes:
                    external_hashes[str(dep)] = sha(dep)
        shutil.copyfile(path, dest)
        if sha(dest) != origin_sha:
            raise ValueError('physical object copy mismatch')
        reused[name] = {'origin': str(path), 'origin_sha256': origin_sha,
                        'copy': str(dest), 'sha256': origin_sha}
    provider_sha = sha(provider)
    sdk_hashes = {}
    for name, h in oldinputs['sdk_libraries'].items():
        target = Path(remap(name))
        if sha(Path(name)) != h or sha(target) != h:
            raise ValueError('SDK library changed from completed full build')
        sdk_hashes[str(target)] = h
    vendor_hashes = {}
    for name, h in oldinputs['vendor_archives'].items():
        target = Path(remap(name))
        if sha(Path(name)) != h or sha(target) != h:
            raise ValueError('vendor archive changed from completed full build')
        vendor_hashes[str(target)] = h
    link_hashes = {word: sha(Path(word)) for word in oldlink
                   if word.startswith('/') and word.endswith(('.so', '.a'))
                   and Path(word).is_file() and word != str(oldpackage / 'lib/libtorch_npu.so')}
    inputs = {'parent_attempt': str(parent), 'provider': str(provider),
              'provider_sha256': provider_sha, 'profile_closure': affected,
              'parent_full_inputs_sha256': historical_inputs_sha,
              'source_dependency_pin_basis': 'completed full-build inputs.json.source_files',
              'source_dependency_hashes': source_hashes, 'external_dependency_hashes': external_hashes,
              'sdk_libraries': sdk_hashes, 'vendor_archives': vendor_hashes,
              'parent_link_inputs': link_hashes,
              'reused_objects': reused, 'rebuilt_objects': rebuilt,
              'retained_profiler': {'path': str(package / 'lib/libnpu_profiler.so'),
                                    'sha256': sha(package / 'lib/libnpu_profiler.so')},
              'rebind_driver_sha256': sha(evidence / 'rebind-driver-source.py')}
    (evidence / 'inputs.json').write_text(json.dumps(inputs, indent=2) + '\n')
    for name in affected:
        row = recipes[name]
        run(shlex.split(row['command']), 'recompile-' + Path(name).name)
        rebuilt[name]['sha256'] = sha(build / name)
    newlink = link_argv(build)
    if [word for word in newlink if word.endswith('.o')] != objects:
        raise ValueError('new CMake target object list differs from completed parent')
    run(newlink, 'native-link')
    binding = next(row for row in oldcommands if row['name'] == 'python-bindings')['argv']
    run([remap(word) for word in binding], 'python-bindings')
    for name, row in reused.items():
        if sha(Path(row['origin'])) != row['origin_sha256'] or sha(build / name) != row['sha256']:
            raise ValueError('reused object changed during rebind')
    verify_source_postcheck(source_hashes, oldsource, source, historical_source_pins)
    for name, h in external_hashes.items():
        if sha(Path(name)) != h:
            raise ValueError('external dependency changed during rebind')
    for name, h in {**sdk_hashes, **vendor_hashes}.items():
        if sha(Path(name)) != h:
            raise ValueError('new SDK/vendor input changed during rebind')
    for name, h in link_hashes.items():
        if sha(Path(name)) != h or sha(Path(remap(name))) != h:
            raise ValueError('link dependency changed during rebind')
    if sha(provider) != provider_sha:
        raise ValueError('provider changed during rebind')
    outputs = [package / 'lib/libtorch_npu.so', *package.glob('_C*.so')]
    if len(outputs) != 2:
        raise ValueError('exactly one rebuilt _C required')
    identities = {str(path): {'sha256': sha(path), 'bytes': path.stat().st_size,
                              'elf_notes': subprocess.check_output(['readelf', '-n', str(path)], text=True)}
                  for path in outputs}
    result = {'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'outputs': identities, 'provider_sha256': provider_sha,
              'pythonpath': str(package.parent),
              'ld_library_path': ':'.join([str(package / 'lib'), str(provider.parent),
                  str(source / 'third_party/acl/libs'),
                  '/root/ft-pytorch-repro/envs/py314t/lib/python3.14t/site-packages/torch/lib']),
              'full_compile': False, 'parent_full_compile': str(parent),
              'reused_native_objects': len(reused), 'rebuilt_native_objects': rebuilt,
              'new_python_extension': True}
    (evidence / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
