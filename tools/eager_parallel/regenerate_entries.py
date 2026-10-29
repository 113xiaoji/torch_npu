"""Regenerate native entry sources in an explicitly isolated R26 Linux checkout.

This invokes code generators directly and never fetches, resets submodules,
installs packages, removes ACL source trees, or changes the original model.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--evidence', required=True, type=Path)
    args = parser.parse_args()
    owned = Path('/home/r26-pta-joint-20261029').resolve(strict=True)
    source = args.source.resolve(strict=True)
    evidence = args.evidence.resolve()
    if not source.is_relative_to(owned) or not evidence.is_relative_to(owned):
        raise ValueError('source and evidence must stay in the isolated R26 namespace')
    evidence.mkdir(parents=True, exist_ok=False)
    plugin = source / 'third_party/op-plugin'
    config = plugin / 'op_plugin/config/v2r13'
    config.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTORCH_VERSION='2.13.0', PYTHONUTF8='1', OMP_NUM_THREADS='1')
    env.pop('PYTHONPATH', None)
    commands = [
        ('op-plugin-config', plugin, ['torchnpugen.gen_op_plugin_functions', '--version=2.13.0',
            '--output_dir=' + str(config), '--source_yaml=' + str(plugin / 'op_plugin/config/op_plugin_functions.yaml')]),
        ('op-plugin-interface', plugin, ['torchnpugen.gen_op_backend', '--version=2.13.0',
            '--output_dir=' + str(plugin / 'op_plugin'), '--source_yaml=' + str(config / 'op_plugin_functions.yaml'),
            '--deprecate_yaml=' + str(plugin / 'op_plugin/config/deprecated.yaml')]),
        ('op-plugin-structured', plugin, ['torchnpugen.struct.gen_struct_opapi',
            '--output_dir=' + str(plugin / 'op_plugin/ops/opapi'),
            '--native_yaml=' + str(config / 'op_plugin_functions.yaml'),
            '--struct_yaml=' + str(plugin / 'op_plugin/config/op_plugin_functions.yaml')]),
        ('pta-dispatch', source, ['torchnpugen.gen_backend_stubs',
            '--output_dir=' + str(source / 'torch_npu/csrc/aten'),
            '--source_yaml=' + str(source / 'torch_npu/csrc/aten/npu_native_functions.yaml'),
            '--impl_path=' + str(source / 'torch_npu/csrc/aten'),
            '--op_plugin_impl_path=' + str(plugin / 'op_plugin/ops'),
            '--op_plugin_yaml_path=' + str(config / 'op_plugin_functions.yaml'), '--pytorch_version_dir=v2r13']),
        ('pta-autograd', source, ['torchnpugen.autograd.gen_autograd',
            '--out_dir=' + str(source / 'torch_npu/csrc/aten'),
            '--autograd_dir=' + str(source / 'torchnpugen/autograd'),
            '--npu_native_function_dir=' + str(source / 'torch_npu/csrc/aten/npu_native_functions.yaml')]),
    ]
    receipt = []
    for name, cwd, arguments in commands:
        command = [sys.executable, '-m', *arguments]
        record = {'name': name, 'cwd': str(cwd), 'argv': command}
        receipt.append(record)
        (evidence / 'commands.json').write_text(json.dumps(receipt, indent=2) + '\n')
        with (evidence / (name + '.log')).open('w') as log:
            result = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        record['returncode'] = result.returncode
        (evidence / 'commands.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(name, 'exit', result.returncode, flush=True)
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, command)


if __name__ == '__main__':
    main()
