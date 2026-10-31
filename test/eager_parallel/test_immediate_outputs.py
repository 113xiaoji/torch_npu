"""Compile actual native capture/complete/fail bodies with fault-injected resources.

This proves output ownership and failure control flow, not ACL implementation.
The real adapter translation unit and SDK execution are validated on host-sim.
"""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


class ImmediateOutputLifetimeTests(unittest.TestCase):
    def test_all_outputs_survive_every_post_body_failure(self):
        source=(ROOT/'torch_npu/csrc/core/npu/eager_parallel/ImmediateRuntime.cpp').read_text()
        def function(signature):
            start=source.index(signature)
            brace=source.index('{',start)
            depth=1
            end=brace+1
            while depth:
                depth+=(source[end]=='{')-(source[end]=='}')
                end+=1
            return source[start:end]
        fixture=(ROOT/'test/cpp/eager_parallel/immediate_outputs_fixture.cpp.in').read_text()
        for marker,signature in (
            ('ADD_OUTPUT','void ImmediateGuard::add_output('),
            ('COMPLETE','void ImmediateGuard::complete('),
            ('FAIL','void ImmediateGuard::fail('),
        ):
            fixture=fixture.replace('/*'+marker+'*/',function(signature))
        spec=importlib.util.spec_from_file_location('r27_resource_compiler',
            ROOT/'test/host_regions/test_eager_prepare_integration.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(prefix='r27-immediate-outputs-') as directory:
            path=Path(directory)/'immediate_outputs.cpp'
            path.write_text(fixture)
            module.NativeOwnedJobTests().compile_run(path)


if __name__=='__main__':unittest.main()
