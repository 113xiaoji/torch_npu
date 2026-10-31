"""Execute the production destructor with an allocator bookkeeping callback."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


class ImmediateDestructorTests(unittest.TestCase):
    def test_last_tensor_references_release_inside_internal_scope(self):
        source=(ROOT/'torch_npu/csrc/core/npu/eager_parallel/ImmediateRuntime.cpp').read_text()
        start=source.index('ImmediateGuard::~ImmediateGuard() noexcept')
        brace=source.index('{',start)
        depth,end=1,brace+1
        while depth:
            depth+=(source[end]=='{')-(source[end]=='}')
            end+=1
        fixture=(ROOT/'test/cpp/eager_parallel/immediate_destructor_fixture.cpp.in').read_text()
        fixture=fixture.replace('/*DESTRUCTOR*/',source[start:end])
        spec=importlib.util.spec_from_file_location('r27_destructor_compiler',
            ROOT/'test/host_regions/test_eager_prepare_integration.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(prefix='r27-immediate-destructor-') as directory:
            path=Path(directory)/'immediate_destructor.cpp';path.write_text(fixture)
            module.NativeOwnedJobTests().compile_run(path)


if __name__=='__main__':unittest.main()
