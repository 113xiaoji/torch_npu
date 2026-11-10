"""Execute production lean caller methods with deterministic backend faults."""
import importlib.util
import os
import shutil
import subprocess
from pathlib import Path
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[2]

def generated_fixture():
    source=(ROOT/'torch_npu/csrc/core/npu/eager_parallel/ImmediateRuntime.cpp').read_text()
    methods=[]
    for signature in ('bool ImmediateCallerGuard::begin(', 'void ImmediateCallerGuard::leave(',
        'ImmediateCallerGuard::~ImmediateCallerGuard()', 'void ImmediateCallerGuard::complete(',
        'void ImmediateCallerGuard::add_failed_output(', 'void ImmediateCallerGuard::fail('):
        start=source.index(signature);brace=source.index('{',start);depth=1;end=brace+1
        while depth:
            depth+=(source[end]=='{')-(source[end]=='}');end+=1
        methods.append(source[start:end])
    return (ROOT/'test/cpp/eager_parallel/immediate_selective_fixture.cpp.in').read_text().replace('/*METHODS*/','\n'.join(methods))

class SelectiveNativeTests(unittest.TestCase):
    def test_lean_caller_guard_dependency_metadata_and_failure_paths(self):
        spec=importlib.util.spec_from_file_location('r31_compiler',ROOT/'test/host_regions/test_eager_prepare_integration.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(prefix='r31-selective-') as directory:
            source=Path(directory)/'selective.cpp';source.write_text(generated_fixture())
            module.NativeOwnedJobTests().compile_run(source)
    def test_actual_invoke_failure_lifetime_and_zero_happy_output_visits(self):
        import torch
        torch_root=Path(torch.__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix='r31-invoke-') as directory:
            build=Path(directory);source=ROOT/'test/cpp/eager_parallel/immediate_invoke_lifetime_test.cpp'
            includes=[torch_root/'include',torch_root/'include/torch/csrc/api/include']
            if os.name=='nt':
                vc=Path(r'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat')
                if not vc.is_file():self.skipTest('MSVC unavailable')
                executable=build/'invoke.exe';script=build/'build.cmd'
                script.write_text('@echo off\ncall "'+str(vc)+'" -arch=x64 -host_arch=x64 >nul\n'
                    'if errorlevel 1 exit /b 1\ncl /nologo /std:c++17 /EHsc /MD /W3 /WX /I"'+str(ROOT)+'" '+
                    ' '.join('/external:I"'+str(value)+'"' for value in includes)+
                    ' /external:W0 "'+str(source)+'" /Fe:"'+str(executable)+'" /link /LIBPATH:"'+str(torch_root/'lib')+'" torch.lib torch_cpu.lib c10.lib\n')
                command=['cmd','/d','/c',str(script)]
                environment=dict(os.environ,PATH=str(torch_root/'lib')+os.pathsep+os.environ['PATH'])
            else:
                compiler=shutil.which('c++')
                if not compiler:self.skipTest('C++ compiler unavailable')
                executable=build/'invoke'
                command=[compiler,'-std=c++17','-O1','-pthread','-Wall','-Wextra','-Werror',
                    '-D_GLIBCXX_USE_CXX11_ABI='+str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),
                    '-I'+str(ROOT),*[option for value in includes for option in ('-isystem',str(value))],
                    str(source),'-L'+str(torch_root/'lib'),'-ltorch','-ltorch_cpu','-lc10','-o',str(executable)]
                environment=dict(os.environ,LD_LIBRARY_PATH=str(torch_root/'lib')+os.pathsep+os.environ.get('LD_LIBRARY_PATH',''))
            compiled=subprocess.run(command,cwd=build,capture_output=True,timeout=120)
            self.assertEqual(compiled.returncode,0,(compiled.stdout+compiled.stderr).decode(errors='replace'))
            ran=subprocess.run([str(executable)],cwd=build,env=environment,capture_output=True,timeout=30)
            self.assertEqual(ran.returncode,0,(ran.stdout+ran.stderr).decode(errors='replace'))
            self.assertIn(b'zero successful caller output visits',ran.stdout)

if __name__=='__main__':unittest.main()
