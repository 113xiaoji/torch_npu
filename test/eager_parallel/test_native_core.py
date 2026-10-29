"""Compile and execute the real SDK-independent R26 scheduler, never import torch."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class NativeCoreTests(unittest.TestCase):
    def test_native_protocol(self):
        self.compile_run("dependency_scheduler_test.cpp", b"PASS R26 native protocol")

    def test_native_feature_pipeline(self):
        self.compile_run("feature_pipeline_test.cpp", b"PASS R26 protocol-driven feature pipeline")

    def compile_run(self, filename, expected):
        header = ROOT / "torch_npu/csrc/core/npu/eager_parallel/DependencyScheduler.h"
        self.assertTrue(header.is_file(), "R26 native dependency/completion core is missing")
        source = ROOT / "test/cpp/eager_parallel" / filename
        self.assertTrue(source.is_file(), "R26 executable native scenario is missing: " + filename)
        with tempfile.TemporaryDirectory(prefix="r26-native-core-") as folder:
            build = Path(folder)
            executable = build / ("core.exe" if os.name == "nt" else "core")
            if os.name == "nt":
                vc = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat")
                if not vc.is_file():
                    self.skipTest("MSVC C++ toolchain unavailable")
                script = build / "compile.cmd"
                script.write_text(
                    '@echo off\ncall "' + str(vc) + '" -arch=x64 -host_arch=x64 >nul\n'
                    'if errorlevel 1 exit /b 1\n'
                    'cl /nologo /std:c++17 /EHsc /W4 /WX /I"' + str(ROOT) +
                    '" "' + str(source) + '" /Fe:"' + str(executable) + '"\n', encoding="utf-8")
                command = ["cmd", "/d", "/c", str(script)]
            else:
                compiler = shutil.which("c++")
                if compiler is None:
                    self.skipTest("C++ compiler unavailable")
                command = [compiler, "-std=c++17", "-pthread", "-Wall", "-Wextra", "-Werror",
                           "-I" + str(ROOT), str(source), "-o", str(executable)]
            result = subprocess.run(command, cwd=build, capture_output=True, timeout=90)
            self.assertEqual(result.returncode, 0, (result.stdout + result.stderr).decode(errors="replace"))
            result = subprocess.run([str(executable)], cwd=build, capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 0, (result.stdout + result.stderr).decode(errors="replace"))
            self.assertIn(expected, result.stdout)


if __name__ == "__main__":
    unittest.main()
