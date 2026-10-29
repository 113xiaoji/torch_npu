"""Compile a small CPython extension using the installed compiler, no pip step."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import threading

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CORE = ROOT / "torch_npu/csrc/core/npu/EagerPrepareQueueCore.h"
ORDERED = ROOT / "torch_npu/csrc/core/npu/OrderedMpscQueueCore.h"
SOURCE = HERE / "native_bridge.cpp"
_LOAD_LOCK = threading.Lock()
_LOADED = None


def build_native() -> Path:
    dependencies = [CORE, ORDERED, SOURCE, Path(__file__)]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in dependencies}
    identity = {"sources": hashes, "python": sys.version, "cache_tag": sys.implementation.cache_tag}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    cache = HERE / ".cache" / digest
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / ("_eager_prepare_cpu" + sysconfig.get_config_var("EXT_SUFFIX"))
    if output.is_file():
        return output
    (cache / "build_info.h").write_text(
        "#pragma once\n"
        f"constexpr const char* kCoreSha256 = {json.dumps(hashes[str(CORE.relative_to(ROOT))])};\n"
        f"constexpr const char* kCorePath = {json.dumps(str(CORE))};\n",
        encoding="utf-8",
    )
    includes = [sysconfig.get_path("include"), str(ROOT), str(cache)]
    if os.name == "nt":
        developer = Path(os.environ.get("VSDEVCMD", r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"))
        if not developer.is_file():
            raise FileNotFoundError("Installed VS Build Tools not found; set VSDEVCMD to VsDevCmd.bat")
        library = Path(sys.base_prefix) / "libs"
        command = ["cl", "/nologo", "/std:c++17", "/O2", "/EHsc", "/W4", "/WX", "/LD", "/DNOMINMAX"]
        command += ["/I" + value for value in includes]
        command += [str(SOURCE), "/Fo" + str(cache / "native_bridge.obj"), "/link", "/LIBPATH:" + str(library),
                    f"python{sys.version_info.major}{sys.version_info.minor}.lib", "/OUT:" + str(output)]
        script = cache / "build.cmd"
        script.write_text("@echo off\n" + f'call "{developer}" -arch=x64 -host_arch=x64\n'
                          + "if errorlevel 1 exit /b %errorlevel%\n"
                          + subprocess.list2cmdline(command) + "\nexit /b %errorlevel%\n", encoding="utf-8")
        invocation = ["cmd", "/d", "/c", str(script)]
    else:
        compiler = os.environ.get("CXX") or shutil.which("c++")
        if not compiler:
            raise FileNotFoundError("An installed C++17 compiler is required")
        command = [compiler, "-std=c++17", "-O2", "-shared", "-fPIC", "-pthread", "-Wall", "-Wextra", "-Werror"]
        command += ["-I" + value for value in includes] + [str(SOURCE), "-o", str(output)]
        invocation = command
    (cache / "build_manifest.json").write_text(json.dumps({**identity, "command": command}, indent=2), encoding="utf-8")
    completed = subprocess.run(invocation, cwd=cache, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (cache / "compile.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"Native bridge build failed ({completed.returncode}); {cache / 'compile.log'}\n"
                           + (completed.stdout + completed.stderr)[-14000:])
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != hashes[str(path.relative_to(ROOT))] for path in dependencies):
        output.unlink(missing_ok=True)
        raise RuntimeError("Native bridge sources changed during compilation; rebuild from stable sources")
    return output


def load_native():
    global _LOADED
    with _LOAD_LOCK:
        if _LOADED is not None:
            current_sha = hashlib.sha256(CORE.read_bytes()).hexdigest()
            if _LOADED.core_info()["core_sha256"] != current_sha:
                raise RuntimeError("loaded binary compiled core SHA differs from the current source; restart Python and rebuild")
            return _LOADED
        output = build_native()
        spec = importlib.util.spec_from_file_location("_eager_prepare_cpu", output)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load native bridge: {output}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if module.core_info()["core_sha256"] != hashlib.sha256(CORE.read_bytes()).hexdigest():
            raise RuntimeError("loaded binary compiled core SHA differs from the current source; restart Python and rebuild")
        _LOADED = module
        return module


if __name__ == "__main__":
    print(build_native())
