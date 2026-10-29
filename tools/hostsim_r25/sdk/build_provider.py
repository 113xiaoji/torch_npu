"""Build isolated libopapi provider and native contracts without Torch/CANN."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = HERE / "r25_hostsim_opapi.cpp"
HEADER = HERE / "r25_hostsim_api.h"
TEST = ROOT / "test/hostsim_r25_sdk/native_contract.cpp"


def build_provider(output_dir: Path | None = None, *, native_test: bool = False) -> Path:
    dependencies = [SOURCE, HEADER, Path(__file__)] + ([TEST] if native_test else [])
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in dependencies}
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:20]
    output_dir = Path(output_dir) if output_dir else HERE / ".cache" / identity
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / ("opapi.dll" if os.name == "nt" else "libopapi.so")
    test_output = output_dir / ("native_contract.exe" if os.name == "nt" else "native_contract")
    manifest_path = output_dir / "build_manifest.json"
    if output.is_file() and manifest_path.is_file() and (not native_test or test_output.is_file()):
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved.get("sources") == hashes and saved.get("binary_sha256") == hashlib.sha256(output.read_bytes()).hexdigest():
            return output
    commands = []
    if os.name == "nt":
        developer = Path(os.environ.get("VSDEVCMD", r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"))
        if not developer.is_file():
            raise FileNotFoundError("VS Build Tools required; set VSDEVCMD")
        common = ["cl", "/nologo", "/std:c++17", "/O2", "/EHsc", "/W4", "/WX", "/DNOMINMAX", "/I" + str(HERE)]
        commands.append(common + ["/LD", str(SOURCE), "/Fo" + str(output_dir / "provider.obj"),
            "/link", "/OUT:" + str(output), "/IMPLIB:" + str(output_dir / "opapi.lib")])
        if native_test:
            commands.append(common + [str(TEST), "/Fo" + str(output_dir / "test.obj"),
                "/link", str(output_dir / "opapi.lib"), "/OUT:" + str(test_output)])
        script = output_dir / "build.cmd"
        script.write_text("@echo off\n" + f'call "{developer}" -arch=x64 -host_arch=x64\n'
            + "if errorlevel 1 exit /b %errorlevel%\n"
            + "\n".join(subprocess.list2cmdline(command) + "\nif errorlevel 1 exit /b %errorlevel%" for command in commands)
            + "\nexit /b 0\n", encoding="utf-8")
        invocations = [["cmd", "/d", "/c", str(script)]]
    else:
        compiler = os.environ.get("CXX") or shutil.which("c++")
        if not compiler:
            raise FileNotFoundError("C++17 compiler required")
        common = [compiler, "-std=c++17", "-O2", "-pthread", "-Wall", "-Wextra", "-Werror", "-I" + str(HERE)]
        commands.append(common + ["-shared", "-fPIC", str(SOURCE), "-o", str(output)])
        if native_test:
            commands.append(common + [str(TEST), "-L" + str(output_dir), "-lopapi", "-Wl,-rpath,$ORIGIN", "-o", str(test_output)])
        invocations = commands
    logs = []
    for invocation in invocations:
        completed = subprocess.run(invocation, cwd=output_dir, capture_output=True, text=True, encoding="utf-8", errors="replace")
        logs.append(completed.stdout + completed.stderr)
        (output_dir / "compile.log").write_text("\n".join(logs), encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(f"Provider build failed; {output_dir / 'compile.log'}\n" + logs[-1][-16000:])
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != hashes[str(path.relative_to(ROOT))] for path in dependencies):
        raise RuntimeError("Provider source changed during build; rebuild from stable source")
    (output_dir / "build_manifest.json").write_text(json.dumps({
        "backend_kind": "HOST_SIM_R25", "abi": 1, "numeric_supported": False,
        "real_sdk_qualified": False, "sources": hashes, "commands": commands,
        "binary_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    library = build_provider(args.output_dir, native_test=args.test)
    print(library)
    if args.test:
        executable = library.parent / ("native_contract.exe" if os.name == "nt" else "native_contract")
        subprocess.run([str(executable)], cwd=library.parent, check=True)
