#!/bin/bash
set -euo pipefail
exec > >(tee /work/logs/build-clang18.log) 2>&1
export LD_LIBRARY_PATH=/tools/clang18/usr/lib/aarch64-linux-gnu:/tools/clang18/usr/lib/llvm-18/lib:/tools/gcc13/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONDONTWRITEBYTECODE=1 CINDERX_PLUGIN_ENABLE=0
export PYTHONPATH=/root/ft-pytorch-repro/envs/py314t/lib/python3.14t/site-packages
unset PYTHON_GIL PYTHONJITAUTO PYTHONJITDISABLE CINDERX_DISABLE
cd /work/src
mapfile -t feature_args < <(/work/ft-python -B - <<'PY'
import sys
from ci_pipeline.cmake_options import cmake_feature_options
for name,value in cmake_feature_options(python_root=sys.base_prefix).items():
    print(f'-D{name}={value}')
PY
)
cmake -S /work/src -B /tools/cinderx-clang18-build \
  "${feature_args[@]}" \
  -DPython_EXECUTABLE=/work/ft-python \
  -DPython_INCLUDE_DIR=/root/ft-pytorch-repro/prefixes/py314t/include/python3.14t \
  -DCMAKE_C_COMPILER=/tools/clang18/usr/bin/clang-18 \
  -DCMAKE_CXX_COMPILER=/tools/clang18/usr/bin/clang++-18 \
  -DCMAKE_C_FLAGS=--gcc-toolchain=/tools/gcc13/usr \
  -DCMAKE_CXX_FLAGS=--gcc-toolchain=/tools/gcc13/usr \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DFETCHCONTENT_SOURCE_DIR_FMT=/tools/deps/fmt \
  -DFETCHCONTENT_SOURCE_DIR_PARALLEL-HASHMAP=/tools/deps/parallel-hashmap \
  -DFETCHCONTENT_SOURCE_DIR_USDT=/tools/deps/usdt \
  -DFETCHCONTENT_SOURCE_DIR_CAPSTONE=/tools/deps/capstone \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=/tools/cinderx-clang18-pylib
cp /tools/cinderx-clang18-build/CMakeCache.txt /work/logs/CMakeCache-clang18-ft.txt
cp /tools/cinderx-clang18-build/compile_commands.json /work/logs/compile_commands-clang18-ft.json
mkdir -p /tools/cinderx-clang18-pylib
cp -a /tools/cinderx-pylib/cinderx /tools/cinderx-clang18-pylib/
cmake --build /tools/cinderx-clang18-build --parallel 8
sha256sum /tools/cinderx-clang18-pylib/_cinderx.so
printf 'CLANG18_FT_BUILD_DONE\n'
