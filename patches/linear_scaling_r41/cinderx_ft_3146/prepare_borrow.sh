#!/bin/bash
set -euo pipefail
exec > >(tee /work/logs/prepare-borrow.log) 2>&1
mkdir -p /tools/cpython3146 /tools/borrowtool
curl -fL --connect-timeout 15 --max-time 120 https://codeload.github.com/python/cpython/tar.gz/refs/tags/v3.14.6 -o /tools/downloads/cpython3146.tar.gz
sha256sum /tools/downloads/cpython3146.tar.gz
printf '%s\n' '7f77ccf3613ddadc74dfb8d6f8082581a8835115a25e1307d189f03aa750893e  /tools/downloads/cpython3146.tar.gz' | sha256sum -c -
tar -xzf /tools/downloads/cpython3146.tar.gz -C /tools/cpython3146 --strip-components=1
PYTHONDONTWRITEBYTECODE=1 python3 -m pip install --disable-pip-version-check --no-cache-dir --target /tools/borrowtool 'clang==14.0.6' 'click==8.1.8'
ls /usr/lib/llvm-14/lib/libclang.so*
printf 'BORROW_TOOLS_READY\n'
