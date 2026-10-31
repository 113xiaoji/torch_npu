"""One unchanged original DLRM forward through the numerical SDK."""
import argparse
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--upstream',type=Path,required=True);parser.add_argument('--sdk',type=Path,required=True)
    parser.add_argument('--deps',type=Path,required=True);args=parser.parse_args()
    sys.path.append(str(args.deps))
    import torch
    import torch_npu
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    runtime=ctypes.CDLL(str(args.sdk/'libr27_hostsim_runtime.so'))
    opapi=ctypes.CDLL(str(args.sdk/'libopapi.so'))
    for lib,name in [(runtime,'R27HostSimDiagnosticsJson'),(runtime,'R27HostSimLastError'),(opapi,'R27HostSimOpDiagnosticsJson')]:
        getattr(lib,name).restype=ctypes.c_char_p
    try:
        spec=importlib.util.spec_from_file_location('r27_original_reference',args.reference)
        reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
        model,inputs,provenance=reference.load_original_model(args.upstream)
        with torch.inference_mode():
            expected=model(*inputs)
            model=model.to('npu:0')
            inputs=(inputs[0].to('npu:0'),[v.to('npu:0') for v in inputs[1]],[v.to('npu:0') for v in inputs[2]])
            torch.npu.synchronize()
            actual=model(*inputs)
            torch.npu.synchronize()
            actual=actual.cpu()
            error=reference.error_record(actual,expected,rtol=1e-4,atol=1e-5)
        print(json.dumps(dict(passed=error['passed'],error=error,source=provenance,
            runtime=json.loads(runtime.R27HostSimDiagnosticsJson()),opapi=json.loads(opapi.R27HostSimOpDiagnosticsJson()),
            maps=[line for line in Path('/proc/self/maps').read_text().splitlines() if any(name in line for name in ('libopapi.so','libascendcl.so','libacl_op_compiler.so'))],gil_enabled=sys._is_gil_enabled())))
        return 0 if error['passed'] else 1
    except BaseException:
        traceback.print_exc()
        print(runtime.R27HostSimLastError().decode(),flush=True)
        print(runtime.R27HostSimDiagnosticsJson().decode(),flush=True)
        print(opapi.R27HostSimOpDiagnosticsJson().decode(),flush=True)
        return 1


if __name__=='__main__':raise SystemExit(main())
