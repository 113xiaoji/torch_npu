"""Report R26 entry guards in generated C++ (source evidence, not NPU qualification)."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ENTRY = re.compile(r'eager_parallel::Invoke\("([^"]+)",\s*"([^"]+)"')
PIPELINE = (
    'aten::_embedding_bag', 'aten::_embedding_bag_forward_only', 'aten::cat',
    'aten::view', 'aten::reshape', 'aten::transpose.int', 'aten::slice.Tensor',
    'aten::bmm', 'aten::index.Tensor', 'aten::addmm', 'aten::relu', 'aten::sigmoid',
    'aten::copy_', 'aten::_local_scalar_dense', 'aten::resize_', 'aten::set_.source_Storage',
)


def audit(paths):
    entries = []
    files = []
    for path in paths:
        raw = path.read_bytes()
        found = ENTRY.findall(raw.decode('utf-8'))
        files.append({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
                      'entry_guards': len(found)})
        entries.extend(found)
    schemas = sorted({schema for schema, _ in entries})
    return {
        'schema': 1,
        'scope': 'generated C++ source; no runtime or CANN qualification',
        'files': files,
        'guard_count_by_route': dict(sorted(Counter(route for _, route in entries).items())),
        'schemas': schemas,
        'pipeline_entry_observed': {schema: schema in schemas for schema in PIPELINE},
        'full_entry_coverage_qualified': False,
        'unproven_boundaries': [
            'direct op_api/acl_op C++ calls bypassing generated OpInterface',
            'handwritten PTA entries and custom registration outside these generated files',
            'PyTorch core direct data/storage mutation and tensor containers not emitted here',
            'allocation ownership, event publication, and device stream dependencies',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('generated_cpp', type=Path, nargs='+')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = json.dumps(audit(args.generated_cpp), indent=2) + '\n'
    if args.output:
        args.output.write_text(report, encoding='utf-8')
    else:
        print(report, end='')


if __name__ == '__main__':
    main()
