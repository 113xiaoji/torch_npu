"""Generate target-version borrows with the repository TemplateFileProcessor.

The repository's clang_parser imports Meta-only python.clang_parser. This
qualification driver supplies its small FileParser contract with libclang and
explicit selected-runtime flags. It does not replace production functions by
hand or silently regenerate unrelated, locally patched borrows.
"""
from pathlib import Path
from types import ModuleType, SimpleNamespace
import difflib
import importlib.util
import json
import os
import sys

sys.path.insert(0,'/tools/borrowtool')
from clang.cindex import Config, CursorKind, Diagnostic, Index
Config.set_library_file('/usr/lib/llvm-14/lib/libclang.so.1')
root=Path('/work/src')
directory=root/'cinderx/UpstreamBorrow'
upstream=Path('/tools/cpython3146')
include='/root/ft-pytorch-repro/prefixes/py314t/include/python3.14t'
class FileParser:
    def __init__(self, compile_commands):
        self.index=Index.create()
        self.cache={}
    def parse(self, source_file):
        if source_file not in self.cache:
            path=upstream/source_file
            arguments=['-x','c','-std=gnu11','-DPy_BUILD_CORE','-DPy_GIL_DISABLED=1','-DNDEBUG',
                '-I'+include,'-I'+include+'/internal','-I'+str(upstream/'Include'),
                '-I'+str(upstream/'Include/internal')]
            tu=self.index.parse(str(path),args=arguments)
            diagnostics=[str(d) for d in tu.diagnostics]
            print(json.dumps({'source':str(path),'arguments':arguments,'diagnostics':diagnostics}),flush=True)
            assert not any(d.severity>=Diagnostic.Error for d in tu.diagnostics)
            self.cache[source_file]=SimpleNamespace(source_file=str(path),translation_unit=tu,lines=path.read_text().split('\n'))
        return self.cache[source_file]

package=ModuleType('r41borrow'); package.__path__=[str(directory)]
sys.modules['r41borrow']=package
adapter=ModuleType('r41borrow.clang_parser')
adapter.FileParser=FileParser; adapter.ParsedFile=SimpleNamespace
sys.modules[adapter.__name__]=adapter
spec=importlib.util.spec_from_file_location('r41borrow.UpstreamBorrow',directory/'UpstreamBorrow.py')
module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=module
spec.loader.exec_module(module)

stem='borrowed-3.14.6.free-threading-instrumentation'
fragment_template=directory/(stem+'.c.template')
template_contents='''// Copyright (c) Meta Platforms, Inc. and affiliates.
// Source: CPython v3.14.6, Python/instrumentation.c.
// Reproduce with /work/adapt_ft_borrows.py and the recorded source/tool manifest.
// @Borrow function _Py_Instrumentation_GetLine from Python/instrumentation.c [3.14.6]
// @Borrow function _Py_call_instrumentation_line from Python/instrumentation.c [3.14.6]
'''
if '--generate-only' in sys.argv:
    assert fragment_template.read_text()==template_contents
else:
    fragment_template.write_text(template_contents)
fragment=directory/(stem+'.gen_cached.h')
output_fragment=Path(os.environ.get('R41_BORROW_OUTPUT',str(fragment)))
module.TemplateFileProcessor(str(fragment_template),str(output_fragment),FileParser(None),'3.14.6').process()
if '--generate-only' in sys.argv:
    assert output_fragment.read_bytes()==fragment.read_bytes(), 'fresh borrow replay differs'
    print('BORROW_REPLAY_IDENTICAL',flush=True)
    raise SystemExit(0)

changes={}
def update(path,new):
    old=path.read_text()
    backup=Path('/work/baseline')/path.relative_to(root)
    backup.parent.mkdir(parents=True,exist_ok=True)
    backup.write_text(old)
    path.write_text(new)
    changes[str(path.relative_to(root))]=(old,new)

template=directory/'borrowed-3.14.free-threading.c.template'
text=template.read_text()
pair='''// @Borrow function _Py_Instrumentation_GetLine from Python/instrumentation.c [3.14]
// @Borrow function _Py_call_instrumentation_line from Python/instrumentation.c [3.14]'''
assert text.count(pair)==1
guard='''#if PY_VERSION_HEX >= 0x030E06F0
#include "cinderx/UpstreamBorrow/borrowed-3.14.6.free-threading-instrumentation.gen_cached.h"
#else
'''
text=text.replace(pair,guard+pair+'\n#endif')
debug_pair='''// @Borrow optional function reftotal_add from Objects/object.c
// @Borrow optional function _Py_AddRefTotal from Objects/object.c'''
assert text.count(debug_pair)==1
text=text.replace(debug_pair,'#ifdef Py_REF_DEBUG\n'+debug_pair+'\n#endif')
update(template,text)

cached=directory/'borrowed-3.14.free-threading.gen_cached.c'
text=cached.read_text()
start=text.index('int\n_Py_Instrumentation_GetLine(')
end=text.index('Py_NO_INLINE int\n_Py_call_instrumentation_instruction(',start)
text=text[:start]+guard+text[start:end]+'#endif\n'+text[end:]
start=text.index('static inline void\nreftotal_add(')
end=text.index('Py_ssize_t\n_Py_ExplicitMergeRefcount(',start)
text=text[:start]+'#ifdef Py_REF_DEBUG\n'+text[start:end]+'#endif\n'+text[end:]
update(cached,text)

out=[]
for name,(old,new) in changes.items():
    out.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
for path in (fragment_template,fragment):
    name=str(path.relative_to(root))
    out.extend(difflib.unified_diff([],path.read_text().splitlines(True),fromfile='/dev/null',tofile='b/'+name))
Path('/work/ft-borrows-3146.patch').write_text(''.join(out))
print('BORROW_ADAPTATION_DONE',flush=True)
