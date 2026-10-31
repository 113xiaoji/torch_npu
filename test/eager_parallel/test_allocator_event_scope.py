"""Execute actual allocator event call sites and native Boundary/scope bodies."""
import importlib.util
from pathlib import Path
import re
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


def generated_fixture():
    runtime=(ROOT/'torch_npu/csrc/core/npu/eager_parallel/ImmediateRuntime.cpp').read_text()
    def function(signature):
        start=runtime.index(signature);brace=runtime.index('{',start);depth,end=1,brace+1
        while depth:
            depth+=(runtime[end]=='{')-(runtime[end]=='}');end+=1
        return runtime[start:end]
    fixture=(ROOT/'test/cpp/eager_parallel/allocator_event_scope_fixture.cpp.in').read_text()
    for marker,signature in [('SCOPE_CONSTRUCTOR','AllocatorEventScope::AllocatorEventScope() noexcept'),
                             ('SCOPE_DESTRUCTOR','AllocatorEventScope::~AllocatorEventScope() noexcept')]:
        # An absent scope is represented as inert solely to execute the old
        # unguarded call sites and observe the regression before implementation.
        body=function(signature) if signature in runtime else signature+' {}'
        fixture=fixture.replace('/*'+marker+'*/',body)
    fixture=fixture.replace('/*BOUNDARY*/',function('void ImmediateBoundary('))
    wrappers=[]
    names=[]
    for name in ('NPUCachingAllocator.cpp','CachingHostAllocator.cpp'):
        text=(ROOT/'torch_npu/csrc/core/npu'/name).read_text(encoding='utf-8')
        scopes=list(re.finditer(r'\{\s*c10_npu::eager_parallel::AllocatorEventScope \w+;\s*(?:event->record\(stream\);|isEventCompleted = event->query\(\);)\s*\}',text))
        for match in re.finditer(r'event->(record\(stream\)|query\(\))',text):
            query=match.group(1).startswith('query')
            enclosing=next((scope for scope in scopes if scope.start()<=match.start()<scope.end()),None)
            snippet=enclosing.group() if enclosing else ('isEventCompleted = event->query();' if query else 'event->record(stream);')
            function_name='site_'+str(len(names));names.append(function_name)
            prelude='bool isEventCompleted=false;' if query else 'int stream=1;'
            tail='assert(isEventCompleted);' if query else ''
            wrappers.append('void '+function_name+'(FakeEvent* event) {'+prelude+snippet+tail+'}')
    if len(names)!=7:raise AssertionError('allocator event coverage changed: '+str(len(names)))
    wrappers.append('using Operation=void(*)(FakeEvent*); Operation operations[]={'+','.join(names)+'};')
    return fixture.replace('/*CALL_SITES*/','\n'.join(wrappers))


class AllocatorEventScopeTests(unittest.TestCase):
    def test_allocator_events_do_not_reenter_failed_domain(self):
        spec=importlib.util.spec_from_file_location('r27_allocator_scope_compiler',ROOT/'test/host_regions/test_eager_prepare_integration.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(prefix='r27-allocator-event-scope-') as directory:
            path=Path(directory)/'allocator_event_scope.cpp';path.write_text(generated_fixture())
            module.NativeOwnedJobTests().compile_run(path)


if __name__=='__main__':unittest.main()
