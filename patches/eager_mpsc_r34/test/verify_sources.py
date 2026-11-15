"""Read-only checks that the R34 overlay preserves the old SDK/inline contracts."""
from hashlib import sha256
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
UTILS = ROOT / 'patches/eager_mpsc_r34/op-plugin/op_plugin/utils'
OLD = ROOT / 'patches/eager_prepare_r25/op-plugin/op_plugin/utils'


def function(text, signature):
    start = text.index(signature)
    brace = text.index('{', start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


old_cpp = (OLD / 'op_api_common.cpp').read_text(encoding='utf-8')
new_cpp = (UTILS / 'op_api_common.cpp').read_text(encoding='utf-8')
for signature in ('const SdkProfile& VerifiedSdkProfile() noexcept',
                  'bool OwnedOpApiSdkContractAvailable() noexcept'):
    assert function(old_cpp, signature) == function(new_cpp, signature), signature

old_header = (OLD / 'op_api_eager_prepare_r24.h').read_text(encoding='utf-8')
new_header = (UTILS / 'op_api_eager_prepare_r24.h').read_text(encoding='utf-8')
for signature in ('inline bool EagerSdkQualified(const char* name) noexcept',
                  'bool EagerArgumentDomain(const char* name,const Tuple& args)',
                  'void EnqueueEagerArguments(const char* name,bool v1,Args&&... values)'):
    assert function(old_header, signature) == function(new_header, signature), signature
start, end = 'template<class Arguments>\nclass EagerOwnedCommand', '\ntemplate<class Tuple,std::size_t... Index>'
assert old_header[old_header.index(start):old_header.index(end)] == new_header[new_header.index(start):new_header.index(end)]

old_public = subprocess.check_output(
    ['git', 'show', 'c9890d2:torch_npu/csrc/core/npu/EagerPrepare.h'], cwd=ROOT, text=True)
new_public = (ROOT / 'torch_npu/csrc/core/npu/EagerPrepare.h').read_text(encoding='utf-8')
signature = 'inline bool SdkContractAvailable(const char* operation) noexcept'
assert function(old_public, signature) == function(new_public, signature)

common = (UTILS / 'op_api_common.h').read_bytes()
restored = common.replace(b'HDB_EAGER_PREPARE_R34(aclnn_api,', b'HDB_EAGER_PREPARE_R24(aclnn_api,')
restored = restored.replace(b'hdb_opapi::EagerSdkQualifiedR34(#aclnn_api)', b'hdb_opapi::EagerSdkQualified(#aclnn_api)')
assert sha256(restored).hexdigest() == '306fe9ed42c4acaf921ee26f14a6ba399b64ce97f58577fd1ff525cae2376c95'
assert sha256((UTILS / 'op_api_common_base.h').read_bytes()).hexdigest() == \
    '120cf8bfc022ee5170f4fd752a9099410fbb84f6765df5c3b3c819556c91504a'
print('PASS unchanged 920/R22/legacy-inline bodies and original R32 macro/base sources')
