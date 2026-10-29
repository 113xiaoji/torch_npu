"""Check, or explicitly apply, the separate R24 eager-Prepare overlay to its pinned submodule.

No fetch, checkout, index update, commit, SDK or build is performed. All eight
sources/destinations are checked before writing. Run with repository edits and
builds idle; an I/O failure stops rather than repairing a partial overlay.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile

BASE_COMMIT = '312921458e7d396c54a1e92e90385d7d5f02dede'
SUBMODULE = 'third_party/op-plugin'
MANIFEST = 'patches/eager_prepare_r24/op-plugin-manifest.json'
OVERLAY_SHA256 = {'op_plugin/utils/op_api_common.cpp': '1019fa662f67484855f27f8cdcf665962564551f23b52066825f572d8eaba4b1',
 'op_plugin/utils/op_api_common.h': '306fe9ed42c4acaf921ee26f14a6ba399b64ce97f58577fd1ff525cae2376c95',
 'op_plugin/utils/op_api_common_base.h': '120cf8bfc022ee5170f4fd752a9099410fbb84f6765df5c3b3c819556c91504a',
 'op_plugin/utils/op_api_eager_prepare_r24.h': '94302b23bd9e405771fe380328668d0896be10efd7762bc7961445c6939011a9',
 'op_plugin/utils/op_api_owned_pack_r22.h': 'dfaea9ea90a7bbf8caebdf8754712b62b5e474d5c3dbaa6d6493a4ac260d033b',
 'op_plugin/utils/op_api_owned_payload_r22.h': '5eff67885603249f92914c1b0745d8b14e50322ff6ad51d8dd4d2d259b1ac507',
 'op_plugin/utils/op_api_owned_r22.h': '5a1c7d166f3087aff1f56542e7a9b0d53155c14811584df0e806012ac6d58444',
 'op_plugin/utils/op_api_sdk_profile_r22.h': '77e00c02e2ba0e7048a1b0674bccd989c534122b21c181dc58c8b470d9e98f55'}
PARENT_OVERLAY_SHA256 = {'op_plugin/utils/op_api_common.cpp': 'fd79118ee7f97f8cd6297e8ad8e6a2a8cb70fccffd09826fe525c3df7a8f16eb',
 'op_plugin/utils/op_api_common.h': 'd2b77155457632e1547e461d5bb1043fc165c0ce6819c2a8577e3adf4bffa169',
 'op_plugin/utils/op_api_common_base.h': '4f7512633d3bdf287d83759b1cc826f4be8031f6804dedadf1cd6204c0606f89',
 'op_plugin/utils/op_api_owned_pack_r22.h': 'dfaea9ea90a7bbf8caebdf8754712b62b5e474d5c3dbaa6d6493a4ac260d033b',
 'op_plugin/utils/op_api_owned_payload_r22.h': '5eff67885603249f92914c1b0745d8b14e50322ff6ad51d8dd4d2d259b1ac507',
 'op_plugin/utils/op_api_owned_r22.h': '5a1c7d166f3087aff1f56542e7a9b0d53155c14811584df0e806012ac6d58444',
 'op_plugin/utils/op_api_sdk_profile_r22.h': '77e00c02e2ba0e7048a1b0674bccd989c534122b21c181dc58c8b470d9e98f55'}
EXISTING = frozenset(('op_plugin/utils/op_api_common.cpp', 'op_plugin/utils/op_api_common.h',
                      'op_plugin/utils/op_api_common_base.h'))
MAX_FILE = 4 * 1024 * 1024


class OverlayError(ValueError):pass


def require(value, message):
    if not value:raise OverlayError(message)


def digest(raw):return hashlib.sha256(raw).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate manifest key: ' + key)
        result[key] = value
    return result


def git(repo, *arguments):
    environment = {key:value for key,value in os.environ.items() if not key.startswith('GIT_')}
    environment.update(GIT_TERMINAL_PROMPT='0', GIT_OPTIONAL_LOCKS='0', LC_ALL='C')
    try:
        result = subprocess.run(['git', '-C', str(repo), *arguments], env=environment,
                                capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise OverlayError('Git metadata read failed: ' + str(error)) from error
    require(result.returncode == 0, 'Git metadata read failed: ' + result.stderr.decode(errors='replace').strip())
    return result.stdout


def physical_directory(path):
    require(path.is_absolute() and path.is_dir() and path.resolve(strict=True) == path,
            'an existing physical directory is required: ' + str(path))


def read_file(path):
    require(path.parent.resolve(strict=True) == path.parent and not path.is_symlink(),
            'nonphysical file path refused: ' + str(path))
    with path.open('rb') as stream:
        first = os.fstat(stream.fileno())
        require(stat.S_ISREG(first.st_mode) and first.st_size <= MAX_FILE, 'nonregular/unbounded input: ' + str(path))
        raw = stream.read(MAX_FILE + 1); last = os.fstat(stream.fileno())
    identity = lambda value:(value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns)
    require(len(raw) <= MAX_FILE and identity(first) == identity(last) == identity(path.stat()),
            'file changed during read: ' + str(path))
    return raw


def target_bytes(path):
    if not path.exists() and not path.is_symlink():return None
    return read_file(path)


def repository(root):
    physical_directory(root)
    top = Path(git(root, 'rev-parse', '--show-toplevel').decode().strip()).resolve(strict=True)
    require(top == root and git(root,'rev-parse','--is-bare-repository').strip() == b'false',
            'repo-root must be the actual nonbare Git worktree root')
    sub = root / SUBMODULE;physical_directory(sub)
    require((sub/'.git').is_file() and not (sub/'.git').is_symlink(), 'op-plugin is not an initialized submodule')
    superproject = git(sub, 'rev-parse', '--show-superproject-working-tree').decode().strip()
    require(superproject and Path(superproject).resolve(strict=True) == root,
            'op-plugin does not belong to this physical superproject')
    require(Path(git(sub,'rev-parse','--show-toplevel').decode().strip()).resolve(strict=True) == sub,
            'op-plugin metadata resolves to a different worktree')
    require(git(sub,'rev-parse','HEAD').decode().strip() == BASE_COMMIT, 'op-plugin HEAD is not exact pinned base ' + BASE_COMMIT)
    index = git(root,'ls-files','--stage','-z','--',SUBMODULE)
    expected = ('160000 ' + BASE_COMMIT + ' 0\t' + SUBMODULE + '\0').encode()
    require(index == expected, 'superproject index does not contain the pinned stage0 op-plugin gitlink')
    head = git(root,'ls-tree','-z','HEAD','--',SUBMODULE)
    require(head == ('160000 commit ' + BASE_COMMIT + '\t' + SUBMODULE + '\0').encode(),
            'superproject HEAD does not contain the pinned op-plugin gitlink')
    return sub


def sources(root):
    path = root / MANIFEST
    manifest = json.loads(read_file(path), object_pairs_hook=unique_object)
    require(type(manifest) is dict and set(manifest) == {'schema','base_commit','files','payload_root'}
            and type(manifest['schema']) is int and manifest['schema'] == 1
            and manifest['base_commit'] == BASE_COMMIT and manifest['payload_root'] == 'op-plugin',
            'manifest schema/base/payload root differs')
    files = manifest['files']
    require(type(files) is dict and set(files) == set(OVERLAY_SHA256), 'exact eight R24 paths required')
    payload = path.parent / manifest['payload_root'];physical_directory(payload)
    values = {}
    for name in sorted(OVERLAY_SHA256):
        row = files[name]
        require(type(row) is dict and set(row) == {'bytes','sha256'} and type(row['bytes']) is int
                and 0 <= row['bytes'] <= MAX_FILE and row['sha256'] == OVERLAY_SHA256[name],
                'manifest R24 pin differs: ' + name)
        raw = read_file(payload/name)
        require(len(raw) == row['bytes'] and digest(raw) == row['sha256'], 'payload size/SHA differs: ' + name)
        values[name] = raw
    return values


def base_files(sub):
    raw = git(sub,'ls-tree','-z','HEAD','--',*sorted(OVERLAY_SHA256))
    result = {}
    for entry in raw.split(b'\0'):
        if not entry:continue
        metadata, name = entry.split(b'\t',1);mode,kind,oid = metadata.split()
        require(mode in (b'100644',b'100755') and kind == b'blob', 'nonregular base entry refused')
        result[name.decode()] = git(sub,'cat-file','blob',oid.decode())
    require(set(result) == EXISTING, 'pinned base must have three existing and five new overlay paths')
    return result


def snapshot(sub):return {name:target_bytes(sub/name) for name in sorted(OVERLAY_SHA256)}


def parent_sources(root):
    payload = root / 'patches/host_regions/op-plugin'
    values = {}
    for name, pin in PARENT_OVERLAY_SHA256.items():
        raw = read_file(payload/name)
        require(digest(raw) == pin, 'pinned R22 parent source changed: ' + name)
        values[name] = raw
    return values


def baseline_state(values, parent, base, before):
    require(set(before) == set(values), 'incomplete destination snapshot')
    if before == values:
        return 'applied'
    if all(before[name] == base.get(name) for name in values):
        return 'base'
    if all(before[name] == parent.get(name) for name in values):
        return 'parent'
    raise OverlayError('dirty or partial R24/parent destinations refused')


def execute(repo_root, *, apply=False):
    root = Path(repo_root).absolute();sub = repository(root)
    values = sources(root);base = base_files(sub);before = snapshot(sub)
    parent = parent_sources(root)
    baseline = baseline_state(values, parent, base, before)
    if baseline == 'applied':
        return dict(mode='apply' if apply else 'check',state='applied',base_commit=BASE_COMMIT,
                    checked_files=8,written_files=0,idempotent=True)
    require(not git(sub,'diff-index','--cached','--name-only','--no-ext-diff','HEAD','--',*sorted(values)),
            'staged edits to overlay destinations are refused')
    if not apply:
        return dict(mode='check',state='ready',base_commit=BASE_COMMIT,checked_files=8,written_files=0,idempotent=False)
    try:
        with tempfile.TemporaryDirectory(prefix='.host-regions-overlay-',dir=sub) as folder:
            temporary = Path(folder)
            require(temporary.resolve().is_relative_to(sub), 'owned staging directory escaped the submodule')
            staged = {}
            for index,(name,raw) in enumerate(values.items()):
                path = temporary / str(index)
                with path.open('xb') as stream:
                    require(stream.write(raw) == len(raw), 'short staging write');stream.flush();os.fsync(stream.fileno())
                mode = stat.S_IMODE((sub/name).stat().st_mode) if before[name] is not None else 0o644
                os.chmod(path,mode);staged[name] = path
            require(sources(root) == values and snapshot(sub) == before, 'source/destination changed after full preflight')
            repository(root)  # No Git operation after target mutation begins.
            require(not git(sub,'diff-index','--cached','--name-only','--no-ext-diff','HEAD','--',*sorted(values)),
                    'staged overlay destinations changed after preflight')
            for name,raw in values.items():
                target = sub/name
                require(target_bytes(target) == before[name], 'destination changed before installation: ' + name)
                if before[name] is not None:
                    os.replace(staged[name],target)
                else:
                    with target.open('xb') as stream:
                        require(stream.write(raw) == len(raw), 'short new-file write');stream.flush();os.fsync(stream.fileno())
                    os.chmod(target,0o644)
            require(snapshot(sub) == values, 'installed overlay byte verification failed')
    except OSError as error:
        raise OverlayError('apply stopped; inspect targeted files before retrying: ' + str(error)) from error
    return dict(mode='apply',state='applied',base_commit=BASE_COMMIT,checked_files=8,written_files=8,idempotent=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--apply',action='store_true',help='write the eight files after complete preflight')
    arguments = parser.parse_args(argv)
    try:result = execute(arguments.repo_root,apply=arguments.apply)
    except (OverlayError,OSError,json.JSONDecodeError) as error:
        print(json.dumps(dict(error=str(error),mode='apply' if arguments.apply else 'check')),file=sys.stderr)
        return 1
    print(json.dumps(result,sort_keys=True));return 0


if __name__ == '__main__':sys.exit(main())
