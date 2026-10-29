"""Check/apply the isolated R26 source overlay; never alter Git state or R25 files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

BASE = '312921458e7d396c54a1e92e90385d7d5f02dede'
FILES = ('torchnpugen/eager_parallel.py', 'torchnpugen/gen.py',
         'torchnpugen/templates/OpInterface.cpp')


def digest(value):
    return hashlib.sha256(value).hexdigest()


def git(root, *arguments):
    environment = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
    environment['GIT_TERMINAL_PROMPT'] = '0'
    return subprocess.run(['git', '-C', str(root), *arguments], check=True, capture_output=True,
                          env=environment, timeout=30).stdout


def checked_path(root, relative):
    result = root / relative
    if result.is_symlink() or result.parent.resolve() != result.parent:
        raise ValueError('symlink/redirected overlay path refused: ' + str(result))
    return result


def plan(payload, target, manifest):
    if manifest.get('schema') != 1 or manifest.get('base_commit') != BASE:
        raise ValueError('R26 manifest schema/base differs')
    rows = manifest.get('files', {})
    if set(rows) != set(FILES):
        raise ValueError('R26 exact three-file overlay required')
    changes = []
    for name in FILES:
        row = rows[name]
        source = checked_path(payload, name).read_bytes()
        if digest(source) != row['sha256'] or len(source) != row['bytes']:
            raise ValueError('overlay payload differs: ' + name)
        path = checked_path(target, name)
        current = path.read_bytes() if path.exists() else None
        # Windows checkouts may use CRLF; Git's pinned blob is canonical LF.
        normalized = current.replace(b'\r\n', b'\n') if current is not None else None
        current_hash = digest(normalized) if normalized is not None else None
        if current_hash not in (row['base_sha256'], row['sha256']):
            raise ValueError('target has unrelated modifications: ' + name)
        if normalized != source:
            changes.append((path, current, source))
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[2])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true')
    group.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    root = args.repo_root.resolve(strict=True)
    sub = root / 'third_party/op-plugin'
    if Path(git(root, 'rev-parse', '--show-toplevel').decode().strip()).resolve() != root:
        raise ValueError('repo-root must be the worktree root')
    if git(sub, 'rev-parse', 'HEAD').decode().strip() != BASE:
        raise ValueError('op-plugin HEAD differs from pinned R26 base')
    if Path(git(sub, 'rev-parse', '--show-superproject-working-tree').decode().strip()).resolve() != root:
        raise ValueError('op-plugin belongs to another superproject')
    overlay = root / 'patches/eager_parallel_r26'
    manifest = json.loads((overlay / 'op-plugin-manifest.json').read_text(encoding='utf-8'))
    for name, row in manifest['files'].items():
        if name not in FILES:
            raise ValueError('unexpected manifest path')
        if row['base_sha256'] is not None and digest(git(sub, 'show', BASE + ':' + name)) != row['base_sha256']:
            raise ValueError('manifest baseline differs from pinned Git blob: ' + name)
    changes = plan(overlay / 'op-plugin', sub, manifest)
    if args.apply:
        # All paths and contents are validated before the first write. Recheck
        # each destination immediately before replacement to reject local races.
        for path, before, after in changes:
            if (path.read_bytes() if path.exists() else None) != before:
                raise ValueError('target changed after preflight: ' + str(path))
            descriptor, temporary = tempfile.mkstemp(prefix='.r26-', dir=path.parent)
            try:
                with os.fdopen(descriptor, 'wb') as stream:
                    stream.write(after)
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
    print('R26 overlay check passed; ' + str(len(changes)) + (' files applied' if args.apply else ' files pending'))


if __name__ == '__main__':
    main()
