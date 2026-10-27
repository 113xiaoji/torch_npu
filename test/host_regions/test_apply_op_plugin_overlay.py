"""Local Git fixtures for the byte overlay; never access a real checkout or network."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
TOOL = HERE.parents[1] / 'tools/host_regions/apply_op_plugin_overlay.py'
PATHS = tuple('op_plugin/utils/' + name for name in (
    'op_api_common.cpp', 'op_api_common.h', 'op_api_common_base.h',
    'op_api_owned_pack_r22.h', 'op_api_owned_payload_r22.h', 'op_api_owned_r22.h',
    'op_api_sdk_profile_r22.h'))
EXISTING = PATHS[:3]
SUBMODULE = 'third_party/op-plugin'
SUBJECT = 'test: seed fixture'
BODY = 'Create isolated Git objects for overlay preflight tests.\nNo real repository or remote is modified.'


def git(repo, *args, check=True):
    env = {key:value for key,value in os.environ.items() if not key.startswith('GIT_')}
    env.update(GIT_TERMINAL_PROMPT='0', GIT_OPTIONAL_LOCKS='0')
    result = subprocess.run(['git', '-C', str(repo), *args], env=env, capture_output=True, timeout=30)
    if check and result.returncode:
        raise AssertionError(result.stderr.decode(errors='replace'))
    return result


def init(repo):
    repo.mkdir()
    git(repo, 'init', '--object-format=sha1', '--initial-branch=main')
    git(repo, 'config', 'user.name', 'Overlay Fixture')
    git(repo, 'config', 'user.email', 'fixture@example.invalid')
    git(repo, 'config', 'core.autocrlf', 'false')


def commit_seed(repo, paths):
    # The commit-validator skill was read before authoring these fixtures.
    assert len(SUBJECT) <= 50 and SUBJECT.startswith('test: ')
    assert all(len(line) <= 72 for line in BODY.splitlines())
    before = git(repo, 'status', '--porcelain').stdout
    assert before
    git(repo, 'log', '--oneline', '-5', check=False)  # Empty repository is expected initially.
    git(repo, 'diff', 'HEAD', check=False)
    git(repo, 'add', '--', *paths)
    assert git(repo, 'diff', '--cached', '--stat').stdout
    git(repo, '-c', 'commit.gpgsign=false', 'commit', '-m', SUBJECT, '-m', BODY)
    return git(repo, 'rev-parse', 'HEAD').stdout.decode().strip()


class OverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared = tempfile.TemporaryDirectory(prefix='overlay-seed-')
        cls.seed = Path(cls.shared.name).resolve() / 'seed'
        init(cls.seed)
        for name in EXISTING:
            path = cls.seed/name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(('base:' + name + '\n').encode())
        (cls.seed/'unrelated.txt').write_bytes(b'original unrelated file\n')
        cls.base = commit_seed(cls.seed, [*EXISTING, 'unrelated.txt'])

    @classmethod
    def tearDownClass(cls):cls.shared.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='overlay-case-'); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()/'repo'; init(self.root)
        (self.root/'README').write_text('isolated test repo\n')
        git(self.root, '-c', 'core.autocrlf=false', '-c', 'protocol.file.allow=always',
            'submodule', 'add', str(self.seed), SUBMODULE)
        commit_seed(self.root, ['README', '.gitmodules', SUBMODULE])
        self.sub = self.root/SUBMODULE
        git(self.sub, 'config', 'core.autocrlf', 'false')
        for name in EXISTING:
            self.assertEqual((self.sub/name).read_bytes(),git(self.sub,'show','HEAD:'+name).stdout)
        self.payload = self.root/'patches/host_regions/op-plugin'
        self.values = {name:('overlay:' + name + '\r\n').encode() for name in PATHS}
        self.pins = {name:hashlib.sha256(raw).hexdigest() for name,raw in self.values.items()}
        for name,raw in self.values.items():
            path = self.payload/name; path.parent.mkdir(parents=True, exist_ok=True);path.write_bytes(raw)
        self.manifest = self.root/'patches/host_regions/op-plugin-manifest.json'
        self.manifest.write_text(json.dumps(dict(schema=1, base_commit=self.base, payload_root='op-plugin',
            files={name:dict(bytes=len(raw),sha256=self.pins[name]) for name,raw in self.values.items()})))

    def module(self):
        self.assertTrue(TOOL.is_file(), 'portable op-plugin overlay helper is not implemented')
        spec = importlib.util.spec_from_file_location('_overlay_tested', TOOL)
        value = importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
        # Only in-process fixture constants are substituted. The public CLI has
        # no base/pin bypass; the production file retains the actual312/source13.
        value.BASE_COMMIT = self.base;value.OVERLAY_SHA256 = self.pins.copy()
        return value

    def target_snapshot(self):
        return {name:(self.sub/name).read_bytes() if (self.sub/name).exists() else None for name in PATHS}

    def reject_without_writes(self):
        before = self.target_snapshot();m = self.module()
        with self.assertRaises(m.OverlayError):m.execute(self.root, apply=True)
        self.assertEqual(self.target_snapshot(), before)

    def test_default_check_does_not_write_target_or_indexes(self):
        m = self.module();before = self.target_snapshot()
        indexes = [git(p,'ls-files','--stage').stdout for p in (self.root,self.sub)]
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):code = m.main(['--repo-root',str(self.root)])
        self.assertEqual(code,0);self.assertEqual(json.loads(stream.getvalue())['state'],'ready')
        self.assertEqual(self.target_snapshot(),before)
        self.assertEqual([git(p,'ls-files','--stage').stdout for p in (self.root,self.sub)],indexes)

    def test_apply_is_byte_exact_idempotent_and_preserves_unrelated_files(self):
        m = self.module();other = self.sub/'unrelated.txt';other.write_bytes(b'unrelated local edit\n')
        result = m.execute(self.root,apply=True)
        self.assertEqual(result['written_files'],7);self.assertEqual(self.target_snapshot(),self.values)
        times = {n:(self.sub/n).stat().st_mtime_ns for n in PATHS}
        again = m.execute(self.root,apply=True)
        self.assertEqual(again['state'],'applied');self.assertEqual(again['written_files'],0)
        self.assertEqual({n:(self.sub/n).stat().st_mtime_ns for n in PATHS},times)
        self.assertEqual(other.read_bytes(),b'unrelated local edit\n')
        self.assertEqual(git(self.sub,'rev-parse','HEAD').stdout.decode().strip(),self.base)

    def test_dirty_existing_file_is_never_overwritten(self):
        (self.sub/EXISTING[0]).write_bytes(b'user edit\n');self.reject_without_writes()

    def test_unknown_new_destination_is_never_overwritten(self):
        (self.sub/PATHS[4]).write_bytes(b'unknown preexisting file\n');self.reject_without_writes()

    def test_wrong_submodule_head_is_rejected(self):
        git(self.sub,'config','user.name','Overlay Fixture');git(self.sub,'config','user.email','fixture@example.invalid')
        (self.sub/'divergent').write_text('different base\n');commit_seed(self.sub,['divergent'])
        self.reject_without_writes()

    def test_changed_payload_sha_is_rejected_before_any_target_write(self):
        (self.payload/PATHS[-1]).write_bytes(b'changed source payload\n');self.reject_without_writes()

    def test_last_destination_failure_is_found_before_first_replacement(self):
        (self.sub/PATHS[-1]).write_bytes(self.values[PATHS[-1]])
        self.reject_without_writes()  # Partial overlay is not an idempotent seven-file state.

    def test_manifest_cannot_repin_another_source_profile(self):
        data = json.loads(self.manifest.read_text());data['files'][PATHS[-1]]['sha256']='0'*64
        self.manifest.write_text(json.dumps(data));self.reject_without_writes()

    def test_staged_target_edit_is_preserved_when_worktree_was_restored_to_base(self):
        path = self.sub/EXISTING[0];base = path.read_bytes()
        path.write_bytes(b'user staged source\n');git(self.sub,'add','--',EXISTING[0]);path.write_bytes(base)
        index = git(self.sub,'ls-files','--stage').stdout
        self.reject_without_writes()
        self.assertEqual(git(self.sub,'ls-files','--stage').stdout,index)


if __name__ == '__main__':unittest.main()
