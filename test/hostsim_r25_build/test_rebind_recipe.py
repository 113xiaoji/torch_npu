"""Regression for effective shell RPATH when replaying the native Ninja link."""
import importlib.util
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('r25_rebind_recipe',
    ROOT / 'tools/hostsim_r25/build/rebind_provider_ft.py')
RECIPE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECIPE)


class LinkRecipeTests(unittest.TestCase):
    def parse(self, line):
        with patch.object(RECIPE.subprocess, 'check_output', return_value=line + '\n'):
            return RECIPE.link_argv(Path('/unused'))

    def test_ninja_double_quote_escape_matches_parent_elf_rpath(self):
        line = r': && /usr/bin/c++ -shared -Wl,-rpath,"\$ORIGIN/lib/:\$ORIGIN/" -o candidate.so && :'
        argv = self.parse(line)
        # Captured by readelf -d on the actual successful attempt-002 main ELF.
        self.assertEqual(next(x for x in argv if x.startswith('-Wl,-rpath,')),
                         '-Wl,-rpath,$ORIGIN/lib/:$ORIGIN/')

    def test_already_literal_origin_is_unchanged(self):
        line = ": && /usr/bin/c++ '-Wl,-rpath,$ORIGIN/lib/:$ORIGIN/' -o candidate.so && :"
        self.assertIn('-Wl,-rpath,$ORIGIN/lib/:$ORIGIN/', self.parse(line))

    def test_unrelated_argv_escapes_are_preserved(self):
        line = r": && /usr/bin/c++ '-DPROOF=\$KEEP' '-Wl,-rpath,$ORIGIN/lib/:$ORIGIN/' && :"
        self.assertIn(r'-DPROOF=\$KEEP', self.parse(line))

    def test_unknown_rpath_shape_is_rejected(self):
        line = r': && /usr/bin/c++ -Wl,-rpath,"\$OTHER/lib" && :'
        with self.assertRaises(ValueError):
            self.parse(line)


class HistoricalSourcePinsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.parent, self.candidate = root/'parent', root/'candidate'
        for source in (self.parent, self.candidate):
            (source/'include').mkdir(parents=True)
            (source/'include/example.h').write_bytes(b'completed full-build header\n')
        self.dep = self.parent/'include/example.h'
        self.pins = {'include/example.h': RECIPE.sha(self.dep)}

    def verify(self, pins=None):
        observed = {}
        RECIPE.verify_source_dependency(self.dep, self.parent, self.candidate,
                                        self.pins if pins is None else pins, observed)
        return observed

    def test_parent_and_candidate_joint_drift_is_rejected_against_history(self):
        for source in (self.parent, self.candidate):
            (source/'include/example.h').write_bytes(b'changed after completed build\n')
        with self.assertRaises(ValueError):
            self.verify()

    def test_missing_historical_source_pin_is_rejected(self):
        with self.assertRaises(ValueError):
            self.verify({})

    def test_unchanged_source_uses_completed_build_pin(self):
        self.assertEqual(self.verify(), self.pins)

    def test_parent_only_drift_is_rejected(self):
        self.dep.write_bytes(b'changed parent\n')
        with self.assertRaises(ValueError):
            self.verify()

    def test_candidate_only_drift_is_rejected(self):
        (self.candidate/'include/example.h').write_bytes(b'changed candidate\n')
        with self.assertRaises(ValueError):
            self.verify()

    def test_postcheck_rejects_joint_drift_after_initial_verification(self):
        observed = self.verify()
        for source in (self.parent, self.candidate):
            (source/'include/example.h').write_bytes(b'changed during rebind\n')
        with self.assertRaises(ValueError):
            RECIPE.verify_source_postcheck(observed, self.parent, self.candidate, self.pins)

    def test_postcheck_cannot_replace_history_with_a_current_digest(self):
        for source in (self.parent, self.candidate):
            (source/'include/example.h').write_bytes(b'changed after completed build\n')
        current = {'include/example.h': RECIPE.sha(self.dep)}
        with self.assertRaises(ValueError):
            RECIPE.verify_source_postcheck(current, self.parent, self.candidate, self.pins)


if __name__ == '__main__':
    unittest.main()
