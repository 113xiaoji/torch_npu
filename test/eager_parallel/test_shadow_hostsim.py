import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]


class ShadowHostSimTests(unittest.TestCase):
    def setUp(self):
        path=ROOT/'benchmarks/eager_parallel_r26/shadow_hostsim.py'
        self.assertTrue(path.is_file(),'actual framework shadow verification runner missing')
        spec=importlib.util.spec_from_file_location('r26_shadow_hostsim',path)
        self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)

    @staticmethod
    def snapshot():
        return dict(schema=1,backend='shadow',device_parallel_qualified=False,
            real_sdk_qualified=False,active_calls=0,first_error=None,coverage_incomplete=False,
            foreign_thread_calls=0,operator_counts={
                'aten::_embedding_bag_forward_only':dict(entered=26,completed=26,failed=0,inputs=78,outputs=104),
                'aten::cat':dict(entered=2,completed=2,failed=0,inputs=29,outputs=2),
                'aten::bmm':dict(entered=1,completed=1,failed=0,inputs=2,outputs=1),
                'aten::index.Tensor':dict(entered=1,completed=1,failed=0,inputs=3,outputs=1)},
            closure=dict(passed=True,admission_closed=True,no_owned_device_work=True))

    def test_actual_chain_requires_all_entries_and_four_embedding_outputs(self):
        row=self.snapshot()
        self.assertTrue(self.module.validate_shadow(row,1)['passed'])
        row['operator_counts']['aten::_embedding_bag_forward_only']['outputs']=26
        with self.assertRaises(ValueError):self.module.validate_shadow(row,1)

    def test_cat_container_count_missing_or_all_fallback_is_not_coverage(self):
        row=self.snapshot();row['operator_counts']['aten::cat']['inputs']=0
        with self.assertRaises(ValueError):self.module.validate_shadow(row,1)
        row=self.snapshot();row['operator_counts']={}
        with self.assertRaises(ValueError):self.module.validate_shadow(row,1)

    def test_shadow_never_claims_device_execution_or_clean_incomplete_trace(self):
        for field in ('device_parallel_qualified','real_sdk_qualified','coverage_incomplete'):
            row=self.snapshot();row[field]=True
            with self.assertRaises(ValueError):self.module.validate_shadow(row,1)

    def test_late_validation_failure_never_writes_a_success_receipt(self):
        def fail_after_result(config,output,report):
            report['passed']=True
            raise ValueError('artifact changed during run')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);config=root/'config.json';config.write_text('{}')
            with patch.object(self.module,'run',side_effect=fail_after_result):
                with self.assertRaisesRegex(ValueError,'artifact changed'):
                    self.module.main(['--config',str(config),'--output',str(root/'out')])
            result=json.loads((root/'out/result.json').read_text())
            self.assertIs(result['passed'],False)
            self.assertEqual(result['error']['message'],'artifact changed during run')


if __name__=='__main__':unittest.main()
