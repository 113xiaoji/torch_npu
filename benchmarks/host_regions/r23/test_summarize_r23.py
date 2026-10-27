"""Two small synthetic aggregation checks; no SDK/device or raw-producer calls."""
import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent


def load(name):
    path = HERE/(name+'.py')
    if not path.is_file(): raise AssertionError('summary script missing')
    spec = importlib.util.spec_from_file_location(name+'_test', path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


class SummaryTests(unittest.TestCase):
    def rows(self):
        rows = []
        for abi in ('std', 'ft'):
            for label in ('original-serial', 'candidate-serial', 'worker4'):
                for round_index in (1, 2, 3):
                    candidate = 100+round_index*10+(100 if abi == 'ft' else 0)
                    off = candidate*(2 if label == 'worker4' else 1); on = off+20
                    stages = dict(bottom=10, emb=20, interaction=5, top=10, **{'forward.misc': 5, 'tail_sync': 10, 'outer.misc': on-60})
                    rows.append(dict(abi=abi, label=label, round=round_index, _off_ns=[off]*2000,
                                     _on_ns=[on]*2000, _stage_ns={k:[v]*2000 for k,v in stages.items()}))
        return rows

    def test_actual_6000_sample_quantiles_and_stage_totals_are_conserved(self):
        m, analyzer = load('summarize_r23'), load('analyze_profile_r23')
        result = m.aggregate(self.rows(), analyzer)
        self.assertEqual(len(result['pooled']), 6)
        row = next(r for r in result['pooled'] if r['abi']=='std' and r['label']=='candidate-serial')
        self.assertEqual(row['off']['count'], 6000); self.assertEqual(row['off']['p99_ns'], 130)
        self.assertEqual(sum(stage['mean_ns'] for stage in row['stages'].values()), row['on']['mean_ns'])
        self.assertTrue(row['stage_total_conserved'])
        self.assertEqual(result['same_candidate_pooled'][0]['worker4_over_candidate_serial_mean'], 2)

    def test_missing_or_duplicate_round_is_rejected(self):
        m, analyzer = load('summarize_r23'), load('analyze_profile_r23')
        with self.assertRaises(ValueError): m.aggregate(self.rows()[:-1], analyzer)
        rows = self.rows(); rows[-1] = dict(rows[-2])
        with self.assertRaises(ValueError): m.aggregate(rows, analyzer)


if __name__ == '__main__': unittest.main()
