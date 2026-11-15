import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'benchmarks/forward_concurrency_r35/requests.py'


class RequestBankTests(unittest.TestCase):
    def module(self):
        self.assertTrue(SOURCE.is_file(), 'deterministic request bank is not implemented')
        spec = importlib.util.spec_from_file_location('r35_requests', SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_replay_varies_values_preserves_ragged_offsets_and_empty_table(self):
        bank = self.module().build_request_bank([[.1, .2], [.3, .4]],
            [[0, 1], [0, 0]], [[1, 1, 2], []], vocabulary=1024, size=8)
        self.assertEqual(len(bank), 8)
        self.assertEqual(bank[0], dict(dense=((.1, .2), (.3, .4)),
            offsets=((0, 1), (0, 0)), indices=((1, 1, 2), ())))
        self.assertEqual(bank[1]['dense'], ((.3, .4), (.1, .2)))
        self.assertEqual(bank[1]['indices'], ((18, 18, 19), ()))
        self.assertEqual({row['offsets'] for row in bank}, {bank[0]['offsets']})
        self.assertEqual(len({repr(row) for row in bank}), 8)

    def test_bank_is_immutable_and_does_not_mutate_template(self):
        dense, offsets, indices = [[1.], [2.]], [[0, 1]], [[3, 4]]
        before = repr((dense, offsets, indices))
        bank = self.module().build_request_bank(dense, offsets, indices, vocabulary=8, size=2)
        self.assertEqual(repr((dense, offsets, indices)), before)
        with self.assertRaises(TypeError):
            bank[0]['indices'][0][0] = 7

    def test_bad_shapes_ids_and_offsets_fail_before_tensor_creation(self):
        module = self.module()
        cases = [([], [[0]], [[1]], 8, 1),
                 ([[1.], [2.]], [[0]], [[1]], 8, 1),
                 ([[1.], [2.]], [[0, 3]], [[1]], 8, 1),
                 ([[1.], [2.]], [[0, 1]], [[8, 1]], 8, 1),
                 ([[1.], [2.]], [[0, 1]], [[1, 1]], 0, 1),
                 ([[1.], [2.]], [[0, 1]], [[1, 1]], 8, 0)]
        for dense, offsets, indices, vocabulary, size in cases:
            with self.subTest(dense=dense, offsets=offsets, indices=indices), self.assertRaises(ValueError):
                module.build_request_bank(dense, offsets, indices, vocabulary=vocabulary, size=size)


if __name__ == '__main__':
    unittest.main()
