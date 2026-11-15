import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).parents[2]


def load_worker():
    path = ROOT / 'benchmarks/eager_mpsc_r34/real_device.py'
    if not path.exists():
        raise AssertionError('R34 real worker is not implemented')
    spec = importlib.util.spec_from_file_location('r34_worker_test', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class WorkerTests(unittest.TestCase):
    def test_environment_rejects_tq0_forced_gil_preload_and_wrong_ingress(self):
        worker = load_worker()
        env = dict(TASK_QUEUE_ENABLE='1', PER_STREAM_QUEUE='0', ASCEND_LAUNCH_BLOCKING='0',
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   TORCH_DEVICE_BACKEND_AUTOLOAD='0', TORCH_NPU_EAGER_PREPARE_INGRESS='mpsc',
                   TORCH_NPU_EXPERIMENTAL_FREE_THREADED='1')
        worker.validate_environment(env, 'mpsc2')
        for key, value in [('TASK_QUEUE_ENABLE', '0'), ('PYTHON_GIL', '0'),
                           ('LD_PRELOAD', '/tmp/interposer.so'), ('QWEN_ACL_HOST_SIM', '1'),
                           ('TORCH_NPU_EAGER_PREPARE_INGRESS', 'mutex')]:
            bad = dict(env, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                worker.validate_environment(bad, 'mpsc2')

    def test_pin_check_detects_changed_helper_before_import(self):
        worker = load_worker()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'helper.py'
            path.write_bytes(b'original')
            pins = {str(path): worker.sha(path)}
            worker.verify_pins(pins)
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'pin'):
                worker.verify_pins(pins)

    def test_map_check_requires_exact_real_provider_and_rejects_hostsim(self):
        worker = load_worker()
        with tempfile.TemporaryDirectory() as folder:
            paths = []
            for name in ('libascendcl.so', 'libopapi_nn.so', 'libopapi_math.so', 'libtorch_npu.so'):
                path = Path(folder) / name
                path.write_bytes(name.encode())
                paths.append(str(path.resolve()))
            pins = {path: worker.sha(path) for path in paths}
            result = worker.validate_mappings(paths, pins)
            self.assertEqual(len(result), 4)
            with self.assertRaises(ValueError):
                worker.validate_mappings(paths + [str(Path(folder) / 'libhostsim.so')], pins)
            with self.assertRaises(ValueError):
                worker.validate_mappings(paths, {p: h for p, h in pins.items() if 'opapi_nn' not in p})


if __name__ == '__main__':
    unittest.main()
