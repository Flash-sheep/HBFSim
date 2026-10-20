"""CPU fixtures for child-preload isolation and unchanged GPU safety thresholds."""
from contextlib import ExitStack
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('discovery_v2', Path(__file__).resolve().parents[1] / 'native_discovery_v2.py')
d = importlib.util.module_from_spec(spec); spec.loader.exec_module(d)


class V2Tests(unittest.TestCase):
    def test_worker_clears_child_preload_before_gate_and_stops_without_approval(self):
        observed = []
        def no_approval(*args):
            observed.append(os.environ.get('LD_PRELOAD'))
            raise ValueError('NO_USER_APPROVAL')
        with patch.dict(os.environ, {'LD_PRELOAD': '/already-loaded-agent.so'}), ExitStack() as stack:
            stack.enter_context(patch.object(d, 'approval_gate', side_effect=no_approval))
            stack.enter_context(patch.object(d, 'verify', side_effect=AssertionError('verify before approval')))
            stack.enter_context(patch.object(d.C, 'CDLL', side_effect=AssertionError('CUDA/DSO before approval')))
            with self.assertRaisesRegex(ValueError, 'NO_USER_APPROVAL'):
                d.worker(Path('/does-not-exist'), None, 'capture')
        self.assertEqual(observed, [None])

    def test_execute_stops_at_gate_without_starting_actors(self):
        with patch.object(d, 'approval_gate', side_effect=ValueError('NO_USER_APPROVAL')):
            with patch.object(d.subprocess, 'Popen', side_effect=AssertionError('must not launch')):
                with self.assertRaisesRegex(ValueError, 'NO_USER_APPROVAL'):
                    d.execute(Path('/does-not-exist'), None)

    def test_native_reexecution_is_explicitly_outside_v2(self):
        with self.assertRaisesRegex(RuntimeError, 'only the capture'):
            d.worker(Path('/does-not-exist'), None, 'native')

    def test_gpu_safety_reasons_keep_independent_thresholds(self):
        sample = {'temperature_c': 43, 'memory_mib': 1024}
        self.assertEqual(d.gpu_stop_reasons(sample, []), [])
        self.assertEqual(d.gpu_stop_reasons(dict(sample, temperature_c=80), []), ['temperature>=80C'])
        self.assertEqual(d.gpu_stop_reasons(dict(sample, memory_mib=12288), []), ['GPU memory>=12288MiB'])
        self.assertEqual(d.gpu_stop_reasons(sample, ['unexpected child PID']), ['unexpected GPU process'])
        self.assertEqual(len(d.gpu_stop_reasons({'temperature_c': 80, 'memory_mib': 12288}, ['foreign'])), 3)


if __name__ == '__main__':
    unittest.main()
