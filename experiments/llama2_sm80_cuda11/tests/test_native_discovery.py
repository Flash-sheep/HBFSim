"""CPU safety and phase-window fixtures; never start a subprocess or CUDA."""
import importlib.util
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('discovery', Path(__file__).resolve().parents[1] / 'native_discovery.py')
discovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(discovery)


class DiscoveryTests(unittest.TestCase):
    def test_both_entries_stop_at_approval_before_external_actions(self):
        for arm in [None, 'native', 'capture']:
            with self.subTest(arm=arm), ExitStack() as stack:
                stack.enter_context(patch.object(discovery, 'approval_gate', side_effect=ValueError('NO_USER_APPROVAL')))
                stack.enter_context(patch.object(discovery, 'verify', side_effect=AssertionError('file/torch check before approval')))
                stack.enter_context(patch.object(discovery.C, 'CDLL', side_effect=AssertionError('DSO before approval')))
                stack.enter_context(patch.object(discovery.subprocess, 'Popen', side_effect=AssertionError('process before approval')))
                with self.assertRaisesRegex(ValueError, 'NO_USER_APPROVAL'):
                    if arm is None:
                        discovery.execute(Path('/nonexistent'), None)
                    else:
                        discovery.worker(Path('/nonexistent'), None, arm)

    def test_capture_window_excludes_setup_and_trailing_checks(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'log.jsonl'
            setup = b'{"kernel":"setup"}\n'
            phase = b'{"kernel":"original_forward"}\n'
            p.write_bytes(setup + phase + b'{"kernel":"trailing_check"}\n')
            self.assertEqual(discovery.log_window(p, len(setup), len(setup) + len(phase)),
                             [{'kernel': 'original_forward'}])

    def test_parent_environment_drops_external_preload_and_HBF_controls(self):
        with patch.dict(discovery.os.environ, {'LD_PRELOAD': '/bad.so', 'HBFSIM_INSTRUMENTATION_POLICY': 'auto',
                    'BPFTIME_GLOBAL_SHM_NAME': 'someone_else', 'PYTHONPATH': '/other'}):
            env = discovery.base_env({'python': '/private/bin/python', 'library_path': '/private/lib'})
        self.assertNotIn('LD_PRELOAD', env)
        self.assertNotIn('HBFSIM_INSTRUMENTATION_POLICY', env)
        self.assertNotIn('BPFTIME_GLOBAL_SHM_NAME', env)
        self.assertNotIn('PYTHONPATH', env)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], discovery.GPU)


if __name__ == '__main__':
    unittest.main()
