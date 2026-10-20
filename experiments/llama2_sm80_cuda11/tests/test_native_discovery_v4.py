"""CPU fixtures for child-preload isolation and unchanged GPU safety thresholds."""
from contextlib import ExitStack
import copy
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('discovery_v4', Path(__file__).resolve().parents[1] / 'native_discovery_v4.py')
d = importlib.util.module_from_spec(spec); spec.loader.exec_module(d)


class V4Tests(unittest.TestCase):
    def driver_rows(self):
        return [dict(event='driver_launch', supported_parameter_layout=True,
            function='0x123',kernel=name,context_uid=1,correlation_id=i,
            context='0x456',api='cuLaunchKernel',site=site,
            result=None if site=='ENTER' else 0)
            for i,name in enumerate(['known_torch','additional']) for site in ['ENTER','EXIT']]

    def test_cupti_full_driver_pairs_preserve_actual_handles(self):
        result=d.validate_cupti_records(self.driver_rows(),{'known_torch'})
        self.assertEqual(result['launches'],2)
        self.assertEqual(result['additional_entries'],['additional'])
        self.assertFalse(result['binding_validated'])

    def test_cupti_missing_bad_or_incomplete_inventory_is_rejected(self):
        for change in [{'supported_parameter_layout':False},{'result':801},
                       {'function':None},{'kernel':''},{'function':'0x999'}]:
            rows=self.driver_rows();rows[-1].update(change)
            with self.subTest(change=change),self.assertRaises(AssertionError):
                d.validate_cupti_records(rows,{'known_torch'})
        for rows in [[],self.driver_rows()[:-1],self.driver_rows()[:2],
                     self.driver_rows()+self.driver_rows()[:1]]:
            with self.subTest(rows=rows),self.assertRaises(AssertionError):
                d.validate_cupti_records(rows,{'known_torch'})
        with self.assertRaises(AssertionError):
            d.validate_cupti_records(self.driver_rows(),{'missing_torch'})

    def records(self):
        return {'coverage': [{'allowed': True, 'modeled': False,
                              'requires_instrumented_execution': False}],
                'strict-runtime': [{'runtime_domain': 'cudart11', 'runtime_result': 0,
                    'outcome': 'ORIGINAL', 'gate_decision': 1, 'exact_alias_found': False,
                    'driver_calls': 0, 'original_completion': True, 'gate_runtime_result': 0}],
                'strict-bridge': []}

    def test_authenticated_original_completion_without_driver_log(self):
        d.validate_forward_records(self.records())

    def test_unauthenticated_and_rejected_records_fail(self):
        for change in [{'original_completion': False}, {'runtime_result': 801},
                       {'gate_decision': 0}, {'outcome': 'REJECTED_DRIVER_CORRELATION'},
                       {'exact_alias_found': True}, {'gate_runtime_result': 999},
                       {'driver_calls': 2}]:
            rows = copy.deepcopy(self.records())
            rows['strict-runtime'][0].update(change)
            with self.subTest(change=change), self.assertRaises((AssertionError, KeyError)):
                d.validate_forward_records(rows)

    def test_driver_observation_requires_successful_matching_bridge(self):
        rows = self.records()
        rows['strict-runtime'][0].update(driver_calls=1, driver_result=0,
            original_function='0x123', selected_function='0x123')
        with self.assertRaises(AssertionError):
            d.validate_forward_records(rows)
        rows['strict-bridge'] = [dict(gate_decision=1, selected_path='ORIGINAL',
            cuda_result=0, original_function='0x123', selected_function='0x123')]
        d.validate_forward_records(rows)
        rows['strict-bridge'][0]['selected_path'] = 'PATCHED'
        with self.assertRaises(AssertionError):
            d.validate_forward_records(rows)

    def test_missing_coverage_and_foreign_runtime_do_not_pass(self):
        for field in ['coverage', 'strict-runtime']:
            rows = self.records(); rows[field] = []
            with self.subTest(field=field), self.assertRaises(AssertionError):
                d.validate_forward_records(rows)
        rows = self.records(); rows['strict-runtime'][0]['runtime_domain'] = 'unversioned'
        with self.assertRaises(AssertionError):
            d.validate_forward_records(rows)

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

    def test_native_reexecution_is_explicitly_outside_v4(self):
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
