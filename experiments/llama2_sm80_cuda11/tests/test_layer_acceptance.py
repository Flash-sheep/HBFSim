"""Synthetic CPU fixtures: reject false coverage; never GPU evidence."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import layer_acceptance as check


def fixture():
    root = Path(__file__).resolve().parents[3]
    expected = json.loads((root / 'artifacts/llama2-layer15-full-validation-v1/frozen/original_weight_manifest.json').read_text())
    arena = copy.deepcopy(expected)
    arena['hbf_base_addr'] = 2**32
    for weight in arena['tensors']:
        weight['address'] = arena['hbf_base_addr'] + weight['offset_bytes']
        weight['end_address_exclusive'] = weight['address'] + weight['size_bytes']
    payload = struct.pack('<4096e', *([1.0] * 4096))
    digest = hashlib.sha256(payload).hexdigest()
    module = 'ptx:sha256:' + 'a' * 64
    c = dict.fromkeys(check.COUNTERS, 0)
    c.update(supported_accesses=2, supported_bytes=4,
             in_range_accesses=2, in_range_intersection_bytes=4,
             modeled_admitted_accesses=2, modeled_admitted_bytes=4,
             service_completed_accesses=2, service_completed_bytes=4, service_requests=1)
    scopes = []
    for index, weight in enumerate(arena['tensors'], 1):
        snapshot = {'status': 'COMPLETE', 'epoch': index, 'module_count': 1,
                    'disable_complete': True, 'synchronization': {'before_snapshot': True, 'after_disable': True},
                    'aggregate': dict(c), 'per_module': [{'identity': module, 'status': 'COMPLETE', 'counters': dict(c)}]}
        scopes.append({'weight': weight['name'], 'call_count': 1,
                       'direct_parameter_names': [weight['name']], 'access': snapshot,
                       'bound_module_ids': [module], 'gate_decisions': [{
                           'allowed': True, 'modeled': True, 'address': weight['address'],
                           'requires_instrumented_execution': True, 'opaque_unmodeled': False,
                           'cubin_only': False, 'range_policy': 'timing_backed',
                           'module_id': module, 'ptx_target': 'sm_80'}]})
    before = dict(requests_submitted=0, requests_completed=0, fast_requests=0,
                  reference_requests=0, fast_modeled_ns=0)
    after = dict(before, requests_submitted=9, requests_completed=9, fast_requests=9,
                 fast_modeled_ns=90000)
    e = {'is_test_fixture': True, 'schema': 'llama2-layer15-service-receipt-v1',
         'service_path': 'ordinary_hbf_timing', 'eval_delay_enabled': False,
         'workload': {'layer_index': 15, 'batch_size': 1, 'query_length': 1,
                      'context_length': 32, 'dtype': 'float16', 'seed': 1},
         'forward_calls': 1, 'native_output_sha256': digest, 'instrumented_output_sha256': digest,
         'arena': arena, 'weight_scopes': scopes, 'stats_before': before, 'stats_after': after,
         'full_forward_strict_bridge': [dict(cuda_result=0, gate_decision=2,
                  exact_alias_found=True, selected_path='PATCHED', original_function='0x1234',
                  patched_function='0x5678', selected_function='0x5678') for _ in scopes],
         'full_forward_gate_rejections': 0}
    return e, expected, payload


class AcceptanceTests(unittest.TestCase):
    def test_synthetic_ordinary_service_closes_nine_weights(self):
        e, expected, output = fixture()
        result = check.check_layer(e, expected, output, output)
        self.assertEqual(result['status'], 'RECEIPT_CHECK_PASS')
        self.assertEqual(len(result['per_weight']), 9)
        self.assertFalse(result['GPU_launched_by_checker'])

    def test_aggregate_success_cannot_hide_missing_weight(self):
        e, expected, output = fixture()
        e['weight_scopes'].pop()
        with self.assertRaisesRegex(ValueError, 'per-weight measurement'):
            check.check_layer(e, expected, output, output)

    def test_error_or_incomplete_snapshot_cannot_pass(self):
        for fault in ['counter_overflow', 'failed_after_issue_accesses', 'translation_failed_bytes']:
            with self.subTest(fault=fault):
                e, expected, output = fixture()
                for counters in [e['weight_scopes'][0]['access']['aggregate'],
                                 e['weight_scopes'][0]['access']['per_module'][0]['counters']]:
                    counters[fault] = 1
                with self.assertRaisesRegex(ValueError, 'error/overflow'):
                    check.check_layer(e, expected, output, output)
        e, expected, output = fixture()
        e['weight_scopes'][0]['access']['status'] = 'INCOMPLETE'
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            check.check_layer(e, expected, output, output)

    def test_scoped_address_and_request_closure_are_required(self):
        for fault in ['address', 'requests']:
            with self.subTest(fault=fault):
                e, expected, output = fixture()
                if fault == 'address':
                    e['weight_scopes'][0]['gate_decisions'][0]['address'] = e['arena']['tensors'][1]['address']
                else:
                    e['stats_after']['requests_completed'] = 8
                with self.assertRaises(ValueError):
                    check.check_layer(e, expected, output, output)

    def test_fixed_delay_and_native_weight_bypass_are_rejected(self):
        for fault in ['fixed_delay', 'alias', 'decision']:
            with self.subTest(fault=fault):
                e, expected, output = fixture()
                if fault == 'fixed_delay':
                    e['eval_delay_enabled'] = True
                elif fault == 'alias':
                    e['full_forward_strict_bridge'][0]['exact_alias_found'] = False
                else:
                    e['weight_scopes'][0]['gate_decisions'][0]['modeled'] = False
                with self.assertRaises(ValueError):
                    check.check_layer(e, expected, output, output)

    def test_nonfinite_or_different_outputs_are_rejected(self):
        output = fixture()[2]
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            check.compare_fp16(output, struct.pack('<e', float('nan')) + output[2:])
        with self.assertRaisesRegex(ValueError, 'bytes differ'):
            check.compare_fp16(output, struct.pack('<e', 1.001) + output[2:])


if __name__ == '__main__':
    unittest.main()
