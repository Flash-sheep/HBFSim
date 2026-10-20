#!/usr/bin/env python3
"""Read-only CPU acceptance checks for future full-layer GPU receipts.

This module neither launches CUDA nor creates execution/approval evidence.
Per-weight counters must come from the ordinary service path. Fixed-delay
experiments cannot satisfy this contract.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


COUNTERS = (
    'supported_accesses', 'supported_bytes', 'in_range_accesses',
    'in_range_intersection_bytes', 'native_out_of_range_accesses',
    'native_out_of_range_bytes', 'modeled_admitted_accesses',
    'modeled_admitted_bytes', 'service_completed_accesses',
    'service_completed_bytes', 'failed_after_issue_accesses',
    'failed_after_issue_bytes', 'unsupported_preissue_accesses',
    'unsupported_preissue_bytes', 'failed_preissue_accesses',
    'failed_preissue_bytes', 'translation_failed_accesses',
    'translation_failed_bytes', 'service_requests', 'unclassified_accesses',
    'unclassified_bytes', 'counter_overflow',
)
ZERO_COUNTERS = tuple(name for name in COUNTERS if name.startswith((
    'failed_', 'unsupported_', 'translation_failed_', 'unclassified_',
    'counter_overflow')))


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def uint(value, label):
    require(type(value) is int and 0 <= value < 2**64, label + ': invalid uint64')
    return value


def counter_closure(counters):
    require(set(counters) == set(COUNTERS), 'counter fields differ from actual v2 ABI')
    for name in COUNTERS:
        uint(counters[name], name)
    require(all(counters[name] == 0 for name in ZERO_COUNTERS), 'error/overflow counter')
    a = counters
    require(a['in_range_accesses'] > 0 and a['in_range_intersection_bytes'] > 0,
            'no registered-weight accesses')
    require(a['service_requests'] > 0, 'no ordinary modeled service requests')
    require(a['in_range_accesses'] == a['modeled_admitted_accesses'] ==
            a['service_completed_accesses'], 'access admission/completion does not close')
    require(a['in_range_intersection_bytes'] == a['modeled_admitted_bytes'] ==
            a['service_completed_bytes'], 'byte admission/completion does not close')
    require(a['supported_accesses'] == a['in_range_accesses'] +
            a['native_out_of_range_accesses'], 'supported access partition does not close')
    require(a['supported_bytes'] == a['in_range_intersection_bytes'] +
            a['native_out_of_range_bytes'], 'supported byte partition does not close')


def check_snapshot(snapshot):
    require(snapshot['status'] == 'COMPLETE', 'incomplete access-accounting snapshot')
    require(snapshot['disable_complete'] is True and
            all(snapshot['synchronization'][k] is True for k in
                ['before_snapshot', 'after_disable']), 'accounting synchronization failed')
    epoch = uint(snapshot['epoch'], 'epoch')
    require(epoch > 0, 'zero accounting epoch')
    rows = snapshot['per_module']
    require(snapshot['module_count'] == len(rows) and len(rows) > 0, 'empty/module count')
    identities = [row['identity'] for row in rows]
    require(len(set(identities)) == len(identities), 'duplicate accounting module')
    require(all(row['status'] == 'COMPLETE' for row in rows), 'incomplete module')
    for row in rows:
        c = row['counters']
        require(set(c) == set(COUNTERS), 'per-module counter fields differ')
        for name in COUNTERS:
            uint(c[name], name)
    aggregate = {name: sum(row['counters'][name] for row in rows) for name in COUNTERS}
    require(aggregate == snapshot['aggregate'], 'per-module/aggregate counters differ')
    counter_closure(aggregate)
    return epoch, aggregate


def compare_fp16(native, instrumented):
    require(len(native) == len(instrumented) == 4096 * 2, 'FP16 output shape/size differs')
    left = struct.unpack('<4096e', native)
    right = struct.unpack('<4096e', instrumented)
    require(all(math.isfinite(x) for x in left + right), 'nonfinite layer output')
    # An instrumentation correctness check uses exact bytes. Any difference
    # remains a failure for diagnosis; this checker never relaxes tolerances.
    require(native == instrumented, 'native/instrumented FP16 output bytes differ')
    return hashlib.sha256(native).hexdigest()


def check_layer(evidence, expected, native, instrumented):
    require(evidence['schema'] == 'llama2-layer15-service-receipt-v1', 'receipt schema')
    require(evidence['service_path'] == 'ordinary_hbf_timing' and
            evidence['eval_delay_enabled'] is False, 'fixed-delay path is not ordinary HBF service')
    require(evidence['workload'] == {
        'layer_index': 15, 'batch_size': 1, 'query_length': 1,
        'context_length': 32, 'dtype': 'float16', 'seed': 1,
    }, 'workload identity differs')
    require(evidence['forward_calls'] == 1, 'full-layer call count differs')
    native_sha = compare_fp16(native, instrumented)
    require(evidence['native_output_sha256'] == native_sha and
            evidence['instrumented_output_sha256'] == native_sha, 'output receipt hash differs')
    arena = evidence['arena']
    require(arena['tensor_count'] == len(arena['tensors']) == len(expected['tensors']) == 9,
            'expected all nine weights')
    require(arena['weight_payload_bytes'] == arena['hbf_size'] == 404766720, 'weight extent differs')
    base = uint(arena['hbf_base_addr'], 'arena base')
    require(base > 0 and base % 256 == 0 and base + arena['hbf_size'] < 2**64,
            'arena bounds/alignment')
    weights = {}
    previous_end = base
    for actual, ref in zip(arena['tensors'], expected['tensors']):
        require(actual['dtype'] == 'torch.float16', 'parameter dtype differs')
        for key in ['name', 'shape', 'offset_bytes', 'size_bytes']:
            require(actual[key] == ref[key], 'weight identity/layout differs: ' + key)
        address = uint(actual['address'], 'weight address')
        end = address + actual['size_bytes']
        require(address == base + actual['offset_bytes'] and address >= previous_end and
                end == actual['end_address_exclusive'] and end <= base + arena['hbf_size'],
                'weight address/layout differs')
        require(actual['name'] not in weights, 'duplicate weight')
        weights[actual['name']] = actual
        previous_end = end
    scopes = evidence['weight_scopes']
    require(len(scopes) == 9 and {x['weight'] for x in scopes} == set(weights),
            'per-weight measurement missing/duplicated')
    epochs = set()
    sums = {name: 0 for name in COUNTERS}
    per_weight = []
    for scope in scopes:
        require(scope['call_count'] == 1 and scope['direct_parameter_names'] == [scope['weight']],
                'weight scope is not one direct-parameter module call')
        epoch, c = check_snapshot(scope['access'])
        require(epoch not in epochs, 'reused accounting epoch')
        epochs.add(epoch)
        weight = weights[scope['weight']]
        decisions = scope['gate_decisions']
        require(decisions and all(x['allowed'] is True for x in decisions), 'rejected/missing gate decision')
        modeled = [x for x in decisions if x['modeled']]
        require(modeled, 'weight scope has no modeled gate address')
        for row in modeled:
            require(weight['address'] <= row['address'] < weight['end_address_exclusive'],
                    'modeled gate address does not belong to scoped weight')
            require(row['requires_instrumented_execution'] is True and not row['opaque_unmodeled']
                    and not row['cubin_only'] and row['range_policy'] == 'timing_backed',
                    'opaque/native weight execution cannot certify coverage')
            require(row['module_id'] in scope['bound_module_ids'] and
                    row['ptx_target'] == 'sm_80', 'missing exact bound sm80 module identity')
        for name in COUNTERS:
            sums[name] += c[name]
        per_weight.append({'weight': scope['weight'], 'addressed': True,
                           'modeled_accesses': c['modeled_admitted_accesses'],
                           'modeled_bytes': c['modeled_admitted_bytes'],
                           'completed_requests': c['service_requests']})
    counter_closure(sums)
    before, after = evidence['stats_before'], evidence['stats_after']
    for name in ['requests_submitted', 'requests_completed', 'fast_requests',
                 'reference_requests', 'fast_modeled_ns']:
        uint(before[name], name); uint(after[name], name)
        require(after[name] >= before[name], 'service statistics moved backwards')
    submitted = after['requests_submitted'] - before['requests_submitted']
    completed = after['requests_completed'] - before['requests_completed']
    require(submitted == completed == sums['service_requests'] > 0,
            'ordinary request totals do not close against scoped counters')
    require((after['fast_requests'] - before['fast_requests']) +
            (after['reference_requests'] - before['reference_requests']) == completed,
            'fast/reference request totals do not close')
    bridge = evidence['full_forward_strict_bridge']
    require(bridge, 'missing actual strict bridge launch records')
    patched = 0
    for row in bridge:
        require(row['cuda_result'] == 0, 'failed strict bridge launch')
        if row['gate_decision'] == 2:
            require(row['exact_alias_found'] is True and row['selected_path'] == 'PATCHED' and
                    row['selected_function'] == row['patched_function'] and
                    row['original_function'] != row['patched_function'] and
                    row['patched_function'] is not None, 'weight launch lacks exact patched alias')
            patched += 1
        else:
            require(row['gate_decision'] == 1 and row['selected_path'] == 'ORIGINAL' and
                    row['selected_function'] == row['original_function'] and
                    row['original_function'] is not None, 'rejected/invalid native bridge launch')
    require(patched >= sum(sum(bool(row['modeled']) for row in scope['gate_decisions'])
                           for scope in scopes), 'missing patched launches for modeled decisions')
    require(evidence['full_forward_gate_rejections'] == 0, 'strict gate rejected a launch')
    return {'status': 'RECEIPT_CHECK_PASS', 'GPU_launched_by_checker': False,
            'output_comparison': 'EXACT_FP16_BYTES', 'output_sha256': native_sha,
            'per_weight': per_weight, 'counter_sums': sums,
            'limits': 'Checks supplied receipts; does not authenticate authorship, launch CUDA, prove whole-weight byte coverage or establish physical calibration.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['receipt', 'expected-layout', 'native-output', 'instrumented-output', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    evidence = json.loads(a.receipt.read_text())
    require(evidence.get('is_test_fixture') is False, 'production CLI refuses test/missing fixture identity')
    result = check_layer(evidence, json.loads(a.expected_layout.read_text()),
                         a.native_output.read_bytes(), a.instrumented_output.read_bytes())
    result['receipt_sha256'] = hashlib.sha256(a.receipt.read_bytes()).hexdigest()
    with a.output.open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
