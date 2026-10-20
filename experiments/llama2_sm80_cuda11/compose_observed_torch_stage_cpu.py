#!/usr/bin/env python3
"""Publish new scoped provenance for successfully prepared observed entries."""
import json
from pathlib import Path
import shutil
from stage_observed_torch_cpu import ROOT, load_module, save, sha

w = Path('/var/tmp/hbfsim-yjl/llama2-layer15-full-v1')
old, follow = w/'observed-torch-stage-v1', w/'observed-torch-finish-v1'
out = w/'observed-torch-scoped-v1'; out.mkdir(); stage=out/'stage'; stage.mkdir()
collection = w/'logs/postbuild-v1/torch-cuda-original-collection/native-ptx-manifest.json'
variants, pass_rows, selected, chains, rejected = [], [], [], [], []
for process in sorted(old.glob('*/PROCESS.json')):
    spec = json.loads((process.parent/'SPEC.json').read_text())
    m = spec['member']; raw=m['member_sha256']; current=process.parent
    transformed=current/'TRANSFORM_RESULT.json'
    if not transformed.exists() or any(r['status']!='SUPPORTED_TRANSFORMED'
            for r in json.loads(transformed.read_text())['entry_results']):
        current=follow/raw; transformed=current/'TRANSFORM_RESULT.json'
    if not transformed.exists() or any(r['status']!='SUPPORTED_TRANSFORMED'
            for r in json.loads(transformed.read_text())['entry_results']):
        rejected.append(dict(module_sha256=raw,entries=spec['entries'],
            reason='Transform not supported; raw receipts preserved',receipt=str(transformed)))
        continue
    result=json.loads(transformed.read_text()); derived=json.loads((current/'version87/VERSION_COPY.json').read_text())
    p=Path(derived['output']); assert sha(p)==derived['output_sha256']
    target=stage/(raw+'.ptx');shutil.copyfile(p,target)
    variants.append(dict(raw_sha256=raw,raw_bytes=m['size_bytes'],module_id='ptx:sha256:'+raw,
        sources=[m['saved_path']],kernel_entries=m['entries'],selected_entries=spec['entries'],
        entry_results=result['entry_results'],staged_path=str(target),staged_sha256=sha(target),
        staged_bytes=target.stat().st_size,status='PARTIAL_READY'))
    selected.extend(spec['entries']);pass_rows.extend(result['manifest_records'])
    chains.append(dict(module_sha256=raw,original_transform_receipt=str(transformed),
        original_transform_receipt_sha256=sha(transformed),version_copy_receipt=str(current/'version87/VERSION_COPY.json'),
        version_copy_receipt_sha256=sha(current/'version87/VERSION_COPY.json'),final_sha256=sha(target)))
manifest=out/'pass-manifests.jsonl'
with manifest.open('x') as f:
    for row in pass_rows:f.write(json.dumps(row,sort_keys=True)+'\n')
save(stage/'ptx-staging-manifest.json',dict(schema='hbfsim.ptx_staging_manifest.v1',policy='strict',status='PARTIAL_READY',variants=variants))
save(stage/'COMPLETE.json',dict(status='PARTIAL_READY',manifest_sha256=sha(stage/'ptx-staging-manifest.json'),
    pass_manifest_path=str(manifest),pass_manifest_sha256=sha(manifest),
    artifact_sha256={r['staged_path']:r['staged_sha256'] for r in variants}))
join=load_module('scoped_join',ROOT/'tools/native_ptx_reproduction/scoped_join/join_scoped_staging_provenance.py')
result=join.build_join(collection,stage,manifest,selected)
save(out/'SCOPED_PROVENANCE.json',result)
save(out/'VERSION_CHAINS.json',chains)
save(out/'RESULT.json',dict(status='SCOPED_READY_CPU_NOT_GPU_VALIDATED',selected_entries=len(selected),
    selected_members=len(variants),excluded=rejected,PTX_target='sm_80',derived_version='8.7',
    whole_layer_ready=False,HBF_timing_validated=False,GPU_tasks_started=0,
    limits='Read-only source/derived identity join only; selected-entry assembly is not full runtime linking/load validation'))
print(json.dumps(dict(status=result['status'],selected_entries=len(selected),selected_members=len(variants),excluded=len(rejected))))
