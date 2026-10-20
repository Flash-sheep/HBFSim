#!/usr/bin/env python3
"""Copy legacy sm80 PTX, changing only its one version directive to 8.7."""
import argparse,hashlib,json,re
from pathlib import Path

def sha(data):return hashlib.sha256(data).hexdigest()
def copy_version(source,out,expected_sha):
    source=Path(source).resolve();out=Path(out)
    raw=source.read_bytes()
    if sha(raw)!=expected_sha:raise ValueError('input SHA256 mismatch')
    versions=list(re.finditer(rb'(?m)^([ \t]*\.version[ \t]+)([0-9]+\.[0-9]+)([ \t]*(?://[^\n]*)?)$',raw))
    targets=re.findall(rb'(?m)^[ \t]*\.target[ \t]+(sm_[0-9]+)(?=[, \t\r\n]|$)',raw)
    if len(versions)!=1 or versions[0][2] not in (b'7.4',b'7.7'):raise ValueError('requires exactly one reviewed legacy version 7.4/7.7')
    if targets!=[b'sm_80']:raise ValueError('requires exactly one original sm_80 target; architecture conversion is forbidden')
    m=versions[0];updated=raw[:m.start(2)]+b'8.7'+raw[m.end(2):]
    assert updated[:m.start(2)]==raw[:m.start(2)] and updated[m.start(2)+3:]==raw[m.end(2):]
    out.mkdir(parents=True,exist_ok=False)
    path=out/'version-adapted.ptx'
    with path.open('xb') as f:f.write(updated)
    unchanged=source.read_bytes()==raw
    result={'status':'DERIVED_VERSION_COPY' if unchanged else 'FAILED_SOURCE_CHANGED','evidence_type':'DERIVED',
            'input':str(source),'input_sha256':sha(raw),'output':str(path.resolve()),'output_sha256':sha(updated),
            'old_version':m[2].decode(),'new_version':'8.7','target_before':'sm_80','target_after':'sm_80',
            'only_version_token_changed':True,'source_unchanged':unchanged,
            'assembly_or_GPU_equivalence':'NOT_VALIDATED','GPU_tasks_started':0}
    with (out/'VERSION_COPY.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    if not unchanged:raise ValueError('source changed during copying; derived attempt is FAILED')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--input-sha256',required=True);a=p.parse_args()
    print(json.dumps(copy_version(a.input,a.output,a.input_sha256)))
if __name__=='__main__':main()
