"""Static inventory only. No CUDA API initialization or kernel execution."""
import datetime, hashlib, json, pathlib, re, subprocess
root=pathlib.Path(__file__).resolve().parent
site=root/'env/lib/python3.11/site-packages'
rows=[]
for relative in ('nvidia/cublas/lib/libcublas.so.12','nvidia/cublas/lib/libcublasLt.so.12','torch/lib/libtorch_cuda.so'):
    lib=site/relative
    p=subprocess.run([str(root/'cuda/bin/cuobjdump'),'--list-ptx',str(lib)],capture_output=True,text=True,timeout=120)
    label=lib.name
    (root/'logs'/f'{label}.list-ptx.stdout').write_text(p.stdout)
    (root/'logs'/f'{label}.list-ptx.stderr').write_text(p.stderr)
    targets=sorted(set(re.findall(r'sm_\d+[a-z]?',p.stdout)))
    digest=hashlib.sha256()
    with lib.open('rb') as stream:
        for chunk in iter(lambda:stream.read(16*1024*1024),b''): digest.update(chunk)
    rows.append({'path':str(lib),'sha256':digest.hexdigest(),'bytes':lib.stat().st_size,'returncode':p.returncode,'ptx_member_count':len(re.findall(r'^PTX file\s+\d+:',p.stdout,re.M)),'targets':targets,'sm80_named_ptx_available':'sm_80' in targets})
result={'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Installed cuBLAS/cuBLASLt/Torch CUDA containers only; no claim about other libraries or dynamic kernel coverage','gpu_workload_launched':False,'libraries':rows}
(root/'PTX_INVENTORY.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
