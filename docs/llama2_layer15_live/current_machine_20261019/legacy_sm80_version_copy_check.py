import datetime,hashlib,json,os,pathlib,re,resource,subprocess,time
out=pathlib.Path(__file__).resolve().parent
root=out.parent
inputs={
 'raw':root/'alternate-library-audit-20261019T081156Z/raw-cublas/libcublas.so.11.10.3.1.sm_80.ptx',
 'staged':root/'legacy-sm80-ptx-software-check-v1/staged.ptx',
}
libraries=[pathlib.Path('/usr/local/cuda-11.7/targets/x86_64-linux/lib/libcublas.so.11.10.3.66'),pathlib.Path('/usr/local/cuda-11.7/targets/x86_64-linux/lib/libcublasLt.so.11.10.3.66')]
ptxas=root/'cuda/bin/ptxas'
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for chunk in iter(lambda:stream.read(16*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
before={str(p):digest(p) for p in [*inputs.values(),*libraries]}
meta={'scope':'One fixed CPU-only version-directive compatibility check; no GPU execution or installed-library replacement','input_hashes':before,'compiler':{'path':str(ptxas),'sha256':digest(ptxas)},'planned_change':{'copies_only':True,'version_from':'7.7','version_to':'8.7','target_preserved':'sm_80','all_non_version_bytes_preserved':True},'resources':{'serial_cpu_subprocesses':True,'omp_blas_threads':1,'per_command_timeout_s':120,'per_child_address_space_gib':4},'gpu_workload_launched':False,'limitations':['One sample module/entry; not established as Llama workload-selected kernel','Compilation only; no driver loading, plugin ABI binding, numerical equivalence or performance verification','PTX language-version change is not a proof of semantic equivalence','Earlier unmodified staged compile failure remains immutable']}
(out/'PRECHECK.json').write_text(json.dumps(meta,indent=2)+'\n')
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
def limit():resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
receipts=[]
for label,source in inputs.items():
 raw=source.read_bytes()
 matches=list(re.finditer(rb'(?m)^\.version[ \t]+7\.7[ \t]*$',raw))
 assert len(matches)==1
 m=matches[0]; modified=raw[:m.start()]+b'.version 8.7'+raw[m.end():]
 assert re.findall(rb'(?m)^\.target[^\n]*',raw)==re.findall(rb'(?m)^\.target[^\n]*',modified)
 assert re.sub(rb'(?m)^\.version[^\n]*',b'',raw)==re.sub(rb'(?m)^\.version[^\n]*',b'',modified)
 path=out/(label+'.ptx87.sm80.ptx');path.write_bytes(modified)
 cubin=out/(label+'.ptx87.sm80.cubin')
 argv=[str(ptxas),'--gpu-name=sm_80',str(path),'-o',str(cubin)]
 start=time.monotonic();p=subprocess.run(argv,capture_output=True,text=True,env=env,timeout=120,preexec_fn=limit)
 (out/(label+'.stdout')).write_text(p.stdout);(out/(label+'.stderr')).write_text(p.stderr)
 row={'label':label,'argv':argv,'returncode':p.returncode,'wall_s':time.monotonic()-start,'output_ptx_sha256':digest(path),'all_non_version_bytes_preserved':True}
 if cubin.exists(): row['cubin']={'path':str(cubin),'sha256':digest(cubin),'bytes':cubin.stat().st_size}
 receipts.append(row)
after={str(p):digest(p) for p in [*inputs.values(),*libraries]}
meta['receipts']=receipts;meta['input_and_cuda_libraries_unchanged']=before==after;meta['postcheck_hashes']=after
meta['status']='CPU_COMPILATION_PASS_GPU_NOT_RUN' if all(r['returncode']==0 for r in receipts) and before==after else 'FAILED'
meta['completed_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
(out/'RESULT.json').write_text(json.dumps(meta,indent=2)+'\n')
(out/('DONE' if meta['status']!='FAILED' else 'FAILED')).write_text(meta['status']+'\n')
print(json.dumps({'status':meta['status'],'input_and_cuda_libraries_unchanged':meta['input_and_cuda_libraries_unchanged'],'receipts':receipts},indent=2))
