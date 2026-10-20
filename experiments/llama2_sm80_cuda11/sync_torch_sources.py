#!/usr/bin/env python3
"""Resume exact Torch gitlinks with bounded, transport-only retries."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--work',required=True,type=Path);p.add_argument('--attempt',required=True);a=p.parse_args()
w=a.work.resolve();src=w/'pytorch-src';logs=w/'logs'/('torch-source-sync-'+a.attempt);logs.mkdir()
assert subprocess.check_output(['git','-C',str(src),'rev-parse','HEAD'],text=True).strip()=='e4ee3be4063b7c430974252fdf7db42273388d86'
paths=[]
for line in subprocess.check_output(['git','-C',str(src),'ls-tree','-r','HEAD'],text=True).splitlines():
 meta,_,name=line.partition('\t')
 if meta.startswith('160000 commit '):paths.append(name)
start=time.monotonic();rows=[];env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',GIT_TERMINAL_PROMPT='0')
for i,name in enumerate(paths):
 rc=None
 for retry in range(1,6):
  if time.monotonic()-start>=900:break
  cmd=['git','-c','http.version=HTTP/1.1','-c','http.lowSpeedLimit=1','-c','http.lowSpeedTime=30','-C',str(src),'submodule','update','--init','--recursive','--depth','1','--jobs','1','--',name]
  with (logs/(str(i)+'-'+str(retry)+'.stdout')).open('x') as out,(logs/(str(i)+'-'+str(retry)+'.stderr')).open('x') as err:
   proc=subprocess.Popen(cmd,env=env,stdout=out,stderr=err,start_new_session=True)
   try:rc=proc.wait(timeout=min(120,900-(time.monotonic()-start)))
   except subprocess.TimeoutExpired:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    rc=124
  if rc==0:break
  # No global Git settings, no alternate source or pin changes.
  time.sleep(1)
 rows.append({'path':name,'returncode':rc,'attempts':retry})
 (logs/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
 if time.monotonic()-start>=900:break
status=subprocess.check_output(['git','-C',str(src),'submodule','status','--recursive'],text=True)
ready=len(rows)==len(paths) and all(x['returncode']==0 for x in rows) and all(x.startswith(' ') for x in status.splitlines())
r={'status':'READY' if ready else 'INCOMPLETE','source_git':'e4ee3be4063b7c430974252fdf7db42273388d86','wall_seconds':time.monotonic()-start,'rows':rows,'submodules':status,'GPU_tasks_started':0,'code_changes':False,'transport':'per-command HTTP1.1, five bounded retries per pinned top-level submodule, 900s task watchdog'}
root=Path(__file__).resolve().parents[2];e=root/'artifacts/llama2-layer15-full-validation-v1/framework-cpu';e.mkdir(exist_ok=True);f=e/('source-sync-'+a.attempt+'.json');assert not f.exists();f.write_text(json.dumps(r,indent=2)+'\n');(logs/('DONE' if ready else 'FAILED')).write_text(json.dumps(r)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ['submodules','rows']}));raise SystemExit(0 if ready else 1)
