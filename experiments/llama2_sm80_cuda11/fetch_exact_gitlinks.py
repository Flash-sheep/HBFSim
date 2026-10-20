#!/usr/bin/env python3
"""Fetch only pinned commit objects; avoid refreshing unrelated branch history."""
import argparse,json,os,signal,subprocess,time,urllib.parse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--work',required=True,type=Path);p.add_argument('--attempt',required=True);a=p.parse_args()
w=a.work.resolve();src=w/'pytorch-src';out=w/'logs'/('exact-gitlinks-'+a.attempt);out.mkdir();start=time.monotonic();rows=[];counter=0
root=Path(__file__).resolve().parents[2];e=root/'artifacts/llama2-layer15-full-validation-v1/framework-cpu';e.mkdir(exist_ok=True)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',GIT_TERMINAL_PROMPT='0',GIT_LFS_SKIP_SMUDGE='1')
(e/('exact-gitlinks-'+a.attempt+'.running.json')).write_text(json.dumps({'pid':os.getpid(),'completion_notification':'NOT_CONFIGURED','DONE':str(out/'DONE'),'FAILED':str(out/'FAILED'),'watchdog_seconds':600},indent=2)+'\n')
def git(repo,args,check=True):
 global counter
 if time.monotonic()-start>=600:raise TimeoutError('600s source transport watchdog')
 cmd=['git','-c','http.version=HTTP/1.1','-c','http.lowSpeedLimit=1','-c','http.lowSpeedTime=20','-C',str(repo)]+args
 proc=subprocess.Popen(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
 try:stdout,stderr=proc.communicate(timeout=min(60,600-(time.monotonic()-start)))
 except subprocess.TimeoutExpired:
  os.killpg(proc.pid,signal.SIGTERM)
  try:stdout,stderr=proc.communicate(timeout=5)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);stdout,stderr=proc.communicate()
 (out/(str(counter)+'.stdout')).write_text(stdout);(out/(str(counter)+'.stderr')).write_text(stderr)
 counter+=1
 if check and proc.returncode:raise RuntimeError(str(cmd)+' '+stderr[-1000:])
 return proc.returncode,stdout.strip()
def variants(url):
 if url.startswith('git@github.com:'):url='https://github.com/'+url.split(':',1)[1]
 assert url.startswith('https://'),('unexpected source transport',url)
 result=[url]
 if url.startswith('https://github.com/'):
  bare=url.removesuffix('.git').rstrip('/')
  result += [bare,bare.lower()+'.git',bare.lower()]
 return list(dict.fromkeys(result))
def visit(repo,parent_url):
 listing=git(repo,['ls-tree','-r','HEAD'])[1]
 links=[(line.split('\t',1)[1],line.split()[2]) for line in listing.splitlines() if line.startswith('160000 commit ')]
 if not links:return
 # Register declared URLs in this isolated repository's local Git metadata.
 # This performs no fetch and changes neither .gitmodules nor global configuration.
 git(repo,['submodule','init'])
 config=git(repo,['config','-f',str(repo/'.gitmodules'),'--get-regexp',r'^submodule\..*\.path$'])[1]
 urls={}
 for line in config.splitlines():
  key,path=line.split(None,1);url=git(repo,['config','-f',str(repo/'.gitmodules'),'--get',key[:-5]+'.url'])[1]
  if url.startswith('.'):url=urllib.parse.urljoin(parent_url.rstrip('/')+'/',url)
  urls[path]=url
 for path,commit in links:
  child=repo/path;url=urls[path];relative=str(child.relative_to(src));errors=[];used=None
  try:
   if not (child/'.git').exists():
    child.mkdir(parents=True,exist_ok=True)
    if any(child.iterdir()):raise RuntimeError('unrecognized nonempty source directory')
    git(child,['init']);git(child,['remote','add','origin',url])
   present=git(child,['cat-file','-e',commit+'^{commit}'],False)[0]==0
   if not present:
    for candidate in variants(url):
     rc,_=git(child,['fetch','--no-tags','--depth','1',candidate,commit],False)
     if rc==0 and git(child,['cat-file','-e',commit+'^{commit}'],False)[0]==0:used=candidate;break
     errors.append(candidate)
    else:raise RuntimeError('all official URL spellings failed for fixed commit')
   assert not git(child,['diff','--name-only','--ignore-submodules=all'])[1], 'unexpected tracked source edit'
   git(child,['checkout','--detach',commit])
   actual=git(child,['rev-parse','HEAD'])[1];assert actual==commit
   rows.append({'path':relative,'status':'EXACT','expected':commit,'actual':actual,'transport_url':used or 'already present object','declared_url':url})
   visit(child,url)
  except TimeoutError:raise
  except Exception as exc:rows.append({'path':relative,'status':'FAILED','expected':commit,'error':str(exc),'failed_urls':errors})
  (out/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
error=None
try:
 assert git(src,['rev-parse','HEAD'])[1]=='e4ee3be4063b7c430974252fdf7db42273388d86'
 visit(src,'https://github.com/pytorch/pytorch.git')
except Exception as exc:error=str(exc)
status_output=subprocess.run(['git','-C',str(src),'submodule','status','--recursive'],capture_output=True,text=True)
ready=error is None and rows and all(x['status']=='EXACT' for x in rows) and status_output.returncode==0 and all(x.startswith(' ') for x in status_output.stdout.splitlines())
r={'status':'READY' if ready else 'INCOMPLETE','error':error,'wall_seconds':time.monotonic()-start,'rows':rows,'submodules':status_output.stdout,'status_stderr':status_output.stderr,'GPU_tasks_started':0,'method':'only expected commit object; official URL spelling fallbacks; local submodule metadata registration; no global config/.gitmodules/source changes'}
p=e/('exact-gitlinks-'+a.attempt+'.json');assert not p.exists();p.write_text(json.dumps(r,indent=2)+'\n');(out/('DONE' if ready else 'FAILED')).write_text(json.dumps(r)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ['rows','submodules','status_stderr']}));raise SystemExit(0 if ready else 1)
