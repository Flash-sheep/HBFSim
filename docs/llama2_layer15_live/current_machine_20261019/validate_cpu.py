"""Current-host CPU input preparation; does not replace the missing donor script."""
import datetime, hashlib, importlib.metadata, json, os, pathlib, struct, subprocess, sys
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU validation requires CUDA_VISIBLE_DEVICES empty'
import torch
import transformers
import safetensors
from safetensors import safe_open
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
root = pathlib.Path(__file__).resolve().parent
repo = pathlib.Path('/home/yjl/HBFSim')
model = pathlib.Path('/home/yjl/yjl-r760/llama2-7b-hf-chat')
original = json.loads((repo/'docs/llama2_layer15_live/evidence/input_identity.json').read_text())
records = []
for item in original['files']:
    name = pathlib.Path(item['source']).name
    if name not in ('config.json', 'model.safetensors.index.json'):
        continue
    path = model/name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    records.append({'path':str(path), 'sha256':digest, 'matches_archive':digest == item['sha256']})
tensors = []
with safe_open(str(model/'model-00001-of-00002.safetensors'), framework='pt', device='cpu') as checkpoint:
    for item in original['layer15_checkpoint_tensors']:
        tensor = checkpoint.get_tensor(item['name'])
        raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
        digest = hashlib.sha256(raw).hexdigest()
        tensors.append({'name':item['name'], 'shape':list(tensor.shape), 'dtype':str(tensor.dtype), 'bytes':len(raw), 'sha256':digest, 'matches_archive':digest == item['sha256']})
assert all(x['matches_archive'] for x in records+tensors)
assert sum(x['bytes'] for x in tensors)==404766848
assert not torch.cuda.is_initialized()
result = {'status':'PASS', 'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'python':sys.version, 'python_executable':sys.executable, 'packages':{n:importlib.metadata.version(n) for n in ('torch','transformers','safetensors','numpy')}, 'torch_cuda_build':torch.version.cuda, 'cuda_initialized':torch.cuda.is_initialized(), 'gpu_workload_launched':False, 'configuration_identity':records, 'tensors':tensors, 'total_tensor_bytes':sum(x['bytes'] for x in tensors), 'scope':'CPU imports and checkpoint tensor loading/hash verification only; no forward; missing original workload scripts remain a separate prerequisite'}
(root/'CPU_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('status','packages','total_tensor_bytes','cuda_initialized','gpu_workload_launched')}))
