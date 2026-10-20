import hashlib,importlib.util,tempfile,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('copy',Path(__file__).parents[1]/'version_copy.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class VersionCopy(unittest.TestCase):
 def fixture(self,version='7.7',target='sm_80'):
  return f'// exact source\n.version {version}\n.target {target}\n.address_size 64\n.visible .entry k() {{ ret; }}\n'.encode()
 def test_two_original_versions_preserve_every_other_byte(self):
  for version in ['7.4','7.7']:
   with self.subTest(version=version),tempfile.TemporaryDirectory() as t:
    root=Path(t);raw=self.fixture(version);src=root/'raw.ptx';src.write_bytes(raw)
    r=m.copy_version(src,root/'copy',hashlib.sha256(raw).hexdigest())
    self.assertEqual(src.read_bytes(),raw)
    self.assertEqual(Path(r['output']).read_bytes(),raw.replace(f'.version {version}'.encode(),b'.version 8.7',1))
    self.assertEqual(r['status'],'DERIVED_VERSION_COPY')
 def test_reject_architecture_masquerading_hash_drift_and_multiple_headers(self):
  for raw in [self.fixture(target='sm_90'),self.fixture()+b'.version 7.7\n',self.fixture(version='8.7')]:
   with tempfile.TemporaryDirectory() as t:
    p=Path(t)/'raw.ptx';p.write_bytes(raw)
    with self.assertRaises(ValueError):m.copy_version(p,Path(t)/'copy',hashlib.sha256(raw).hexdigest())
    self.assertFalse((Path(t)/'copy').exists())
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'raw.ptx';p.write_bytes(self.fixture())
   with self.assertRaises(ValueError):m.copy_version(p,Path(t)/'copy','0'*64)
 def test_refuse_overwriting_an_existing_attempt(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'raw.ptx';raw=self.fixture();p.write_bytes(raw);out=Path(t)/'copy';out.mkdir();(out/'keep').write_text('original')
   with self.assertRaises(FileExistsError):m.copy_version(p,out,hashlib.sha256(raw).hexdigest())
   self.assertEqual((out/'keep').read_text(),'original')
if __name__=='__main__':unittest.main()
