import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('assertproof', Path(__file__).resolve().parents[1]/'assertfail_proof_cpu.py')
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)


def fixture():
    head = '.extern .func __assertfail (.param .b64 a,.param .b64 b,.param .b32 c,.param .b64 d,.param .b64 e);\n'
    for name in ['s1','s2','s3']: head += '.global .align 1 .b8 '+name+'[2] = {65,0};\n'
    block = '''mov.u64 %rd1, s1;
cvta.global.u64 %rd2, %rd1;
mov.u64 %rd3, s2;
cvta.global.u64 %rd4, %rd3;
mov.u64 %rd5, s3;
cvta.global.u64 %rd6, %rd5;
mov.u32 %r1, 78;
mov.u64 %rd7, 1;
{
.reg .b32 temp_param_reg;
.param .b64 param0;
st.param.b64 [param0+0], %rd2;
.param .b64 param1;
st.param.b64 [param1+0], %rd4;
.param .b32 param2;
st.param.b32 [param2+0], %r1;
.param .b64 param3;
st.param.b64 [param3+0], %rd6;
.param .b64 param4;
st.param.b64 [param4+0], %rd7;
call.uni __assertfail,(param0,param1,param2,param3,param4);
}
'''
    return (head+'.visible .entry test() {\n'+block+block+'ret;\n}\n').encode()


class ProofTests(unittest.TestCase):
    def test_multiple_static_diagnostic_calls(self):
        self.assertEqual(p.prove(fixture(), 'test')['call_count'], 2)

    def test_dynamic_branch_overwrite_unknown_and_nonstring_rejected(self):
        data = fixture()
        changes = [data.replace(b'mov.u32 %r1, 78;', b'ld.param.u32 %r1, [input];'),
            data.replace(b'cvta.global.u64 %rd2, %rd1;', b'bra skip;\ncvta.global.u64 %rd2, %rd1;'),
            data.replace(b'.param .b64 param0;', b'mov.u64 %rd2, %rd99;\n.param .b64 param0;'),
            data.replace(b'call.uni __assertfail,', b'call.uni unknown,'),
            data.replace(b'{65,0}', b'{65,66}'),
            data.replace(b'mov.u64 %rd7, 1;', b'mov.u64 %rd7, 2;')]
        for value in changes:
            with self.subTest(value=value[:40]), self.assertRaises(ValueError):
                p.prove(value, 'test')


if __name__ == '__main__': unittest.main()
