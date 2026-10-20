"""Read-only proof of static diagnostic arguments in original CUDA11 PTX.

Accept only the observed eight-instruction setup and exact five-argument ABI.
This does not rewrite PTX or infer that the diagnostic branches are unreachable.
"""
import hashlib
import importlib.util
from pathlib import Path
import re

_path = Path(__file__).resolve().parents[2] / 'tools/native_ptx_reproduction/execution/entry_proof.py'
_spec = importlib.util.spec_from_file_location('lexical_entry_proof', _path)
_proof = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_proof)


def prove(payload, entry):
    _, body, span = _proof.entry_body(payload, entry)
    calls = list(_proof.CALL_TARGET_RE.finditer(body))
    if not calls or len(calls) != len(_proof.CALL_RE.findall(body)):
        raise ValueError('absent or unparsed calls')
    if any(c.group(1) != b'__assertfail' for c in calls):
        raise ValueError('unknown callee; only exact __assertfail is admitted')
    clean = _proof.lexical_mask(payload)
    if not _proof.assertfail_abi_ok(clean):
        raise ValueError('exact extern assertfail ABI missing')
    R, S = _proof.REG, _proof.SYM
    setup = re.compile(
        rb'(?m)^[ \t]*mov\.u64\s+('+R+rb')\s*,\s*('+S+rb')\s*;[ \t]*\r?\n'
        rb'^[ \t]*cvta\.global\.u64\s+('+R+rb')\s*,\s*\1\s*;[ \t]*\r?\n'
        rb'^[ \t]*mov\.u64\s+('+R+rb')\s*,\s*('+S+rb')\s*;[ \t]*\r?\n'
        rb'^[ \t]*cvta\.global\.u64\s+('+R+rb')\s*,\s*\4\s*;[ \t]*\r?\n'
        rb'^[ \t]*mov\.u64\s+('+R+rb')\s*,\s*('+S+rb')\s*;[ \t]*\r?\n'
        rb'^[ \t]*cvta\.global\.u64\s+('+R+rb')\s*,\s*\7\s*;[ \t]*\r?\n'
        rb'^[ \t]*mov\.u32\s+('+R+rb')\s*,\s*(\d+)\s*;[ \t]*\r?\n'
        rb'^[ \t]*mov\.u64\s+('+R+rb')\s*,\s*1\s*;\s*\Z')
    results = []
    for call in calls:
        left = body.rfind(b'{', 0, call.start()); right = body.find(b'}', call.end())
        if left < 0 or right < 0:
            raise ValueError('unbounded parameter block')
        match = setup.search(body[:left])
        if match is None:
            raise ValueError('setup is not the exact unconditional static sequence')
        registers = [match.group(i) for i in (1,3,4,6,7,9,10,12)]
        if len(set(registers)) != len(registers):
            raise ValueError('setup registers alias')
        values = [match.group(i) for i in (3,6,10,9,12)]
        exact = rb'\{\s*(?:\.reg\s+\.b32\s+temp_param_reg\s*;\s*)?'
        for i, width in enumerate((64,64,32,64,64)):
            exact += (rb'\.param\s+\.b'+str(width).encode()+rb'\s+param'+str(i).encode()+rb'\s*;\s*'
                rb'st\.param\.b'+str(width).encode()+rb'\s+\[param'+str(i).encode()+rb'\+0\]\s*,\s*'
                + re.escape(values[i])+rb'\s*;\s*')
        exact += rb'call(?:\.uni)?\s+__assertfail\s*,\s*\(\s*param0\s*,\s*param1\s*,\s*param2\s*,\s*param3\s*,\s*param4\s*\)\s*;\s*\}'
        block = body[left:right+1]
        if re.fullmatch(exact, block) is None:
            raise ValueError('parameter block differs or contains extra instructions')
        strings = []
        for symbol in [match.group(i) for i in (2,5,8)]:
            declarations = re.findall(rb'(?m)^\s*\.global\s+\.align\s+1\s+\.b8\s+'
                + re.escape(symbol)+rb'\s*\[(\d+)\]\s*=\s*\{([^}]*)\}\s*;', clean)
            if len(declarations) != 1:
                raise ValueError('static diagnostic global is not unique')
            size, initializer = declarations[0]
            data = [int(v.strip()) for v in initializer.split(b',') if v.strip()]
            if not data or len(data) > int(size) or any(v < 0 or v > 255 for v in data):
                raise ValueError('invalid string initializer')
            data += [0] * (int(size)-len(data))
            if 0 not in data:
                raise ValueError('diagnostic string is not NUL terminated')
            strings.append(dict(symbol=symbol.decode(), bytes=len(data),
                initializer_sha256=hashlib.sha256(bytes(data)).hexdigest()))
        results.append(dict(line=int(match.group(11)), char_size=1, strings=strings,
            setup_sha256=hashlib.sha256(match.group()).hexdigest(),
            parameter_block_sha256=hashlib.sha256(block).hexdigest()))
    return dict(proof='EXACT_STATIC_ASSERTFAIL_ARGUMENTS', entry=entry, **span,
        call_count=len(calls), calls=results, unknown_callees=0,
        limitation='System diagnostic calls remain present; no assertion branch is assumed unreachable')
