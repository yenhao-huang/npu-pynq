"""Durable official-model traces and physical acceptance diagnostics."""
from pathlib import Path
import time
import numpy as np
from acceptance import save, memory, sha256, PROMPT, MAX_NEW

SOURCE_SHA256 = 'a7c33f9ef85d06734cc9d1f943f7e2ba57c77e769df727f4d3e217d9f672b0cc'


def trace(engine, output, label='generation', display_token=None):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    engine.reset()
    prompt_ids = engine.tokenizer.encode(engine.tokenizer.chat([{'role':'user','content':PROMPT}]))
    npu = engine.graph.integer_gemm
    before = dict(npu.stats) if npu else {}
    start = time.monotonic()
    engine.prefill(prompt_ids)
    prefill = time.monotonic()-start
    ids, rows, times = [], [], []
    for step in range(MAX_NEW):
        row = engine.logits[0].copy()
        assert np.isfinite(row).all()
        ids.append(int(row.argmax()))
        rows.append(row)
        text = engine.tokenizer.decode(ids)
        if display_token: display_token(text)
        np.savez(output / (label+'.npz'),ids=ids,logits=np.stack(rows))
        print(label,step+1,'/',MAX_NEW,repr(text),flush=True)
        if step+1 < MAX_NEW:
            start = time.monotonic()
            engine.decode(ids[-1])
            times.append(time.monotonic()-start)
    elapsed = sum(times)
    result = {'prompt':PROMPT,'prompt_ids':prompt_ids,'ids':ids,'text':engine.tokenizer.decode(ids),
              'prefill_seconds':prefill,'decode_seconds':elapsed,'per_token_seconds':times,
              'decode_tokens_per_second':len(times)/elapsed,'ttft_seconds':prefill,
              'tpot_seconds':elapsed/len(times),'memory':memory(),
              'npu':{k:npu.stats[k]-before[k] for k in before} if npu else {}}
    save(output/(label+'.json'),result)
    return result


def compare(actual,reference):
    with np.load(actual) as a, np.load(reference) as b:
        assert a['logits'].shape == b['logits'].shape == (MAX_NEW,49152)
        error = np.max(np.abs(a['logits']-b['logits']),axis=1)
        result = {'token_ids_equal':bool(np.array_equal(a['ids'],b['ids'])),
                  'logits_allclose':bool(np.allclose(a['logits'],b['logits'],rtol=1e-3,atol=1e-3)),
                  'max_abs_error':float(error.max()),'per_step_max_abs_error':error.tolist()}
    return result
