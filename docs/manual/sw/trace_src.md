# Export and run trace

How a model travels from a config file to tokens on the board, call by call.
Line numbers refer to the files as of the `pynqo-middle-end` / `pynqo-backend`
rename; recheck them after a large edit.

```
# Export SmolLM2: one package with a model.so for the board and one for this host
python -m src.export.cli build examples/smollm-safetensors/configs/export.yml --out dist/smollm/package

# Run it. On the board the NPU runs the delegates; elsewhere libnpu_rt simulates the ISA.
python -m src.runtime.cli run dist/smollm/package \
  --prompt "What is the capital of France?" --chat --stream
```

The config decides everything the export does:

```yaml
# examples/smollm-safetensors/configs/export.yml
model: smollm2-135m-instruct
load: {framework: huggingface, source: examples/smollm-safetensors/model/checkpoint}
quantization: w8a8
graph: {kind: llm, max_seq: 256, chunk: 16}   # both become static shapes
devices: [cpu, npu]                           # [cpu]: no delegates
cpus: [arm-cortex-a9, native]                 # one model.so each
```

### The package

```
dist/smollm/package/
├─ manifest.json        format pynqo-package/3: entry points, argument names, delegate
│                       table, libraries, NPU limits, LLM metadata
├─ compile.json         what pynqo-backend wrote; manifest.json is built on top of it
├─ weights.bin          every weight, 64-byte aligned; NPU weights in B-tile layout
├─ programs.bin         NPU ISA programs, 64-bit words
├─ programs.json        one entry per NPU delegate: program, segments, relocations
├─ programs.txt         the same programs, readable
├─ tokenizer.json       copied from the checkpoint by the framework adapter
├─ arm-cortex-a9/
│  └─ model.so          the board: armv7a, cortex-a9, +neon,+vfp3, hard float
├─ <arch>-<cpu>/        this host (native), e.g. x86_64-raptorlake/
│  └─ model.so
└─ ir/
   ├─ torch.mlir        to_mlir's output: the torch dialect
   └─ model.mlir        pynqo-middle-end's output: linalg on tensors
```

## Overview

```
src/export/cli.py  build <config> --out <dir>
  └─ pipeline.export_package(config, out)            src/export/pipeline.py:182
       ├─ load                frontend/load           -> nn.Module (float)
       ├─ quantize            frontend/quantization   -> nn.Module (W8A8 layers)
       ├─ export_graph        frontend/export_graph   -> {name: ExportedProgram}
       ├─ to_mlir             frontend/to_mlir        -> ir/torch.mlir
       ├─ pynqo-middle-end      middle-end/ (C++)       -> ir/model.mlir
       └─ pynqo-backend         backend/ (C++)          -> weights.bin, programs.*, */model.so, compile.json


src/runtime/cli.py  run <package>
  └─ ExportedLLM(package, overlay)                   src/runtime/core/delegated.py:108
       └─ generate(prompt)   prefill per chunk, then decode per token;
                             each call is model.so's _mlir_ciface_<entry>
```

The front end is Python and runs wherever torch runs. Both C++ tools need
Linux with LLVM 22; on Windows, `pipeline._linux` runs them in WSL and
`_linux_path` turns `C:\x` into `/mnt/c/x`.

## 1. Export

### load (pipeline.py:205)

```
load_model(spec['framework'], src, **extra)          frontend/load/__init__.py
  └─ framework(name).load(source)                    frontend/load/frameworks/
       ├─ huggingface          AutoModelForCausalLM from a checkpoint or a Hub id
       └─ torchvision_models   resnet18 + a .pth file or a weights name
```

A relative `source` that exists is resolved against the repository root. Any
other key under `load:` (such as `model: resnet18`) goes to the adapter.

### export_graph (pipeline.py:219)

`torch.export` traces the model once per entry point with example inputs.
Only their shapes and dtypes matter, and they become static in the graph.

```
export_llm(model, max_seq=256, chunk=16)             frontend/export_graph/export.py:129
  ├─ cache_spec()          layers, kv_heads, max_seq, head_dim from model.config
  └─ for decode (T=1) and prefill (T=chunk):
       ├─ Step(model, spec, prefill)                 export.py:93
       │    ├─ functional_cache()   update() returns new cache tensors (torch.where
       │    │                       over the whole cache), so the graph stays functional
       │    ├─ an additive causal mask over all max_seq slots
       │    └─ model(input_ids, position_ids, past_key_values=cache, logits_to_keep=...)
       └─ torch.export.export(step, example_inputs(...), strict=False)
            undecomposed: each nn.Linear is still one aten.linear, and so on

    decode (tokens [1,1], pos [1], k0, v0, ..., )        -> (logits [1,1,V], k0', v0', ...)
    prefill(tokens [1,T], pos [1], last [1], k0, v0, ...) -> (logits [1,1,V], k0', v0', ...)

export_classifier(model, (224, 224, 3))              export.py:154
  └─ Classify: forward(image [H,W,C]) = model(image.permute(2,0,1).unsqueeze(0))
       -> {'forward': ep}
```

### quantize (pipeline.py:230)

```
quantize(programs, 'w8a8', calibration)              frontend/quantization/quantize.py:45
  ├─ 1. Calibrator(schemes.W8A8).run (fx/calibrator.py) -> scales, per layer_key (op name +
  │    float weight storage): an fx.Interpreter runs the float prefill (or forward) program
  │    on the config's calibration: inputs (pipeline.calibration_inputs) and records the
  │    largest |input| of each node of calibrated_ops (linear, conv2d); then a Scales per
  │    layer: max(amax, eps) / weight_qmax per output channel of the float weight, and
  │    max(amax, eps) / activation_qmax as the [1] activation scale of a calibrated op
  ├─ 2. Quantizer(W8A8QuantizedLayer(), ops, scales).run (fx/quantizer/base.py) -> plan,
  │    over every program's selected nodes: layer.check each, then once per layer key the
  │    method named after the op (fx/quantizer/w8a8.py), linear / conv2d / embedding(weight,
  │    scales): the weight laid out for the op and rounded to INT8 with its scale, in a
  │    QuantizedOp (INT8 weight, weight scale, activation scale); shared by every program
  └─ 3. Rewriter(ops, plan, activation_qmax).run (fx/rewriter.py), per program:
       ├─ ep.module(), weights as get_attr
       ├─ b. insert_qdq_op, per node of the scheme's ops: plan[layer_key]'s tensors become
       │    buffers, the INT8 weight replaces the float one and, with an activation scale,
       │    x -> quant.quantize_per_tensor -> op(x_q, W_q, bias) -> quant.dequantize
       ├─ a. mismatch_type_handler, one rule per op: linear -> reshape to rows, conv2d ->
       │    pad + (ky, kx, c) im2col, before the quantize; the op -> torch._int_mm; the bias
       │    added after the dequantize; reshape / back to NCHW; embedding -> INT8 lookup
       │    times the row scale:
       │      quant.quantize_per_tensor -> _int_mm -> quant.dequantize -> + bias
       │      (the quant.* ops are custom ops, quantization/fx/custom_ops.py)
       ├─ Rewriter.finish: the float weights no node reads any more are dropped
       └─ torch.export.export(gm, ep.example_inputs)   traces those calls; each
                                                      quant.* op stays one node
```

The programs are still undecomposed ExportedPrograms; `ep.module()` runs one in eager PyTorch.

### to_mlir (pipeline.py:239)

```
to_mlir(programs, out/'ir')                          frontend/to_mlir/__init__.py:12
  └─ to_torch_dialect(programs)                      to_mlir/torch_mlir/to_torch_dialect.py:16
       ├─ FxImporter(context)
       ├─ ep.run_decompositions()   default ATen decompositions; _int_mm and the
       │                            quant.* custom ops survive (torch.operator)
       ├─ copy non-persistent buffers (RoPE inv_freq) from ep.constants into ep._state_dict
       └─ importer.import_frozen_program(ep, func_name=name)   once per program
            parameters and buffers become torch.vtensor.literal constants,
            one func.func per entry point: @decode, @prefill (or @forward)
  └─ write ir/torch.mlir   UTF-8, LF, because Linux LLVM parses it
```

### pynqo-middle-end (pipeline.py:245)

```
build_middle_end()      cmake + ninja in build/pynqo-middle-end; a no-op when unchanged
middle_end(torch.mlir, ir/model.mlir, devices)
  └─ pynqo-middle-end torch.mlir -o model.mlir --emit=linalg --devices cpu,npu
                      --delegates-dir ir/delegates            middle-end/pynqo_middle_end.cpp
       ├─ our rewrites on the torch dialect           middle-end/torch/transforms/
       │    ├─ DecomposeQuantOps   quant.quantize_per_tensor / quant.dequantize -> their ATen
       │    │                      arithmetic (torch-mlir cannot lower torch.operator)
       │    ├─ FuseConvBatchNorm   fold inference BatchNorm into the conv before it
       │    │                      (float: weight and bias; W8A8: w_scale and bias)
       │    ├─ DecomposeIntMm      _int_mm -> aten.mm on i8 (torch-mlir cannot lower _int_mm)
       │    └─ AnyDimToAnyDims     aten.any.dim -> aten.any.dims (same gap)
       ├─ torch-mlir's pipelines
       │    ├─ torch-function-to-torch-backend-pipeline
       │    └─ torch-backend-to-linalg-on-tensors-backend-pipeline
       ├─ pynqo-assign-devices    pynqo.device on each op a device claims   middle-end/linalg/transforms/
       │                          (the NPU: npu::isClaimable, backend/delegates/npu/lib/Match.cpp)
       ├─ pynqo-partition-devices each claimed op -> @npu_<id> in the device module,
       │                          a call to a bodiless @__delegate_<id> in its place
       ├─ device modules          -> ir/delegates/npu.mlir, removed from the graph
       └─ canonicalize, cse, linalg-fuse-elementwise-ops on the CPU graph
```

`model.mlir` (the CPU graph) and `delegates/npu.mlir` hold only upstream
dialects (linalg, tensor, arith, func) with the weights still inline as
constants. `--emit=torch` stops after our rewrites.

### pynqo-backend (pipeline.py:255)

```
build_backend()         cmake + ninja in build/pynqo-backend; a no-op when unchanged
pynqo-backend model.mlir --out <dir> --devices cpu,npu --cpus arm-cortex-a9,native
            --delegates-dir ir/delegates --rows 16 --columns 16 --max-k 256
                                                     backend/pynqo_backend.cpp
  ├─ read the entry-point signatures before any pass rewrites them
  ├─ registerDevices(devices)                        backend/device_register.cpp
  │    [cpu]: no delegates;  [cpu, npu]: createNpuDelegate()
  ├─ the delegate table from the @__delegate_<id> declarations: ids 0..n-1,
  │    device, input and output bytes (the subgraph's argument and result)
  ├─ npu delegate compile(ir/delegates/npu.mlir)     backend/delegates/npu/lower.cpp
  │    per @npu_<id> in id order:
  │    ├─ subgraph: the claimed linalg.matmul's M, K, N, constant weight and
  │    │    tk = min(K, MAX_K) (libnpu_rt sums the K chunks: M x N result)
  │    ├─ Pack:   its weight as B tiles, into the arena (weights.bin)
  │    └─ Encode: its ISA program, relocated against that weight block
  │    -> programs.bin, programs.txt, programs.json
  ├─ lowerDelegateCalls(module)                      backend/cpu/delegate_calls.cpp
  │    each call to @__delegate_<id> -> func.call @__delegate__(%id, %in, %out)
  │    (byte buffers in and out)
  ├─ externalizeWeights(module, arena)               backend/package.cpp
  │    every remaining large constant moves into the arena and is read back as
  │    memref.view %ws[offset]; ws becomes each entry point's first argument
  ├─ arena.write(weights.bin)
  ├─ for each CPU in --cpus: codegen(clone, target)   backend/cpu/toolchain.cpp
  │    bufferize -> loops -> LLVM dialect -> LLVM IR -> llc -> ld.lld
  │    -> <model>/model.so   (__delegate__ stays undefined for the runtime)
  └─ compile.json: entry points, abi, delegate table, weights, libraries, npu_limits
```

### manifest (pipeline.py:272)

```
manifest = compile.json + {format: pynqo-package/3, model, quantization, compile_seconds}
  ├─ copy framework(...).files(src), e.g. tokenizer.json
  ├─ llm:        entry_points.*.names = tokens, pos, (last,) k0, v0, ...
  │              manifest['llm'] = vocab, max_seq, chunk, cache shape, eos
  └─ classifier: entry_points.forward.names = image -> logits
                 manifest['classifier'] = input contract, classes
  -> manifest.json   (pipeline.py:297)
```

## 2. Run

### Loading the package

```
cli.cmd_run()                                        src/runtime/cli.py
  ├─ backend pynq (default on the board): pynq.Overlay(npu_accelerator.bit)
  └─ ExportedLLM(package, overlay)                   delegated.py:108
       └─ ExportedPackage(root, overlay)             delegated.py:60
            ├─ manifest['format'] must be pynqo-package/3
            ├─ Dispatcher()                          runtime/delegate/__init__.py
            │    builds libnpu_delegate.so from dispatch.c on first use and loads it
            │    with RTLD_GLOBAL, so model.so resolves __delegate__ against it
            ├─ NpuBackend(root, overlay)             runtime/delegate/npu.py
            │    ├─ builds libnpu_rt.so from vendors/npu_pynq/driver/npu_rt on first use
            │    ├─ board: check_overlay (ISA front end, rows/columns/max_k match),
            │    │         pynq.allocate the io, stage and code buffers
            │    └─ npu_rt_load(root, platform)   reads programs.*; platform None = simulator
            ├─ dispatcher.install(delegates)   id -> (npu_rt_execute, device index)
            ├─ ctypes.CDLL(<cpu model>/model.so)   picked by platform.machine()
            └─ np.memmap(weights.bin)              passed as ws on every call
       ├─ two KV cache sets (cache, spare), [1, kv_heads, max_seq, head_dim] each
       └─ Tokenizer(tokenizer.json), eos ids
```

The two native libraries are compiled on the machine that runs the package,
which on the board means the board's own compiler, the first time and after
any source change.

### generate

```
ExportedLLM.generate(prompt, max_new, params, chat, on_token)    delegated.py:163
  ├─ chat: tokenizer.chat([...]); then tokenizer.encode() -> ids
  ├─ reset(): zero the cache, pos = 0
  ├─ prefill(ids)                                    delegated.py:143
  │    └─ for each chunk of 16 ids (padded with zeros):
  │         _step('prefill', [tokens [1,16], pos, last])
  ├─ for each new token:
  │    ├─ sampler(logits, history) -> next id; stop at eos
  │    ├─ on_token(id, text) when streaming
  │    └─ decode(id) -> _step('decode', [tokens [1,1], pos])   delegated.py:156
  └─ return (ids, text, stats: prefill/decode seconds, tok/s, NPU counters)

_step(entry, inputs)                                 delegated.py:136
  ├─ package.call(entry, inputs + cache, [logits] + spare)
  │    └─ _mlir_ciface_<entry>(ws, *inputs, *outputs)   memref descriptors   delegated.py:88
  │         ├─ CPU code in model.so: embedding, RMSNorm, RoPE, attention, softmax, ...
  │         └─ at each claimed matmul: __delegate__(id, in, out)
  │              └─ dispatch.c -> npu_rt_execute(index, ...)
  │                   board: run program `index` on the NPU over MMIO
  │                   host:  the ISA simulator
  │                   then sum the K chunks into the M x N result
  ├─ dispatcher.check()   raise if a delegate failed
  └─ swap cache and spare: this call's output cache is the next call's input
```

ResNet-18 follows the same path through `DelegatedImageEngine`
(`delegated.py:199`): one `forward` call per normalized NHWC image, no cache.
