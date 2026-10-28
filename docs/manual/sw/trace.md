
```
source /Users/yenhaohuang/Desktop/side_project/npu/worktrees/.venv-sw/bin/activate
cd /Users/yenhaohuang/Desktop/side_project/npu/worktrees/npu-pynq-issue119-a

# Export SmolLM2；預設同時產生 host 與 PYNQ-Z1 版本
python -m src.export.cli export smollm --out build/pkg/smollm --awq

# 在主機執行，NPU 部分使用 C ISA 模擬器
python -m src.runtime.cli run build/pkg/smollm \
  --backend sim --prompt "What is the capital of France?" --chat --stream
```

### export 產生的 package
```
build/pkg/smollm/
├─ manifest.json          套件索引：模型、入口函式、library 路徑、權重布局
├─ weights.bin            所有權重的實際 bytes
├─ programs.bin           NPU ISA 指令，連續的 64-bit words
├─ programs.json          NPU 任務、segments、權重位置與位址 relocation
├─ partition.json         CPU/NPU 分工報告
├─ tokenizer.json         SmolLM 的 tokenizer；由 graph.files 複製
├─ host/
│  └─ model.dylib         macOS 主機版；Linux 上會是 model.so
├─ pynq/
│  └─ model.so            Cortex-A9 版，含編譯出的 CPU 計算與 C runtime
└─ ir/                    預設保留，方便檢查編譯過程
   ├─ model.mlir          原始計算圖
   ├─ partitioned.mlir    CPU/NPU 分工後：host subgraph + npu.module（NPU subgraph）
   ├─ npu.mlir            NPU subgraph（只剩 npu.module），由此產生 programs
   ├─ host.mlir           host subgraph，npu.call 改成 runtime 呼叫後
   ├─ weights.json        編譯時使用的權重 offset 表
   ├─ host/               主機版的 LLVM 中間產物
   └─ pynq/               Cortex-A9 版的 LLVM 中間產物
```

## Detail
cli.py
```
<cmd_export>: export package
build -> export

cli.cmd_export()
  └─ load("smollm")
       └─ import src.export.graphs.smollm.model #選 frontend 模型 
  └─ smollm.model.build(opts=LLMOptions(...))
       └─ 回傳 (graph, cfg)
  └─ export(graph, ...)


<cmd_run>: run
cli.cmd_run()
  ├─ LLMEngine(package, backend)
  │    ├─ 載入編譯好的套件
  │    └─ 準備 tokenizer 與 KV cache
  ├─ engine.generate(prompt, ...)
  │    ├─ 將 prompt 轉成 token
  │    ├─ prefill() → 處理 prompt
  │    ├─ decode() → 逐步產生新 token
  │    └─ 回傳文字與統計
  └─ 顯示結果
```

### 1. Export Package
#### Build
build: 1. 讀模型資料 (config、weight) 2. 準備常數和權重 3. 建立計算圖 (包含 prefill graph 與 decode graph)
```
build(opts=LLMOptions(...))
      ├─ 找模型檔、讀 config.json 和 safetensors
      ├─ 建立空的 ModelGraph (line#65new_graph)
      ├─ 準備每層的量化權重: embedding、RoPE 表 (line#75 quantize_layer_weights)
      ├─ 定義一個「模型前向計算」body()
      ├─ build_entry_points() 呼叫 body() 兩次
      │    ├─ 產生 decode 函式，輸入 1 個 token
      │    └─ 產生 prefill 函式，輸入固定 chunk 個 token
      └─ 回傳 (graph, cfg)
```

#### Export
export: 1. 原始計算圖建立 MLIR 2. 決定 CPU/NPU 分工 3. 打包權重 4. 產生 NPU 指令 5. 產生 CPU 指令
```
export(graph, out, targets=("host", "pynq")) # src/export/cli.py
      ├─ 輸出原始計算圖 (line 105–106)
      │    └─ graph.module.render() → ir/model.mlir
      ├─ 決定 CPU / NPU 分工 (line 107–110)
      │    └─ 挑出 NPU 能做的矩陣乘法，改成 npu.matmul (call src/vendors/npu_pynq/compiler/mlir/tools/npu-opt.cpp)
      │    ├─ 每個 npu.matmul 外提 (outline) 到 npu.module @npu（NPU subgraph），
      │    │  原位置換成 npu.call @npu::@task_N（host subgraph）
      │    └─ npu-partition → ir/partitioned.mlir、partition.json
      ├─ 打包權重
      │    ├─ NPU 使用的權重：沿 K、N 切成 tiles，每個 tile 內按 row-major 排放
      │    ├─ 其他權重：維持原本的 dense row-major 排放
      │    └─ 產生 weights.bin、ir/weights.json
      ├─ 兩個 subgraph 各自 lower（同時進行）
      │    ├─ NPU subgraph：npu-encode 將每個 task 編成 ISA
      │    │    └─ 產生 ir/npu.mlir、programs.bin、programs.json
      │    └─ host subgraph：npu-lower-host 將 npu.call 換成 call @npu_rt_gemm(task id, ...)
      │         ├─ 產生 ir/host.mlir
      │         ├─ host → host/model.dylib 或 model.so
      │         └─ PYNQ → pynq/model.so
      ├─ 複製 graph.files 中的附帶檔案，例如 tokenizer.json (line 137–138)
      ├─ 寫入 manifest.json (line 139–157)
      └─ 回傳套件目錄 out (line 164)
```

目前的建立 graph 架構

```python
computation graph
     |
operator (src/export/compilers/mlir/ops/)
     |
MLIR 組語 (arith.addf、mulf、divf)
```




### 2. Run
```
LLMEngine.generate(prompt, ...) # src/runtime/core/engine.py
      ├─ 建立 Sampler，套用溫度、top-k 等取樣設定 (line 104–105)
      ├─ 準備 prompt (line 106–110)
      │    ├─ chat=True：先套對話格式
      │    └─ tokenizer.encode() → prompt token IDs
      ├─ reset()：清空 KV cache，將位置歸零 (line 111)
      ├─ 處理 prompt (line 114–120)
      │    ├─ use_chunked_prefill=True
      │    │    ├─ 每 chunk 個 token 分一段 (line 83–86)
      │    │    └─ model.call("prefill", ...) → 更新 KV cache、取得 logits (line 87–90)
      │    └─ use_chunked_prefill=False：改為逐 token 呼叫 decode()
      ├─ 逐步生成新 token (line 124–134)
      │    ├─ sampler(logits, history) → 選出下一個 token
      │    ├─ 遇到結束 token 就停止
      │    ├─ 記錄 token；若有 on_token callback，立即輸出
      │    └─ decode(token)
      │         └─ model.call("decode", ...) → 更新 KV cache、取得下一組 logits (line 92–97)
      ├─ 計算 prefill、decode 時間與 NPU 使用統計 (line 120、135–138)
      └─ tokenizer.decode(out) → 回傳 (token IDs, 文字, 統計) (line 139–140)
```
