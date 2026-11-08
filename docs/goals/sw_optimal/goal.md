goal; refactor 開一個 or 多個 worktree 優化目前的軟體 stack 包含
a. 優化 compiler stack
1. 能夠串接 MLIR+LLVM 的 compiler 鏈路 (參考 smoLM, qwen, resnet-18.pdf)
2. 自定義 NPU ISA，並且硬體上增加 IF、ID stage


編譯時
模型 → MLIR 優化 → 分配 CPU / NPU task
                  ├─ CPU task → LLVM IR → LLVM Arm backend → Cortex-A9 程式
                  └─ NPU task → NPU 指令編碼器 → 自訂 ISA 指令串

執行時
Cortex-A9 程式
  ├─ 自己執行 CPU task
  └─ 呼叫 ARM runtime，把 NPU 指令串和資料送給 FPGA
                                      ↓
                          FPGA 解碼指令 → NPU 執行
                                      ↓
                          結果傳回 ARM

b. 以 MLIR+LLVM 為引擎實現架設自己的推理框架，包含

支援模型: smoLM, qwen, resnet-18

LLM 模型架構 
1. 基本: tied embedding、RMSNorm、RoPE、GQA/MQA、attention、MLP
    1. 不同 attention 實作模式 
2. 進階: MoE、Linear attention 

推理框架的技術
1. 讀權重 
2. Tokenizer 
3. prefill 和 decode 分開 
4. KV cache 
5. Sampling 

量化
- uniform quantization: INT8、INT4
- AWQ
- Quantization in KV cache



Notes;
1. 盡量讓目錄具有解耦性，比如不同模型獨立開來，只以算子為單位共用
2. 開一個 worktree 開發，並將中途的實驗放在 <worktree>/exp/ 內


驗收：
* moLM, qwen, resnet-18 能夠成功 export 並且能夠在 mac and npu 上執行
* 最終整理 1 個主要 worktees 包含所有子 worktree 的修改 
    * 何時會有子 worktee? 當做到一個重要部分分裂成一個 worktree 然後發 pr
* 將最終結果 Rresults.md 與 Reproduce.md 到主要 worktees/docs/goals/sw_optimal/, 並且 RESULTS 只放重要事項。細節連結到 worktees/docs/goals/sw_optimal/details/
