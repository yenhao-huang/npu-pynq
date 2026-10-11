優化 NPU 架構

開一個 issue 並且接這個 issue 開一個 worktree 基底是 (npu/issue119-a) 開發

參考 HC2023 Qualcomm Hexagon NPU.pdf 與 hexagon_npy.png 建立模組化異質 NPU，在既有模組上 (DMA/BUS=dma_axi、TCM=memory、Tenser unit=systolic array)
* 新增 L1/L2 cache、scalar unit、vector unit
* 加入 FP16、INT16 等資料寬度

驗收條件
* 每增加一個組件，都要撰寫專屬 testbench，通過 sim tool 功能驗證、run 過 PPA tool，比較與加入前的 PPA metrics (power、peformance、area) 
    * PPA metrics 寫在 goas/hexagon_npu/PPA/<modules>.md
* 撰寫 Report.md、Reproduce.md 寫在驗收目錄
* 當一個 agent 完後開啟另一個 subagent 檢查哪裡可以改善。這樣的 loop 持續 3 輪
* 能夠 deploy 在板子上，運行 smollm (http://192.168.2.99:9090/tree?)
* 等全部確認後，發 PR

Notes
* 驗收目錄寫在 goas/hexagon_npu/
* 新增的模組要模組化，不要耦合在一起


