npu-issue119-a 開個 worktree 
1. 將 src/deprecate 改成 src/jit 體系 (just in time)
2. 將需要 src/deprecate 的檔案整理出來 move 到 src/jit/ 裡面, e.g., examples/resnet18-> src/jit/examples
3. 目前的 src/export 與 src/runtime 要能支援 examples/resnet18 

驗收條件
1. examples/resnet18 : 目前的 src/export 與 src/runtime (ahead-of-time 的模式)
2. resnet18.ipynb 的 block 邏輯要保留 一樣輸入圖片預測
3. 實驗 紀錄不同 block 總時間。
4. 加  pynq 板子上成功執行 resnet-15