goal: 實現 paper-based eda-tool-exploration agent 能夠讓目前的 ic-tools 更深入

1. 幫我自行 survey 論文 (iccad、dac 等)
2. 找尋與 AI-assisted RTL  Optimization 相關的論文
3. 把想法萃取成工具
4. 實作 tools
5. 驗證
6. 提出 PR

Notes:
1. 設計的 tools 要具有耦合性
依據類別到不同種類 tools/ lint、sim、debug、synth
如果是一些 pipline 則放在 tools/pipeline/<論文名稱縮寫>

2. 實驗要放在 exp/tool-exploration/exp-tool-<tool-id>-<論文名稱縮寫>/ 裡。並且依然具有 module 性

3. 從 dev branch 開 worktree 出來開發

驗收/停止條件
1. 至少要 explore 10 個 tools 至少 8 個跟 PPA 優化有關
2. 成功後整理 report.md (包含這些 tool 作用與實驗成果, e.g., PPA 優化比例)、reproduce.md 如何復現這些 tools 的實驗成果、將 workflow、notes、驗收條件整理到 skills/eda-tool-exploration/
skills 當中某些變成互動式 question 詢問 user
a. explore N 個 tools, 特別注重在那些 topic (e.g., PPA)
b. 實驗的路徑
3. 當 codex usage 低於 30% 不管上面條件直接暫停
